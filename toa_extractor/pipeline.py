import copy

import luigi
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import log
from astropy.table import Table
from hendrics.ml_timing import ml_pulsefit, normalized_template

from .data_setup import (
    GetInfo,
    GetPhaseogram,
    GetTemplate,
    _get_and_normalize_phaseogram,
    _plot_phaseogram,
)
from .utils import encode_image_file, output_name, search_substring_in_list
from .utils.config import load_yaml_file
from .utils.fit_crab_profiles import (
    _plot_profile_and_fit,
    create_template_from_profile_table,
    default_crab_model,
)


def plot_complete_diagnostics(
    phaseogram_files, model_fit, phase_max=None, model_init=None, output_fname=None
):
    """Make a pretty plot to show how the fit went."""

    n_phas_rows = len(phaseogram_files)

    phaseograms = []
    for phaseogram in phaseogram_files:
        phaseograms.append(Table.read(phaseogram))

    phases = phaseograms[0].meta["phase"]
    profile = 0.0
    profile_raw = 0.0
    durations = []
    for phaseogram_table in phaseograms:
        phaseogram = phaseogram_table["profile"]
        local_profile = np.sum(phaseogram, axis=0)
        profile_raw += local_profile
        if "expo" in phaseogram_table.meta:
            expo = phaseogram_table.meta["expo"]
            local_profile = local_profile / expo
        durations.append(phaseogram_table.meta["mjdstop"] - phaseogram_table.meta["mjdstart"])
        profile += local_profile

    durations = np.array(durations)
    if phase_max is None:
        phase_max = 0

    if np.min(phases) >= 0 and np.max(phases) <= 1:
        phases = np.concatenate([phases - 1, phases, phases + 1])
        profile = np.concatenate([profile, profile, profile])
        if profile_raw is not None:
            profile_raw = np.concatenate([profile_raw, profile_raw, profile_raw])

    # External GridSpec
    if n_phas_rows < 3:
        height_ratios = [0.65, 0.35] + [1] * n_phas_rows
        hspace = 0.1
    else:
        max_height = 1 * np.log2(n_phas_rows) / n_phas_rows
        height_scale = durations / durations.max() * max_height
        height_ratios = [0.65, 0.35] + list(height_scale)
        hspace = 0

    figure_width = 10
    figure_height = 3 * np.sum(height_ratios)

    fig = plt.figure(figsize=(figure_width, figure_height))
    padding_border = 0.001
    padding_width = 0.075
    rescale_height = figure_width / figure_height

    gs_external = fig.add_gridspec(
        2 + n_phas_rows,
        3,
        width_ratios=[2, 1, 1],
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=0.05,
    )
    plt.subplots_adjust(
        left=padding_width,
        right=1 - padding_border,
        top=1 - padding_border * rescale_height,
        bottom=padding_width * rescale_height,
    )

    axes_full_profile = [fig.add_subplot(gs_external[i, 0]) for i in range(2)]
    axes_zoom_peak1 = [fig.add_subplot(gs_external[i, 1]) for i in range(2)]
    axes_zoom_peak2 = [fig.add_subplot(gs_external[i, 2]) for i in range(2)]

    axes_all_profiles = [axes_full_profile, axes_zoom_peak1, axes_zoom_peak2]
    axes_all_phaseograms = [
        [fig.add_subplot(gs_external[i + 2, j], sharex=axes_all_profiles[j][0]) for j in range(3)]
        for i in range(n_phas_rows)
    ]

    for phaseogram, axrow in zip(phaseogram_files, axes_all_phaseograms):
        local_phases, time_hrs, phas, meantime_mjd = _get_and_normalize_phaseogram(phaseogram)
        _plot_phaseogram(
            local_phases,
            time_hrs,
            phas,
            meantime_mjd,
            axrow[0],
            title=None,
            label_y=n_phas_rows < 3,
        )
        for ax in axrow[1:]:
            _plot_phaseogram(
                local_phases,
                time_hrs,
                phas,
                meantime_mjd,
                ax,
                title=None,
                label_y=False,
            )

    for axpair in [axes_full_profile, axes_zoom_peak1, axes_zoom_peak2]:
        axfit, axres = axpair[0], axpair[1]

        _plot_profile_and_fit(
            phases,
            profile,
            model_fit,
            axfit,
            axres,
            model_init=model_init,
            phase_max=phase_max,
            profile_raw=profile_raw,
        )

    for ax in axes_full_profile:
        ax.set_xlim([-0.5, 0.5])
    for ax in axes_zoom_peak1:
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlim([phase_max - 0.125, phase_max + 0.125])

    for ax in axes_zoom_peak2:
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlim([phase_max + 0.25, phase_max + 0.5])

    for ax in [axes_full_profile[1], axes_zoom_peak1[1], axes_zoom_peak2[1]]:
        plt.setp(ax.get_xticklabels(), visible=False)

    if n_phas_rows > 2:
        for ax_row in axes_all_phaseograms[:-1]:
            for ax in ax_row:
                ax.xaxis.label.set_visible(False)
                plt.setp(ax.get_xticklabels(), visible=False)

    for ax in axes_all_phaseograms[-1]:
        ax.xaxis.label.set_visible(True)
        ax.set_xlabel("Pulse Phase")
        ax.axvline(0, ls="--", color="k")

    for ax in [axes_full_profile[0], axes_zoom_peak1[0], axes_zoom_peak2[0]]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axes_full_profile[0].set_ylabel("Counts")
    axes_full_profile[1].set_ylabel("Residuals")
    axes_zoom_peak2[0].legend()

    if output_fname is not None:
        plt.savefig(output_fname, dpi=150)
        plt.close(fig)
    else:
        plt.show()


class TOAPipeline(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield PlotDiagnostics(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetProfileFit(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetPhaseogram(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_results.txt"))

    def run(self):
        residual_file = (
            GetResidual(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        image_files_list = (
            PlotDiagnostics(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        profile_fit_file = (
            GetProfileFit(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        image_files = list(filter(None, open(image_files_list, "r").read().splitlines()))
        image_file = image_files[0]

        residual_dict = load_yaml_file(residual_file)
        profile_fit_table = Table.read(profile_fit_file)
        # best_freq_table = Table.read(best_freq_file)
        residual_dict["phase_max"] = profile_fit_table.meta["phase_max"]
        residual_dict["phase_max_err"] = profile_fit_table.meta["phase_max_err"]
        residual_dict["fit_residual"] = (
            profile_fit_table.meta["phase_max"] / profile_fit_table.meta["F0"]
        )
        residual_dict["fit_residual_err"] = (
            profile_fit_table.meta["phase_max_err"] / profile_fit_table.meta["F0"]
        )

        outfile = self.output().path.replace(".txt", ".yaml")

        output_files = [outfile]

        if "profile_1" in profile_fit_table.colnames:
            for col in profile_fit_table.colnames:
                if not col.startswith("profile_") or "raw" in col:
                    continue
                label = col.replace("profile", "")
                meta = profile_fit_table[col].meta
                local_residual_dict = copy.deepcopy(residual_dict)
                local_residual_dict.update(meta)
                local_residual_dict["fit_residual"] = meta["phase_max"] / meta["F0"]
                local_residual_dict["fit_residual_err"] = meta["phase_max_err"] / meta["F0"]
                local_image_file = search_substring_in_list(label + ".jpg", image_files)[0]
                local_residual_dict["img"] = encode_image_file(local_image_file)
                local_outfile = self.output().path.replace(".txt", f"{label}.yaml")

                with open(local_outfile, "w") as f:
                    yaml.dump(local_residual_dict, f)

                output_files.append(local_outfile)

        residual_dict["img"] = encode_image_file(image_file)
        with open(outfile, "w") as f:
            yaml.dump(residual_dict, f)

        with open(self.output().path, "w") as f:
            for outfile in output_files:
                print(outfile, file=f)


class PlotDiagnostics(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield GetResidual(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetFoldedProfile(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetProfileFit(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetPhaseogram(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_diagnostics.txt"))

    def run(self):
        profile_fit_file = (
            GetProfileFit(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        phaseogram_file = (
            GetPhaseogram(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        profile_fit_table = Table.read(profile_fit_file)

        init_model = default_crab_model(init_pars=profile_fit_table.meta["model_init"])
        best_fit_model = default_crab_model(init_pars=profile_fit_table.meta["best_fit"])
        phaseograms = open(phaseogram_file).read().splitlines()

        root = self.output().path.replace(".txt", "")
        outfile = root + ".jpg"
        plot_complete_diagnostics(
            phaseograms,
            model_fit=best_fit_model,
            model_init=init_model,
            output_fname=outfile,
        )
        outfiles = [outfile]
        if "profile_1" in profile_fit_table.colnames:
            for col in profile_fit_table.colnames:
                if not col.startswith("profile_") or "raw" in col:
                    continue

                label = col.replace("profile", "")

                init_model = default_crab_model(init_pars=profile_fit_table[col].meta["model_init"])
                best_fit_model = default_crab_model(
                    init_pars=profile_fit_table[col].meta["best_fit"]
                )
                local_phaseograms = search_substring_in_list(
                    label + ".hdf5",
                    phaseograms,
                )

                outfile = root + label + ".jpg"
                plot_complete_diagnostics(
                    local_phaseograms,
                    model_fit=best_fit_model,
                    model_init=init_model,
                    output_fname=outfile,
                )
                outfiles.append(outfile)

        with open(self.output().path, "w") as f:
            for outfile in outfiles:
                print(outfile, file=f)


class GetResidual(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetFoldedProfile(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_residual.yaml"))

    def run(self):
        prof_file = (
            GetFoldedProfile(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        prof_table = Table.read(prof_file)
        template_file = (
            GetTemplate(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )

        template_table = Table.read(template_file, format="ascii.ecsv")

        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout).output().path
        )
        info = load_yaml_file(infofile)

        prof = prof_table["profile"]
        template = template_table["profile"]
        template = normalized_template(template_table["profile"], tomax=False)

        pars, errs = ml_pulsefit(prof, template, calculate_errors=True, fit_base=True)

        phase_res, phase_res_err = pars[1], errs[1]

        output = {}
        output.update(info)
        for key, val in prof_table.meta.items():
            if key not in output:
                try:
                    output[key] = float(val)
                except Exception:
                    output[key] = val
        output["residual"] = float(phase_res / prof_table.meta["F0"])
        output["residual_err"] = float(phase_res_err / prof_table.meta["F0"])

        with open(self.output().path, "w") as f:
            yaml.dump(output, f)


class GetProfileFit(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetFoldedProfile(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_fit_template.hdf5"))

    def run(self):
        prof_file = (
            GetFoldedProfile(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        out = self.output().path
        create_template_from_profile_table(
            prof_file,
            output_template_fname=out,
            nbins=512,
        )


class GetFoldedProfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield GetPhaseogram(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_folded.hdf5"))

    def run(self):
        phaseogram_file = (
            GetPhaseogram(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        phaseogram_files = open(phaseogram_file).read().splitlines()

        phaseograms = []
        for phaseogram in phaseogram_files:
            phaseograms.append(Table.read(phaseogram))

        phases = phaseograms[0].meta["phase"]
        profile = 0.0
        profile_raw = 0.0

        result_table = Table()
        for subprofile_count, phaseogram_table in enumerate(phaseograms):
            phaseogram = phaseogram_table["profile"]
            local_profile_raw = np.sum(phaseogram, axis=0)
            profile_raw += local_profile_raw
            if "expo" in phaseogram_table.meta:
                expo = phaseogram_table.meta["expo"]
                local_profile = local_profile_raw / expo
            else:
                local_profile = local_profile_raw

            if "phase" not in result_table.colnames:
                result_table["phase"] = phases
                result_table.meta.update(phaseogram_table.meta)

            result_table[f"profile_{subprofile_count}"] = local_profile
            result_table[f"profile_raw_{subprofile_count}"] = local_profile_raw
            result_table[f"profile_{subprofile_count}"].meta.update(phaseogram_table.meta)

            profile += local_profile

        if subprofile_count == 0:
            log.info("Only one profile found in the observation")
            result_table.rename_column("profile_0", "profile")
        else:
            log.info("Creating summed profile")
            result_table["profile"] = np.sum(
                [result_table[f"profile_{i}"] for i in range(len(phaseograms))], axis=0
            )
            result_table["profile_raw"] = np.sum(
                [result_table[f"profile_raw_{i}"] for i in range(len(phaseograms))],
                axis=0,
            )

        result_table.write(self.output().path, serialize_meta=True)


def get_outputs(task):
    """Get all outputs from a given luigi task and his dependencies."""
    outputs = []
    local_require = task

    while 1:
        try:
            outputs.append(local_require.output().path)
            local_require = local_require.requires()
        except AttributeError:
            break

    return outputs


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("files", help="Input binary files", type=str, nargs="+")
    parser.add_argument("--config", help="Config file", type=str, default="none")
    parser.add_argument("-v", "--version", help="Version", type=str, default="none")
    parser.add_argument(
        "-N",
        "--nmax",
        help="Maximum number of data files from a given directory",
        type=int,
        default=None,
    )

    args = parser.parse_args(args)

    config_file = args.config

    import os
    import random

    fnames = args.files
    if args.nmax is not None:
        log.info(f"Analyzing only {args.nmax} files per directory, chosen randomly")
        dirs = list(set([os.path.split(fname)[0] for fname in args.files]))
        fnames = []
        for d in dirs:
            good_files = [f for f in args.files if f.startswith(d)]
            if len(good_files) > args.nmax:
                good_files = random.sample(good_files, k=args.nmax)
            fnames += good_files

    _ = luigi.build(
        [TOAPipeline(fname, config_file, args.version) for fname in fnames],
        local_scheduler=True,
        log_level="INFO",
        workers=4,
    )
