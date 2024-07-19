import numpy as np
import luigi
import yaml
from astropy import log
from astropy.table import Table

from stingray.pulse.pulsar import get_model
from hendrics.ml_timing import ml_pulsefit, normalized_template

from pulse_deadtime_fix.core import _create_weights

import matplotlib.pyplot as plt

from .utils import output_name, encode_image_file
from .utils.data_manipulation import get_events_from_fits
from .utils.config import load_yaml_file
from .utils.fold import calculate_profile, get_phase_func_from_ephemeris_file
from .utils.fit_crab_profiles import create_template_from_profile_table
from .data_setup import GetInfo, GetParfile, GetTemplate, GetPulseFreq, PlotPhaseogram


class TOAPipeline(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield PlotDiagnostics(
            self.fname, self.config_file, self.version, self.worker_timeout
        )
        yield GetProfileFit(
            self.fname, self.config_file, self.version, self.worker_timeout
        )
        yield GetPulseFreq(
            self.fname, self.config_file, self.version, self.worker_timeout
        )
        yield PlotPhaseogram(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_results.yaml"))

    def run(self):
        residual_file = (
            GetResidual(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        profile_fit_file = (
            GetProfileFit(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
            .output()
            .path
        )
        best_freq_file = (
            GetPulseFreq(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
            .output()
            .path
        )
        residual_dict = load_yaml_file(residual_file)
        profile_fit_table = Table.read(profile_fit_file)
        best_freq_table = Table.read(best_freq_file)
        residual_dict["phase_max"] = profile_fit_table.meta["phase_max"]
        residual_dict["phase_max_err"] = profile_fit_table.meta["phase_max_err"]
        residual_dict["fit_residual"] = (
            profile_fit_table.meta["phase_max"] / profile_fit_table.meta["F0"]
        )
        residual_dict["fit_residual_err"] = (
            profile_fit_table.meta["phase_max_err"] / profile_fit_table.meta["F0"]
        )

        image_file = output_name(self.fname, self.version, "_fit_diagnostics.jpg")
        phas_image_file = (
            PlotPhaseogram(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
            .output()
            .path
        )
        residual_dict["img"] = encode_image_file(image_file)
        residual_dict["phas_img"] = encode_image_file(phas_image_file)

        residual_dict["local_best_freq"] = float(best_freq_table["f"][0])
        residual_dict["local_best_freq_err_n"] = float(best_freq_table["f_err_n"][0])
        residual_dict["local_best_freq_err_p"] = float(best_freq_table["f_err_p"][0])
        residual_dict["initial_freq_estimate"] = float(
            best_freq_table["initial_freq_estimate"][0]
        )

        with open(self.output().path, "w") as f:
            yaml.dump(residual_dict, f)


class PlotDiagnostics(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetResidual(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(
            output_name(self.fname, self.version, "_diagnostics.jpg")
        )

    def run(self):
        prof_file = (
            GetFoldedProfile(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
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

        residual_file = (
            GetResidual(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        residual_dict = load_yaml_file(residual_file)

        fig = plt.figure()
        pphase = prof_table["phase"]
        pphase = np.concatenate([pphase - 2, pphase - 1, pphase, pphase + 1])

        def normalize_profile(ref_profile):
            ref_std_prof = np.std(np.diff(ref_profile)) / 1.4
            ref_min = np.median(
                ref_profile[ref_profile < ref_profile.min() + 3 * ref_std_prof]
            )
            ref_max = np.median(
                ref_profile[ref_profile > ref_profile.max() - 3 * ref_std_prof]
            )
            prof = ref_profile - ref_min
            prof /= ref_max - ref_min

            return prof

        prof = normalize_profile(prof_table["profile"])
        prof = np.concatenate([prof, prof, prof, prof])
        prof_raw = None
        if "profile_raw" in prof_table.colnames:
            prof_raw = normalize_profile(prof_table["profile_raw"])
            prof_raw = np.concatenate([prof_raw, prof_raw, prof_raw, prof_raw])

        tphase = pphase / prof_table.meta["F0"]

        temp = template_table["profile"] - template_table["profile"].min()
        temp = temp / temp.max()
        temp = np.concatenate([temp, temp, temp, temp])

        minres = residual_dict["residual"] - residual_dict["residual_err"]
        maxres = residual_dict["residual"] + residual_dict["residual_err"]

        main_ax = plt.gca()
        # inset Axes....
        x1, x2, y1, y2 = (
            residual_dict["residual"] - 2e-3,
            residual_dict["residual"] + 2e-3,
            -0.1,
            1.1,
        )  # subregion of the original image
        axins = main_ax.inset_axes(
            [0.65, 0.03, 0.47, 0.94],
            xlim=(x1, x2),
            ylim=(y1, y2),
            xticklabels=[],
            yticklabels=[],
            zorder=20,
        )
        axins.tick_params(axis="x", direction="in", pad=-15)
        import matplotlib.ticker as ticker

        ticks = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x * 1000))
        axins.xaxis.set_major_formatter(ticks)
        axins.set_xlabel("Residual (ms)", labelpad=-30)
        main_ax.indicate_inset_zoom(axins, edgecolor="black")
        if prof_raw is not None:
            main_ax.plot(
                tphase,
                prof_raw,
                color="red",
                alpha=0.5,
                zorder=0,
                ds="steps-mid",
            )

        for ax in main_ax, axins:
            ax.plot(tphase, prof, color="red", ds="steps-mid")
            ax.plot(
                tphase,
                temp,
                color="grey",
                zorder=0,
                lw=1,
                alpha=0.5,
            )
            ax.plot(
                tphase + residual_dict["residual"],
                temp,
                color="k",
                zorder=10,
                lw=2,
            )

            ax.axvspan(minres, maxres, color="#9999dd")
        # main_ax.axvline(0, color="k")
        axins.axvline(0, color="k", alpha=0.5, ls=":")
        main_ax.set_xlabel("Time (s)")
        main_ax.set_ylabel("Flux (arbitrary units)")
        main_ax.set_xlim(
            -1 / prof_table.meta["F0"] + 1.5e-2, 1 / prof_table.meta["F0"] + 1.5e-2
        )

        plt.tight_layout()
        plt.savefig(self.output().path, dpi=100)

        plt.close(fig)


class GetResidual(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetFoldedProfile(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(
            output_name(self.fname, self.version, "_residual.yaml")
        )

    def run(self):
        prof_file = (
            GetFoldedProfile(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
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
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
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
        return GetFoldedProfile(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(
            output_name(self.fname, self.version, "_fit_template.hdf5")
        )

    def run(self):
        prof_file = (
            GetFoldedProfile(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
            .output()
            .path
        )
        out = self.output().path
        create_template_from_profile_table(
            prof_file,
            output_template_fname=out,
            plot=True,
            plot_file=output_name(self.fname, self.version, "_fit_diagnostics.jpg"),
            nbins=512,
        )


class GetFoldedProfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield GetParfile(
            self.fname, self.config_file, self.version, self.worker_timeout
        )
        yield GetTemplate(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_folded.hdf5"))

    def run(self):
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        info = load_yaml_file(infofile)
        events = get_events_from_fits(self.fname)
        ephem = info["ephem"]
        mjdstart, mjdstop = info["mjdstart"], info["mjdstop"]
        parfile_list = (
            GetParfile(
                self.fname,
                self.config_file,
                self.version,
                worker_timeout=self.worker_timeout,
            )
            .output()
            .path
        )
        # Read list of file ignoring blank lines
        parfiles = list(filter(None, open(parfile_list, "r").read().splitlines()))
        models = [get_model(p) for p in parfiles]
        epochs = np.asarray([m.PEPOCH.value for m in models])
        mjd_edges = [mjdstart, mjdstop]
        if len(set(epochs)) > 1:
            additional_mjd_edges = (epochs[1:] + epochs[:-1]) / 2
            mjd_edges = np.concatenate([[mjdstart], additional_mjd_edges, [mjdstop]])

        mjds = events.time / 86400 + events.mjdref
        nbin = 512
        edge_idxs = np.searchsorted(mjds, mjd_edges)

        result_table = Table()
        for i, (mjdstart, mjdstop) in enumerate(zip(mjd_edges[:-1], mjd_edges[1:])):
            correction_fun = get_phase_func_from_ephemeris_file(
                mjdstart,
                mjdstop,
                parfiles[i],
                ephem=ephem,
                return_sec_from_mjdstart=True,
            )

            good = slice(edge_idxs[i], edge_idxs[i + 1])
            times_from_mjdstart = events.time[good] - (mjdstart - events.mjdref) * 86400
            phase = correction_fun(times_from_mjdstart)

            expo = None
            if hasattr(events, "prior") and (
                np.any(events.prior != 0) or np.any(np.isnan(events.prior))
            ):
                phases_livetime_start = correction_fun(
                    times_from_mjdstart - events.prior[good]
                )
                phase_edges = np.linspace(0, 1, nbin + 1)
                weights = _create_weights(
                    phases_livetime_start.astype(float),
                    phase.astype(float),
                    phase_edges,
                )
                expo = 1 / weights

            phase -= np.floor(phase)

            model = get_model(parfiles[i])
            table = calculate_profile(phase, nbin=nbin, expo=expo)
            if "phase" not in result_table.colnames:
                result_table["phase"] = table["phase"]
                result_table.meta["F0"] = model.F0.value
            result_table[f"profile_{i}"] = table["profile"]
            result_table[f"profile_raw_{i}"] = table["profile_raw"]

            result_table[f"profile_{i}"].meta["F0"] = model.F0.value
            result_table[f"profile_{i}"].meta["F1"] = model.F1.value
            result_table[f"profile_{i}"].meta["F2"] = model.F2.value

        result_table["profile"] = np.sum(
            [result_table[f"profile_{i}"] for i in range(len(parfiles))], axis=0
        )
        result_table["profile_raw"] = np.sum(
            [result_table[f"profile_raw_{i}"] for i in range(len(parfiles))], axis=0
        )

        result_table.write(self.output().path)


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
