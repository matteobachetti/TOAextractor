import io
import base64
import shutil
import numpy as np
import luigi
import yaml
import warnings
from astropy.table import Table
from stingray.pulse.pulsar import get_model
from hendrics.ml_timing import ml_pulsefit
from pulse_deadtime_fix.core import _create_weights
from PIL import Image

import matplotlib.pyplot as plt
from .utils.crab import get_crab_ephemeris
from .utils import output_name
from .utils.data_manipulation import get_observing_info, get_events_from_fits
from .utils.config import get_template, load_yaml_file
from .utils.fold import calculate_profile, get_phase_func_from_ephemeris_file


class TOAPipeline(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return PlotDiagnostics(
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
        residual_dict = load_yaml_file(residual_file)
        image_file = (
            PlotDiagnostics(
                self.fname, self.config_file, self.version, self.worker_timeout
            )
            .output()
            .path
        )

        foo = Image.open(image_file)
        # Get image file
        image_file = open(image_file, "rb")

        foo = foo.resize((128, 96), Image.LANCZOS)

        # From https://stackoverflow.com/questions/42503995/how-to-get-a-pil-image-as-a-base64-encoded-string
        in_mem_file = io.BytesIO()
        foo.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()

        base64_encoded_result_bytes = base64.b64encode(img_bytes)
        base64_encoded_result_str = base64_encoded_result_bytes.decode("ascii")

        residual_dict["img"] = base64_encoded_result_str

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
        pphase = np.concatenate([pphase - 1, pphase])

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
        prof = np.concatenate([prof, prof])
        prof_raw = None
        if "profile_raw" in prof_table.colnames:
            prof_raw = normalize_profile(prof_table["profile_raw"])
            prof_raw = np.concatenate([prof_raw, prof_raw])

        tphase = template_table["phase"]
        tphase = np.concatenate([tphase - 1, tphase])
        temp = template_table["profile"] - template_table["profile"].min()
        temp = temp / temp.max()
        temp = np.concatenate([temp, temp])

        plt.plot(pphase / prof_table.meta["F0"], prof, color="red")
        plt.plot(tphase / prof_table.meta["F0"], temp, color="k", zorder=1)
        if prof_raw is not None:
            plt.plot(
                pphase / prof_table.meta["F0"],
                prof_raw,
                color="red",
                alpha=0.5,
                zorder=0,
            )

        minres = residual_dict["residual"] - residual_dict["residual_err"]
        maxres = residual_dict["residual"] + residual_dict["residual_err"]
        plt.axvspan(minres, maxres, color="#aaaaff")
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (arbitrary units)")
        plt.xlim(-1 / prof_table.meta["F0"], 1 / prof_table.meta["F0"])

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

        prof = prof_table["profile"]
        template = template_table["profile"]
        # mean_amp, std_amp, phase_res, phase_res_err = fftfit(prof, template=template)

        pars, errs = ml_pulsefit(prof, template, calculate_errors=True, fit_base=True)
        phase_res, phase_res_err = pars[1], errs[1]

        output = {}
        output.update(prof_table.meta)
        output["residual"] = float(phase_res / prof_table.meta["F0"])
        output["residual_err"] = float(phase_res_err / prof_table.meta["F0"])

        with open(self.output().path, "w") as f:
            yaml.dump(output, f)


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

        correction_fun = get_phase_func_from_ephemeris_file(
            mjdstart,
            mjdstop,
            parfiles,
            ephem=ephem,
            return_sec_from_mjdstart=True,
        )

        times_from_mjdstart = events.time - (mjdstart - events.mjdref) * 86400
        phase = correction_fun(times_from_mjdstart)

        nbin = 512
        expo = None
        if hasattr(events, "prior") and (
            np.any(events.prior != 0) or np.any(np.isnan(events.prior))
        ):
            phases_livetime_start = correction_fun(times_from_mjdstart - events.prior)
            phase_edges = np.linspace(0, 1, nbin + 1)
            weights = _create_weights(
                phases_livetime_start.astype(float), phase.astype(float), phase_edges
            )
            expo = 1 / weights
        phase -= np.floor(phase)

        table = calculate_profile(phase, nbin=nbin, expo=expo)
        table.meta.update(info)
        model = get_model(parfiles[0])
        table.meta["F0"] = model.F0.value

        table.write(self.output().path)


class GetParfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, ".txt"))

    def run(self):
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        info = load_yaml_file(infofile)
        ephem = info["ephem"]
        crab_names = ["crab", "b0531+21", "j0534+22"]
        found_crab = False
        name_compare = info["source"].lower() if info["source"] is not None else ""
        for name in crab_names:
            if name in name_compare:
                found_crab = True
                break

        if not found_crab:
            warnings.warn("Parfiles only available for the Crab")
        # Detect whether start and end of observation have different files
        fname = self.output().path
        model1 = get_crab_ephemeris(info["mjdstart"], ephem=ephem)
        model2 = get_crab_ephemeris(info["mjdstop"], ephem=ephem)

        if model1.PEPOCH.value != model2.PEPOCH.value:
            warnings.warn(f"Different models for start and stop of {self.fname}")
            fname1 = fname.replace(".txt", "_start.par")
            model1.write_parfile(fname1)
            fname2 = fname.replace(".txt", "_stop.par")
            model2.write_parfile(fname2)
            parfiles = [fname1, fname2]
        else:
            fname1 = fname.replace(".txt", ".par")
            model1.write_parfile(fname1)
            parfiles = [fname1]

        with open(fname, "w") as fobj:
            for pf in parfiles:
                print(pf, file=fobj)


class GetTemplate(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, ".template"))

    def run(self):
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        info = load_yaml_file(infofile)
        template_file = get_template(info["source"], info)
        shutil.copyfile(template_file, self.output().path)


class GetInfo(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, ".info"))

    def run(self):
        info = get_observing_info(self.fname)
        with open(self.output().path, "w") as f:
            yaml.dump(info, f, default_flow_style=False, sort_keys=False)


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
    parser.add_argument("--version", help="Version", type=str, default="none")

    args = parser.parse_args(args)

    config_file = args.config

    _ = luigi.build(
        [TOAPipeline(fname, config_file, args.version) for fname in args.files],
        local_scheduler=True,
        log_level="INFO",
        workers=4,
    )
