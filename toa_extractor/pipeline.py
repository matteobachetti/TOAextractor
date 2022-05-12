import shutil
import numpy as np
import luigi
import yaml
import warnings
from astropy.table import Table
from stingray.pulse.pulsar import get_model
from stingray.pulse.pulsar import fftfit
# from stingray.events import EventList
import matplotlib.pyplot as plt
from .utils.crab import get_crab_ephemeris
from .utils import root_name, output_name
from .utils.data_manipulation import get_observing_info, get_events_from_fits
from .utils.config import get_template, load_yaml_file
from .utils.fold import calculate_profile, get_phase_from_ephemeris_file


class PlotDiagnostics(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetResidual(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_diagnostics.jpg"))

    def run(self):
        prof_file = (
            GetFoldedProfile(self.fname, self.config_file, self.worker_timeout).output().path
        )
        prof_table = Table.read(prof_file)

        template_file = GetTemplate(self.fname, self.config_file, self.worker_timeout).output().path
        template_table = Table.read(template_file, format="ascii.ecsv")

        residual_file = GetResidual(self.fname, self.config_file, self.worker_timeout).output().path
        residual_dict = load_yaml_file(residual_file)

        fig = plt.figure()
        pphase = prof_table["phase"]
        pphase = np.concatenate([pphase - 1, pphase])
        prof = prof_table["profile"] - prof_table["profile"].min()
        prof /= prof.max()
        prof = np.concatenate([prof, prof])

        tphase = template_table["phase"]
        tphase = np.concatenate([tphase - 1, tphase])
        temp = template_table["profile"] - template_table["profile"].min()
        temp = temp / temp.max()
        temp = np.concatenate([temp, temp])

        plt.plot(pphase / prof_table.meta["F0"], prof, color="red")
        plt.plot(tphase / prof_table.meta["F0"], temp, color="k", zorder=1)

        minres = residual_dict["residual"] - residual_dict["residual_err"]
        maxres = residual_dict["residual"] + residual_dict["residual_err"]
        plt.axvspan(minres, maxres, color="#aaaaff", alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (arbitrary units)")

        plt.savefig(self.output().path)
        plt.close(fig)


class GetResidual(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetFoldedProfile(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_residual.yaml"))

    def run(self):
        prof_file = (
            GetFoldedProfile(self.fname, self.config_file, self.worker_timeout).output().path
        )
        prof_table = Table.read(prof_file)

        template_file = GetTemplate(self.fname, self.config_file, self.worker_timeout).output().path
        template_table = Table.read(template_file, format="ascii.ecsv")

        prof = prof_table["profile"]
        template = template_table["profile"]
        mean_amp, std_amp, phase_res, phase_res_err = fftfit(prof, template=template)
        output = {}
        output.update(prof_table.meta)
        output["residual"] = float(phase_res / prof_table.meta["F0"])
        output["residual_err"] = float(phase_res_err / prof_table.meta["F0"])
        with open(self.output().path, "w") as f:
            yaml.dump(output, f)


class GetFoldedProfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield GetParfile(self.fname, self.config_file, self.worker_timeout)
        yield GetTemplate(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_folded.hdf5"))

    def run(self):
        infofile = GetInfo(self.fname, self.config_file, self.worker_timeout).output().path
        info = load_yaml_file(infofile)
        events = get_events_from_fits(self.fname)
        mjdstart, mjdstop = info["mjdstart"], info["mjdstop"]
        parfile = GetParfile(self.fname, self.config_file, self.worker_timeout).output().path
        correction_fun = get_phase_from_ephemeris_file(
            mjdstart, mjdstop, parfile, ephem=info["ephem"]
        )
        mjds = events.time / 86400 + events.mjdref
        phase = correction_fun(mjds)
        phase -= np.floor(phase)
        table = calculate_profile(phase)
        table.meta.update(info)
        model = get_model(parfile)
        for attr in ["F0", "F1", "F2"]:
            table.meta[attr] = getattr(model, attr).value
        table.meta["epoch"] = model.PEPOCH.value
        # table.meta['mjd'] = (local_events[0] + local_events[-1]) / 2
        table.write(self.output().path)


class GetParfile(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetInfo(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, ".par"))

    def run(self):
        infofile = GetInfo(self.fname, self.config_file, self.worker_timeout).output().path
        info = load_yaml_file(infofile)
        crab_names = ["crab", "b0531+21", "j0534+22"]
        found_crab = False
        name_compare = info["source"].lower() if info["source"] is not None else ""
        for name in crab_names:
            if name in name_compare:
                found_crab = True
                break

        if not found_crab:
            warnings.warn("Parfiles only available for the Crab")

        get_crab_ephemeris(info["mjd"], fname=self.output().path)


class GetTemplate(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetInfo(self.fname, self.config_file, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, ".template"))

    def run(self):
        infofile = GetInfo(self.fname, self.config_file, self.worker_timeout).output().path
        info = load_yaml_file(infofile)
        template_file = get_template(info["source"], info)
        shutil.copyfile(template_file, self.output().path)


class GetInfo(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter()
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
        [PlotDiagnostics(fname, config_file, args.version) for fname in args.files],
        local_scheduler=True,
        log_level="INFO",
        workers=4,
    )
