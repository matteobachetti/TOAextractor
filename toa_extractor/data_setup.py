import shutil
import warnings
import luigi
import yaml
import numpy as np
from astropy import log
from astropy import units as u
from astropy.table import vstack

from stingray.pulse.pulsar import get_model
from hendrics.efsearch import (
    search_with_qffa,
    EFPeriodogram,
    pf_upper_limit,
    _analyze_qffa_results,
)

from .utils.crab import get_crab_ephemeris
from .utils.config import get_template, load_yaml_file
from .utils import output_name
from .utils.data_manipulation import get_observing_info
from .utils.data_manipulation import get_events_from_fits


class GetPulseFreq(luigi.Task):
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
        return luigi.LocalTarget(
            output_name(self.fname, self.version, "_best_cands.ecsv")
        )

    def run(self):
        N = 6
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout)
            .output()
            .path
        )
        info = load_yaml_file(infofile)
        events = get_events_from_fits(self.fname)
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

        result_table = []
        for i, (mjdstart, mjdstop) in enumerate(zip(mjd_edges[:-1], mjd_edges[1:])):
            model = get_model(parfiles[i])

            good = slice(edge_idxs[i], edge_idxs[i + 1])
            events_to_analyze = events.time[good]
            length = events_to_analyze[-1] - events_to_analyze[0]

            ref_time = (events_to_analyze[-1] + events_to_analyze[0]) / 2
            ref_mjd = ref_time / 86400 + events.mjdref

            secs_from_pepoch = (ref_mjd - model.PEPOCH.value) * 86400
            central_freq = (
                model.F0.value
                + secs_from_pepoch * model.F1.value
                + 0.5 * secs_from_pepoch**2 * model.F2.value
            )
            f0_err = 2 / length
            frequency_range = [central_freq - f0_err, central_freq + f0_err]

            log.info(
                f"Searching for pulsations in interval {frequency_range[0]}-{frequency_range[1]}"
            )
            log.info(f"The central frequency is {central_freq}")

            frequencies, stats, step, length = search_with_qffa(
                events_to_analyze,
                *frequency_range,
                nbin=nbin,
                n=N,
                search_fdot=False,
                oversample=nbin // 2,
            )
            efperiodogram = EFPeriodogram(
                frequencies,
                stats,
                "Z2n",
                nbin,
                N,
                mjdref=events.mjdref,
                pepoch=ref_time,
                oversample=N * 8,
            )
            efperiodogram.upperlim = pf_upper_limit(
                np.max(stats), events.time.size, n=N
            )
            efperiodogram.ncounts = events.time.size
            best_cand_table = _analyze_qffa_results(efperiodogram)
            best_cand_table["initial_freq_estimate"] = central_freq
            log.info(best_cand_table[0])
            result_table.append(best_cand_table[0])
        result_table = vstack(result_table)
        result_table.write(self.output().path, format="ascii.ecsv", overwrite=True)


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
        force_parameters = None
        if "ra_bary" in info and info["ra_bary"] is not None:
            log.info(
                "Trying to set coordinates to the values found in the FITS file header"
            )
            force_parameters = {
                "RAJ": info["ra_bary"] * u.deg,
                "DECJ": info["dec_bary"] * u.deg,
            }

        model1 = get_crab_ephemeris(
            info["mjdstart"], ephem=ephem, force_parameters=force_parameters
        )
        model2 = get_crab_ephemeris(
            info["mjdstop"], ephem=ephem, force_parameters=force_parameters
        )

        if model1.PEPOCH.value != model2.PEPOCH.value:
            warnings.warn(f"Different models for start and stop of {self.fname}")
            fname1 = fname.replace(".txt", "_start.par")
            model1.write_parfile(fname1, include_info=False)
            fname2 = fname.replace(".txt", "_stop.par")
            model2.write_parfile(fname2, include_info=False)
            parfiles = [fname1, fname2]
        else:
            fname1 = fname.replace(".txt", ".par")
            model1.write_parfile(fname1, include_info=False)
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
