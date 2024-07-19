import shutil
import warnings
import luigi
import yaml
import numpy as np
from astropy import log
from astropy import units as u
from astropy.table import vstack, Table
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from stingray.pulse.pulsar import get_model
from hendrics.efsearch import (
    search_with_qffa,
    EFPeriodogram,
    pf_upper_limit,
    _analyze_qffa_results,
)
from .utils.fit_crab_profiles import normalize_phase_0d5
from .utils.crab import get_crab_ephemeris
from .utils.config import get_template, load_yaml_file
from .utils import output_name
from .utils.data_manipulation import get_observing_info
from .utils.data_manipulation import get_events_from_fits
from .utils.fold import calculate_dyn_profile, get_phase_func_from_ephemeris_file
from pulse_deadtime_fix.core import _create_weights


def _get_and_normalize_phaseogram(phaseogram_file, time_units="hr", smooth_window=None):

    table = Table.read(phaseogram_file)
    normalized_phaseogram = (table["profile"].T).astype(float)

    for iph in range(normalized_phaseogram.shape[1]):
        profile = normalized_phaseogram[:, iph]
        if sum(profile) < 10:
            continue
        if len(profile) > 200:
            window_length = profile.size / 50
            polyorder = min(3, window_length - 1)
            profile = savgol_filter(
                profile, window_length, polyorder, mode="wrap", cval=0.0
            )
        profile = profile - profile.min()
        profile = profile / profile.max()
        normalized_phaseogram[:, iph] = profile

    meantime_mjd = round(np.mean(table.meta["time"]) / 86400 + table.meta["mjdref"], 1)
    meantime = (meantime_mjd - table.meta["mjdref"]) * 86400
    time_factor = (1 * u.s / u.Unit(time_units)).to("").value
    times = (table.meta["time"] - meantime) * time_factor
    phases = table.meta["phase"]

    return phases, times, normalized_phaseogram, meantime_mjd


def _plot_phaseogram(
    phases, time_hrs, phas, meantime_mjd, ax, title=None, label_y=True
):
    from stingray.pulse.search import plot_phaseogram

    plot_phaseogram(
        phas,
        phases - 1,
        time_hrs,
        ax=ax,
        cmap="Greys",
    )
    plot_phaseogram(
        phas,
        phases,
        time_hrs,
        ax=ax,
        cmap="Greys",
    )
    plot_phaseogram(
        phas,
        phases + 1,
        time_hrs,
        ax=ax,
        cmap="Greys",
    )
    ax.set_title(title)
    if label_y:
        ax.set_ylabel(f"Time (hr) since {meantime_mjd:.1f}")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        # plt.setp(ax.get_ylabel(), None)
        ax.yaxis.label.set_visible(False)
    ax.grid(True)


class PlotPhaseogram(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        return GetPhaseogram(
            self.fname, self.config_file, self.version, self.worker_timeout
        )

    def output(self):
        return luigi.LocalTarget(
            output_name(self.fname, self.version, "_phaseograms.jpg")
        )

    def run(self):
        import matplotlib.gridspec as gridspec

        phaseograms = open(self.input().path, "r").read().splitlines()
        nphaseograms = len(phaseograms)

        ncol = 2
        nrow = nphaseograms

        fig = plt.figure(figsize=(7.5 * ncol, 5 * nrow))
        gs = gridspec.GridSpec(nrow, ncol)

        for i, phaseogram in enumerate(phaseograms):
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            phases, time_hrs, phas, meantime_mjd = _get_and_normalize_phaseogram(
                phaseogram
            )

            _plot_phaseogram(
                phases,
                time_hrs,
                phas,
                meantime_mjd,
                ax1,
                title=f"Phaseogram {i}",
                label_y=True,
            )
            _plot_phaseogram(
                phases,
                time_hrs,
                phas,
                meantime_mjd,
                ax2,
                title=f"Phaseogram {i}",
                label_y=False,
            )
            imax = np.argmax(phas.sum(axis=1))

            imax = np.argmax(phas.sum(axis=1))
            phmax = normalize_phase_0d5(phases[imax]) + 1
            print(imax, phmax)
            ax2.set_xlim(phmax - 0.1, phmax + 0.1)

        plt.tight_layout()
        plt.savefig(self.output().path)


class GetPhaseogram(luigi.Task):
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
            output_name(self.fname, self.version, "_phaseograms.txt")
        )

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
        output_files = []

        for i, (mjdstart, mjdstop) in enumerate(zip(mjd_edges[:-1], mjd_edges[1:])):
            if edge_idxs[i] >= events.time.size:
                warnings.warn("No events in this interval")
                continue

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

            tot_phots = times_from_mjdstart.size
            tot_time = mjdstop - mjdstart
            ntimebin = int(max(tot_phots // 200_000, tot_time * 4, 10))
            expo = None
            # In principle, this should be applied to each sub-interval of the phaseogram.
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
            result_table = calculate_dyn_profile(
                events.time[good], phase, nbin=nbin, ntimebin=ntimebin, expo=expo
            )
            result_table.meta["F0"] = model.F0.value
            result_table.meta["F1"] = model.F1.value
            result_table.meta["F2"] = model.F2.value
            result_table.meta["mjdref"] = events.mjdref
            new_file_name = output_name(
                self.fname, self.version, f"_dynprof_{i:000d}.hdf5"
            )
            result_table.write(new_file_name, overwrite=True, serialize_meta=True)
            output_files.append(new_file_name)

        with open(self.output().path, "w") as fobj:
            for fname in output_files:
                print(fname, file=fobj)


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
            if edge_idxs[i] >= events.time.size:
                warnings.warn("No events in this interval")
                continue

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
        result_table.write(
            self.output().path, format="ascii.ecsv", overwrite=True, serialize_meta=True
        )


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
