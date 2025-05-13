import shutil
import warnings

import luigi
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import log
from astropy import units as u
from astropy.table import Table, vstack
from hendrics.efsearch import (
    EFPeriodogram,
    _analyze_qffa_results,
    pf_upper_limit,
    search_with_qffa,
)
from pulse_deadtime_fix.core import _create_weights
from scipy.signal import savgol_filter
from stingray import EventList
from stingray.gti import cross_two_gtis, split_gtis_by_exposure
from stingray.io import FITSTimeseriesReader
from stingray.pulse.pulsar import get_model

from .utils import output_name
from .utils.config import get_template, load_yaml_file

# from .utils.fit_crab_profiles import normalize_phase_0d5
from .utils.crab import get_crab_ephemeris
from .utils.data_manipulation import get_events_from_fits, get_observing_info
from .utils.fold import calculate_dyn_profile, get_phase_func_from_ephemeris_file


def hrc_true_rate(rate: float):
    if isinstance(rate, np.ndarray):
        return np.asarray([hrc_true_rate(r) for r in rate])
    if rate < 12:
        return rate
    if rate > 24:
        warnings.warn("The rate is too high for HRC correction")
    rate_lim = 2058.4 / 75.6
    if rate > rate_lim:
        rate = rate_lim
    corr = 45.91 - (2058.4 - 75.6 * rate) ** 0.5
    return corr


_RATE_CORRECTION_FUNC = {
    "hrc": hrc_true_rate,
    "hrc-s": hrc_true_rate,
    "hrc-i": hrc_true_rate,
}


def split_gtis_at_times_and_exposure(gti, times, max_exposure=np.inf):
    """Split gtis with various criteria.

    Examples
    --------
    >>> gti = np.array([[0, 10], [20, 30]])
    >>> times = [5, 25]
    >>> res = split_gtis_at_times_and_exposure(gti, times)
    >>> assert np.allclose(res[0], np.array([[0, 5]]))
    >>> assert np.allclose(res[1], np.array([[5, 10], [20, 25]]))
    >>> assert np.allclose(res[2], np.array([[25, 30]]))
    >>> res = split_gtis_at_times_and_exposure(gti, [10], max_exposure=5)
    >>> assert np.allclose(res[0], np.array([[0, 5]]))
    >>> assert np.allclose(res[1], np.array([[5, 10]]))
    >>> assert np.allclose(res[2], np.array([[20, 25]]))
    >>> assert np.allclose(res[3], np.array([[25, 30]]))
    """
    duration = np.sum(gti[:, 1] - gti[:, 0])

    if duration < max_exposure and len(times) == 0:
        return [gti]

    if len(times) == 0:
        split_gtis = [gti]
    elif times[0] > gti[0, 0] and times[-1] < gti[-1, 1]:
        edges = np.concatenate([[gti[0, 0]], times, [gti[-1, 1]]])

        time_intervals = list(zip(edges[:-1], edges[1:]))

        split_gtis = [cross_two_gtis(gti, [t_int]) for t_int in time_intervals]
    elif times[0] < gti[0, 0]:
        raise ValueError("First time is smaller than the first GTI")
    elif times[1] > gti[-1, 1]:
        raise ValueError("Last time is larger than the last GTI")

    if duration < max_exposure:
        return split_gtis

    # Otherwise, further split the observation

    new_split_gtis = []
    for gti in split_gtis:
        local_duration = np.sum(gti[:, 1] - gti[:, 0])
        if local_duration > max_exposure:
            new_split_gtis.extend(split_gtis_by_exposure(gti, max_exposure))
        else:
            new_split_gtis.append(gti)

    return new_split_gtis


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
            profile = savgol_filter(profile, window_length, polyorder, mode="wrap", cval=0.0)
        profile = profile - profile.min()
        profile = profile / profile.max()
        normalized_phaseogram[:, iph] = profile

    meantime_mjd = round(np.mean(table.meta["time"]) / 86400 + table.meta["mjdref"], 1)
    meantime = (meantime_mjd - table.meta["mjdref"]) * 86400
    time_factor = (1 * u.s / u.Unit(time_units)).to("").value
    times = (table.meta["time"] - meantime) * time_factor
    phases = table.meta["phase"]

    return phases, times, normalized_phaseogram, meantime_mjd


def _plot_phaseogram(phases, time_hrs, phas, meantime_mjd, ax, title=None, label_y=True):
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


class GetPhaseogram(luigi.Task):
    fname = luigi.Parameter()
    config_file = luigi.Parameter()
    version = luigi.Parameter(default="none")
    worker_timeout = luigi.IntParameter(default=600)

    def requires(self):
        yield GetParfile(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetTemplate(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_phaseograms.txt"))

    def run(self):
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout).output().path
        )
        info = load_yaml_file(infofile)

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

        log.info("Relevant parameter files: " + ",".join(parfiles))

        model_epochs = np.asarray([get_model(p).PEPOCH.value for p in parfiles])

        # Sort parfiles based on model_epochs
        sorted_indices = np.argsort(model_epochs)
        parfiles = [parfiles[i] for i in sorted_indices]
        model_epochs = model_epochs[sorted_indices]

        fitsreader = FITSTimeseriesReader(
            self.fname, output_class=EventList, additional_columns=["PRIOR"]
        )
        model_epochs_met = (model_epochs - fitsreader.mjdref) * 86400
        current_gtis = fitsreader.gti

        obs_will_be_split = False
        split_at_edges = []
        if len(set(model_epochs_met)) > 1:
            split_at_edges = (model_epochs_met[1:] + model_epochs_met[:-1]) / 2
            obs_will_be_split = True

        nphotons = fitsreader.nphot
        log.info(f"Number of photons in the observation: {nphotons}.")
        # If nphotons is too high, further split the intervals
        photon_max = 5_000_000  # soft upper limit
        max_exposure = np.inf
        if nphotons > photon_max * 1.5:
            max_exposure = fitsreader.exposure * photon_max / nphotons
            obs_will_be_split = True

        if obs_will_be_split:
            log.info("Splitting observation into smaller intervals.")

        split_gti = split_gtis_at_times_and_exposure(
            current_gtis,
            split_at_edges,
            max_exposure=max_exposure,
        )

        nbin = 512
        output_files = []

        for subprofile_count, events in enumerate(fitsreader.apply_gti_lists(split_gti)):
            mjdstart = events.gti[0, 0] / 86400 + events.mjdref
            mjdstop = events.gti[-1, 1] / 86400 + events.mjdref

            mean_met = (events.gti[0, 0] + events.gti[-1, 1]) / 2
            parfile = parfiles[np.argmin(np.abs(model_epochs_met - mean_met))]

            log.info(f"Using {parfile} for {mjdstart} - {mjdstop}")

            correction_fun = get_phase_func_from_ephemeris_file(
                mjdstart,
                mjdstop,
                parfile,
                ephem=ephem,
                return_sec_from_mjdstart=True,
            )

            times_from_mjdstart = events.time - (mjdstart - events.mjdref) * 86400
            phase = correction_fun(times_from_mjdstart)

            tot_phots = times_from_mjdstart.size
            tot_time = mjdstop - mjdstart
            ntimebin = int(max(tot_phots // 200_000, tot_time * 4, 10))
            expo = None

            # In principle, this should be applied to each sub-interval of the phaseogram.
            if hasattr(events, "prior") and (
                np.any(events.prior != 0) or np.any(np.isnan(events.prior))
            ):
                phases_livetime_start = correction_fun(times_from_mjdstart - events.prior)
                phase_edges = np.linspace(0, 1, nbin + 1)
                weights = _create_weights(
                    phases_livetime_start.astype(float),
                    phase.astype(float),
                    phase_edges,
                )
                expo = 1 / weights

            phase -= np.floor(phase)

            model = get_model(parfile)
            result_table = calculate_dyn_profile(
                events.time, phase, nbin=nbin, ntimebin=ntimebin, expo=expo
            )
            if events.instr is not None and events.instr.lower() in list(
                _RATE_CORRECTION_FUNC.keys()
            ):
                log.info(f"Instrument: {events.instr}")
                # HRC
                log.info(f"Using {events.instr} rate correction")
                tot_prof = np.sum(result_table["profile"], axis=0)
                avg_rate = tot_prof * nbin / events.exposure
                fold_correction_fun = _RATE_CORRECTION_FUNC[events.instr.lower()]
                # avg_rate = np.mean(rate, axis=0)
                from scipy.ndimage import gaussian_filter1d

                avg_rate = gaussian_filter1d(avg_rate, 5, mode="wrap")
                if np.any(avg_rate <= 0):
                    warnings.warn("Rate is zero or negative")
                    continue
                expo = avg_rate / fold_correction_fun(avg_rate)

                result_table.meta["expo"] = expo

            result_table.meta["F0"] = model.F0.value
            result_table.meta["F1"] = model.F1.value
            result_table.meta["F2"] = model.F2.value
            result_table.meta["mjdref"] = events.mjdref
            result_table.meta["mjdstart"] = mjdstart
            result_table.meta["mjdstop"] = mjdstop
            result_table.meta["mjd"] = model.PEPOCH.value

            new_file_name = output_name(
                self.fname, self.version, f"_dynprof_{subprofile_count:000d}.hdf5"
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
        yield GetParfile(self.fname, self.config_file, self.version, self.worker_timeout)
        yield GetTemplate(self.fname, self.config_file, self.version, self.worker_timeout)

    def output(self):
        return luigi.LocalTarget(output_name(self.fname, self.version, "_best_cands.ecsv"))

    def run(self):
        N = 6
        infofile = (
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout).output().path
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
            efperiodogram.upperlim = pf_upper_limit(np.max(stats), events.time.size, n=N)
            efperiodogram.ncounts = events.time.size
            best_cand_table = _analyze_qffa_results(efperiodogram)
            best_cand_table["f_err_n"] = [
                -max(step / 2, err) for err in np.abs(best_cand_table["f_err_n"])
            ]
            best_cand_table["f_err_p"] = [
                max(step / 2, err) for err in np.abs(best_cand_table["f_err_p"])
            ]

            best_cand_table["initial_freq_estimate"] = central_freq
            log.info(best_cand_table[0])
            result_table.append(best_cand_table[0])
        result_table = vstack(result_table)
        result_table.write(
            self.output().path,
            format="ascii.ecsv",
            overwrite=True,
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
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout).output().path
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
            log.info("Trying to set coordinates to the values found in the FITS file header")
            force_parameters = {
                "RAJ": info["ra_bary"] * u.deg,
                "DECJ": info["dec_bary"] * u.deg,
            }

        model1 = get_crab_ephemeris(
            info["mjdstart"], ephem=ephem, force_parameters=force_parameters
        )
        model2 = get_crab_ephemeris(info["mjdstop"], ephem=ephem, force_parameters=force_parameters)

        if model1.PEPOCH.value != model2.PEPOCH.value:
            warnings.warn(f"Different models for start and stop of {self.fname}")
            n_months = max(np.rint((info["mjdstop"] - info["mjdstart"]) / 30).astype(int), 2)
            models = [
                get_crab_ephemeris(mjd, ephem=ephem, force_parameters=force_parameters)
                for mjd in np.linspace(info["mjdstart"], info["mjdstop"], n_months)
            ]
            fname_root = fname.replace(".txt", "")
            current_pepoch = -1.0
            parfiles = []
            count = 0
            for m in models:
                local_pepoch = m.PEPOCH.value
                if local_pepoch != current_pepoch:
                    new_parfile = fname_root + f"_{count:03d}.par"
                    m.write_parfile(new_parfile)
                    parfiles.append(new_parfile)
                    current_pepoch = local_pepoch
                    count += 1
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
            GetInfo(self.fname, self.config_file, self.version, self.worker_timeout).output().path
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
