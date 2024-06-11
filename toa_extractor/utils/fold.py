from numba import njit
import numpy as np
from stingray.pulse.pulsar import _load_and_prepare_TOAs, get_model
from scipy.interpolate import interp1d
from astropy.table import Table

ONE_SIXTH = 1 / 6


@njit(nogil=True, parallel=False)
def _hist1d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[1] - ranges[0]) / bins)

    for t in range(tracks.size):
        i = (tracks[t] - ranges[0]) * delta
        if 0 <= i < bins:
            H[int(i)] += 1

    return H


def histogram(a, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> H, xedges = np.histogram(x, bins=5, range=[0., 1.])
    >>> Hn = histogram(x, bins=5, ranges=[0., 1.])
    >>> assert np.all(H == Hn)
    """

    hist_arr = np.zeros((bins,), dtype=a.dtype)

    return _hist1d_numba_seq(hist_arr, a, bins, np.asarray(ranges))


@njit(parallel=True)
def _fast_phase_fddot(ts, mean_f, mean_fdot=0, mean_fddot=0):
    tssq = ts * ts
    phases = ts * mean_f + 0.5 * tssq * mean_fdot + ONE_SIXTH * tssq * ts * mean_fddot
    return phases - np.floor(phases)


def calculate_phase(events_time, model):
    return _fast_phase_fddot(
        events_time,
        model.F0.value.astype(np.double),
        model.F1.value.astype(np.double),
        model.F2.value.astype(np.double),
    )


def calculate_profile(phase, nbin=512, expo=None):
    prof = histogram(phase.astype(float), bins=nbin, ranges=[0, 1])
    prof_corr = prof

    if expo is not None:
        prof_corr = prof / expo

    t = Table(
        {
            "phase": np.linspace(0, 1, nbin + 1)[:-1],
            "profile": prof_corr,
            "profile_raw": prof,
        }
    )
    if expo is not None:
        t["expo"] = expo

    return t


def prepare_TOAs(mjds, ephem):
    toalist = _load_and_prepare_TOAs(mjds, ephem=ephem)

    toalist.clock_corr_info["include_bipm"] = False
    toalist.clock_corr_info["include_gps"] = False
    return toalist


def get_phase_from_ephemeris_file(
    mjdstart,
    mjdstop,
    parfile,
    ntimes=1000,
    ephem="DE405",
    return_sec_from_mjdstart=False,
):
    """Get a correction for orbital motion from pulsar parameter file.

    Parameters
    ----------
    mjdstart, mjdstop : float
        Start and end of the time interval where we want the orbital solution
    parfile : str
        Any parameter file understood by PINT (Tempo or Tempo2 format)

    Other parameters
    ----------------
    ntimes : int
        Number of time intervals to use for interpolation. Default 1000

    Returns
    -------
    correction_mjd : function
        Function that accepts times in MJDs and returns the deorbited times.
    """

    mjds = np.linspace(mjdstart, mjdstop, ntimes)

    toalist = prepare_TOAs(mjds, ephem)
    m = get_model(parfile)
    phase_int, phase_frac = np.array(m.phase(toalist, abs_phase=True))
    if not return_sec_from_mjdstart:
        phases = phase_int + phase_frac

        return interp1d(mjds, phases, fill_value="extrapolate")
    phases = (phase_int - phase_int[0]) + phase_frac
    return interp1d((mjds - mjdstart) * 86400, phases, fill_value="extrapolate")
