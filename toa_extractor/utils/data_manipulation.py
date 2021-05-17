import os
import warnings
import numpy as np
from stingray import EventList
from stingray.io import read_mission_info, get_key_from_mission_info
from stingray.io import high_precision_keyword_read
from astropy import log
from astropy.io import fits
from . import safe_get_key


def read_rmf(rmf_file):
    """Load RMF info.

    .. note:: Preliminary: only EBOUNDS are read.

    Parameters
    ----------
    rmf_file : str
        The rmf file used to read the calibration.

    Returns
    -------
    pis : array-like
        the PI channels
    e_mins : array-like
        the lower energy bound of each PI channel
    e_maxs : array-like
        the upper energy bound of each PI channel
    """

    lchdulist = fits.open(rmf_file, checksum=True)
    lchdulist.verify("warn")
    lctable = lchdulist["EBOUNDS"].data
    pis = np.array(lctable.field("CHANNEL"))
    e_mins = np.array(lctable.field("E_MIN"))
    e_maxs = np.array(lctable.field("E_MAX"))
    lchdulist.close()
    return pis, e_mins, e_maxs


def read_calibration(pis, rmf_file):
    """Read the energy channels corresponding to the given PI channels.

    Parameters
    ----------
    pis : array-like
        The channels to lookup in the rmf

    Other Parameters
    ----------------
    rmf_file : str
        The rmf file used to read the calibration.
    """
    calp, calEmin, calEmax = read_rmf(rmf_file)
    es = np.zeros(len(pis), dtype=float)
    for ic, c in enumerate(calp):
        good = pis == c
        if not np.any(good):
            continue
        es[good] = (calEmin[ic] + calEmax[ic]) / 2

    return es


def get_events_from_fits(evfile):
    log.info(f"Opening file {evfile}")
    events = EventList.read(evfile, format_="hea")

    return events


def calibrate_events(events, rmf_file):
    events.energy = read_calibration(events.pi, rmf_file)
    return events


def get_observing_info(evfile, hduname=1):
    info = {}

    with fits.open(evfile) as hdul:
        header0 = hdul[0].header
        header1 = hdul[0].header
        mission_key = "MISSION"
        if mission_key not in header0 and mission_key not in header1:
            mission_key = "TELESCOP"
        if mission_key not in header0:
            header0 = header1
        mission = header0[mission_key].lower()
        db = read_mission_info(mission)
        instkey = get_key_from_mission_info(db, "instkey", "INSTRUME")
        instr = mode = None
        if instkey in header0:
            instr = header0[instkey].strip()

        modekey = get_key_from_mission_info(db, "dmodekey", None, instr)
        if modekey is not None and modekey in header0:
            mode = header0[modekey].strip()

        if hduname is None:
            hduname = get_key_from_mission_info(db, "events", "EVENTS", instr, mode)

        if hduname not in hdul:
            warnings.warn(f"HDU {hduname} not found. Trying first extension")
            hduname = 1

        hdu = hdul[hduname]
        header = hdu.header

        if "EXPOSURE" in header:
            ctrate = header["NAXIS2"] / header["EXPOSURE"]
        else:
            # Need better estimate
            ctrate = header["NAXIS2"] / (header["TSTOP"] - header["TSTART"])

        info["fname"] = os.path.abspath(evfile)
        info["obsid"] = safe_get_key(header, "OBS_ID", "")
        info["mission"] = mission
        info["instrument"] = instr
        info["mjdref_highprec"] = high_precision_keyword_read(header, "MJDREF")
        info["mjdref"] = float(info["mjdref_highprec"])
        info["mode"] = mode
        info["ephem"] = safe_get_key(header, "PLEPHEM", "JPL-DE200").strip().lstrip("JPL-").lower()
        info["timesys"] = safe_get_key(header, "TIMESYS", "TDB").strip().lower()
        info["timeref"] = safe_get_key(header, "TIMEREF", "SOLARSYSTEM").strip().lower()
        info["tstart"] = safe_get_key(header, "TSTART", None)
        info["tstop"] = safe_get_key(header, "TSTOP", None)
        info["source"] = safe_get_key(header, "OBJECT", None)
        info["ra"] = safe_get_key(header, "RA_OBJ", None)
        info["dec"] = safe_get_key(header, "DEC_OBJ", None)
        info["mjdstart"] = info["tstart"] / 86400 + info["mjdref_highprec"]
        info["mjdstop"] = info["tstop"] / 86400 + info["mjdref_highprec"]
        MJD = (info["mjdstart"] + info["mjdstop"]) / 2
        info["mjd"] = MJD
        info["countrate"] = float(ctrate)

    return info
