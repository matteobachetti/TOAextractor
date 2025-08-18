import os
import sys
import warnings

import astropy
import astroquery
import bokeh
import h5py
import hendrics
import luigi
import matplotlib
import numba
import numpy as np
import pandas
import pint
import pulse_deadtime_fix
import scipy
import statsmodels
import stingray
import toa_extractor
import uncertainties
import yaml

from astropy.io import fits
from stingray.io import (
    get_key_from_mission_info,
    high_precision_keyword_read,
    read_mission_info,
)

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


def load_events_and_gtis(
    fits_file,
    additional_columns=None,
    gtistring=None,
    gti_file=None,
    hduname=None,
    column=None,
    max_events=1e32,
):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str

    Other parameters
    ----------------
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])
    gtistring : str
        Comma-separated list of accepted GTI extensions (default GTI,STDGTI),
        with or without appended integer number denoting the detector
    gti_file : str, default None
        External GTI file
    hduname : str or int, default 1
        Name of the HDU containing the event list
    column : str, default None
        The column containing the time values. If None, we use the name
        specified in the mission database, and if there is nothing there,
        "TIME"
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values

    Returns
    -------
    retvals : Object with the following attributes:
        ev_list : array-like
            Event times in Mission Epoch Time
        gti_list: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            GTIs in Mission Epoch Time
        additional_data: dict
            A dictionary, where each key is the one specified in additional_colums.
            The data are an array with the values of the specified column in the
            fits file.
        t_start : float
            Start time in Mission Epoch Time
        t_stop : float
            Stop time in Mission Epoch Time
        pi_list : array-like
            Raw Instrument energy channels
        cal_pi_list : array-like
            Calibrated PI channels (those that can be easily converted to energy
            values, regardless of the instrument setup.)
        energy_list : array-like
            Energy of each photon in keV (only for NuSTAR, NICER, XMM)
        instr : str
            Name of the instrument (e.g. EPIC-pn or FPMA)
        mission : str
            Name of the instrument (e.g. XMM or NuSTAR)
        mjdref : float
            MJD reference time for the mission
        header : str
            Full header of the FITS file, for debugging purposes
        detector_id : array-like, int
            Detector id for each photon (e.g. each of the CCDs composing XMM's or
            Chandra's instruments)
    """
    from astropy.io import fits as pf
    from stingray.io import (
        AstropyUserWarning,
        EventReadOutput,
        _get_additional_data,
        get_gti_from_all_extensions,
        load_gtis,
        order_list_of_arrays,
        rough_calibration,
    )

    hdulist = pf.open(fits_file)
    probe_header = hdulist[0].header
    # Let's look for TELESCOP here. This is the most common keyword to be
    # found in well-behaved headers. If it is not in header 0, I take this key
    # and the remaining information from header 1.
    if "TELESCOP" not in probe_header:
        probe_header = hdulist[1].header
    mission_key = "MISSION"
    if mission_key not in probe_header:
        mission_key = "TELESCOP"
    mission = probe_header[mission_key].lower()
    db = read_mission_info(mission)
    instkey = get_key_from_mission_info(db, "instkey", "INSTRUME")
    instr = mode = None
    if instkey in probe_header:
        instr = probe_header[instkey].strip()

    modekey = get_key_from_mission_info(db, "dmodekey", None, instr)
    if modekey is not None and modekey in probe_header:
        mode = probe_header[modekey].strip()

    gtistring = get_key_from_mission_info(db, "gti", "GTI,STDGTI", instr, mode)
    if hduname is None:
        hduname = get_key_from_mission_info(db, "events", "EVENTS", instr, mode)

    if hduname not in hdulist:
        warnings.warn(f"HDU {hduname} not found. Trying first extension")
        hduname = 1

    datatable = hdulist[hduname].data
    header = hdulist[hduname].header

    ephem = timeref = timesys = None

    if "PLEPHEM" in header:
        ephem = header["PLEPHEM"].strip().lstrip("JPL-").lower()
    if "TIMEREF" in header:
        timeref = header["TIMEREF"].strip().lower()
    if "TIMESYS" in header:
        timesys = header["TIMESYS"].strip().lower()

    if column is None:
        column = get_key_from_mission_info(db, "time", "TIME", instr, mode)

    if column.lower() not in [col.lower() for col in datatable.columns.names]:
        if "TDB" in datatable.columns.names:
            column = "TDB"
        else:
            raise RuntimeError(f"No valid column names found ({column}, TDB)")
    ev_list = np.array(datatable.field(column), dtype=np.longdouble)

    detector_id = None
    ckey = get_key_from_mission_info(db, "ccol", "NONE", instr, mode)
    if ckey != "NONE":
        detector_id = datatable.field(ckey)
    det_number = None if detector_id is None else list(set(detector_id))

    timezero = np.longdouble(0.0)
    if "TIMEZERO" in header:
        timezero = np.longdouble(header["TIMEZERO"])

    if max_events is not None:
        ev_list = ev_list[:max_events]
        if detector_id is not None:
            detector_id = detector_id[:max_events]

    ev_list += timezero

    t_start = ev_list[0]
    t_stop = ev_list[-1]
    if "TSTART" in header:
        t_start = np.longdouble(header["TSTART"])
    if "TSTOP" in header:
        t_stop = np.longdouble(header["TSTOP"])

    mjdref = np.longdouble(high_precision_keyword_read(header, "MJDREF"))

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(",")

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = get_gti_from_all_extensions(
                hdulist,
                accepted_gtistrings=accepted_gtistrings,
                det_numbers=det_number,
            )
        except Exception:  # pragma: no cover
            warnings.warn(
                "No extensions found with a valid name. "
                "Please check the `accepted_gtistrings` values.",
                AstropyUserWarning,
            )
            gti_list = np.array([[t_start, t_stop]], dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    pi_col = get_key_from_mission_info(db, "ecol", "PI", instr, mode)
    if additional_columns is None:
        additional_columns = [pi_col]
    if pi_col not in additional_columns:
        additional_columns.append(pi_col)

    additional_data = _get_additional_data(datatable, additional_columns)
    if max_events is not None:
        for col in additional_data.keys():
            additional_data[col] = additional_data[col][:max_events]

    hdulist.close()
    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]
    if detector_id is not None:
        detector_id = detector_id[order]

    additional_data = order_list_of_arrays(additional_data, order)

    pi = additional_data[pi_col].astype(np.float32)
    cal_pi = pi

    # EventReadOutput() is an empty class. We will assign a number of attributes to
    # it, like the arrival times of photons, the energies, and some information
    # from the header.
    returns = EventReadOutput()

    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.pi_list = pi
    returns.cal_pi_list = cal_pi
    if "energy" in additional_data:
        returns.energy_list = additional_data["energy"]
    else:
        try:
            returns.energy_list = rough_calibration(cal_pi, mission)
        except ValueError:
            returns.energy_list = None
    returns.instr = instr.lower()
    returns.mission = mission.lower()
    returns.mjdref = mjdref
    returns.header = header.tostring()
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop
    returns.detector_id = detector_id
    returns.ephem = ephem
    returns.timeref = timeref
    returns.timesys = timesys

    return returns


def calibrate_events(events, rmf_file):
    events.energy = read_calibration(events.pi, rmf_file)
    return events


def get_observing_info(evfile, hduname=1):
    info = {}

    with fits.open(evfile) as hdul:
        header0 = hdul[0].header
        header1 = hdul[1].header
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

        nphot = header["NAXIS2"]
        if nphot == 0:
            raise ValueError(f"No photons found in {evfile} in HDU {hduname}")

        if "EXPOSURE" in header:
            exposure = header["EXPOSURE"]
        elif "TSTOP" in header and "TSTART" in header:
            exposure = header["TSTOP"] - header["TSTART"]

        if exposure <= 0:
            raise ValueError(f"Invalid or zero exposure time in {evfile} in HDU {hduname}")

        ctrate = header["NAXIS2"] / exposure

        def float_if_not_none(val):
            return float(val) if val is not None else None

        info["path"] = os.path.abspath(os.path.dirname(evfile))
        info["fname"] = os.path.basename(evfile)
        info["nphots"] = header["NAXIS2"]
        info["obsid"] = safe_get_key(header, "OBS_ID", "")
        info["mission"] = mission
        info["instrument"] = instr
        mjdref_highprec = high_precision_keyword_read(header, "MJDREF")
        info["mjdref_highprec"] = f"{mjdref_highprec:0.15f}"
        info["mjdref"] = float_if_not_none(info["mjdref_highprec"])
        info["mode"] = mode
        info["ephem"] = safe_get_key(header, "PLEPHEM", "JPL-DE200").strip().lstrip("JPL-").lower()
        if "RADECSYS" not in header:
            warnings.warn(
                "RADECSYS not found in header. Assuming FK5 if the ephemeris"
                "is JPL-DE200, and ICRS otherwise. "
                "This may lead to incorrect results."
            )
            info["frame"] = "fk5" if info["ephem"] == "de200" else "icrs"
        else:
            info["frame"] = header["RADECSYS"].strip().lower()

        info["timesys"] = safe_get_key(header, "TIMESYS", "TDB").strip().lower()
        info["timeref"] = safe_get_key(header, "TIMEREF", "SOLARSYSTEM").strip().lower()
        info["tstart"] = float_if_not_none(safe_get_key(header, "TSTART", None))
        info["tstop"] = float_if_not_none(safe_get_key(header, "TSTOP", None))
        info["source"] = safe_get_key(header, "OBJECT", "")
        info["ra"] = float_if_not_none(safe_get_key(header, "RA_OBJ", None))
        info["dec"] = float_if_not_none(safe_get_key(header, "DEC_OBJ", None))
        info["ra_bary"] = info["dec_bary"] = None
        if "RA_BARY" in header and "bary" in header.comments["RA_BARY"]:
            info["ra_bary"] = header["RA_BARY"]
            info["dec_bary"] = header["DEC_BARY"]
        elif "RA_OBJ" in header and "bary" in header.comments["RA_OBJ"]:
            info["ra_bary"] = header["RA_OBJ"]
            info["dec_bary"] = header["DEC_OBJ"]
        info["mjdstart"] = float_if_not_none(info["tstart"] / 86400 + mjdref_highprec)
        info["mjdstop"] = float_if_not_none(info["tstop"] / 86400 + mjdref_highprec)
        MJD = float(info["mjdstart"] + info["mjdstop"]) / 2
        info["mjd"] = MJD
        info["countrate"] = float_if_not_none(ctrate)

        info["astropy_version"] = str(astropy.__version__)
        info["astroquery_version"] = str(astroquery.__version__)
        info["bokeh_version"] = str(bokeh.__version__)
        info["hendrics_version"] = str(hendrics.__version__)
        info["h5py_version"] = str(h5py.__version__)
        info["luigi_version"] = str(luigi.__version__)
        info["matplotlib_version"] = str(matplotlib.__version__)
        info["numba_version"] = str(numba.__version__)
        info["numpy_version"] = str(np.__version__)
        info["pandas_version"] = str(pandas.__version__)
        info["pint_version"] = str(pint.__version__)
        info["pulse_deadtime_fix_version"] = str(pulse_deadtime_fix.__version__)
        info["pyyaml_version"] = str(yaml.__version__)
        info["python_version"] = str(sys.version.split()[0])
        info["scipy_version"] = str(scipy.__version__)
        info["statsmodels_version"] = str(statsmodels.__version__)
        info["stingray_version"] = str(stingray.__version__)
        info["uncertainties_version"] = str(uncertainties.__version__)
        info["version"] = str(toa_extractor.__version__)

    return info
