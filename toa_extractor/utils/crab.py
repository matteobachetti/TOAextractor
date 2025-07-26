import os
import copy
from collections.abc import Iterable
from io import StringIO
from urllib.request import urlopen
import time

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import pint
import pint.fitter
import pint.simulation
from pint.toa import get_TOAs_array
from pint.residuals import Residuals
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from pint.models import get_model

crab_coords = SkyCoord("05h34m31.97232", "22d00m52.069", frame="fk5")


def file_is_outdated(file_name, time_limit=1 * u.day):
    """
    Check if the file is outdated based on its modification time.
    If the file does not exist, it is considered outdated.

    Parameters
    ----------
    file_name : str
        The name of the file to check.
    time_limit : astropy.units.Quantity, optional
        The time limit to consider the file outdated. Default is 1 day.

    Returns
    -------
    is_outdated : bool
        True if the file is outdated or does not exist, False otherwise.

    Examples
    --------
    >>> file_is_outdated("asdfasfathdafasdfa.txt")
    True
    >>> open('asdfasfathdafasdfa.txt', 'w').close()
    >>> file_is_outdated("asdfasfathdafasdfa.txt")
    False
    >>> os.unlink("asdfasfathdafasdfa.txt")
    """
    if not os.path.exists(file_name):
        return True
    mod_time = os.path.getmtime(file_name)
    return bool((time.time() - mod_time) > time_limit.to(u.s).value)


def retrieve_txt_ephemeris():
    """Retrieve the Crab pulsar ephemeris from the Jodrell Bank website in text format.

    This function downloads the ephemeris data from the Jodrell Bank website if the local
    file is outdated or does not exist. The data is processed and returned as an Astropy
    Table with columns for date, frequency, frequency derivative, dispersion measure (DM),
    and other relevant parameters.

    Returns
    -------
    Table
        An Astropy Table containing the Crab pulsar ephemeris data.

    """
    file_name = "Crab.txt"
    url = "https://www.jb.man.ac.uk/~pulsar/crab/crab2.txt"
    if file_is_outdated(file_name):
        response = urlopen(url, timeout=10)
        data = response.read()
        with open(file_name, "wb") as fobj:
            fobj.write(data)

    with open(file_name) as fobj:
        header_data = []
        rows = []
        for line in fobj.readlines():
            line = line.strip()
            if line == "":
                continue
            if line.startswith("#"):
                continue
            line_data = line.replace(")", ") ").replace("(", " (").split()

            if line_data[0] == "JODRELL":
                continue
            if line_data[0] == "NO":
                continue
            if line_data[0] == "Date":
                header_data = line_data
                continue
            if line_data[0] == "sec":
                continue
            new_line = {"Date": "-".join(line_data[:3])}

            for i, key in enumerate(header_data[1:]):
                val = line_data[i + 3].strip("()")
                if "xxxxx" in val:
                    val = np.nan
                try:
                    float_val = np.longdouble(val)
                except ValueError:
                    new_line[key] = val
                    continue
                if float_val % 1 == 0:
                    new_line[key] = int(float_val)
                else:
                    new_line[key] = float_val
            if len(header_data) >= i + 3:
                new_line["Notes"] = line_data[i + 4]
            rows.append(Table(rows=[new_line]))

        return vstack(rows)


def retrieve_cgro_ephemeris():
    """Retrieve the CGRO ephemeris for the Crab pulsar from the Jodrell Bank website.

    This function downloads the CGRO ephemeris data if the local file is outdated or does not exist.
    The data is processed and returned as an Astropy Table with columns for pulsar name,
    right ascension (RA), declination (DEC), Modified Julian Date (MJD),
    frequency (f0), frequency derivative (f1), frequency second derivative (f2),
    root mean square (RMS) of the residuals, and other relevant parameters.

    Returns
    -------
    Table
        An Astropy Table containing the CGRO ephemeris data.
    """
    file_name = "Crab.gro"
    url = "http://www.jb.man.ac.uk/pulsar/crab/all.gro"

    if file_is_outdated(file_name):
        response = urlopen(url, timeout=10)
        data = response.read()
        with open(file_name, "wb") as out_file:
            out_file.write(
                b"PSR_B     RA(J2000)    DEC(J2000)   MJD1  MJD2    t0geo(MJD) "
                b"        f0(s^-1)      f1(s^-2)     f2(s^-3)  RMS O      B    "
                b"Name      Notes\n"
            )
            out_file.write(
                b"------- ------------- ------------ ----- ----- ---------------"
                b" ----------------- ------------ ---------- ---- -  ------ "
                b"------- -----------------\n"
            )

            out_file.write(data)

    return Table.read(file_name, format="ascii.fixed_width_two_line")


def get_best_cgro_row(MJD):
    """Get the best matching row from the CGRO ephemeris for a given MJD.
    Parameters
    ----------
    MJD : float
        The Modified Julian Date for which to get the ephemeris.
    Returns
    -------
    row : astropy.table.Row
        The row from the CGRO ephemeris that best matches the given MJD.
    """
    table = retrieve_cgro_ephemeris()
    good = np.argmin(np.abs(table["t0geo(MJD)"] - MJD))
    row = table[good]
    return row


def get_best_txt_row(MJD):
    """Get the best matching row from the text ephemeris for a given MJD.
    Parameters
    ----------
    MJD : float
        The Modified Julian Date for which to get the ephemeris.
    Returns
    -------
    row : astropy.table.Row
        The row from the text ephemeris that best matches the given MJD.
    """

    table = retrieve_txt_ephemeris()
    good = np.argmin(np.abs(table["MJD"] - MJD))
    row = table[good]
    return row


def create_model_from_cgro(MJD, barycenter_tzr=False):
    """

    From the instructions:
    http://www.jb.man.ac.uk/research/pulsar/crab/CGRO_format.html
    the integer part of the *geocentric* TOA time is also the *TDB*(!) PEPOCH for the ephemeris.
    """

    row = get_best_cgro_row(MJD)
    f0 = np.longdouble(row["f0(s^-1)"])
    f1 = np.longdouble(row["f1(s^-2)"].replace("D", "E"))
    f2 = np.longdouble(row["f2(s^-3)"].replace("D", "E"))

    tzrmjd = np.longdouble(row["t0geo(MJD)"])
    tzrsite = "0"

    pepoch = np.floor(row["t0geo(MJD)"])

    start = np.longdouble(row["MJD1"])
    finish = np.longdouble(row["MJD2"])
    # the RMS here is in milliperiods, converting to microseconds
    rms_us = np.double(row["RMS"]) * 1e3 / f0

    model_str = f"""PSRJ            J0534+2200
RAJ              {crab_coords.ra.to_string(unit=u.hour, sep=":")}
DECJ             {crab_coords.dec.to_string(unit=u.deg, sep=":")}
POSEPOCH         40706
PEPOCH           {pepoch}
F0               {f0} 1
F1               {f1} 1
F2               {f2} 1
EPHEM            DE200
UNITS TDB
CLOCK TT(TAI)
PHOFF            0 1
TZRFRQ           0
TZRMJD           {tzrmjd}
TZRSITE          {tzrsite}
START            {start}
FINISH           {finish}
DM               0
TRES             {rms_us:.2f}
    """
    with StringIO(model_str) as f:
        model_new_start = get_model(f)

    if barycenter_tzr:
        toas = get_TOAs_array(
            (model_new_start.TZRMJD.quantity.jd1 - 2400000.5, model_new_start.TZRMJD.quantity.jd2),
            obs=model_new_start.TZRSITE.value,
            freqs=model_new_start.TZRFRQ.quantity,
            ephem="DE200",
            planets=False,
            include_bipm=False,
            tzr=True,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        bat = model_new_start.get_barycentric_toas(toas)[0]
        model_new_start.TZRMJD.value = bat.value
        model_new_start.TZRSITE.value = "@"

    return model_new_start


def create_model_from_txt(MJD):
    """From crabnotes.ps distributed at JBO
    The information on the start-finish times is taken from the CGRO ephemeris.
    The frequency and its derivatives are taken from the txt ephemeris.
    """

    row = get_best_txt_row(MJD)

    # I get start and finish from the CGRO ephemeris
    cgro_row = get_best_cgro_row(MJD)

    start = np.longdouble(cgro_row["MJD1"])
    finish = np.longdouble(cgro_row["MJD2"])

    f0 = row["nu"]
    f1 = row["nudot"] * 1e-15
    p0 = 1 / f0  # P0 is the period in seconds
    p1 = -f1 / f0**2  # F1 is the first derivative
    f2 = 2.0 * p1**2 / p0**3  # F2 is the second derivative
    # f2 = 2.0 / f1**2 ** f0**7

    f0_err = row["sigma_nu"] * 1e-11
    f1_err = row["sigma_nudot"] * 1e-15
    f2_err = np.sqrt((4 * f1 * f1_err / f0) ** 2 + (2 * f1**2 / f0**2 * f0_err) ** 2)

    dm = row["DM"]
    rms_us = row["t_acc"]

    tzrmjd = (
        np.longdouble(row["MJD"]) + np.longdouble(row["t_JPL"]) / 86400
    )  # t_JPL is in seconds, convert to days
    tzrsite = "@"
    # From crabnotes: The observed barycentric frequency and its first derivative
    # are given at the quoted arrival time
    pepoch = tzrmjd

    model_str = f"""PSRJ            J0534+2200
RAJ              {crab_coords.ra.to_string(unit=u.hour, sep=":")}
DECJ             {crab_coords.dec.to_string(unit=u.deg, sep=":")}
POSEPOCH         40706
PEPOCH           {pepoch}
F0               {f0} 1 {f0_err}
F1               {f1} 1 {f1_err}
F2               {f2} 1 {f2_err}
EPHEM            DE200
UNITS TDB
CLOCK TT(TAI)
PHOFF            0 1
TZRFRQ           0
TZRMJD           {tzrmjd}
TZRSITE          {tzrsite}
START            {start}
FINISH           {finish}
DM               {dm}
TRES             {rms_us:.2f}
    """
    with StringIO(model_str) as f:
        model_new_start = get_model(f)
    return model_new_start


def _default_fname(mjd, ephem="DE200", force_parameters=None):
    """Generate a default filename based on the ephemeris and forced parameters."""
    fname = f"Crab_{mjd}_{ephem}"
    if force_parameters is not None:
        fname += "".join([f"_{k}_{v}" for k, v in force_parameters.items()])
    fname += ".par"
    return fname


def refit_solution(
    model_200,
    new_ephem,
    rms_tolerance=None,
    force_parameters=None,
    plot=True,
    fname=None,
):
    """Refits the model with a new ephemeris and new coordinates.

    Parameters
    ----------
    model_200 : pint.models.timing_model.TimingModel
        The original model based on DE200 ephemeris.
    new_ephem : str
        The new ephemeris to use, e.g., "DE421", "DE430", etc.
    rms_tolerance : astropy.units.Quantity, optional
        The tolerance for the RMS of the residuals. If None, defaults to 1 microsecond.
    force_parameters : dict, optional
        A dictionary of parameters to force in the model. The keys should be the parameter names,
        and the values should be lists containing the value, a flag indicating whether to freeze
        the parameter (1 for frozen, 0 for unfrozen), and the uncertainty.
    Returns
    -------
    out_model : pint.models.timing_model.TimingModel
        The refitted model with the new ephemeris and coordinates.
    """
    if rms_tolerance is None:
        rms_tolerance = 1 * u.us

    if fname is None:
        fname = _default_fname(
            model_200.PEPOCH.value, ephem=new_ephem, force_parameters=force_parameters
        )

    t0_mjd = np.longdouble(model_200.START.value) - 1
    t1_mjd = np.longdouble(model_200.FINISH.value) + 1

    model_new_start = copy.deepcopy(model_200)
    model_new_start.F0.frozen = False
    model_new_start.F1.frozen = False
    model_new_start.F2.frozen = False
    model_new_start.EPHEM.value = new_ephem

    if force_parameters is not None:
        log.info("Forcing parameters:")
        for key, val in force_parameters.items():
            log.info(f"{key} = {val}")
            if not isinstance(val, Iterable) or isinstance(val, str) or isinstance(val, u.Quantity):
                val = [val]
            par = getattr(model_new_start, key)
            par.quantity = val[0]
            if len(val) > 1:
                if val[1] == 1:
                    par.frozen = False
                par.uncertainty = val[2]

    current_phoff = model_200.PHOFF.quantity
    # Create a bunch of geocenter TOAs with the original DE200 model
    fake_geo_toas = pint.simulation.make_fake_toas_uniform(
        t0_mjd,
        t1_mjd,
        501,
        model_200,
        freq=np.inf,
        error=0.1 * u.us,
        add_noise=True,
        subtract_mean=False,
        obs="geocenter",
    )
    # But change the ephemeris to the new one
    fake_geo_toas.ephem = new_ephem
    log.info(f"TOAs: {fake_geo_toas.get_summary()}")

    log.info(str(model_200.compare(model_new_start)))
    if plot:
        r = Residuals(fake_geo_toas, model_new_start, subtract_mean=False, track_mode="nearest")

        plt.errorbar(
            fake_geo_toas.get_mjds(),
            r.time_resids.to_value("us"),
            r.get_data_error().to_value("us"),
            marker="+",
            ls="",
        )
    try:
        # Use the fake TOAs to fit the model with the new ephemeris
        f = pint.fitter.DownhillWLSFitter(fake_geo_toas, model_new_start)
        f.fit_toas()  # fit_toas() returns the final reduced chi squared
    except pint.exceptions.StepProblem as e:
        log.error(f"StepProblem encountered: {e}")
        log.error("This may be due to a bad initial guess for the model parameters.")
        log.error("Try changing the initial guess or using a different ephemeris.")

    if plot:
        r = Residuals(fake_geo_toas, f.model, subtract_mean=False, track_mode="nearest")
        plt.errorbar(
            fake_geo_toas.get_mjds(),
            r.time_resids.to_value("us"),
            r.get_data_error().to_value("us"),
            marker="+",
            ls="",
        )
        plt.savefig(fname.replace(".par", "_residuals.jpg"))
        plt.close(plt.gcf())
    rms = f.resids.rms_weighted()
    if rms > rms_tolerance:
        log.error(f"{rms} > {rms_tolerance} at MJD {t0_mjd:.2}-{t1_mjd:.2}")

    new_model = f.model
    # Substitute with get_TZR_toas, using one of the simulated TOAs?
    new_model.components["AbsPhase"].make_TZR_toa(fake_geo_toas)
    new_model.TZRSITE.value = "0"
    new_model.PHOFF.quantity = current_phoff
    log.info(str(new_model.compare(model_200)))
    new_model.write_parfile(fname, include_info=False)
    return new_model


def get_crab_ephemeris(MJD, fname=None, ephem="DE200", force_parameters=None, format="cgro"):
    """
    Get the Crab ephemeris for a given MJD.

    Parameters
    ----------
    MJD : float
        The Modified Julian Date for which to get the ephemeris.
    fname : str, optional
        The filename to save the ephemeris to. If None, a default name will be generated.
    ephem : str, optional
        The ephemeris to use. Default is "DE200".
    force_parameters : dict, optional
        A dictionary of parameters to force in the model. The keys should be the parameter names,
        and the values should be lists containing the value
    """
    log.info(f"Getting correct ephemeris for MJD {MJD}")

    if format in ["txt", "text"]:
        model_200 = create_model_from_txt(MJD)
    elif format in ["cgro", "gro"]:
        model_200 = create_model_from_cgro(MJD)
    else:
        raise ValueError(f"Unknown format: {format}")

    model_200.write_parfile(_default_fname(MJD, ephem="DE200"), include_info=False)
    if fname is None:
        fname = _default_fname(
            model_200.PEPOCH.value, ephem=ephem, force_parameters=force_parameters
        )

    if ephem.upper() == "DE200" and force_parameters is None:
        return model_200

    log.info(f"ephem={ephem}, REFITTING")

    fit_model = refit_solution(
        model_200,
        ephem,
        rms_tolerance=1 * u.us,
        force_parameters=force_parameters,
        fname=fname,
        plot=True,
    )

    return fit_model
