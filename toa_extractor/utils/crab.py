import os
from io import StringIO
from astropy import log
from urllib.request import urlopen
from astropy.table import Table
from astroquery.heasarc import Heasarc
import numpy as np
import astropy.units as u
import pint
import pint.simulation
from pint.models import get_model
from pint.toa import TOAs, TOA


from astropy.coordinates import SkyCoord

crab_coords = SkyCoord("05h34m31.972", "22d00m52.07", frame="fk5")
crab_coords_icrs = crab_coords.icrs


def retrieve_cgro_ephemeris():
    file_name = "Crab.gro"
    url = "http://www.jb.man.ac.uk/pulsar/crab/all.gro"

    if not os.path.exists(file_name):
        response = urlopen(url)
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
    """

    From the instructions:
    http://www.jb.man.ac.uk/research/pulsar/crab/CGRO_format.html
    the integer part of the *geocentric* TOA time is also the *TDB*(!) PEPOCH for the ephemeris.
    """
    table = retrieve_cgro_ephemeris()
    good = (MJD >= table["MJD1"]) & (MJD < table["MJD2"] + 1)
    if not np.any(good):
        return None
    return table[good][0]


def get_best_cgro_ephemeris(MJD):
    """

    From the instructions:
    http://www.jb.man.ac.uk/research/pulsar/crab/CGRO_format.html
    the integer part of the *geocentric* TOA time is also the *TDB*(!) PEPOCH for the ephemeris.
    """
    row = get_best_cgro_row(MJD)

    if row is None:
        return None

    result = type("result", (object,), {})()
    result.F0 = np.double(row["f0(s^-1)"])
    result.F1 = np.double(row["f1(s^-2)"].replace("D", "E"))
    result.F2 = np.double(row["f2(s^-3)"].replace("D", "E"))

    result.TZRMJD = np.double(row["t0geo(MJD)"])
    result.TZRSITE = "0"
    result.PEPOCH = np.floor(row["t0geo(MJD)"])
    return result


def get_best_txt_ephemeris(mjd):
    """Uses the astroquery.heasarc interface."""
    heasarc = Heasarc()
    table = heasarc.query_object("Crab", mission="crabtime", fields="All")
    good = np.argmin(np.abs(mjd - table["MJD"]))
    row = table[good]

    pepoch = row["MJD"] + row["JPL_TIME"] / 86400
    f = row["NU"]
    fd = row["NU_DOT"]
    p, pd = 1 / f, -1 / f**2 * fd

    # F2 is calculated from P0, P1, as per instructions
    fdd = 2.0 * pd**2 / p**3

    result = type("result", (object,), {})()
    result.F0 = f
    result.F1 = fd
    result.F2 = fdd

    result.TZRMJD = pepoch
    result.TZRSITE = "@"
    result.PEPOCH = pepoch

    return result


def get_model_str(
    ephem, F0, F1, F2, TZRMJD, START, FINISH, include_proper_motion=False
):
    coords = crab_coords

    # if ephem.upper() == "DE200":
    #     coords = crab_coords
    # else:
    #     coords = crab_coords_icrs

    if isinstance(TZRMJD, str):
        pepoch = TZRMJD.split(".")[0]
    else:
        pepoch = int(float(TZRMJD))
    ra = ":".join([f"{v:g}" for v in coords.ra.hms])
    dec = ":".join([f"{v:g}" for v in coords.dec.dms])

    model_str = rf"""PSRJ            J0534+2200
    RAJ              {ra}
    DECJ             {dec}
    POSEPOCH         40706
    PEPOCH           {pepoch}
    F0               {F0} 1
    F1               {F1} 1
    F2               {F2} 1
    EPHEM            {ephem}
    UNITS TDB
    CLOCK TT(TAI)
    TZRFRQ           0
    TZRMJD           {TZRMJD}
    TZRSITE 0
    START            {START}
    FINISH           {FINISH}
    """
    if include_proper_motion:
        model_str += """PMRA            -14.7                         8.000e-01
    PMDEC           2.0                           8.000e-01
    """
    return model_str


def refit_solution(
    model_200, new_ephem, rms_tolerance=None, include_proper_motion=False
):
    if rms_tolerance is None:
        rms_tolerance = 1 * u.us

    t0_mjd = np.longdouble(model_200.START.value) - 1
    t1_mjd = np.longdouble(model_200.FINISH.value) + 1

    # Create a starting model with exactly the same parameters as the original one, but the new ephemeris
    with StringIO(
        get_model_str(
            new_ephem,
            model_200.F0.value,
            model_200.F1.value,
            model_200.F2.value,
            model_200.TZRMJD.value,
            t0_mjd,
            t1_mjd,
            include_proper_motion=include_proper_motion,
        )
    ) as f:
        model_new_start = get_model(f)

    # Create a bunch of geocenter TOAs with the original DE200 model
    fake_geo_toas = pint.simulation.make_fake_toas_uniform(
        t0_mjd, t1_mjd, 101, model_200, freq=np.inf
    )

    # Initial residuals
    # start_residuals = pint.residuals.Residuals(fake_geo_toas, model_new_start)

    # Use the fake TOAs to fit the model with the new ephemeris
    f = pint.fitter.WLSFitter(fake_geo_toas, model_new_start)
    f.fit_toas()  # fit_toas() returns the final reduced chi squared
    # fit_residuals = f.resids.time_resids.to(u.us)

    rms = f.resids.rms_weighted()
    if rms > rms_tolerance:
        print(f"ERROR: {rms} > {rms_tolerance}")

    return f.model


def get_crab_ephemeris(MJD, fname=None, ephem="DE200"):
    log.info("Getting correct ephemeris")
    # ephem_cgro = get_best_cgro_ephemeris(MJD)
    # ephem_txt = get_best_txt_ephemeris(MJD)

    row = get_best_cgro_row(MJD)
    f0, f1, f2, geo_toa, t0_mjd, t1_mjd, rms_mP = (
        row["f0(s^-1)"],
        row["f1(s^-2)"],
        row["f2(s^-3)"],
        row["t0geo(MJD)"],
        row["MJD1"],
        row["MJD2"],
        row["RMS"],
    )

    rms_t = rms_mP / 1000 / f0 * u.s

    with StringIO(get_model_str("DE200", f0, f1, f2, geo_toa, t0_mjd, t1_mjd)) as f:
        model_200 = get_model(f)

    if fname is None:
        fname = f"Crab_{model_200.PEPOCH.value}.par"

    if ephem.upper() == "DE200":
        model_200.write_parfile(fname)
        return model_200

    print(f"ephem={ephem}, REFITTING")

    fit_model = refit_solution(
        model_200,
        ephem,
        rms_tolerance=rms_t / 10,
        include_proper_motion=False,
    )
    fit_model.write_parfile(fname)
    # geo_toas = TOAs(toalist=[TOA(geo_toa, freq=np.inf, obs="0")])
    # geo_toas.compute_TDBs(ephem=ephem)
    # geo_toas.compute_posvels()
    # tdb_ref_toa = fit_model.get_barycentric_toas(geo_toas).value[0]

    # new_cgro_table["f0(s^-1)"][i] = fit_model.F0.value
    # new_cgro_table["f1(s^-2)"][i] = f"{fit_model.F1.value:.5e}"
    # new_cgro_table["f2(s^-3)"][i] = f"{fit_model.F2.value:+.2e}"
    # new_cgro_table["B"][i] = new_ephem
    # new_cgro_table["t0tdb"][i] = tdb_ref_toa
    # new_cgro_table["epoch"][i] = fit_model.PEPOCH.value
    # with open(fname, "w") as fobj:
    #     print("PSRJ            J0534+2200", file=fobj)
    #     print("RAJ             05:34:31.973", file=fobj)
    #     print("DECJ            +22:00:52.06", file=fobj)
    #     print(f"PEPOCH         {fit_model.PEPOCH.value}", file=fobj)
    #     print(f"F0             {fit_model.F0.value}", file=fobj)
    #     print(f"F1             {fit_model.F1.value}", file=fobj)
    #     print(f"F2             {fit_model.F2.value}", file=fobj)
    #     print(f"TZRMJD         {fit_model.TZRMJD.value}", file=fobj)
    #     print(f"TZRSITE        {fit_model.TZRSITE.value}", file=fobj)
    #     print("TZRFRQ          0", file=fobj)
    #     print("EPHEM           ", file=fobj)
    #     print("UNITS           TDB", file=fobj)
    #     print("CLK             TT(TAI)", file=fobj)

    return fit_model
