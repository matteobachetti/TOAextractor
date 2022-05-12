import os
from astropy import log
from urllib.request import urlopen
from astropy.table import Table
from astroquery.heasarc import Heasarc
import numpy as np


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


def get_best_cgro_ephemeris(MJD):
    """

    From the instructions:
    http://www.jb.man.ac.uk/research/pulsar/crab/CGRO_format.html
    the integer part of the *geocentric* TOA time is also the *TDB*(!) PEPOCH for the ephemeris.
    """
    table = retrieve_cgro_ephemeris()
    good = (MJD >= table["MJD1"]) & (MJD < table["MJD2"] + 1)
    if not np.any(good):
        return None
    result = type("result", (object,), {})()
    result.F0 = np.double(table["f0(s^-1)"][good][0])
    result.F1 = np.double(table["f1(s^-2)"][good][0].replace("D", "E"))
    result.F2 = np.double(table["f2(s^-3)"][good][0].replace("D", "E"))

    result.TZRMJD = np.double(table["t0geo(MJD)"][good][0])
    result.TZRSITE = "0"
    result.PEPOCH = np.floor(table["t0geo(MJD)"][good][0])
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
    p, pd = 1/f, -1 / f**2 * fd

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


def get_crab_ephemeris(MJD, fname=None):
    log.info("Getting correct ephemeris")
    ephem_cgro = get_best_cgro_ephemeris(MJD)
    ephem_txt = get_best_txt_ephemeris(MJD)

    if fname is None:
        fname = f"Crab_{ephem_txt.PEPOCH}.par"

    with open(fname, "w") as fobj:
        print("PSRJ            J0534+2200", file=fobj)
        print("RAJ             05:34:31.973", file=fobj)
        print("DECJ            +22:00:52.06", file=fobj)
        print("PEPOCH          ", ephem_txt.PEPOCH, file=fobj)
        print("F0              ", ephem_txt.F0, file=fobj)
        print("F1              ", ephem_txt.F1, file=fobj)
        print("F2              ", ephem_txt.F2, file=fobj)
        print("TZRMJD          ", ephem_cgro.TZRMJD, file=fobj)
        print("TZRSITE         ", ephem_cgro.TZRSITE, file=fobj)
        print("TZRFRQ          0", file=fobj)
        print("EPHEM           DE200", file=fobj)
        print("UNITS           TDB", file=fobj)
        print("CLK             TT(TAI)", file=fobj)

    return ephem_txt
