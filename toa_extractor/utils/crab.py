import os
from astropy import log
from urllib.request import urlopen
from astropy.table import Table
import numpy as np


def retrieve_ephemeris():
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


def get_best_ephemeris(MJD):
    """

    From the instructions:
    http://www.jb.man.ac.uk/research/pulsar/crab/CGRO_format.html
    the integer part of the *geocentric* TOA time is also the *TDB*(!) PEPOCH for the ephemeris.
    """
    table = retrieve_ephemeris()
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


def get_crab_ephemeris(MJD, fname=None):
    log.info("Getting correct ephemeris")
    ephem = get_best_ephemeris(MJD)

    if fname is None:
        fname = f"Crab_{ephem.PEPOCH}.par"

    with open(fname, "w") as fobj:
        print("PSRJ            J0534+2200", file=fobj)
        print("RAJ             05:34:31.973", file=fobj)
        print("DECJ            +22:00:52.06", file=fobj)
        print("PEPOCH          ", ephem.PEPOCH, file=fobj)
        print("F0              ", ephem.F0, file=fobj)
        print("F1              ", ephem.F1, file=fobj)
        print("F2              ", ephem.F2, file=fobj)
        print("TZRMJD          ", ephem.TZRMJD, file=fobj)
        print("TZRSITE         ", ephem.TZRSITE, file=fobj)
        print("EPHEM           DE200", file=fobj)

    return ephem
