import numpy as np
import pytest
from toa_extractor.utils.crab import get_crab_ephemeris
from toa_extractor import check_longdouble_precision


# Check if architecture is Linux and longdouble is at least 80 bits
skip_tests = not check_longdouble_precision()

pytestmark = pytest.mark.skipif(skip_tests, reason="Tests require longdouble with at least 80 bits")


@pytest.mark.parametrize("ephem", ["DE200", "DE430", "DE440"])
def test_refit_solution_ephem(ephem):
    mjd = 57974.71650905952  # Problematic in some previous tests
    model = get_crab_ephemeris(mjd, ephem=ephem, force_parameters=None)

    if ephem.upper() != "DE200":
        assert model.TRES.value < 2


@pytest.mark.parametrize("mjd", np.random.default_rng(20261009).uniform(44000, 70000, 5))
def test_refit_solution_mjd(mjd):
    model = get_crab_ephemeris(mjd, ephem="DE430", force_parameters=None)
    assert model.TRES.value < 2


def test_refit_solution_leap_second():
    """The monthly solution ending on 1989-12-31 must not put FINISH inside the leap second.

    With fake TOAs up to FINISH + 1, the last one fell in the leap second and PINT could not
    represent FINISH as an MJD.
    """
    model = get_crab_ephemeris(47873.269707842075, ephem="DE430", force_parameters=None)
    assert model.TRES.value < 2
