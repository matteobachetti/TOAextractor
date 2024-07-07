import copy
import pytest
import numpy as np
from toa_extractor.utils.fit_crab_profiles import (
    default_crab_model,
    fit_crab_profile,
    fill_template_table,
    normalize_phase_0d5,
)


# From true stories...
ugly_profile_pars = {
    "amplitude_0": 580.5903589923868,
    "amplitude_0_err": 0.00021299270007292819,
    "amplitude_1": 4.612867710759439,
    "amplitude_1_err": 0.017453425559444458,
    "amplitude00_2": 0.1901417048997574,
    "amplitude00_2_err": 0.12543719685127663,
    "amplitude10_2": 0.9611531862833874,
    "amplitude10_2_err": 0.07426486951581325,
    "x00_2": -0.03596863207798328,
    "x00_2_err": 0.004149855533651924,
    "dx00_2": 0.009999999999995854,
    "dx00_2_err": 0.010943609424149495,
    "fwhm00_2": 0.022717488224215883,
    "fwhm00_2_err": 0.017026939919114102,
    "fwhm10_2": 0.08405233582151124,
    "fwhm10_2_err": 0.014533983831333249,
    "fwhm20_2": 0.19999999978155944,
    "fwhm20_2_err": 0.013345730691355482,
    "peak_separation_2": 0.40519483804790085,
    "peak_separation_2_err": 0.005018201010357339,
    "amplitude01_2": 0.2077954418808529,
    "amplitude01_2_err": 0.06493390019688734,
    "amplitude11_2": 1.1983534253104708,
    "amplitude11_2_err": 0.05330261787980353,
    "dx01_2": 0.0029345016863788678,
    "dx01_2_err": 0.004888074684843476,
    "fwhm01_2": 0.021057160774264286,
    "fwhm01_2_err": 0.013121679435918627,
    "fwhm11_2": 0.19999999991737827,
    "fwhm11_2_err": 0.013560870375961664,
    "fwhm21_2": 0.12037012249789268,
    "fwhm21_2_err": 0.009705445389561067,
}


good_profile_pars = {
    "amplitude_0": 2153.918610567955,
    "amplitude_0_err": 9.732494154488217e-06,
    "amplitude_1": 2.035640398880682,
    "amplitude_1_err": 0.0031469108922534397,
    "amplitude00_2": 0.6740095780943907,
    "amplitude00_2_err": 0.13380201470555703,
    "amplitude10_2": 0.8130120179080282,
    "amplitude10_2_err": 0.12571310637921743,
    "x00_2": 0.36583788607189816,
    "x00_2_err": 0.0004333888281412374,
    "dx00_2": -0.009999999999999997,
    "dx00_2_err": 0.0025673688930350456,
    "fwhm00_2": 0.020299779116460737,
    "fwhm00_2_err": 0.003062805688373827,
    "fwhm10_2": 0.05694094514459118,
    "fwhm10_2_err": 0.002098916395629907,
    "fwhm20_2": 0.05760567145278637,
    "fwhm20_2_err": 0.005664821878810293,
    "peak_separation_2": 0.40094540095358183,
    "peak_separation_2_err": 0.0016472804210240512,
    "amplitude01_2": 0.29832520838768883,
    "amplitude01_2_err": 0.09829058440713954,
    "amplitude11_2": 0.43955020559368185,
    "amplitude11_2_err": 0.07578691103080301,
    "dx01_2": -0.009999999999898625,
    "dx01_2_err": 0.00941426206642837,
    "fwhm01_2": 0.03697546621588917,
    "fwhm01_2_err": 0.0073109542740559886,
    "fwhm11_2": 0.14885927102589064,
    "fwhm11_2_err": 0.0088531587276771,
    "fwhm21_2": 0.08142310957237843,
    "fwhm21_2_err": 0.011420604499466004,
}


def _simulate_and_fit(pars, nbin):
    x0_start = pars["x00_2"]
    model_sim = default_crab_model(init_pars=pars)
    phases = np.arange(0, 1, 1 / nbin)
    profile_sim = np.random.poisson(model_sim(phases))
    init, final = fit_crab_profile(phases, profile_sim, fitter=None)
    template_table = fill_template_table(final, nbins=nbin)
    dphase = normalize_phase_0d5([template_table.meta["phase_max"] - x0_start])
    assert np.isclose(
        np.abs(dphase),
        0,
        atol=max(template_table.meta["phase_max_err"] * 3, 0.05),
    )


class TestFit:

    @pytest.mark.parametrize("amplitude0", [100, 100000])
    @pytest.mark.parametrize("amplitude1", [0, 4])
    @pytest.mark.parametrize("x0_start", np.arange(0, 1, 1 / 3))
    @pytest.mark.parametrize("nbin", [64, 512])
    def test_ugly_fit(self, amplitude0, amplitude1, x0_start, nbin):

        pars = copy.deepcopy(ugly_profile_pars)
        pars["amplitude_0"] = amplitude0
        pars["amplitude_1"] = amplitude1
        pars["x00_2"] = x0_start
        _simulate_and_fit(pars, nbin)

    @pytest.mark.parametrize("amplitude0", [100, 100000])
    @pytest.mark.parametrize("amplitude1", [0, 4])
    @pytest.mark.parametrize("x0_start", np.arange(0, 1, 1 / 3))
    @pytest.mark.parametrize("nbin", [64, 512])
    def test_good_fit(self, amplitude0, amplitude1, x0_start, nbin):

        pars = copy.deepcopy(good_profile_pars)
        pars["amplitude_0"] = amplitude0
        pars["amplitude_1"] = amplitude1
        pars["x00_2"] = x0_start
        _simulate_and_fit(pars, nbin)
