import copy

import matplotlib.pyplot as plt
import numpy as np
from astropy import log
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const1D, custom_model
from astropy.table import Table
from scipy.optimize import fmin

from toa_extractor.utils import root_name


def _plot_profile_and_fit(
    phases,
    profile,
    model_fit,
    axfit,
    axres,
    model_init=None,
    phase_max=None,
    profile_raw=None,
):
    residuals = profile - model_fit(phases)

    label = "Data"
    if profile_raw is not None:
        axfit.plot(phases, profile_raw, ds="steps-mid", color="r", alpha=0.5, label="Data")
        label = "Deadtime-corrected Data"
    axfit.plot(phases, profile, ds="steps-mid", color="r", label=label)

    if model_init is not None:
        axfit.plot(phases, model_init(phases), color="grey", alpha=0.5, label="Init model")
    axfit.plot(phases, model_fit(phases), color="k", zorder=20, lw=1.5, label="Fit model")

    peak_colors = ["b", "navy"]
    m = model_fit[2]

    sym_lor_fit0 = lorentzian(phases, amplitude=m.amplitude00, x0=m.x00, fwhm=m.fwhm00)
    sym_lor_fit1 = lorentzian(
        phases, amplitude=m.amplitude01, x0=m.x00 + m.peak_separation, fwhm=m.fwhm01
    )
    asym_lor_fit0 = asymmetric_lorentzian(
        phases,
        amplitude=m.amplitude10,
        x0=m.x00 + m.dx00,
        fwhm1=m.fwhm10,
        fwhm2=m.fwhm20,
    )
    asym_lor_fit1 = asymmetric_lorentzian(
        phases,
        amplitude=m.amplitude11,
        x0=m.x00 + m.peak_separation + m.dx01,
        fwhm1=m.fwhm11,
        fwhm2=m.fwhm21,
    )
    for peak in sym_lor_fit0, asym_lor_fit0:
        axfit.plot(
            phases,
            (peak + model_fit[1](phases)) * model_fit[0](phases),
            color=peak_colors[0],
            zorder=10,
            ls=":",
        )
    for peak in sym_lor_fit1, asym_lor_fit1:
        axfit.plot(
            phases,
            (peak + model_fit[1](phases)) * model_fit[0](phases),
            color=peak_colors[1],
            zorder=10,
            ls=":",
        )

    axfit.grid(True)

    axres.plot(phases, residuals, ds="steps-mid", color="r")
    axres.axhline(0, color="k", ls=":")
    axres.axhspan(-np.std(residuals), np.std(residuals), color="b", alpha=0.2)
    if phase_max is not None:
        axfit.axvline(phase_max, color="k", ls="--")
        axres.axvline(phase_max, color="k", ls="--")
    axres.grid(True)


def normalize_phase_0d5(phase):
    """Normalize phase between -0.5 and 0.5

    Examples
    --------
    >>> normalize_phase_0d5(0.5)
    0.5
    >>> normalize_phase_0d5(-0.5)
    0.5
    >>> normalize_phase_0d5(4.25)
    0.25
    >>> normalize_phase_0d5(-3.25)
    -0.25
    """
    while phase > 0.5:
        phase -= 1
    while phase <= -0.5:
        phase += 1
    return phase


normalize_phase_0d5 = np.vectorize(normalize_phase_0d5)


def _rough_lor(delta_x, fwhm=1):
    """A Lorentzian curve centered at 0"""
    gamma = fwhm / 2
    return gamma**2 / (gamma**2 + delta_x**2)


def lorentzian(x, amplitude=1, x0=0, fwhm=1):
    x = np.asanyarray(x)
    delta_x = x - x0
    delta_x = normalize_phase_0d5(delta_x)

    vals = _rough_lor(delta_x, fwhm=fwhm)
    vals += _rough_lor(delta_x + 1, fwhm=fwhm)
    vals += _rough_lor(delta_x - 1, fwhm=fwhm)
    return vals * amplitude


SingleLorentzian = custom_model(lorentzian)


def _rough_asym_lor(delta_x, fwhm1=1, fwhm2=1):
    vals = np.zeros_like(delta_x)
    gamma1 = fwhm1 / 2
    gamma2 = fwhm2 / 2
    lower = delta_x < 0
    vals[lower] = gamma1**2 / (gamma1**2 + delta_x[lower] ** 2)
    vals[~lower] = gamma2**2 / (gamma2**2 + delta_x[~lower] ** 2)
    return vals


def asymmetric_lorentzian(x, amplitude=1, x0=0, fwhm1=1, fwhm2=1):
    x = np.asanyarray(x)
    delta_x = x - x0
    delta_x = normalize_phase_0d5(delta_x)

    vals = _rough_asym_lor(delta_x, fwhm1=fwhm1, fwhm2=fwhm2)
    vals += _rough_asym_lor(delta_x + 1, fwhm1=fwhm1, fwhm2=fwhm2)
    vals += _rough_asym_lor(delta_x - 1, fwhm1=fwhm1, fwhm2=fwhm2)
    return vals * amplitude


AsymmetricLorentzian = custom_model(asymmetric_lorentzian)


def skewed_peak(x, amplitude0=1, amplitude1=1, x0=0, dx0=0, fwhm0=1, fwhm1=1, fwhm2=1):
    x = np.asanyarray(x)

    sym_lor = lorentzian(x, amplitude=amplitude0, x0=x0, fwhm=fwhm0)
    asym_lor = asymmetric_lorentzian(x, amplitude=amplitude1, x0=x0 + dx0, fwhm1=fwhm1, fwhm2=fwhm2)

    return sym_lor + asym_lor


SkewedPeak = custom_model(skewed_peak)


def full_crab_profile_model(
    x,
    amplitude00=1,
    amplitude10=1,
    x00=0,
    dx00=0,
    fwhm00=1,
    fwhm10=1,
    fwhm20=1,
    peak_separation=0.4,
    amplitude01=1,
    amplitude11=1,
    dx01=0,
    fwhm01=1,
    fwhm11=1,
    fwhm21=1,
):
    peak1 = skewed_peak(
        x,
        amplitude0=amplitude00,
        amplitude1=amplitude10,
        x0=x00,
        dx0=dx00,
        fwhm0=fwhm00,
        fwhm1=fwhm10,
        fwhm2=fwhm20,
    )
    peak2 = skewed_peak(
        x,
        amplitude0=amplitude01,
        amplitude1=amplitude11,
        x0=x00 + peak_separation,
        dx0=dx01,
        fwhm0=fwhm01,
        fwhm1=fwhm11,
        fwhm2=fwhm21,
    )
    res = peak1 + peak2
    # if amplitude10 > amplitude00:
    #     res *= 0.0
    return res


FullCrab = custom_model(full_crab_profile_model)


def default_crab_model(init_pars=None, frozen=None):
    """Default model for the Crab

    The model consists of an external normalization parameter, a DC level,
    and two peaks composed each of a symmetric Lorentzian and a broader asymmetric
    Lorentzian

    Parameters
    ----------
    init_pars: dict
        Initial parameters for the fit. They keys of the dictionary
        must contain the same names as the Astropy model:

        + ``amplitude_0``: external normalization
        + ``amplitude_1``: The DC level
        + ``amplitude00_2``: The amplitude of the main peak's symmetric Lorentzian
        + ``amplitude10_2``: The amplitude of the main peak's asymmetric Lorentzian
        + ``amplitude01_2``: The amplitude of the secondary peak's symmetric Lorentzian
        + ``amplitude11_2``: The amplitude of the secondary peak's asymmetric Lorentzian
        + ``peak_separation_2``: the separation between the symmetric Lorentzians of the two peaks
        + ``x00_2``: the phase of the main peak's symmetric Lorentzian
        + ``dx10_2``: the separation in phase of the main peak's asymmetric Lorentzian
          wrt the symmetric one.
        + ``dx11_2``: the separation in phase of the main peak's asymmetric Lorentzian
          wrt the symmetric one.
        + ``fwhm00_2``: The FWHM of the main peak's symmetric Lorentzian
        + ``fwhm10_2``: The right FWHM of the main peak's asymmetric Lorentzian
        + ``fwhm20_2``: The left FWHM of the main peak's asymmetric Lorentzian
        + ``fwhm01_2``: The FWHM of the secondary peak's symmetric Lorentzian
        + ``fwhm11_2``: The right FWHM of the secondary peak's asymmetric Lorentzian
        + ``fwhm21_2``: The left FWHM of the secondary peak's asymmetric Lorentzian

    frozen: list, default None
        list of frozen parameters
    """
    bounds_fullcrab = dict(
        [(val, (0, np.inf)) for val in ["amplitude00", "amplitude10", "amplitude01", "amplitude11"]]
    )
    bounds_fullcrab.update(
        dict([(val, (0.01, 0.2)) for val in ["fwhm10", "fwhm20", "fwhm11", "fwhm21"]])
    )
    bounds_fullcrab.update(dict([(val, (0.01, 0.1)) for val in ["fwhm00", "fwhm01"]]))
    bounds_fullcrab.update(dict([(val, (-0.01, 0.01)) for val in ["dx00", "dx01"]]))
    bounds_fullcrab["peak_separation"] = (
        0.35,
        0.45,
    )  # Account for the possibility of misinterpreting the first peak

    model_init = Const1D(bounds={"amplitude": (0, np.inf)}) * (
        Const1D(bounds={"amplitude": (0, np.inf)}) + FullCrab(bounds=bounds_fullcrab)
    )
    if init_pars is not None:
        for key, val in init_pars.items():
            setattr(model_init, key, val)

    if frozen is not None:
        for key in frozen:
            if key in model_init.param_names:
                print("Freezing", key)
                getattr(model_init, key).fixed = True
    return model_init


def get_initial_parameters(input_phases, profile):
    """Estimate the initial parameters for the fit.

    The critical point of the operation is finding the main peak.
    In the Crab, we know that the two peaks are spaced by about 0.4 in phase.
    The peak position is estimated as the maximum of the sum of the pulse
    profile by the same profile shifted by 0.4 in phase.
    The rest is pretty straightforward: global amplitudes are estimated by the count
    per bins at the maximum and minimum, the amplitudes of each peak component
    are hardcoded, which is good enough for a first guess.

    """
    from scipy.signal import savgol_filter

    if len(profile) > 200:
        window_length = profile.size / 50
        polyorder = min(3, window_length - 1)
        profile = savgol_filter(profile, window_length, polyorder, mode="wrap", cval=0.0)

    roll_by = int(profile.size * 0.4)

    phases = copy.deepcopy(input_phases)
    phases = normalize_phase_0d5(phases)
    order = np.argsort(phases)
    phases = phases[order]
    profile = copy.deepcopy(profile)[order]

    phases = np.concatenate([phases, phases + 1])
    profile = np.concatenate([profile, profile])
    probe_prof = profile + np.roll(profile, -roll_by)
    idx_1 = np.argmax(probe_prof)

    ph1 = phases[idx_1]
    baseline = np.min(profile)

    amplitude1 = profile[idx_1] - baseline
    if baseline < 0:
        amplitude1 += -baseline
        baseline = 0

    fwhm1 = 0.03
    fwhm2 = 0.08

    # peak 2
    prof_filt2 = copy.deepcopy(profile)
    prof_filt2[np.abs(phases + ph1 - 0.4) > 0.2] = 0
    idx_2 = np.argmax(prof_filt2)
    amplitude2 = max(profile[idx_2] - baseline, 0)

    init_pars = {
        "amplitude_0": amplitude1,  # external normalization
        "amplitude_1": baseline / amplitude1,  # DC level
        "amplitude00_2": 0.75,  # main peak's sym Lor
        "amplitude10_2": 0.5,  # main peak's asym Lor
        "x00_2": ph1,  # phase of main peak's sym Lor
        "dx00_2": -0.01,  # phase separation of main peak's asym Lor
        "fwhm00_2": fwhm1,  # FWHM of main peak's sym Lor
        "fwhm10_2": fwhm1 * 2,  # right FWHM of main peak's asym Lor
        "fwhm20_2": fwhm1 * 2,  # left FWHM of main peak's asym Lor
        "amplitude01_2": amplitude2 / amplitude1 / 10 * 6,  # ampl. of secondary peak's sym Lor
        "amplitude11_2": amplitude2 / amplitude1 / 10 * 4,  # ampl. of secondary peak's asym Lor
        "peak_separation": ph1 + 0.4,  # separation between the sym Lors of the two peaks
        "dx01_2": 0.0,  # separation in phase of secondary peak's asym Lor
        "fwhm01_2": fwhm2,  # FWHM of secondary peak's sym Lor
        "fwhm11_2": fwhm2 * 2,  # right FWHM of secondary peak's asym Lor
        "fwhm21_2": fwhm2 * 2,  # left FWHM of secondary peak's asym Lor
    }
    return init_pars


def fit_crab_profile(phases, profile, fitter=None, init_pars=None, frozen=None):
    """Fit a Crab profile with a mixture of symmetric and asymmetric Lorentzians

    Parameters
    ----------
    phases : array-like
        Pulsation phases of the profile, between 0 and 1
    profile : array-like
        Flux values corresponding to each phase

    Other parameters
    ----------------
    fitter : astropy.modeling.fitting.Fitter
        The fitter to use (default is :class:`astropy.modeling.fitting.TRFLSQFitter`)
    init_pars: dict
        The initial model parameters to use for the fit. If None, a default model is used
    frozen: list
        List of parameters to freeze

    Returns
    -------
    model_init: astropy.modeling.Model
        The initial fit model
    model_fit: astropy.modeling.Model
        The best fit model
    """
    if fitter is None:
        fitter = TRFLSQFitter(calc_uncertainties=True)
    if init_pars is None:
        init_pars = get_initial_parameters(phases, profile)
    model_init = default_crab_model(init_pars=init_pars, frozen=frozen)
    log.info("Initial parameters:")
    for par in model_init.param_names:
        fixed_str = "fixed" if getattr(model_init, par).fixed else "free"
        val = getattr(model_init, par).value
        bounds = getattr(model_init, par).bounds
        log.info(f"    {par}: {val:.2e} {fixed_str} {bounds}")

    weights = None
    if np.all(profile > 20):
        weights = 1 / np.sqrt(profile)

    model_fit = fitter(
        model_init,
        phases,
        profile,
        maxiter=200,
        weights=weights,
        acc=0.001,
    )
    log.info("Fit parameters:")
    log.info(f"External normalization: {model_fit.amplitude_0.value:.2e}")
    log.info(f"DC level: {model_fit.amplitude_1.value:.2e}")
    log.info(f"    Main peak phase: {model_fit.x00_2.value:.2e}")
    log.info(f"    Main peak sym. comp. amplitude: {model_fit.amplitude00_2.value:.2e}")
    log.info(f"    Main peak sym. comp. FWHM: {model_fit.fwhm00_2.value:.2e}")
    log.info(f"    Main peak comp. sep.: {model_fit.dx00_2.value:.2e}")
    log.info(f"    Main peak asym comp. amplitude: {model_fit.amplitude10_2.value:.2e}")
    log.info(f"    Main peak asym comp. FWHM (L): {model_fit.fwhm10_2.value:.2e}")
    log.info(f"    Main peak asym comp. FWHM (R): {model_fit.fwhm20_2.value:.2e}")
    log.info(f"    Peak separation: {model_fit.peak_separation_2.value:.2e}")
    log.info(f"    Secondary peak sym comp. amplitude: {model_fit.amplitude01_2.value:.2e}")
    log.info(f"    Secondary peak sym. comp. FWHM: {model_fit.fwhm01_2.value:.2e}")
    log.info(f"    Secondary peak comp. sep.: {model_fit.dx01_2.value:.2e}")
    log.info(f"    Secondary peak asym comp. amplitude: {model_fit.amplitude11_2.value:.2e}")
    log.info(f"    Secondary peak asym comp. FWHM (L): {model_fit.fwhm11_2.value:.2e}")
    log.info(f"    Secondary peak asym comp. FWHM (R): {model_fit.fwhm21_2.value:.2e}")
    return model_init, model_fit


def fill_template_table(model_fit, nbins=512, template_table=None, model_init=None):
    if template_table is None:
        template_table = Table()
    phase_max = fmin(lambda x: -model_fit(x), model_fit.x00_2)[0]
    phases = np.arange(nbins) / nbins

    renorm_model = copy.deepcopy(model_fit)
    renorm_model.amplitude_0 = 1
    renorm_model.amplitude_1 = 0
    factor = 1 / renorm_model(phase_max)
    template_table["phase"] = phases
    template_table["profile"] = factor * renorm_model(phases + phase_max)
    template_table["profile_raw"] = model_fit(phases)

    free_par_names = [par for par in model_fit.param_names if not getattr(model_fit, par).fixed]

    par = [getattr(model_fit, par).value for par in free_par_names]
    par_err = [model_fit.cov_matrix.cov_matrix[i, i] ** 0.5 for i in range(len(par))]

    template_table.meta["best_fit"] = {}
    for name, p in zip(model_fit.param_names, model_fit.parameters):
        template_table.meta["best_fit"][name] = p

    for name, pe in zip(free_par_names, par_err):
        template_table.meta["best_fit"][name + "_err"] = pe

    if model_init is not None:
        par = model_init.parameters
        template_table.meta["model_init"] = {}
        for name, p in zip(model_init.param_names, par):
            template_table.meta["model_init"][name] = p

    template_table.meta["phase_max"] = phase_max
    template_table.meta["phase_max_err"] = template_table.meta["best_fit"]["x00_2_err"]

    return template_table


def plot_fit_diagnostics(
    phases, profile, model_fit, phase_max=None, model_init=None, plot_file=None
):
    """Make a pretty plot to show how the fit went."""
    if phase_max is None:
        phase_max = 0

    if np.min(phases) >= 0 and np.max(phases) <= 1:
        phases = np.concatenate([phases - 1, phases, phases + 1])
        profile = np.concatenate([profile, profile, profile])

    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gs0 = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

    gs00 = gs0[0].subgridspec(2, 1, hspace=0, height_ratios=[1.5, 1])
    gs01 = gs0[1].subgridspec(2, 1, hspace=0, height_ratios=[1.5, 1])
    gs02 = gs0[2].subgridspec(2, 1, hspace=0, height_ratios=[1.5, 1])
    axs00 = [fig.add_subplot(gs00[i]) for i in range(2)]
    axs01 = [fig.add_subplot(gs01[i]) for i in range(2)]
    axs02 = [fig.add_subplot(gs02[i]) for i in range(2)]

    for axpair in [axs00, axs01, axs02]:
        axfit, axres = axpair[0], axpair[1]

        _plot_profile_and_fit(
            phases,
            profile,
            model_fit,
            axfit,
            axres,
            model_init=model_init,
            phase_max=phase_max,
        )

    for ax in axs00:
        ax.set_xlim([-0.5, 0.5])
    for ax in axs01:
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlim([phase_max - 0.125, phase_max + 0.125])

    for ax in axs02:
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlim([phase_max + 0.25, phase_max + 0.5])

    for ax in [axs00[1], axs01[1], axs02[1]]:
        ax.set_xlabel("Pulse Phase")
    for ax in [axs00[0], axs01[0], axs02[0]]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axs00[0].set_ylabel("Counts")
    axs00[1].set_ylabel("Residuals")
    axs02[0].legend()

    if plot_file is not None:
        plt.savefig(plot_file)
    else:
        plt.show()


def fit_and_save_single_profile(
    phases,
    profile,
    nbins=512,
    additional_meta=None,
    plot=False,
    plot_file=None,
    init_pars=None,
    freeze_most=False,
):
    frozen = None
    if freeze_most:
        frozen = [
            key for key in init_pars.keys() if key not in ["amplitude_0", "amplitude_1", "x00_2"]
        ]

    model_init, model_fit = fit_crab_profile(phases, profile, init_pars=init_pars, frozen=frozen)
    output_table = fill_template_table(model_fit, nbins=nbins, model_init=model_init)
    phase_max = None
    if additional_meta is not None:
        output_table.meta.update(additional_meta)

    phase_max = output_table.meta["phase_max"]

    if plot:
        plot_fit_diagnostics(
            phases,
            profile,
            model_fit,
            phase_max=phase_max,
            model_init=model_init,
            plot_file=plot_file,
        )
    return output_table


def create_template_from_profile_table(
    input_profile_fname,
    output_template_fname=None,
    plot=False,
    plot_file=None,
    nbins=512,
):
    profile_table = Table.read(input_profile_fname)
    phases, profile = profile_table["phase"], profile_table["profile"]

    output_template_table = fit_and_save_single_profile(
        phases,
        profile,
        nbins=nbins,
        additional_meta=profile_table.meta,
        plot=plot,
        plot_file=plot_file,
    )

    if "profile_1" not in profile_table.colnames:
        log.info("Single peak profile, nothing else to do")
        if output_template_fname is not None:
            output_template_table.write(output_template_fname, overwrite=True, serialize_meta=True)
        return output_template_table

    log.info("Fitting subpeaks")

    # Freeze most parameters in sub-intervals, but only if the observation
    # is divided in more than 3 sub-intervals.
    freeze_most = "profile_3" in profile_table.colnames
    for col in profile_table.colnames:
        if not col.startswith("profile_") or "raw" in col:
            continue
        phases, profile = profile_table["phase"], profile_table[col]

        init_pars = copy.deepcopy(output_template_table.meta["best_fit"])
        init_pars["amplitude_0"] *= profile.sum() / profile_table["profile"].sum()
        try:
            output_table = fit_and_save_single_profile(
                phases,
                profile,
                nbins=nbins,
                additional_meta=profile_table.meta,
                init_pars=init_pars,
                freeze_most=freeze_most,
            )
        except Exception as e:
            log.error(f"Error fitting {col}: {e}")
            continue
        output_template_table[col] = output_table["profile"]
        output_template_table[col].meta.update(output_table.meta)
        output_template_table[col].meta.update(profile_table[col].meta)

    if output_template_fname is not None:
        output_template_table.write(output_template_fname, overwrite=True, serialize_meta=True)
    return output_template_table


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Create summary table for toaextract")

    parser.add_argument("fnames", help="Input profile tables", type=str, nargs="+")
    parser.add_argument(
        "-p",
        "--plot",
        help="Create diagnostic plot",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-N", "--nbin", help="Number of bins in output template", type=int, default=512
    )

    args = parser.parse_args(args)

    for fname in args.fnames:
        out_root = root_name(fname)
        ext = fname.replace(out_root, "")

        output_template_fname = out_root + "_fit_template" + ext
        output_plot_fname = None
        if args.plot:
            output_plot_fname = out_root + "_fit_diagnostics.jpg"

        create_template_from_profile_table(
            fname,
            output_template_fname=output_template_fname,
            plot=args.plot,
            plot_file=output_plot_fname,
            nbins=512,
        )
