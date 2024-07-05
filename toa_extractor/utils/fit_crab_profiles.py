import copy
import numpy as np
import warnings
from scipy.optimize import fmin


from astropy.modeling.models import custom_model, Const1D
from astropy.modeling.fitting import TRFLSQFitter
from astropy.table import Table
import matplotlib.pyplot as plt
from . import root_name


@np.vectorize()
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
    asym_lor = asymmetric_lorentzian(
        x, amplitude=amplitude1, x0=x0 + dx0, fwhm1=fwhm1, fwhm2=fwhm2
    )

    return sym_lor + asym_lor


SkewedPeak = custom_model(skewed_peak)


def plot_fit_diagnostics(
    phases, profile, model_fit, phase_max=None, model_init=None, plot_file=None
):
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
    residuals = profile - model_fit(phases)

    for axpair in [axs00, axs01, axs02]:
        axfit, axres = axpair[0], axpair[1]

        axfit.plot(phases, profile, ds="steps-mid", color="r", label="Data")

        if model_init is not None:
            axfit.plot(
                phases, model_init(phases), color="grey", alpha=0.5, label="Init model"
            )
        axfit.plot(phases, model_fit(phases), color="k", zorder=10, label="Fit model")

        for m in (model_fit[2], model_fit[3]):
            axfit.plot(
                phases,
                (m(phases) + model_fit[1](phases)) * model_fit[0](phases),
                color="k",
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


def default_crab_model(init_pars=None):
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
        + ``amplitude_2``: The amplitude of the main peak's symmetric Lorentzian
        + ``x0_2``: the phase of the main peak's symmetric Lorentzian
        + ``fwhm_2``: The FWHM of the main peak's symmetric Lorentzian
        + ``amplitude_3``: The amplitude of the main peak's asymmetric Lorentzian
        + ``x0_3``: the phase of the main peak's asymmetric Lorentzian
        + ``fwhm1_3``: The right FWHM of the main peak's asymmetric Lorentzian
        + ``fwhm2_3``: The left FWHM of the main peak's asymmetric Lorentzian
        + ``amplitude_4``: The amplitude of the main peak's symmetric Lorentzian
        + ``x0_4``: the phase of the main peak's symmetric Lorentzian
        + ``fwhm_4``: The FWHM of the main peak's symmetric Lorentzian
        + ``amplitude_5``: The amplitude of the secondary peak's asymmetric Lorentzian
        + ``x0_5``: the phase of the secondary peak's asymmetric Lorentzian
        + ``fwhm1_5``: The right FWHM of the secondary peak's asymmetric Lorentzian
        + ``fwhm2_5``: The left FWHM of the secondary peak's asymmetric Lorentzian

    """
    bounds_skewed = dict(
        [(val, (0, np.inf)) for val in ["amplitude0", "amplitude1", "fwhm1", "fwhm2"]]
    )
    bounds_skewed["fwhm0"] = (0.01, np.inf)
    bounds_skewed["dx0"] = (-0.02, 0.02)

    model_init = Const1D(bounds={"amplitude": (0, np.inf)}) * (
        Const1D(bounds={"amplitude": (0, np.inf)})
        + SkewedPeak(bounds=bounds_skewed)
        + SkewedPeak(bounds=bounds_skewed)
    )
    if init_pars is not None:
        for key, val in init_pars.items():
            setattr(model_init, key, val)
    return model_init


def get_initial_parameters(input_phases, profile):
    phases = copy.deepcopy(input_phases)
    phases = normalize_phase_0d5(phases)
    order = np.argsort(phases)
    phases = phases[order]
    profile = copy.deepcopy(profile)[order]
    phases = np.concatenate([phases, phases + 1])
    profile = np.concatenate([profile, profile])

    # peak 1
    prof_filt1 = copy.deepcopy(profile)
    # prof_filt1[np.abs(phases) > 0.1] = 0
    idx_1 = np.argmax(prof_filt1)
    ph1 = phases[idx_1]
    baseline = np.min(profile)
    amplitude1 = profile[idx_1] - baseline

    fwhm1 = 0.04
    fwhm2 = 0.1

    # peak 2
    prof_filt2 = copy.deepcopy(profile)
    prof_filt2[np.abs(normalize_phase_0d5(phases - ph1) - 0.35) > 0.2] = 0
    idx_2 = np.argmax(prof_filt2)
    ph2 = phases[idx_2]
    amplitude2 = profile[idx_2] - baseline

    init_pars = {
        "amplitude_0": amplitude1,
        "amplitude_1": baseline / amplitude1,
        "amplitude0_2": 0.75,
        "amplitude1_2": 0.5,
        "x0_2": ph1,
        "dx0_2": 0,
        "fwhm0_2": fwhm1,
        "fwhm1_2": fwhm1 * 2,
        "fwhm2_2": fwhm1 * 2,
        "amplitude0_3": amplitude2 / amplitude1 / 10 * 6,
        "amplitude1_3": amplitude2 / amplitude1 / 10 * 4,
        "x0_3": ph2,
        "dx0_3": 0,
        "fwhm0_3": fwhm2,
        "fwhm1_3": fwhm2 * 2,
        "fwhm2_3": fwhm2 * 2,
    }
    print(init_pars)
    return init_pars


def fit_crab_profile(phases, profile, fitter=None):
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

    Returns
    -------
    model_init: astropy.modeling.Model
        The initial fit model
    model_fit: astropy.modeling.Model
        The best fit model
    """
    if fitter is None:
        fitter = TRFLSQFitter(calc_uncertainties=True)
    init_pars = get_initial_parameters(phases, profile)
    model_init = default_crab_model(init_pars=init_pars)
    model_fit = fitter(model_init, phases, profile, maxiter=200)

    if normalize_phase_0d5(model_fit.x0_3 - model_fit.x0_2) < 0:
        warnings.warn("The two peaks appear swapped. Trying to fix.")
        new_model_fit = copy.deepcopy(model_fit)
        for param in model_fit.param_names:
            if param.endswith("_3"):
                setattr(
                    new_model_fit, param, getattr(model_fit, param.replace("_3", "_2"))
                )
            if param.endswith("_2"):
                setattr(
                    new_model_fit, param, getattr(model_fit, param.replace("_2", "_3"))
                )

        model_fit = new_model_fit
    return model_init, model_fit


def fill_template_table(model_fit, nbins=512, template_table=None):
    if template_table is None:
        template_table = Table()
    phase_max = fmin(lambda x: -model_fit(x), model_fit.x0_2)[0]
    phases = np.arange(nbins) / nbins

    renorm_model = copy.deepcopy(model_fit)
    renorm_model.amplitude_0 = 1
    renorm_model.amplitude_1 = 0
    factor = 1 / renorm_model(phase_max)
    template_table["phase"] = phases
    template_table["profile"] = factor * renorm_model(phases + phase_max)
    template_table["profile_raw"] = model_fit(phases)

    par, par_err = model_fit.parameters, [
        model_fit.cov_matrix.cov_matrix[i, i] ** 0.5
        for i in range(len(model_fit.parameters))
    ]
    template_table.meta["best_fit"] = {}
    for name, p, pe in zip(model_fit.param_names, par, par_err):
        template_table.meta["best_fit"][name] = p
        template_table.meta["best_fit"][name + "_err"] = pe

    template_table.meta["phase_max"] = phase_max
    template_table.meta["phase_max_err"] = template_table.meta["best_fit"]["x0_2_err"]

    return template_table


def create_template_from_profile_table(
    input_profile_fname,
    output_template_fname=None,
    plot=False,
    plot_file=None,
    nbins=512,
):
    profile_table = Table.read(input_profile_fname)
    phases, profile = profile_table["phase"], profile_table["profile"]
    model_init, model_fit = fit_crab_profile(phases, profile)
    empty_template_table = Table()
    empty_template_table.meta.update(profile_table.meta)
    output_template_table = fill_template_table(
        model_fit, nbins=nbins, template_table=empty_template_table
    )
    if output_template_fname is not None:
        output_template_table.write(
            output_template_fname, overwrite=True, serialize_meta=True
        )
    if plot:
        plot_fit_diagnostics(
            phases,
            profile,
            model_fit,
            phase_max=output_template_table.meta["phase_max"],
            model_init=model_init,
            plot_file=plot_file,
        )
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
