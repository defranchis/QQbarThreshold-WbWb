"""Parameter and systematic scans.

Each scan takes a configured :class:`FitCore` instance, sweeps one knob,
and produces a diagnostic plot in ``fit.plot_dir``. All scans are free
functions so they can be opted into independently from the entry script.

Most scans mirror methods of the same names on the original ``fit`` class;
BEC and BES nuisance sweeps share a single :func:`_scan_nuisance` helper.
"""

import copy
import os

import iminuit
import matplotlib.pyplot as plt
import numpy as np

from common.fit_core import ecm_to_str, quadrature_subtract
from common.plots import (
    process_annotation,
    projection_title,
    save_figure,
)


def _impact(arr):
    """``sqrt(arr[i]**2 - arr[0]**2)`` elementwise — the "impact relative to
    the no-syst baseline" pattern."""
    arr = np.asarray(arr)
    return quadrature_subtract(arr, arr[0])


# ---------------------------------------------------------------------------
# Beam-energy resolution
# ---------------------------------------------------------------------------
def scan_beam_resolution(fit, *, lo=0.0, hi=0.5, step=0.01):
    """``doLSscan`` — sweep beam-energy resolution, record statistical uncertainty."""
    if lo == 0:
        lo = 1e-6
    grid = np.arange(lo, hi + step / 2, step)
    scan_keys = [p for p in fit.param_names
                 if p != "alphas" and "BEC" not in p and "BES" not in p]
    if fit.sm_width:
        scan_keys.remove("width")
    if fit.constrain_yukawa:
        scan_keys.remove("yukawa")

    results = {p: [] for p in fit.param_names}

    work = copy.deepcopy(fit)
    work.reinitialise_to_stat()
    work.param_names = [p for p in fit.param_names if "BEC" not in p and "BES" not in p]
    work.bec_nuisances = False
    work.bes_nuisances = False
    for res in grid:
        work.beam_energy_res = res
        work.update()
        fr = work.fit_results(printout=False)
        for i, p in enumerate(scan_keys):
            results[p].append(fr[i].s)

    work.beam_energy_res = fit.beam_energy_res
    work.update()
    work.last_fit_results = work.fit_results(printout=False)
    mass_nom = work.last_fit_results[work._idx["mass"]].s
    width_nom = work.last_fit_results[work._idx["width"]].s

    plt.figure()
    plt.plot(grid, np.array(results["mass"]) * 1e3, "b-",
             label=r"Stat. uncert. in $m_t$", linewidth=2)
    plt.plot(grid, np.array(results["width"]) * 1e3, "g--",
             label=r"Stat. uncert. in $\Gamma_t$", linewidth=2)
    plt.plot(fit.beam_energy_res, mass_nom * 1e3, "ro",
             label=r"Baseline $m_t$", markersize=8)
    plt.plot(fit.beam_energy_res, width_nom * 1e3, "s", color="orange",
             label=r"Baseline $\Gamma_t$", markersize=7)
    plt.legend()
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel("Beam energy spread [%]")
    plt.ylabel("Statistical uncertainty [MeV]")
    process_annotation(fit.card, x=0.95, y=0.37)
    save_figure(fit.plot_dir, "uncert_mass_width_vs_BER")


# ---------------------------------------------------------------------------
# Nuisance-prior scans (BEC / BES)
# ---------------------------------------------------------------------------
def _scan_nuisance(fit, kind, variations, *, axis_unit, axis_label, baseline_uncorr):
    """Generic per-correlation BEC / BES sweep."""
    setter = {"BEC": "set_bec_priors", "BES": "set_bes_priors"}[kind]
    flag_attr = {"BEC": "bec_nuisances", "BES": "bes_nuisances"}[kind]
    kw_uncorr = "prior_uncorr" if kind == "BEC" else "uncert_uncorr"
    kw_corr = "prior_corr" if kind == "BEC" else "uncert_corr"

    results = {"uncorr": {}, "corr": {}}
    for direction in ("uncorr", "corr"):
        l_mass, l_width = [], []
        work = copy.deepcopy(fit)
        if not getattr(fit, flag_attr):
            if kind == "BEC":
                work.add_bec_nuisances(prior_uncorr=1e-6, prior_corr=1e-6)
            else:
                work.add_bes_nuisances(uncert_uncorr=1e-6, uncert_corr=1e-6)
        for v in variations:
            if v < 1e-6:
                v = 1e-6
            v_uncorr = v if direction == "uncorr" else 1e-6
            v_corr = v if direction == "corr" else 1e-6
            getattr(work, setter)(**{kw_uncorr: v_uncorr, kw_corr: v_corr})
            work.fit_parameters(init_minuit=True)
            fr = work.fit_results(printout=False)
            l_mass.append(fr[fit._idx["mass"]].s)
            l_width.append(fr[fit._idx["width"]].s)
        results[direction]["mass"] = _impact(l_mass)
        results[direction]["width"] = _impact(l_width)

    # Baseline single-point reference
    base = {}
    work = copy.deepcopy(fit)
    if not getattr(fit, flag_attr):
        if kind == "BEC":
            work.add_bec_nuisances(prior_uncorr=1e-6, prior_corr=1e-6)
        else:
            work.add_bes_nuisances(uncert_uncorr=1e-6, uncert_corr=1e-6)
    base_mass, base_width = [], []
    for v in (0, baseline_uncorr):
        if v < 1e-6:
            v = 1e-6
        getattr(work, setter)(**{kw_uncorr: v, kw_corr: 1e-6})
        work.fit_parameters(init_minuit=True)
        fr = work.fit_results(printout=False)
        base_mass.append(fr[fit._idx["mass"]].s)
        base_width.append(fr[fit._idx["width"]].s)
    base["mass"] = _impact(base_mass)
    base["width"] = _impact(base_width)

    # Plot ------------------------------------------------------------------
    x = variations * axis_unit
    plt.plot(x, results["uncorr"]["mass"] * 1e3, "b-",
             label=r"Impact on $m_t$ (uncorr.)", linewidth=2)
    plt.plot(x, results["uncorr"]["width"] * 1e3, "g-",
             label=r"Impact on $\Gamma_t$ (uncorr.)", linewidth=2)
    plt.plot(x, results["corr"]["mass"] * 1e3, "b--",
             label=r"Impact on $m_t$ (corr.)", linewidth=2)
    plt.plot(x, results["corr"]["width"] * 1e3, "g--",
             label=r"Impact on $\Gamma_t$ (corr.)", linewidth=2)
    plt.plot(baseline_uncorr * axis_unit, base["mass"][-1] * 1e3, "ro",
             label=r"Baseline $m_t$ (uncorr.)", markersize=8)
    plt.plot(baseline_uncorr * axis_unit, base["width"][-1] * 1e3, "s", color="orange",
             label=r"Baseline $\Gamma_t$ (uncorr.)", markersize=7)

    plt.legend(loc="upper left")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(axis_label)
    plt.ylabel("Impact on fitted parameter [MeV]")
    process_annotation(fit.card, x=0.05, y=0.52, ha="left")
    save_figure(fit.plot_dir, f"uncert_mass_width_vs_{kind}")


def scan_bec(fit, *, lo=0, hi=10, step=0.5):
    variations = np.arange(lo, hi + step / 2, step)
    _scan_nuisance(fit, "BEC", variations,
                   axis_unit=1.0,
                   axis_label=r"Uncertainty in $\sqrt{s}$ [MeV]",
                   baseline_uncorr=fit.card.PRIORS["BEC"]["uncorr"])


def scan_bes(fit, *, lo=0, hi=0.03, step=0.001):
    variations = np.arange(lo, hi + step / 2, step)
    _scan_nuisance(fit, "BES", variations,
                   axis_unit=100.0,
                   axis_label="BES uncertainty [%]",
                   baseline_uncorr=fit.card.PRIORS["BES"]["uncorr"])


# ---------------------------------------------------------------------------
# Luminosity
# ---------------------------------------------------------------------------
def scan_lumi(fit, *, lo=0, hi=3, points=11):
    """``doLumiScans`` — sweep both correlation patterns of the lumi unc.

    ``lumi_uncorr`` / ``lumi_corr`` feed only ``_build_cov``; everything else
    (morph matrix, scenario tensors, smeared templates) is constant across
    the scan. Mutate the lumi attrs + cov caches on ``fit``, run a fresh
    local Minuit per grid point on ``fit.chi2``, restore on exit.
    """
    base = fit.card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(lo, hi, points) * base
    # Cold-start (start=zeros) matches the deepcopy version's init_minuit so
    # hesse cov is bit-identical. Warm-starting from fit.minuit.values
    # converges to the same minimum but gives ~1e-4 different uncertainties.
    start = np.zeros(len(fit.param_names))
    track_yukawa = "yukawa" in fit.param_names
    saved = (fit.lumi_uncorr, fit.lumi_corr,
             fit.cov, fit._cov_factor, fit.lumi_uncorr_ecm)
    res = {}
    l_mass0, l_width0 = [], []
    try:
        for direction in ("uncorr", "corr"):
            l_mass, l_width, l_yuk = [], [], []
            for lumi in l_lumi:
                if direction == "uncorr":
                    fit.lumi_uncorr = lumi
                    fit.lumi_corr = 0
                else:
                    fit.lumi_corr = lumi
                    fit.lumi_uncorr = 0
                fit._build_cov()
                m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
                m.errordef = 1
                m.migrad()
                fr = fit.results_from_minuit(m)
                l_mass.append(fr[fit._idx["mass"]].s * 1000)
                l_width.append(fr[fit._idx["width"]].s * 1000)
                if track_yukawa:
                    l_yuk.append(fr[fit._idx["yukawa"]].s * 100)
            res[direction] = (_impact(l_mass), _impact(l_width), _impact(l_yuk))

        for lumi in (0, base):
            fit.lumi_uncorr = lumi
            fit.lumi_corr = 0
            fit._build_cov()
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            fr = fit.results_from_minuit(m)
            l_mass0.append(fr[fit._idx["mass"]].s * 1000)
            l_width0.append(fr[fit._idx["width"]].s * 1000)
    finally:
        fit.lumi_uncorr, fit.lumi_corr, fit.cov, fit._cov_factor, fit.lumi_uncorr_ecm = saved

    base_pct = l_lumi * 100
    plt.plot(base_pct, res["uncorr"][0], "b-", label=r"Impact on $m_t$ (uncorr.)", linewidth=2)
    plt.plot(base_pct, res["uncorr"][1], "g", label=r"Impact on $\Gamma_t$ (uncorr.)", linewidth=2)
    plt.plot(base_pct, res["corr"][0], "b--", label=r"Impact on $m_t$ (corr.)", linewidth=2)
    plt.plot(base_pct, res["corr"][1], "g--", label=r"Impact on $\Gamma_t$ (corr.)", linewidth=2)

    impact_mass = _impact(l_mass0)
    impact_width = _impact(l_width0)
    plt.plot(base * 100, impact_mass[-1], "ro", label=r"Baseline $m_t$ (uncorr.)", markersize=8)
    plt.plot(base * 100, impact_width[-1], "s", color="orange",
             label=r"Baseline $\Gamma_t$ (uncorr.)", markersize=7)

    plt.legend()
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel("Integrated luminosity uncertainty [%]")
    plt.ylabel("Impact on fitted parameter [MeV]")
    process_annotation(fit.card, x=0.92, y=0.14)
    save_figure(fit.plot_dir, "uncert_mass_width_vs_lumi")

    if not fit.scenario_dict["add_last_ecm"] or "yukawa" not in fit.param_names:
        return

    x = np.linspace(0.5, 1.5, 11)
    plt.plot(x, res["uncorr"][2], "r-", label=r"Impact on $y_t$ (uncorr)", linewidth=2)
    plt.plot(x, res["corr"][2], "r--", label=r"Impact on $y_t$ (corr)", linewidth=2)
    plt.plot(1, res["uncorr"][2][list(x).index(1)], "ro", label="Nominal value", markersize=8)
    plt.legend()
    plt.title(projection_title(fit.scenario_dict["last_lumi"], unit="ab", fmt="{:.2f}"),
              loc="right", fontsize=20)
    plt.xlabel("Luminosity uncert. / nominal value")
    plt.ylabel(r"Luminosity uncert. on fitted $y_t$ [%]")
    process_annotation(fit.card, x=0.96, y=0.27, include_reference=True)
    plt.text(0.96, 0.07,
             f"nominal uncorr (corr) uncert. = {fit.lumi_uncorr_ecm[-1]*100:.3f} ({fit.lumi_corr*100:.2f}) %",
             fontsize=21, transform=plt.gca().transAxes, ha="right")
    save_figure(fit.plot_dir, "uncert_yukawa_vs_lumi")


# ---------------------------------------------------------------------------
# alpha_s
# ---------------------------------------------------------------------------
def scan_alphas(fit, *, hi=3e-4, step=1e-5):
    """``doAlphaSscans`` — sweep the externally-imposed alpha_s prior.

    ``input_uncert_alphas`` feeds only the chi2 alpha_s constraint term — not
    cov, not the morph matrix — so we mutate ``fit`` in place and rebuild a
    fresh local Minuit per grid point instead of cloning the whole FitCore.
    """
    grid = np.arange(1e-10, hi + step / 2, step)
    start = list(fit.minuit.values)
    step_mass = fit.parameters.step("mass")
    step_width = fit.parameters.step("width")
    saved = fit.input_uncert_alphas
    l_mass, l_width = [], []
    try:
        for u in grid:
            fit.input_uncert_alphas = u
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            l_mass.append(step_mass * m.errors[fit._idx["mass"]] * 1000)
            l_width.append(step_width * m.errors[fit._idx["width"]] * 1000)
    finally:
        fit.input_uncert_alphas = saved
    l_mass = np.array(l_mass)
    l_width = np.array(l_width)

    nominal_mass = fit.last_fit_results[fit._idx["mass"]].s * 1000
    nominal_width = fit.last_fit_results[fit._idx["width"]].s * 1000
    nominal_mass = quadrature_subtract(nominal_mass, l_mass[0])
    nominal_width = quadrature_subtract(nominal_width, l_width[0])

    l_mass = _impact(l_mass)
    l_width = _impact(l_width)

    baseline = fit.card.PRIORS["alphas"]["default"]
    plt.plot(grid * 1e3, l_mass, "b-", label=r"Impact on $m_t$", linewidth=2)
    plt.plot(grid * 1e3, l_width, "g--", label=r"Impact on $\Gamma_t$", linewidth=2)
    plt.plot(baseline * 1e3, nominal_mass, "ro", label=r"Baseline $m_t$", markersize=8)
    plt.plot(baseline * 1e3, nominal_width, "s", color="orange",
             label=r"Baseline $\Gamma_t$", markersize=7)
    plt.legend(loc="upper left")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(r"Uncertainty in $\alpha_\mathrm{S} (m_\mathrm{Z}^2) [x10^3]$")
    plt.ylabel("Impact on fitted parameter [MeV]")
    process_annotation(fit.card, x=0.92, y=0.17)
    save_figure(fit.plot_dir, "uncert_mass_width_vs_alphas")


# ---------------------------------------------------------------------------
# Yukawa
# ---------------------------------------------------------------------------
def scan_yukawa_constraint(fit, *, hi=0.05, step=0.001):
    """``doYukawaScan`` — sweep the Yukawa prior strength.

    Like :func:`scan_alphas`, ``input_uncert_yukawa`` only feeds the chi2
    constraint term, so we mutate ``fit`` in place and use a fresh local
    Minuit per grid point.
    """
    grid = np.arange(1e-10, hi + step / 2, step)
    start = list(fit.minuit.values)
    step_mass = fit.parameters.step("mass")
    step_width = fit.parameters.step("width")
    saved_unc = fit.input_uncert_yukawa
    saved_constrain = fit.constrain_yukawa
    l_mass, l_width = [], []
    try:
        fit.constrain_yukawa = True
        for u in grid:
            fit.input_uncert_yukawa = u
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            l_mass.append(step_mass * m.errors[fit._idx["mass"]] * 1000)
            l_width.append(step_width * m.errors[fit._idx["width"]] * 1000)
    finally:
        fit.input_uncert_yukawa = saved_unc
        fit.constrain_yukawa = saved_constrain
    l_mass = np.array(l_mass)
    l_width = np.array(l_width)

    nominal_mass = fit.last_fit_results[fit._idx["mass"]].s * 1000
    nominal_width = fit.last_fit_results[fit._idx["width"]].s * 1000
    nominal_mass = quadrature_subtract(nominal_mass, l_mass[0])
    nominal_width = quadrature_subtract(nominal_width, l_width[0])

    l_mass = _impact(l_mass)
    l_width = _impact(l_width)

    baseline = fit.card.PRIORS["yukawa"]["default"]
    plt.plot(grid * 100, l_mass, "b-", label=r"Impact on $m_t$", linewidth=2)
    plt.plot(grid * 100, l_width, "g--", label=r"Impact on $\Gamma_t$", linewidth=2)
    plt.plot(baseline * 100, nominal_mass, "ro", label=r"Baseline $m_t$", markersize=8)
    plt.plot(baseline * 100, nominal_width, "s", color="orange",
             label=r"Baseline $\Gamma_t$", markersize=7)
    plt.legend(loc="upper left")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(r"Uncertainty in $y_t$ [%]")
    plt.ylabel("Impact on fitted parameter [MeV]")
    process_annotation(fit.card, x=0.92, y=0.17)
    save_figure(fit.plot_dir, "uncert_mass_width_vs_yukawa")


def scan_yukawa_theory(fit, *, max_shift=0.01, step=0.001):
    """``doYukawaTheoryScan`` — shift the above-threshold xsec and read fitted y_t."""
    shifts = np.arange(-max_shift, max_shift + step / 2, step) + 1
    work = copy.deepcopy(fit)
    l_yuk = []
    for s in shifts:
        work.scale_var_scenario[-1] *= s
        work.fit_parameters()
        fr = work.fit_results(printout=False)
        l_yuk.append(fr[fit._idx["yukawa"]].n
                     - fit.last_fit_results[fit._idx["yukawa"]].n)
        work.scale_var_scenario[-1] /= s

    plt.plot(shifts, l_yuk, "b-", label=r"Shift in fitted $y_t$", linewidth=2)
    plt.plot(1, 0, "ro", label="Starting point", markersize=8)
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.legend()
    plt.xlabel("Shift in cross section")
    plt.ylabel(r"Shift in fitted $y_t$")
    process_annotation(fit.card, x=0.60, y=0.17, include_reference=True)
    save_figure(fit.plot_dir, "uncert_yukawa_vs_xsec_shift")


# ---------------------------------------------------------------------------
# Width (only meaningful with --SMwidth)
# ---------------------------------------------------------------------------
def scan_width(fit, *, hi=10, step=0.1):
    """Sweep the SM-width theory uncertainty. Only the WbWb-specific
    ``physical_fit_params`` hook reads ``input_uncert_SM_width``, so we
    mutate ``fit`` in place and use a fresh local Minuit per grid point."""
    if not fit.sm_width:
        raise ValueError("scan_width requires sm_width=True")
    grid = np.arange(1e-10, hi + step / 2, step)
    start = list(fit.minuit.values)
    step_mass = fit.parameters.step("mass")
    saved = fit.input_uncert_SM_width
    l_mass = []
    try:
        for u in grid:
            fit.input_uncert_SM_width = u
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            l_mass.append(step_mass * m.errors[fit._idx["mass"]] * 1000)
    finally:
        fit.input_uncert_SM_width = saved
    l_mass = np.array(l_mass)

    nominal_mass = fit.last_fit_results[fit._idx["mass"]].s * 1000
    baseline = fit.input_uncert_SM_width
    impact = quadrature_subtract(nominal_mass, l_mass[0])
    l_mass = _impact(l_mass)

    plt.plot(grid, l_mass, "b-", label=r"Impact on fitted $m_t$", linewidth=2)
    plt.plot(baseline, impact, "ro",
             label=r"$N^{3}LO$ in QCD [arXiv:2309.01937]", markersize=8)
    plt.legend(loc="upper left")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(r"Uncertainty SM prediction for $\Gamma_t$ [MeV]")
    plt.ylabel(r"Impact on fitted $m_t$ [MeV]")
    process_annotation(fit.card, x=0.92, y=0.17, include_reference=True)
    save_figure(fit.plot_dir, "uncert_mass_vs_width")


# ---------------------------------------------------------------------------
# Scale variation
# ---------------------------------------------------------------------------
def scan_scale_vars(fit):
    """``doScaleVars`` — sweep the renormalisation scale and read fitted shift.

    ``scale_var_scenario`` feeds only ``_xsec_base`` in ``_build_chi2_caches``.
    Mutate both on ``fit``, run a fresh local Minuit per scale, restore on exit.
    """
    l_vars, l_mass, l_width = [], [], []
    track_yukawa = not fit.constrain_yukawa and "yukawa" in fit.param_names
    l_yuk = [] if track_yukawa else None
    saved = (fit.scale_var_scenario, fit._xsec_base)
    # See scan_lumi for why we cold-start (start=zeros) instead of warm-starting.
    start = np.zeros(len(fit.param_names))
    try:
        for v in fit.scale_vars:
            if v < 70:
                continue
            tag = f"scaleM_{v:.1f}"
            if tag not in fit.xsec_dict:
                continue
            l_vars.append(v)
            nominal_scen = fit.slice_to_scenario(fit.template())
            var_scen = fit.slice_to_scenario(fit.template(tag))

            fit.scale_var_scenario = np.array(var_scen["xsec"]) / np.array(nominal_scen["xsec"])
            fit._xsec_base = np.asarray(fit.xsec_scenario["xsec"]) * np.asarray(fit.scale_var_scenario)
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            fr = fit.results_from_minuit(m)
            l_mass.append(fr[fit._idx["mass"]].n
                          - fit.last_fit_results[fit._idx["mass"]].n)
            l_width.append(fr[fit._idx["width"]].n
                           - fit.last_fit_results[fit._idx["width"]].n)
            if track_yukawa:
                l_yuk.append(fr[fit._idx["yukawa"]].n
                             - fit.last_fit_results[fit._idx["yukawa"]].n)
    finally:
        fit.scale_var_scenario, fit._xsec_base = saved

    plt.plot(l_vars, np.array(l_mass) * 1e3, "b-",
             label=r"Shift in fitted $m_t$", linewidth=2)
    plt.plot(l_vars, np.array(l_width) * 1e3, "g--",
             label=r"Shift in fitted $\Gamma_t$", linewidth=2)
    plt.plot(fit.mass_scale, 0, "ro", label="Starting point", markersize=8)
    plt.legend()
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(r"Renormalisation scale $\mu$ [GeV]")
    plt.ylabel("Shift in fitted parameter [MeV]")
    process_annotation(fit.card, x=0.6, y=0.17, include_reference=True)
    save_figure(fit.plot_dir, "uncert_mass_width_vs_scale")

    if track_yukawa:
        plt.plot(l_vars, np.array(l_yuk), "b-",
                 label=r"Shift in fitted $y_t$", linewidth=2)
        plt.plot(fit.mass_scale, 0, "ro", label="Starting point", markersize=8)
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel(r"Renormalisation scale $\mu$ [GeV]")
        plt.ylabel("Shift in fitted parameter")
        process_annotation(fit.card, x=0.97, y=0.12, include_reference=True)
        save_figure(fit.plot_dir, "uncert_yukawa_vs_scale")


# ---------------------------------------------------------------------------
# True-value scan
# ---------------------------------------------------------------------------
def scan_true_value(fit):
    """``doTrueValueScan`` — fit each pseudo-data template in ``INPUT_DIRS.pseudo``.

    Mutate ``scenario_dict[scan_list]`` + ``lumi_uncorr`` per iteration
    (baseline + coarse sub-scan) and feed each file's smeared template as
    ``create_scenario`` pseudodata. Use a fresh local Minuit; restore on exit.
    """
    indir = fit.card.INPUT_DIRS["pseudo"]
    results_baseline = {}
    results_coarse = {}
    results_baseline_width = {}
    results_coarse_width = {}

    saved_scan_list = list(fit.scenario_dict["scan_list"])
    saved_lumi_uncorr = fit.lumi_uncorr
    # See scan_lumi for why we cold-start (start=zeros) instead of warm-starting.
    start = np.zeros(len(fit.param_names))

    def _fit_once(pseudo):
        fit.create_scenario(**fit.scenario_dict, init_vars=True, pseudodata=pseudo)
        fit._build_cov()
        fit._build_chi2_caches()
        m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
        m.errordef = 1
        m.migrad()
        return fit.results_from_minuit(m)

    try:
        for fname in sorted(os.listdir(indir)):
            mass = fname.split("_")[4].replace("mass", "")
            pseudo = fit.smear(fit.read_xsec(os.path.join(indir, fname)))

            fit.scenario_dict["scan_list"] = saved_scan_list
            fit.lumi_uncorr = saved_lumi_uncorr
            fr = _fit_once(pseudo)
            bias = fr[fit._idx["mass"]].n - float(mass)
            mass_unc = fr[fit._idx["mass"]].s
            if abs(bias) > mass_unc * 0.7:
                continue
            results_baseline[mass] = mass_unc * 1000
            results_baseline_width[mass] = fr[fit._idx["width"]].s * 1000

            coarse_scan = [ecm_to_str(e) for e in np.arange(340.5, 345 + 0.5, 1.0)]
            fit.scenario_dict["scan_list"] = coarse_scan
            fit.lumi_uncorr = saved_lumi_uncorr / 2 ** 0.5
            fr = _fit_once(pseudo)
            results_coarse[mass] = fr[fit._idx["mass"]].s * 1000
            results_coarse_width[mass] = fr[fit._idx["width"]].s * 1000
    finally:
        fit.scenario_dict["scan_list"] = saved_scan_list
        fit.lumi_uncorr = saved_lumi_uncorr
        fit.create_scenario(**fit.scenario_dict, init_vars=True)
        fit._build_cov()
        fit._build_chi2_caches()

    masses = np.array([float(k) for k in results_baseline.keys()])
    for label, ylabel, baseline_d, coarse_d in (
        ("mass", r"Uncertainty in fitted $m_t$ [MeV]", results_baseline, results_coarse),
        ("width", r"Uncertainty in fitted $\Gamma_t$ [MeV]", results_baseline_width, results_coarse_width),
    ):
        errs = np.array(list(baseline_d.values()))
        errs_coarse = np.array(list(coarse_d.values()))
        plt.plot(masses, errs, "b-", label=f"Uncertainty in ${label}_t$", linewidth=2)
        plt.plot(masses, errs_coarse, "g--", label=f"Uncertainty in ${label}_t$ (coarse scan)", linewidth=2)
        plt.xlabel(r"True value of $m_t$ [GeV]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        process_annotation(fit.card, x=0.92, y=0.17)
        save_figure(fit.plot_dir, f"uncert_{label}_vs_true_mass")


# ---------------------------------------------------------------------------
# Shift the entire scan grid
# ---------------------------------------------------------------------------
def scan_shift(fit, *, max_abs_shift_neg=2.0, max_abs_shift_pos=2.5, step=0.1):
    """Shift the scan ecms uniformly left/right and read the relative change
    in mass/width uncertainty.

    Requires ``shift_scan=True`` on the fit so that BEC/BES/sw2 morph
    templates (which only exist on the original ecm grid) are skipped.
    """
    if not fit.shift_scan:
        raise ValueError("scan_shift requires the fit to be built with shift_scan=True")
    scan_list = np.array(fit.scenario_dict["scan_list"], dtype=float)
    shifts = np.arange(-max_abs_shift_neg, max_abs_shift_pos + step / 2, step)

    # Sanity check: every shifted ecm has to exist in the template ecm list,
    # otherwise create_scenario will throw a cryptic per-point error mid-scan.
    template_ecms = {float(e) for e in fit.l_ecm}
    have_min = min(template_ecms)
    have_max = max(e for e in template_ecms if e < fit.last_ecm - 0.5)  # exclude the above-threshold lever
    need_min = float(scan_list.min()) - max_abs_shift_neg
    need_max = float(scan_list.max()) + max_abs_shift_pos
    if need_min < have_min - 1e-6 or need_max > have_max + 1e-6:
        raise ValueError(
            f"scan_shift needs templates spanning [{need_min:.1f}, {need_max:.1f}] GeV "
            f"but {fit.input_dir} only provides [{have_min:.1f}, {have_max:.1f}] GeV.\n"
            f"To extend the range, edit ``sqrt_s_min`` / ``sqrt_s_max`` in "
            f"xsec_calculator/ttThresholdScanISR.cpp, rebuild "
            f"(xsec_calculator/compile_calc.sh), and regenerate the cross-section "
            f"templates in {fit.input_dir}.\n"
            f"Alternatively, reduce ``max_abs_shift_neg`` / ``max_abs_shift_pos`` "
            f"so that the shifted ecms stay inside the existing template range."
        )
    saved_scan_list = list(fit.scenario_dict["scan_list"])
    # See scan_lumi for why we cold-start (start=zeros) instead of warm-starting.
    start = np.zeros(len(fit.param_names))
    l_mass, l_width = [], []
    try:
        for shift in shifts:
            fit.scenario_dict["scan_list"] = [ecm_to_str(e) for e in scan_list + shift]
            fit.create_scenario(**fit.scenario_dict, init_vars=True)
            fit._build_cov()
            fit._build_chi2_caches()
            m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
            m.errordef = 1
            m.migrad()
            fr = fit.results_from_minuit(m)
            l_mass.append(fr[fit._idx["mass"]].s)
            l_width.append(fr[fit._idx["width"]].s)
    finally:
        # Restore scenario tensors + chi2 caches in place; fit.minuit was
        # never touched.
        fit.scenario_dict["scan_list"] = saved_scan_list
        fit.create_scenario(**fit.scenario_dict, init_vars=True)
        fit._build_cov()
        fit._build_chi2_caches()
    l_mass = np.array(l_mass)
    l_width = np.array(l_width)

    nominal_mass = fit.last_fit_results[fit._idx["mass"]].s
    nominal_width = fit.last_fit_results[fit._idx["width"]].s

    rel_mass = (l_mass / nominal_mass - 1) * 100
    rel_width = (l_width / nominal_width - 1) * 100

    plt.plot(shifts, rel_mass, "b-", label=r"$m_t$", linewidth=2)
    plt.plot(shifts, rel_width, "g--", label=r"$\Gamma_t$", linewidth=2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Shift in scan range [GeV]")
    plt.ylabel("Relative increase in uncertainty [%]")
    plt.legend()
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    process_annotation(fit.card, x=0.85, y=0.87)
    save_figure(fit.plot_dir, "shift_scan")


# ---------------------------------------------------------------------------
# Chi2 scans
# ---------------------------------------------------------------------------
def _scan_keys(fit):
    out = []
    for p in fit.param_names:
        if p == "alphas" or "BEC" in p or "BES" in p:
            continue
        if p == "yukawa" and fit.constrain_yukawa:
            continue
        if p == "width" and fit.sm_width:
            continue
        out.append(p)
    return out


def scan_chi2(fit):
    """``doChi2Scans`` — 1D and 2D chi2 profile scans for the floating parameters.

    Builds a fresh :class:`iminuit.Minuit` per grid point on the shared
    ``fit.chi2`` callable instead of ``deepcopy(fit)`` — the FitCore state
    (cov factor, morph matrix, smeared templates) is read-only inside
    ``chi2``, so cloning the whole object per iteration is wasted work.
    """
    labels = fit.card.PARAM_LABELS
    keys = _scan_keys(fit)
    start = list(fit.minuit.values)  # warm-start every profile fit from the global best

    def profile_chi2(fixed_idx, fixed_val):
        m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
        m.errordef = 1
        for k, v in zip(fixed_idx, fixed_val):
            m.values[k] = v
            m.fixed[k] = True
        m.migrad()
        return m.fval

    for p in keys:
        i = fit._idx[p]
        grid = np.linspace(fit.minuit.values[i] - 3 * fit.minuit.errors[i],
                           fit.minuit.values[i] + 3 * fit.minuit.errors[i], 101)
        l_chi2 = [profile_chi2([i], [v]) for v in grid]
        plt.plot(fit.value_from_param(grid, p), l_chi2, label=p)
        plt.xlabel(labels.get(p, p))
        plt.ylabel(r"$\chi^2$")
        plt.legend()
        save_figure(fit.plot_dir, f"chi2_scan_{p}", also_pdf=False)

    for ia, pa in enumerate(keys):
        ia_full = fit._idx[pa]
        for ib, pb in enumerate(keys):
            if ib <= ia:
                continue
            ib_full = fit._idx[pb]
            gA = np.linspace(fit.minuit.values[ia_full] - 3 * fit.minuit.errors[ia_full],
                             fit.minuit.values[ia_full] + 3 * fit.minuit.errors[ia_full], 51)
            gB = np.linspace(fit.minuit.values[ib_full] - 3 * fit.minuit.errors[ib_full],
                             fit.minuit.values[ib_full] + 3 * fit.minuit.errors[ib_full], 51)
            grid_chi2 = np.array([
                [profile_chi2([ia_full, ib_full], [va, vb]) for vb in gB]
                for va in gA
            ])
            plt.contour(fit.value_from_param(gA, pa),
                        fit.value_from_param(gB, pb),
                        grid_chi2,
                        levels=[fit.minuit.fval + 1, fit.minuit.fval + 4],
                        colors=["#377eb8", "#4daf4a"], linewidths=2)
            plt.plot(fit.value_from_param(fit.minuit.values[ia_full], pa),
                     fit.value_from_param(fit.minuit.values[ib_full], pb),
                     "k*", label="Best fit value (input)", markersize=10)
            plt.plot([], [], color="#377eb8", label="68% C.L.")
            plt.plot([], [], color="#4daf4a", label="95% C.L.")
            plt.xlabel(labels.get(pa, pa))
            plt.ylabel(labels.get(pb, pb))
            plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
            process_annotation(fit.card, x=0.95, y=0.15, include_reference=True)
            plt.legend(loc="upper left")
            y_offset = -0.003 if pb == "width" and pa == "mass" else 0
            plt.ylim(fit.value_from_param(gB[0], pb) + y_offset,
                     fit.value_from_param(gB[-1], pb) + y_offset)
            plt.xticks(np.round(np.linspace(fit.value_from_param(gA[0], pa),
                                            fit.value_from_param(gA[-1], pa), 5), 2))
            save_figure(fit.plot_dir, f"chi2_scan_{pa}_{pb}")
