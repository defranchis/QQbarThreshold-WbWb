"""WbWb-specific scan helpers.

These complement the process-agnostic scans in :mod:`common.scans`:

* :func:`scan_yukawa_constraint` and :func:`scan_yukawa_theory` are
  WbWb-only (the WW analysis doesn't carry a yukawa parameter at all).
* :func:`scan_lumi_yukawa_ratio` and :func:`scan_scale_vars_yukawa`
  recover the legacy WbWb yukawa-impact-vs-lumi-ratio /
  vs-renormalisation-scale plots. They each re-run a thin version of
  the corresponding generic scan so they remain a single
  function call per panel at the call site — the migrad cost (~10
  re-fits) is negligible compared to the lumi/scale scan that
  preceded them.

The yukawa-specific second panels of :func:`common.scans.scan_lumi`
and :func:`common.scans.scan_scale_vars` used to live inline; moving
them here keeps ``common/`` strictly process-agnostic.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from common.fit_core import quadrature_subtract
from common.plots import process_annotation, projection_title, save_figure
from common.scans import _impact, _poi_symbol, _run_local_migrad, _scan_constraint


def scan_width(fit, *, hi=10, step=0.1):
    """Sweep the SM-width theory uncertainty and read its impact on the
    fitted top mass. Requires ``sm_width=True`` so the SM-width hook
    (:meth:`WbWbFit.physical_fit_params`) is reading ``input_uncert_SM_width``.
    """
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
            m = _run_local_migrad(fit, start)
            l_mass.append(step_mass * m.errors[fit._idx["mass"]] * 1000)
    finally:
        fit.input_uncert_SM_width = saved
    l_mass = np.array(l_mass)

    nominal_mass = fit.last_fit_results[fit._idx["mass"]].s * 1000
    baseline = fit.input_uncert_SM_width
    impact = quadrature_subtract(nominal_mass, l_mass[0])
    l_mass = _impact(l_mass)

    sym_m = _poi_symbol(fit, "mass")
    sym_w = _poi_symbol(fit, "width")
    plt.plot(grid, l_mass, "b-", label=rf"Impact on fitted ${sym_m}$", linewidth=2)
    plt.plot(baseline, impact, "ro",
             label=r"$N^{3}LO$ in QCD [arXiv:2309.01937]", markersize=8)
    plt.legend(loc="upper left")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(rf"Uncertainty SM prediction for ${sym_w}$ [MeV]")
    plt.ylabel(rf"Impact on fitted ${sym_m}$ [MeV]")
    process_annotation(fit.card, x=0.92, y=0.17, include_reference=True)
    save_figure(fit.plot_dir, "uncert_mass_vs_width")


def scan_yukawa_constraint(fit, *, hi=0.05, step=0.001):
    """``doYukawaScan`` — sweep the Yukawa prior width.

    Force-enables the Yukawa constraint for the scan duration so the
    sweep is meaningful even when the caller invoked the fit with
    Yukawa floating (``--fitYukawa``).
    """
    saved_constrain = fit.constrain_yukawa
    try:
        fit.constrain_yukawa = True
        grid = np.arange(1e-10, hi + step / 2, step)
        _scan_constraint(fit, "yukawa", grid,
                         axis_unit=100,
                         axis_label=r"Uncertainty in $y_t$ [%]",
                         plot_filename_stem="yukawa")
    finally:
        fit.constrain_yukawa = saved_constrain


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


def scan_lumi_yukawa_ratio(fit, *, lo=0, hi=3, points=11):
    """WbWb legacy yukawa-vs-lumi-ratio panel.

    Independent mini-scan: sweep ``lumi_uncorr`` and ``lumi_corr`` (one
    direction at a time, same recipe as :func:`common.scans.scan_lumi`),
    collect only the fitted-yukawa hesse uncertainty, and plot it
    against the lumi-uncert / nominal-uncert ratio in the legacy WbWb
    range [0.5, 1.5]. No-op unless ``add_last_ecm`` is on and yukawa
    is a fit parameter.
    """
    if not fit.scenario_dict["add_last_ecm"] or "yukawa" not in fit.param_names:
        return
    base = fit.card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(lo, hi, points) * base
    start = np.zeros(len(fit.param_names))
    saved_lumi_uncorr = fit.lumi_uncorr
    saved_lumi_corr = fit.lumi_corr
    yuk_raw = {"uncorr": [], "corr": []}
    try:
        for direction in ("uncorr", "corr"):
            for lumi in l_lumi:
                if direction == "uncorr":
                    fit.lumi_uncorr = lumi
                    fit.lumi_corr = 0
                else:
                    fit.lumi_corr = lumi
                    fit.lumi_uncorr = 0
                fit._build_cov()
                m = _run_local_migrad(fit, start)
                fr = fit.results_from_minuit(m)
                yuk_raw[direction].append(fr[fit._idx["yukawa"]].s * 100)
    finally:
        fit.lumi_uncorr = saved_lumi_uncorr
        fit.lumi_corr = saved_lumi_corr
        fit._build_cov()

    yuk_uncorr = _impact(np.array(yuk_raw["uncorr"]))
    yuk_corr = _impact(np.array(yuk_raw["corr"]))
    # x-axis is the ratio of swept lumi to the nominal lumi uncertainty.
    x = l_lumi / base
    nominal_idx = int(np.argmin(np.abs(x - 1)))
    plt.plot(x, yuk_uncorr, "r-", label=r"Impact on $y_t$ (uncorr)", linewidth=2)
    plt.plot(x, yuk_corr, "r--", label=r"Impact on $y_t$ (corr)", linewidth=2)
    plt.plot(x[nominal_idx], yuk_uncorr[nominal_idx], "ro", label="Nominal value", markersize=8)
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


def scan_scale_vars_yukawa(fit):
    """WbWb legacy yukawa-shift-vs-renormalisation-scale panel.

    Independent mini-scan: sweep ``scale_var_scenario`` over the same
    grid as :func:`common.scans.scan_scale_vars` and record only the
    fitted-yukawa shift relative to the nominal fit. No-op unless
    yukawa is a free fit parameter (``not constrain_yukawa`` and in
    ``param_names``).
    """
    if fit.constrain_yukawa or "yukawa" not in fit.param_names:
        return
    saved = (fit.scale_var_scenario, fit._xsec_base)
    start = np.zeros(len(fit.param_names))
    l_vars, l_yuk = [], []
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
            m = _run_local_migrad(fit, start)
            fr = fit.results_from_minuit(m)
            l_yuk.append(fr[fit._idx["yukawa"]].n
                         - fit.last_fit_results[fit._idx["yukawa"]].n)
    finally:
        fit.scale_var_scenario, fit._xsec_base = saved

    plt.plot(l_vars, np.array(l_yuk), "b-",
             label=r"Shift in fitted $y_t$", linewidth=2)
    plt.plot(fit.mass_scale, 0, "ro", label="Starting point", markersize=8)
    plt.legend()
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.xlabel(r"Renormalisation scale $\mu$ [GeV]")
    plt.ylabel("Shift in fitted parameter")
    process_annotation(fit.card, x=0.97, y=0.12, include_reference=True)
    save_figure(fit.plot_dir, "uncert_yukawa_vs_scale")
