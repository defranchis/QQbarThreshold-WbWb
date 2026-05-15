"""Systematic-table machinery.

The strategy is the same as in the original ``printSystTable``: for every
systematic in turn, switch off its prior, refit, then compute the impact
of that systematic in quadrature against the total uncertainty.

The hardcoded ``syst_list`` of the original has been replaced by
:func:`systematic_list`, which derives the list from the fit state (which
nuisances were actually added).
"""

import copy
import os

import numpy as np

from common.fit_core import _OFF


# ---------------------------------------------------------------------------
# Which systematics apply, given the fit state?
# ---------------------------------------------------------------------------
def systematic_list(fit):
    out = ["total", "stat", "alphaS"]
    if fit.constrain_yukawa and "yukawa" in fit.param_names:
        out.append("Yukawa")
    if fit.sw2_nuisance:
        out.append("sw2")
    if fit.bes_nuisances:
        out += ["BES_uncorr", "BES_corr"]
    if fit.bec_nuisances:
        out += ["BEC_uncorr", "BEC_corr"]
    out += ["lumi_uncorr", "lumi_corr"]
    return out


# ---------------------------------------------------------------------------
# Per-syst evaluation
# ---------------------------------------------------------------------------
_TURN_OFF = {
    "alphaS":     ("input_uncert_alphas", _OFF),
    "Yukawa":     ("input_uncert_yukawa", _OFF),
    "BES_corr":   ("bes_prior_corr", _OFF),
    "BES_uncorr": ("bes_prior_uncorr", _OFF),
    "BEC_corr":   ("bec_prior_corr", _OFF),
    "BEC_uncorr": ("bec_prior_uncorr", _OFF),
    "lumi_corr":  ("lumi_corr", _OFF),
    "lumi_uncorr":("lumi_uncorr", _OFF),
    "sw2":        ("sw2_prior", _OFF),
}


def _capture(fit, syst_mass, syst_width, syst_yukawa, name):
    res = fit.fit_results(printout=False)
    syst_mass[name] = res[fit.param_names.index("mass")].s * 1000
    syst_width[name] = res[fit.param_names.index("width")].s * 1000
    if syst_yukawa is not None:
        syst_yukawa[name] = res[fit.param_names.index("yukawa")].s * 100


def _quadrature_subtract(d, name):
    if name in ("stat", "total"):
        return
    d[name] = (d["total"] ** 2 - d[name] ** 2) ** 0.5


def _estimate_stat(fit, syst_mass, syst_width, syst_yukawa, breakdown_parametric):
    fit.reinitialise_to_stat()
    if not breakdown_parametric:
        fit.fit_parameters()
        _capture(fit, syst_mass, syst_width, syst_yukawa, "stat")
        fit.reinitialise_to_nominal()
        return

    free_params = ["mass", "width"]
    if syst_yukawa is not None:
        free_params.append("yukawa")

    stat_dict = {}
    for p in free_params:
        fit.minuit.fixed = [True] * len(fit.param_names)
        fit.minuit.fixed[fit.param_names.index(p)] = False
        fit.fit_parameters(init_minuit=False)
        stat = fit.fit_results(printout=False)[fit.param_names.index(p)].s
        if p == "mass":
            syst_mass["stat"] = stat * 1000
        elif p == "width":
            syst_width["stat"] = stat * 1000
        else:
            syst_yukawa["stat"] = stat * 100
        stat_dict[p] = stat

    fit.reinitialise_to_nominal()
    for p in free_params:
        fit.minuit.fixed = [False] * len(fit.param_names)
        fit.minuit.fixed[fit.param_names.index(p)] = True
        fit.fit_parameters(init_minuit=False)
        res = fit.fit_results(printout=False)
        for other in free_params:
            if other == p:
                continue
            unc = res[fit.param_names.index(other)].s * (1000 if other != "yukawa" else 100)
            tgt = {"mass": syst_mass, "width": syst_width, "yukawa": syst_yukawa}[other]
            stat_dict[f"{other}_{p}"] = (tgt["total"] ** 2 - unc ** 2) ** 0.5

    for p in free_params:
        nan = float("nan")
        syst_mass[p] = stat_dict[f"mass_{p}"] if p != "mass" else nan
        syst_width[p] = stat_dict[f"width_{p}"] if p != "width" else nan
        if syst_yukawa is not None:
            syst_yukawa[p] = stat_dict[f"yukawa_{p}"] if p != "yukawa" else nan

    fit.minuit.fixed = [False] * len(fit.param_names)
    fit.reinitialise_to_nominal()


def estimate_systematic(fit, name, syst_mass, syst_width, syst_yukawa,
                        breakdown_parametric=False):
    if name == "stat":
        _estimate_stat(fit, syst_mass, syst_width, syst_yukawa, breakdown_parametric)
        return
    if name == "total":
        fit.fit_parameters()
        _capture(fit, syst_mass, syst_width, syst_yukawa, "total")
        return
    if name not in _TURN_OFF:
        raise ValueError(f"Unknown systematic: {name}")
    attr, off_value = _TURN_OFF[name]
    setattr(fit, attr, off_value)
    fit.fit_parameters()
    _capture(fit, syst_mass, syst_width, syst_yukawa, name)
    _quadrature_subtract(syst_mass, name)
    _quadrature_subtract(syst_width, name)
    if syst_yukawa is not None:
        _quadrature_subtract(syst_yukawa, name)
    fit.reinitialise_to_nominal()


# ---------------------------------------------------------------------------
# Orchestration: build the table, print, write LaTeX
# ---------------------------------------------------------------------------
def print_syst_table(fit, *, latex_path="systematics_table.tex"):
    work = copy.deepcopy(fit)

    syst_mass, syst_width = {}, {}
    syst_yukawa = None if work.constrain_yukawa else {}

    for syst in systematic_list(work):
        # Subtle: the parametric stat breakdown needs to know whether Yukawa
        # is constrained, *not* whether we track it as a syst.
        if syst == "stat":
            estimate_systematic(work, syst, syst_mass, syst_width, syst_yukawa,
                                breakdown_parametric=not work.constrain_yukawa)
        else:
            estimate_systematic(work, syst, syst_mass, syst_width, syst_yukawa)

    total_mass = syst_mass.pop("total")
    total_width = syst_width.pop("total")
    yukawa_central = None
    total_yukawa = None
    if syst_yukawa is not None:
        yukawa_central = fit.last_fit_results[fit.param_names.index("yukawa")].n
        total_yukawa = syst_yukawa.pop("total") / yukawa_central

    th = fit.card.THEORY_UNC

    if syst_yukawa is None:
        _print_no_yukawa(syst_mass, syst_width, total_mass, total_width, th)
    else:
        _print_with_yukawa(syst_mass, syst_width, syst_yukawa,
                           total_mass, total_width, total_yukawa,
                           yukawa_central, th)

    if latex_path:
        _write_latex(syst_mass, syst_width, total_mass, total_width, th, latex_path)


def _print_no_yukawa(syst_mass, syst_width, total_mass, total_width, th):
    print()
    print(f"{'Systematic':<12} {'Mass [MeV]':<12} {'Width [MeV]':<12}")
    print("-" * 36)
    for syst, mass_unc in syst_mass.items():
        print(f"{syst:<12} {mass_unc:<12.1f} {syst_width[syst]:<12.1f}")
    print("-" * 36)
    print(f"{'total exp':<12} {total_mass:<12.1f} {total_width:<12.1f}")
    print(f"{'theory':<12} {th['mass']:<12.0f} {th['width']:<12.0f}")


def _print_with_yukawa(syst_mass, syst_width, syst_yukawa,
                        total_mass, total_width, total_yukawa,
                        yukawa_central, th):
    print()
    print(f"{'Systematic':<12} {'Mass [MeV]':<12} {'Width [MeV]':<12} {'Yukawa [%]':<12}")
    print("-" * 48)
    for syst, mass_unc in syst_mass.items():
        y_unc = syst_yukawa[syst] / yukawa_central
        print(f"{syst:<12} {mass_unc:<12.1f} {syst_width[syst]:<12.1f} {y_unc:<12.1f}")
    print("-" * 48)
    print(f"{'total exp':<12} {total_mass:<12.1f} {total_width:<12.1f} {total_yukawa:<12.1f}")
    print(f"{'theory':<12} {th['mass']:<12.0f} {th['width']:<12.0f} {th.get('yukawa', 0):<12.0f}")


def _write_latex(syst_mass, syst_width, total_mass, total_width, th, path):
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{|l|r|r|}",
        r"\hline",
        r"Systematic & Mass Uncertainty (MeV) & Width Uncertainty (MeV) \\",
        r"\hline",
    ]
    for syst, mass_unc in syst_mass.items():
        lines.append(f"{syst} & {mass_unc:.1f} & {syst_width[syst]:.1f} \\\\")
    lines.append(r"\hline")
    lines.append(f"total & {total_mass:.1f} & {total_width:.1f} \\\\")
    lines.append(f"theory & {th['mass']:.0f} & {th['width']:.0f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Systematic uncertainties on mass and width.}")
    lines.append(r"\label{tab:syst_unc}")
    lines.append(r"\end{table}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
