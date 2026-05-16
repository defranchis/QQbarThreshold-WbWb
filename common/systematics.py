"""Systematic-table machinery.

The strategy is the same as in the original ``printSystTable``: for every
systematic in turn, switch off its prior, refit, then compute the impact
of that systematic in quadrature against the total uncertainty.

The hardcoded ``syst_list`` of the original has been replaced by
:func:`systematic_list`, which derives the list from the fit state (which
nuisances were actually added).
"""

import os

from common.fit_core import _OFF, quadrature_subtract


# ---------------------------------------------------------------------------
# Which systematics apply, given the fit state?
# ---------------------------------------------------------------------------
def systematic_list(fit):
    """Build the list of systematics from FitCore state (data-driven over
    ``_constraints`` / ``_active_binned_nuisances`` / ``_active_global_nuisances``
    + the fixed lumi entries), in the order declared by the card's
    ``SYST_TABLE_ORDER``. Any active systematic not listed in the card is
    appended alphabetically at the end so future card additions still
    appear without reshuffling the legacy rows."""
    active = set()
    for name, c in fit._constraints.items():
        if c["active"]:
            active.add(name)
    for kind in fit._active_binned_nuisances:
        active.add(f"{kind}_uncorr")
        active.add(f"{kind}_corr")
    for kind in fit._active_global_nuisances:
        active.add(kind)
    active.add("lumi_uncorr")
    active.add("lumi_corr")

    ordered = []
    for n in fit.card.SYST_TABLE_ORDER:
        if n in active:
            # Direct match (single-name entries: alphas, yukawa, sw2, ...)
            ordered.append(n)
        else:
            # Binned-nuisance shorthand: "BEC" → "BEC_uncorr" + "BEC_corr"
            # (lumi works the same way: "lumi" → "lumi_uncorr" + "lumi_corr")
            for suffix in ("_uncorr", "_corr"):
                expanded = f"{n}{suffix}"
                if expanded in active:
                    ordered.append(expanded)
    extras = sorted(n for n in active if n not in ordered)
    return ["total", "stat", *ordered, *extras]


# ---------------------------------------------------------------------------
# Per-syst evaluation
# ---------------------------------------------------------------------------
def _turn_off(fit, name):
    """Pin the named systematic at its centre by setting its prior to _OFF.

    Dispatches generically over the FitCore canonical stores; doesn't need
    to be updated when a new card-declared constraint / nuisance is added.
    """
    if name in fit._constraints:
        fit._constraints[name]["sigma"] = _OFF
        return
    for kind in fit._active_binned_nuisances:
        if name == f"{kind}_uncorr":
            fit._nuisance_priors[kind]["uncorr"] = _OFF
            return
        if name == f"{kind}_corr":
            fit._nuisance_priors[kind]["corr"] = _OFF
            return
    if name in fit._active_global_nuisances:
        fit._nuisance_priors[name]["prior"] = _OFF
        return
    if name == "lumi_uncorr":
        fit.lumi_uncorr = _OFF
        return
    if name == "lumi_corr":
        fit.lumi_corr = _OFF
        return
    raise ValueError(f"Unknown systematic: {name}")


def _capture(fit, syst_mass, syst_width, syst_yukawa, name):
    res = fit.fit_results(printout=False)
    syst_mass[name] = res[fit._idx["mass"]].s * 1000
    syst_width[name] = res[fit._idx["width"]].s * 1000
    if syst_yukawa is not None:
        syst_yukawa[name] = res[fit._idx["yukawa"]].s * 100


def _subtract_from_total(d, name):
    if name in ("stat", "total"):
        return
    d[name] = float(quadrature_subtract(d["total"], d[name]))


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
        fit.minuit.fixed[fit._idx[p]] = False
        fit.fit_parameters(init_minuit=False)
        stat = fit.fit_results(printout=False)[fit._idx[p]].s
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
        fit.minuit.fixed[fit._idx[p]] = True
        fit.fit_parameters(init_minuit=False)
        res = fit.fit_results(printout=False)
        for other in free_params:
            if other == p:
                continue
            unc = res[fit._idx[other]].s * (1000 if other != "yukawa" else 100)
            tgt = {"mass": syst_mass, "width": syst_width, "yukawa": syst_yukawa}[other]
            stat_dict[f"{other}_{p}"] = float(quadrature_subtract(tgt["total"], unc))

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
    _turn_off(fit, name)
    fit.fit_parameters()
    _capture(fit, syst_mass, syst_width, syst_yukawa, name)
    _subtract_from_total(syst_mass, name)
    _subtract_from_total(syst_width, name)
    if syst_yukawa is not None:
        _subtract_from_total(syst_yukawa, name)
    fit.reinitialise_to_nominal()


# ---------------------------------------------------------------------------
# Orchestration: build the table, print, write LaTeX
# ---------------------------------------------------------------------------
def print_syst_table(fit, *, latex_path="systematics_table.tex"):
    """Iterate the configured systematics, capturing each one's quadrature
    contribution to the total uncertainty.

    Mutates ``fit`` in place (priors + ``fit.minuit``). ``estimate_systematic``
    restores the priors after each entry via ``reinitialise_to_nominal``;
    a final ``fit_parameters()`` in the ``finally`` block puts ``fit.minuit``
    back to its pre-call nominal-migrad state.
    """
    syst_mass, syst_width = {}, {}
    syst_yukawa = None if fit.constrain_yukawa else {}

    try:
        for syst in systematic_list(fit):
            # Subtle: the parametric stat breakdown needs to know whether Yukawa
            # is constrained, *not* whether we track it as a syst.
            if syst == "stat":
                estimate_systematic(fit, syst, syst_mass, syst_width, syst_yukawa,
                                    breakdown_parametric=not fit.constrain_yukawa)
            else:
                estimate_systematic(fit, syst, syst_mass, syst_width, syst_yukawa)
    finally:
        # estimate_systematic / _estimate_stat leave fit.minuit at the last
        # iteration's migrad result; rerun the nominal fit so fit.minuit is
        # bit-identical to its pre-call state.
        fit.reinitialise_to_nominal()
        fit.fit_parameters()

    total_mass = syst_mass.pop("total")
    total_width = syst_width.pop("total")
    yukawa_central = None
    total_yukawa = None
    if syst_yukawa is not None:
        yukawa_central = fit.last_fit_results[fit._idx["yukawa"]].n
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
