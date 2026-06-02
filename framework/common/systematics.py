"""Systematic-table machinery.

The strategy is the same as in the original ``printSystTable``: for every
systematic in turn, switch off its prior, refit, then compute the impact
of that systematic in quadrature against the total uncertainty.

The hardcoded ``syst_list`` of the original has been replaced by
:func:`systematic_list`, which derives the list from the fit state (which
nuisances were actually added). The set of POIs whose impact is tracked
is read from ``card.POI_DISPLAY`` — adding a new POI is a card-only
edit.
"""

import math
import os

from framework.common.fit_core import OFF, quadrature_subtract


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
    """Pin the named systematic at its centre by setting its prior to OFF.

    Dispatches generically over the FitCore canonical stores; doesn't need
    to be updated when a new card-declared constraint / nuisance is added.
    """
    if name in fit._constraints:
        fit._constraints[name]["sigma"] = OFF
        return
    for kind in fit._active_binned_nuisances:
        if name == f"{kind}_uncorr":
            fit._nuisance_priors[kind]["uncorr"] = OFF
            return
        if name == f"{kind}_corr":
            fit._nuisance_priors[kind]["corr"] = OFF
            return
    if name in fit._active_global_nuisances:
        fit._nuisance_priors[name]["prior"] = OFF
        return
    if name == "lumi_uncorr":
        fit.lumi_uncorr = OFF
        return
    if name == "lumi_corr":
        fit.lumi_corr = OFF
        return
    raise ValueError(f"Unknown systematic: {name}")


def _capture(fit, syst, name):
    """Record this iteration's per-POI uncertainty into ``syst[poi][name]``.

    Stored values are in display units (raw uncert * POI_DISPLAY[poi].scale);
    the optional ``relative`` divide-by-central is applied at print time.
    Quadrature subtraction works on these because (c·a)² − (c·b)² = c²(a²−b²).
    """
    res = fit.fit_results(printout=False)
    for poi in syst:
        scale = fit.card.POI_DISPLAY[poi]["scale"]
        syst[poi][name] = res[fit._idx[poi]].s * scale


def _subtract_from_total(syst, name):
    if name in ("stat", "total"):
        return
    for poi in syst:
        syst[poi][name] = float(quadrature_subtract(syst[poi]["total"], syst[poi][name]))


def _estimate_stat(fit, syst, breakdown_parametric):
    fit.reinitialise_to_stat()
    if not breakdown_parametric:
        fit.fit_parameters()
        _capture(fit, syst, "stat")
        fit.reinitialise_to_nominal()
        return

    free_params = list(syst.keys())
    stat_dict = {}
    for p in free_params:
        fit.minuit.fixed = [True] * len(fit.param_names)
        fit.minuit.fixed[fit._idx[p]] = False
        fit.fit_parameters(init_minuit=False)
        stat = fit.fit_results(printout=False)[fit._idx[p]].s
        syst[p]["stat"] = stat * fit.card.POI_DISPLAY[p]["scale"]
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
            unc = res[fit._idx[other]].s * fit.card.POI_DISPLAY[other]["scale"]
            stat_dict[f"{other}_{p}"] = float(quadrature_subtract(syst[other]["total"], unc))

    for p in free_params:
        for poi in free_params:
            syst[poi][p] = stat_dict[f"{poi}_{p}"] if p != poi else float("nan")

    fit.minuit.fixed = [False] * len(fit.param_names)
    fit.reinitialise_to_nominal()


def estimate_systematic(fit, name, syst, breakdown_parametric=False):
    if name == "stat":
        _estimate_stat(fit, syst, breakdown_parametric)
        return
    if name == "total":
        fit.fit_parameters()
        _capture(fit, syst, "total")
        return
    _turn_off(fit, name)
    fit.fit_parameters()
    _capture(fit, syst, name)
    _subtract_from_total(syst, name)
    fit.reinitialise_to_nominal()


# ---------------------------------------------------------------------------
# Orchestration: build the table, print, write LaTeX
# ---------------------------------------------------------------------------
def compute_syst_breakdown(fit):
    """Iterate the configured systematics, capturing each one's quadrature
    contribution to the total uncertainty on each POI in ``card.POI_DISPLAY``.

    Returns ``(syst, totals, centrals)`` — ``syst[poi][source]`` the per-source
    contribution, ``totals[poi]`` the total experimental uncertainty,
    ``centrals[poi]`` the best-fit central — without printing or writing any
    table (so callers such as the scenario comparison can collect breakdowns
    across several fit geometries and emit a consolidated table themselves).

    Mutates ``fit`` in place (priors + ``fit.minuit``). ``estimate_systematic``
    restores the priors after each entry via ``reinitialise_to_nominal``;
    a final ``fit_parameters()`` in the ``finally`` block puts ``fit.minuit``
    back to its pre-call nominal-migrad state.
    """
    syst = {poi: {} for poi in fit.tracked_pois()}
    try:
        breakdown = fit.stat_breakdown_default()
        for s in systematic_list(fit):
            if s == "stat":
                estimate_systematic(fit, s, syst, breakdown_parametric=breakdown)
            else:
                estimate_systematic(fit, s, syst)
    finally:
        # estimate_systematic / _estimate_stat leave fit.minuit at the last
        # iteration's migrad result; rerun the nominal fit so fit.minuit is
        # bit-identical to its pre-call state. ``last_fit_results`` is only
        # populated by ``fit_results(printout=True)``, so set it explicitly
        # here — this makes the breakdown self-contained (callers need not
        # have done a prior printout fit to populate the nominal central).
        fit.reinitialise_to_nominal()
        fit.fit_parameters()
        fit.last_fit_results = fit.fit_results(printout=False)

    totals = {poi: syst[poi].pop("total") for poi in syst}
    centrals = {poi: fit.last_fit_results[fit._idx[poi]].n for poi in syst}
    return syst, totals, centrals


def print_syst_table(fit, *, latex_path=None):
    """Compute the systematics breakdown (:func:`compute_syst_breakdown`) and
    print it, optionally writing the LaTeX table.

    ``latex_path`` defaults to ``fit.card.SYST_TABLE_PATH``; pass ``""`` to
    suppress LaTeX output entirely.
    """
    if latex_path is None:
        latex_path = fit.card.SYST_TABLE_PATH
    syst, totals, centrals = compute_syst_breakdown(fit)
    _print_table(fit.card, syst, totals, centrals)
    if latex_path:
        _write_latex(fit.card, syst, totals, centrals, latex_path)


def _display(raw, disp, central):
    """Apply the optional relative-mode divide-by-central. ``raw`` is
    already pre-scaled (see ``_capture``)."""
    if disp.get("relative"):
        return raw / central
    return raw


def _fmt_cell(raw, disp, central, width=12):
    """Render one syst-table cell; NaN (diagonal POI-on-self entries from
    ``_estimate_stat``) prints as a dash."""
    if not math.isfinite(raw):
        return f"{'--':<{width}}"
    return f"{_display(raw, disp, central):<{width}.1f}"


def _emit_rows(card, syst, totals, centrals, cell, total_cell, theory_cell):
    """Build the syst-table rows once, shared by the text and LaTeX emitters.

    Returns ``(body, total, theory)`` where ``body`` is a list of
    ``(syst_name, [cell_str, ...])`` in source order, ``total`` is the list of
    total-row cell strings, and ``theory`` is the theory-row cell strings or
    ``None`` when the card carries no ``THEORY_UNC``. Row order, the
    ``next(iter(syst.values()))`` body source, the per-POI column iteration and
    the ``THEORY_UNC`` gate live here so the two emitters cannot structurally
    drift; each emitter supplies its own cell formatters, labels and glue."""
    pois = list(syst.keys())
    body = []
    if pois:
        for s in next(iter(syst.values())):
            body.append((s, [cell(syst[poi][s], card.POI_DISPLAY[poi], centrals[poi])
                             for poi in pois]))
    total = [total_cell(totals[poi], card.POI_DISPLAY[poi], centrals[poi])
             for poi in pois]
    theory = getattr(card, "THEORY_UNC", None)
    theory_cells = (None if theory is None
                    else [theory_cell(theory.get(poi, 0)) for poi in pois])
    return body, total, theory_cells


def _print_table(card, syst, totals, centrals):
    pois = list(syst.keys())
    sep_len = 12 * (1 + len(pois))
    headers = [f"{poi.capitalize()} [{card.POI_DISPLAY[poi]['unit']}]" for poi in pois]
    body, total, theory = _emit_rows(
        card, syst, totals, centrals,
        _fmt_cell,
        lambda raw, disp, central: f"{_display(raw, disp, central):<12.1f}",
        lambda raw: f"{raw:<12.0f}")
    print()
    print(f"{'Systematic':<12} " + " ".join(f"{h:<12}" for h in headers))
    print("-" * sep_len)
    for label, cells in body:
        print(f"{label:<12} {' '.join(cells)}")
    print("-" * sep_len)
    print(f"{'total exp':<12} {' '.join(total)}")
    if theory is not None:
        print(f"{'theory':<12} {' '.join(theory)}")


def _write_latex(card, syst, totals, centrals, path):
    pois = list(syst.keys())
    cols = "|l|" + "r|" * len(pois)
    header_cells = " & ".join(
        f"{poi.capitalize()} Uncertainty ({card.POI_DISPLAY[poi]['unit']})"
        for poi in pois)

    def _latex_cell(raw, disp, central):
        if not math.isfinite(raw):
            return "--"
        return f"{_display(raw, disp, central):.1f}"

    body, total, theory = _emit_rows(
        card, syst, totals, centrals,
        _latex_cell,
        lambda raw, disp, central: f"{_display(raw, disp, central):.1f}",
        lambda raw: f"{raw:.0f}")
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        f"Systematic & {header_cells} \\\\",
        r"\hline",
    ]
    for label, cells in body:
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines.append(r"\hline")
    lines.append(f"total & {' & '.join(total)} \\\\")
    if theory is not None:
        lines.append(f"theory & {' & '.join(theory)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if len(pois) > 2:
        poi_caption = ", ".join(pois[:-1]) + ", and " + pois[-1]
    else:
        poi_caption = " and ".join(pois) if pois else "(none)"
    lines.append(rf"\caption{{Systematic uncertainties on {poi_caption}.}}")
    lines.append(r"\label{tab:syst_unc}")
    lines.append(r"\end{table}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
