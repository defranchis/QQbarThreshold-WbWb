"""Parameter and systematic scans.

Each scan takes a configured :class:`FitCore` instance, sweeps one knob,
and produces a diagnostic plot in ``fit.plot_dir``. All scans are free
functions so they can be opted into independently from the entry script.

Most scans mirror methods of the same names on the original ``fit`` class;
BEC and BES nuisance sweeps share a single :func:`_scan_nuisance` helper.
"""

import copy
import multiprocessing as mp
import os
from multiprocessing.connection import wait

import iminuit
import matplotlib.pyplot as plt
import numpy as np

from framework.common.fit_core import ecm_to_str, quadrature_subtract
from framework.common.plots import (
    _labels_module,
    process_annotation,
    projection_title,
    save_figure,
)


def run_parallel(jobs, max_workers=6):
    """Run a list of zero-argument callables in parallel via fork.

    Each scan helper is self-contained — it reads from ``fit`` without mutating
    it (see ``scripts/audit_scans.py`` for the per-scan invariant check) and
    writes a plot file under ``fit.plot_dir``. That makes parallelisation
    trivial under ``fork``: child processes inherit ``fit`` from the parent's
    memory, do their work, exit. No pickling, no shared mutable state.

    Why ``multiprocessing.Process`` rather than ``Pool`` /
    ``ProcessPoolExecutor``: both of those route tasks through a queue that
    pickles its arguments, which would force two awkward changes here — the
    inline ``lambda``\\s in ``doFit_wbwb.py`` that capture ``fit`` aren't
    picklable, and the ``FitCore`` itself carries enough state (smeared
    DataFrames, the cached morph matrix, the Minuit object) that round-tripping
    it through a queue per dispatch would dwarf the scan work. With ``fork``
    and a bare ``Process``, ``self._target`` lives in inherited memory and is
    never serialised.

    Parameters
    ----------
    jobs : sequence of callables
        Each callable should produce its own side effects (typically a
        plot file written under ``fit.plot_dir``). Return values are
        ignored.
    max_workers : int
        Hard ceiling on the number of concurrent child processes. New jobs
        are launched as earlier ones finish, so a long list of short jobs
        still completes promptly.

    Returns
    -------
    None — the function blocks until every job has exited.

    Raises
    ------
    RuntimeError if any child exited with a non-zero status; the rest of
    the jobs are still allowed to finish first so partial output is
    preserved.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    ctx = mp.get_context("fork")
    procs = []
    active = []

    for job in jobs:
        while len(active) >= max_workers:
            # Block until at least one child exits — no polling needed,
            # the kernel wakes us via the sentinel pipes.
            wait([p.sentinel for p in active])
            active = [p for p in active if p.is_alive()]
            for p in procs:
                if p not in active and p.exitcode is None:
                    p.join()
        p = ctx.Process(target=job)
        p.start()
        procs.append(p)
        active.append(p)

    for p in procs:
        p.join()

    failures = [p for p in procs if p.exitcode != 0]
    if failures:
        names = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in failures)
        raise RuntimeError(f"{len(failures)} scan worker(s) failed: {names}")


def impact(arr):
    """``sqrt(arr[i]**2 - arr[0]**2)`` elementwise — the "impact relative to
    the no-syst baseline" pattern."""
    arr = np.asarray(arr)
    return quadrature_subtract(arr, arr[0])


# Colour + linestyle + marker cycles used to render one impact line per POI.
_POI_COLORS  = ["b", "g", "r", "orange", "purple", "brown"]
_POI_STYLES  = ["-", "--", "-.", ":"]
_POI_MARKERS = ["o", "s", "^", "D", "v", "P"]
_POI_MARKER_COLORS = ["red", "orange", "purple", "brown", "teal", "olive"]


def _poi_line(i):
    return _POI_COLORS[i % len(_POI_COLORS)], _POI_STYLES[i % len(_POI_STYLES)]


def _poi_marker(i):
    return _POI_MARKER_COLORS[i % len(_POI_MARKER_COLORS)], _POI_MARKERS[i % len(_POI_MARKERS)]


def _pois_by_unit(fit):
    """Group ``fit.tracked_pois()`` by their POI_DISPLAY unit. Returns an
    ordered dict ``{unit: [poi_names...]}`` preserving the card's
    POI_DISPLAY iteration order."""
    out = {}
    for poi in fit.tracked_pois():
        unit = fit.card.POI_DISPLAY[poi]["unit"]
        out.setdefault(unit, []).append(poi)
    return out


def poi_symbol(fit, poi):
    """Math symbol declared in ``card.POI_DISPLAY[poi]["symbol"]`` (rendered
    via LaTeX in plot legends). Falls back to the bare POI name."""
    return fit.card.POI_DISPLAY[poi].get("symbol", poi)


def _impact_pois_filename(stem, pois):
    """Build ``uncert_{stem}_vs_X`` → ``uncert_{p1}_{p2}_vs_X``."""
    return "_".join(["uncert", *pois, "vs", stem])


def _centrals(fit, pois):
    """Last-fit central values for ``pois`` (used in relative-mode display)."""
    return {poi: fit.last_fit_results[fit._idx[poi]].n for poi in pois}


def _maybe_relative(val, fit, poi, centrals):
    """Apply the POI_DISPLAY ``relative`` divide-by-central if requested."""
    if fit.card.POI_DISPLAY[poi].get("relative"):
        return val / centrals[poi]
    return val


def _local_minuit(fit, start):
    """Fresh local ``Minuit`` instance on ``fit.chi2`` at the given start
    vector, errordef=1 (chi²). Doesn't run migrad — callers that want
    to fix parameters before fitting (``scan_chi2``) can set ``m.fixed``
    first."""
    m = iminuit.Minuit(fit.chi2, start, name=fit.param_names)
    m.errordef = 1
    return m


def run_local_migrad(fit, start):
    """Like ``_local_minuit`` but also runs migrad. The common pattern for
    deepcopy-free scan helpers that don't touch ``fit.minuit``."""
    m = _local_minuit(fit, start)
    m.migrad()
    return m


def sweep_lumi(fit, l_lumi, pois):
    """Sweep ``lumi_uncorr`` / ``lumi_corr`` (one direction at a time) over
    ``l_lumi``, recording per-POI hesse uncertainties at each grid point.
    Returns ``{"uncorr": {poi: [.s...]}, "corr": {poi: [.s...]}}`` of raw
    (unscaled, physical-unit) values. Saves/restores the lumi attrs and
    rebuilds the cov on exit.

    Shared between :func:`scan_lumi` and WbWb's yukawa-vs-lumi-ratio panel.
    """
    start = np.zeros(len(fit.param_names))
    saved_uncorr = fit.lumi_uncorr
    saved_corr = fit.lumi_corr
    raw = {d: {poi: [] for poi in pois} for d in ("uncorr", "corr")}
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
                m = run_local_migrad(fit, start)
                fr = fit.results_from_minuit(m)
                for poi in pois:
                    raw[direction][poi].append(fr[fit._idx[poi]].s)
    finally:
        fit.lumi_uncorr = saved_uncorr
        fit.lumi_corr = saved_corr
        fit._build_cov()
    return raw


def sweep_scale_vars(fit, pois):
    """Sweep the renormalisation scale over ``fit.scale_vars`` (filtered to
    valid mass-scale tags), recording per-POI fitted shifts relative to
    ``fit.last_fit_results``. Returns ``(l_vars, {poi: [shifts...]})`` —
    only points whose template tag exists in ``fit.xsec_dict`` appear in
    ``l_vars``. Saves/restores ``scale_var_scenario`` + ``_xsec_base``.

    Shared between :func:`scan_scale_vars` and WbWb's yukawa-shift panel.
    """
    saved = (fit.scale_var_scenario, fit._xsec_base)
    start = np.zeros(len(fit.param_names))
    l_vars = []
    shifts = {poi: [] for poi in pois}
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
            m = run_local_migrad(fit, start)
            fr = fit.results_from_minuit(m)
            for poi in pois:
                shifts[poi].append(fr[fit._idx[poi]].n
                                   - fit.last_fit_results[fit._idx[poi]].n)
    finally:
        fit.scale_var_scenario, fit._xsec_base = saved
    return l_vars, shifts


# ---------------------------------------------------------------------------
# Beam-energy resolution
# ---------------------------------------------------------------------------
def scan_beam_resolution(fit, *, lo=0.0, hi=0.5, step=0.01):
    """``doLSscan`` — sweep beam-energy resolution, record statistical
    uncertainty on each POI (grouped by unit)."""
    if lo == 0:
        lo = 1e-6
    grid = np.arange(lo, hi + step / 2, step)
    pois = [p for p in fit.tracked_pois() if fit.is_scannable_poi(p)]
    results = {poi: [] for poi in pois}

    work = copy.deepcopy(fit)
    work.reinitialise_to_stat()
    work.param_names = [p for p in fit.param_names if "BEC" not in p and "BES" not in p]
    work._active_binned_nuisances.discard("BEC")
    work._active_binned_nuisances.discard("BES")
    for res in grid:
        work.beam_energy_res = res
        work.update()
        fr = work.fit_results(printout=False)
        for poi in pois:
            results[poi].append(fr[work._idx[poi]].s)

    work.beam_energy_res = fit.beam_energy_res
    work.update()
    work.last_fit_results = work.fit_results(printout=False)
    centrals = _centrals(work, pois)
    baselines = {poi: work.last_fit_results[work._idx[poi]].s for poi in pois}

    for unit, unit_pois in _pois_by_unit(fit).items():
        unit_pois = [p for p in unit_pois if p in pois]
        if not unit_pois:
            continue
        plt.figure()
        for i, poi in enumerate(unit_pois):
            scale = fit.card.POI_DISPLAY[poi]["scale"]
            vals = _maybe_relative(np.array(results[poi]) * scale, fit, poi, centrals)
            base = _maybe_relative(baselines[poi] * scale, fit, poi, centrals)
            color, ls = _poi_line(i)
            mcolor, marker = _poi_marker(i)
            sym = poi_symbol(fit, poi)
            plt.plot(grid, vals, color=color, linestyle=ls,
                     label=rf"Stat. uncert. in ${sym}$", linewidth=2)
            plt.plot(fit.beam_energy_res, base, marker=marker, color=mcolor,
                     linestyle="None", label=rf"Baseline ${sym}$", markersize=8)
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel("Beam energy spread [%]")
        plt.ylabel(f"Statistical uncertainty [{unit}]")
        process_annotation(fit.card, x=0.95, y=0.37)
        save_figure(fit.plot_dir, _impact_pois_filename("BER", unit_pois))


# ---------------------------------------------------------------------------
# Nuisance-prior scans (binned: BEC / BES)
# ---------------------------------------------------------------------------
def _scan_nuisance(fit, kind, variations, *, axis_unit, axis_label):
    """Generic binned-nuisance prior sweep.

    For a binned nuisance ``kind`` (registered in ``card.SYSTEMATICS``
    with ``type=binned`` and already activated via ``add_binned_nuisance``),
    sweep its ``PRIORS[kind]["uncorr"]`` and ``PRIORS[kind]["corr"]`` in
    turn over ``variations`` (in physical units) and plot the impact on
    each POI in ``card.POI_DISPLAY``, one panel per unit-group.

    Mutates ``fit._nuisance_priors[kind]`` in place, runs a fresh local
    Minuit (cold-start) per grid point on ``fit.chi2``, restores on exit.
    """
    if kind not in fit._active_binned_nuisances:
        raise ValueError(
            f"_scan_nuisance({kind!r}) needs the nuisance active — "
            f"call fit.add_binned_nuisance({kind!r}) first."
        )
    baseline_uncorr = fit.card.PRIORS[kind]["uncorr"]
    saved_priors = dict(fit._nuisance_priors[kind])
    start = np.zeros(len(fit.param_names))
    pois = fit.tracked_pois()

    def _fit(uncorr, corr):
        fit.set_binned_nuisance_priors(kind, uncorr=uncorr, corr=corr)
        m = run_local_migrad(fit, start)
        fr = fit.results_from_minuit(m)
        return {poi: fr[fit._idx[poi]].s for poi in pois}

    results = {"uncorr": {poi: [] for poi in pois},
               "corr":   {poi: [] for poi in pois}}
    baseline_raw = {poi: [] for poi in pois}
    try:
        for direction in ("uncorr", "corr"):
            for v in variations:
                v = max(v, 1e-6)
                v_uncorr = v if direction == "uncorr" else 1e-6
                v_corr = v if direction == "corr" else 1e-6
                vals = _fit(v_uncorr, v_corr)
                for poi in pois:
                    results[direction][poi].append(vals[poi])
        for v in (0, baseline_uncorr):
            v = max(v, 1e-6)
            vals = _fit(v, 1e-6)
            for poi in pois:
                baseline_raw[poi].append(vals[poi])
    finally:
        fit._nuisance_priors[kind] = saved_priors

    centrals = _centrals(fit, pois)
    impacts = {d: {} for d in ("uncorr", "corr")}
    baselines = {}
    for poi in pois:
        scale = fit.card.POI_DISPLAY[poi]["scale"]
        impacts["uncorr"][poi] = impact(results["uncorr"][poi]) * scale
        impacts["corr"][poi] = impact(results["corr"][poi]) * scale
        baselines[poi] = impact(baseline_raw[poi])[-1] * scale

    x = variations * axis_unit
    for unit, unit_pois in _pois_by_unit(fit).items():
        plt.figure()
        for i, poi in enumerate(unit_pois):
            color, _ = _poi_line(i)
            mcolor, marker = _poi_marker(i)
            sym = poi_symbol(fit, poi)
            plt.plot(x, _maybe_relative(impacts["uncorr"][poi], fit, poi, centrals),
                     color=color, linestyle="-",
                     label=rf"Impact on ${sym}$ (uncorr.)", linewidth=2)
            plt.plot(x, _maybe_relative(impacts["corr"][poi], fit, poi, centrals),
                     color=color, linestyle="--",
                     label=rf"Impact on ${sym}$ (corr.)", linewidth=2)
            plt.plot(baseline_uncorr * axis_unit, _maybe_relative(baselines[poi], fit, poi, centrals),
                     marker=marker, color=mcolor, linestyle="None",
                     label=rf"Baseline ${sym}$ (uncorr.)", markersize=8)
        plt.legend(loc="upper left")
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel(axis_label)
        plt.ylabel(f"Impact on fitted parameter [{unit}]")
        process_annotation(fit.card, x=0.05, y=0.52, ha="left")
        save_figure(fit.plot_dir, _impact_pois_filename(kind, unit_pois))


def scan_bec(fit, *, lo=0, hi=10, step=0.5):
    variations = np.arange(lo, hi + step / 2, step)
    _scan_nuisance(fit, "BEC", variations,
                   axis_unit=1.0,
                   axis_label=r"Uncertainty in $\sqrt{s}$ [MeV]")


def scan_bes(fit, *, lo=0, hi=0.03, step=0.001):
    variations = np.arange(lo, hi + step / 2, step)
    _scan_nuisance(fit, "BES", variations,
                   axis_unit=100.0,
                   axis_label="BES uncertainty [%]")


# ---------------------------------------------------------------------------
# Luminosity
# ---------------------------------------------------------------------------
def scan_lumi(fit, *, lo=0, hi=3, points=11):
    """``doLumiScans`` — sweep both correlation patterns of the lumi unc.

    ``lumi_uncorr`` / ``lumi_corr`` feed only ``_build_cov``; everything else
    (morph matrix, scenario tensors, smeared templates) is constant across
    the scan. Mutate the lumi attrs + cov caches on ``fit``, run a fresh
    local Minuit per grid point on ``fit.chi2``, restore on exit.

    Produces one figure per non-``%`` unit-group of ``fit.tracked_pois()``.
    The WbWb-specific yukawa-vs-lumi-ratio panel lives in
    :func:`process.wbwb.scans.scan_lumi_yukawa_ratio` — call it
    alongside if needed.

    Under ``LUMI_MODE='nuisance'`` the cov-matrix lumi terms are zero, so
    the cov-mode scan loop is a no-op. Dispatch instead to the generic
    binned-nuisance prior scanner — it sweeps the same ``card.PRIORS["lumi"]``
    range via ``set_binned_nuisance_priors("lumi", ...)`` and writes the
    same ``impact_lumi_<units>.pdf`` filenames, so downstream
    consumers (plot panels, allFits sweeps) see no API difference.
    """
    if "lumi" in fit._active_binned_nuisances:
        base = fit.card.PRIORS["lumi"]["uncorr"]
        variations = np.linspace(lo, hi, points) * base
        _scan_nuisance(fit, "lumi", variations,
                       axis_unit=100.0,
                       axis_label="Integrated luminosity uncertainty [%]")
        return
    base = fit.card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(lo, hi, points) * base
    pois = fit.tracked_pois()
    raw = sweep_lumi(fit, l_lumi, pois)
    # Baseline marker: lumi=0 (stat-only) and lumi=base. Two extra fits;
    # quadrature-subtracting the stat-only point gives the lumi impact.
    start = np.zeros(len(fit.param_names))
    saved_uncorr = fit.lumi_uncorr
    saved_corr = fit.lumi_corr
    baseline_raw = {poi: [] for poi in pois}
    try:
        for lumi in (0, base):
            fit.lumi_uncorr = lumi
            fit.lumi_corr = 0
            fit._build_cov()
            m = run_local_migrad(fit, start)
            fr = fit.results_from_minuit(m)
            for poi in pois:
                baseline_raw[poi].append(fr[fit._idx[poi]].s)
    finally:
        fit.lumi_uncorr = saved_uncorr
        fit.lumi_corr = saved_corr
        fit._build_cov()

    centrals = _centrals(fit, pois)
    impacts = {d: {} for d in ("uncorr", "corr")}
    baselines = {}
    for poi in pois:
        scale = fit.card.POI_DISPLAY[poi]["scale"]
        # Pre-scale before impact() so the internal sqrt(a²-b²) operates on
        # display-unit values — keeps the float precision identical to the
        # legacy code that scaled inline.
        impacts["uncorr"][poi] = impact(np.array(raw["uncorr"][poi]) * scale)
        impacts["corr"][poi] = impact(np.array(raw["corr"][poi]) * scale)
        baselines[poi] = impact(np.array(baseline_raw[poi]) * scale)[-1]

    base_pct = l_lumi * 100
    for unit, unit_pois in _pois_by_unit(fit).items():
        if unit == "%":
            continue  # fractional-uncertainty POIs get a per-process helper
        plt.figure()
        for i, poi in enumerate(unit_pois):
            color, _ = _poi_line(i)
            mcolor, marker = _poi_marker(i)
            sym = poi_symbol(fit, poi)
            plt.plot(base_pct, _maybe_relative(impacts["uncorr"][poi], fit, poi, centrals),
                     color=color, linestyle="-",
                     label=rf"Impact on ${sym}$ (uncorr.)", linewidth=2)
            plt.plot(base_pct, _maybe_relative(impacts["corr"][poi], fit, poi, centrals),
                     color=color, linestyle="--",
                     label=rf"Impact on ${sym}$ (corr.)", linewidth=2)
            plt.plot(base * 100, _maybe_relative(baselines[poi], fit, poi, centrals),
                     marker=marker, color=mcolor, linestyle="None",
                     label=rf"Baseline ${sym}$ (uncorr.)", markersize=8)
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel("Integrated luminosity uncertainty [%]")
        plt.ylabel(f"Impact on fitted parameter [{unit}]")
        process_annotation(fit.card, x=0.92, y=0.14)
        save_figure(fit.plot_dir, _impact_pois_filename("lumi", unit_pois))


# ---------------------------------------------------------------------------
# Statistical point-to-point correlation
# ---------------------------------------------------------------------------
def scan_stat_correlation(fit, *, lo=0.0, hi=0.99, points=21, outdir=None):
    """Sweep the global point-to-point statistical correlation ρ between the
    measured cross sections at different √s from ``lo`` to ``hi``, recording the
    STAT-ONLY uncertainty on each POI.

    All systematics are switched off (``reinitialise_to_stat``) so the data
    covariance is purely statistical: ``cov_ij = ρ·σ_i·σ_j`` for i≠j, ``σ_i²``
    on the diagonal. As ρ→1 the *common* (all-points-together) statistical
    fluctuation becomes the only poorly-measured mode, while the point-to-point
    *differences* — i.e. the line **shape**, hence m_W/Γ_W — are pinned ever more
    tightly (their inverse-covariance weight ∝ 1/(1−ρ)). So σ(m_W) and σ(Γ_W)
    *decrease* toward 0 as ρ→1: positive statistical correlation between √s
    points *helps* a shape/slope extraction (the common normalisation cancels).
    ρ=1 exactly is singular (rank-1, cov = σσᵀ); the default grid stops at
    ``hi=0.99`` to avoid that degenerate endpoint (``_build_cov`` still caps any
    ρ passed ≥1 just below 1 as a safety net). The per-point σ_i carries the
    selection-efficiency 1/√ε inflation — "stat-only" means "no systematics",
    not "ε = 1".

    Mutates ``fit`` in place (priors → stat-only, ``_stat_corr`` + cov per grid
    point) and restores it on exit (same save/restore-in-``finally`` pattern as
    :func:`scan_lumi`; no ``deepcopy``, which the theory generator can't survive).
    Runs a fresh local Minuit each point (the morph matrix / smeared templates
    are constant). Saves one figure (all POIs share the MeV unit group) to
    ``outdir`` (default ``fit.plot_dir``).
    """
    grid = np.linspace(lo, hi, points)
    pois = [p for p in fit.tracked_pois() if fit.is_scannable_poi(p)]
    start = np.zeros(len(fit.param_names))
    results = {poi: [] for poi in pois}
    saved_stat_corr = getattr(fit, "_stat_corr", 0.0)
    fit.reinitialise_to_stat()
    try:
        for rho in grid:
            fit._stat_corr = float(rho)
            fit._build_cov()
            m = run_local_migrad(fit, start)
            fr = fit.results_from_minuit(m)
            for poi in pois:
                results[poi].append(fr[fit._idx[poi]].s)
    finally:
        # _stat_corr restored before reinitialise_to_nominal so its cov rebuild
        # uses the pre-scan correlation (reinitialise_to_nominal calls _build_cov).
        fit._stat_corr = saved_stat_corr
        fit.reinitialise_to_nominal()

    outdir = outdir or fit.plot_dir
    for unit, unit_pois in _pois_by_unit(fit).items():
        unit_pois = [p for p in unit_pois if p in pois]
        if not unit_pois:
            continue
        plt.figure()
        for i, poi in enumerate(unit_pois):
            scale = fit.card.POI_DISPLAY[poi]["scale"]
            vals = np.array(results[poi]) * scale
            color, ls = _poi_line(i)
            sym = poi_symbol(fit, poi)
            plt.plot(grid, vals, color=color, linestyle=ls,
                     label=rf"Stat. uncert. in ${sym}$", linewidth=2)
        plt.legend(loc="upper left")
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel(r"Statistical correlation $\rho$ between $\sqrt{s}$ points")
        plt.ylabel(f"Statistical uncertainty [{unit}]")
        process_annotation(fit.card, x=0.05, y=0.55, ha="left")
        save_figure(outdir, _impact_pois_filename("statcorr", unit_pois))


# ---------------------------------------------------------------------------
# Cross-section systematic as a fraction of the per-point stat
# ---------------------------------------------------------------------------
def scan_xsec_syst(fit, *, lo=0.0, hi=None, points=21, outdir=None):
    """Sweep a flat *relative* cross-section systematic — a single fractional
    uncertainty on the measured σ(√s), the SAME percentage at every √s — and
    record the TOTAL uncertainty (stat ⊕ this one syst) on each POI. The
    fully-correlated and fully-uncorrelated components are swept independently
    (the other held at zero) and drawn as two lines per POI.

    The component scales with the per-point cross section (``pseudo_data_scenario``,
    a normalisation uncertainty — uncorr → ``diag((δ·σ_i)²)``, corr →
    ``outer(δ·σ, δ·σ)``), NOT with the per-point stat. The x-axis is therefore a
    flat percentage on σ. The default sweep runs from ``lo`` to ``hi``; when
    ``hi`` is ``None`` it is set to twice the SMALLEST per-√s *relative*
    statistical uncertainty ``min_i(σ^stat_i/σ_i)`` — so the scan spans up to
    where the flat syst reaches 2× the tightest point's stat. A vertical
    reference line marks that smallest relative stat uncertainty and is labelled
    with the √s at which it occurs.

    "Only stat + this one systematic": all OTHER systematics are switched off
    (``reinitialise_to_stat``) so the data covariance is stat ⊕ the swept syst.
    At δ=0 both lines start from the stat-only σ; positive δ inflates it.

    Mutates ``fit`` in place (priors → stat-only, ``xsec_syst_*_rel`` + cov per
    grid point) and restores it on exit (same save/restore-in-``finally`` pattern
    as :func:`scan_stat_correlation`; no ``deepcopy``, which the theory generator
    can't survive). Runs a fresh local Minuit each point (the morph matrix /
    smeared templates are constant). Saves one figure per unit-group of the
    scannable POIs to ``outdir`` (default ``fit.plot_dir``).
    """
    # Per-point relative statistical uncertainty σ^stat_i/σ_i; the smallest sets
    # the natural x-axis ceiling (2×) and the reference line.
    xsec_vec = np.asarray(fit.pseudo_data_scenario, dtype=float)
    rel_stat = np.asarray(fit.unc_pseudodata_scenario, dtype=float) / xsec_vec
    i_min = int(np.argmin(rel_stat))
    min_rel_stat = float(rel_stat[i_min])
    ecm_min = float(list(fit.scenario.keys())[i_min])
    if hi is None:
        hi = 2.0 * min_rel_stat

    grid = np.linspace(lo, hi, points)
    pois = [p for p in fit.tracked_pois() if fit.is_scannable_poi(p)]
    start = np.zeros(len(fit.param_names))
    results = {"uncorr": {poi: [] for poi in pois},
               "corr":   {poi: [] for poi in pois}}
    fit.reinitialise_to_stat()
    try:
        for direction in ("uncorr", "corr"):
            for delta in grid:
                fit.xsec_syst_uncorr_rel = float(delta) if direction == "uncorr" else 0.0
                fit.xsec_syst_corr_rel = float(delta) if direction == "corr" else 0.0
                fit._build_cov()
                m = run_local_migrad(fit, start)
                fr = fit.results_from_minuit(m)
                for poi in pois:
                    results[direction][poi].append(fr[fit._idx[poi]].s)
    finally:
        # Drop the flat-relative terms, then reinitialise_to_nominal rebuilds the
        # production covariance (the %-of-stat fractions) cleanly.
        fit.xsec_syst_uncorr_rel = 0.0
        fit.xsec_syst_corr_rel = 0.0
        fit.reinitialise_to_nominal()

    outdir = outdir or fit.plot_dir
    x = grid * 1e4  # flat relative cross-section systematic, in units of 1e-4
    proc_label = _labels_module(fit.card).process_label_short(fit.card)
    # Subscript the (muon) neutrino for this plot only; targeted so it matches
    # the inclusive μνqq̄ final state and leaves any other channel untouched.
    proc_label = proc_label.replace(r"\mu\nu q", r"\mu\nu_\mu q")
    for unit, unit_pois in _pois_by_unit(fit).items():
        unit_pois = [p for p in unit_pois if p in pois]
        if not unit_pois:
            continue
        plt.figure()
        ax = plt.gca()
        for i, poi in enumerate(unit_pois):
            color, _ = _poi_line(i)
            sym = poi_symbol(fit, poi)
            # Relative increase over the stat-only value (the δ=0 grid point,
            # common to both directions): tot/stat − 1, so the origin is (0, 0).
            base = results["uncorr"][poi][0]
            plt.plot(x, np.array(results["uncorr"][poi]) / base - 1,
                     color=color, linestyle="-",
                     label=rf"${sym}$ (uncorr.)", linewidth=2)
            plt.plot(x, np.array(results["corr"][poi]) / base - 1,
                     color=color, linestyle="--",
                     label=rf"${sym}$ (corr.)", linewidth=2)
        # Cap the y-axis at a 50% increase (curves run off the top beyond that).
        plt.ylim(top=0.5)
        # Reference line at the smallest per-√s relative stat uncertainty.
        plt.axvline(min_rel_stat * 1e4, color="0.4", linestyle=":", linewidth=1.5)
        ymin, ymax = plt.ylim()
        plt.text(min_rel_stat * 1e4, ymin + 0.04 * (ymax - ymin),
                 rf" stat. uncertainty at {ecm_min:.1f} GeV",
                 rotation=90, va="bottom", ha="left", color="0.4", fontsize=17)
        # Reference line at a 10% increase in the total uncertainty.
        plt.axhline(0.10, color="0.4", linestyle="--", linewidth=1.5)
        xmin, xmax = plt.xlim()
        plt.text(xmax, 0.10, "10% increase ",
                 va="bottom", ha="right", color="0.4", fontsize=17)
        # Process label outside the frame, upper-left (mirrors the projection
        # title, which sits outside upper-right).
        ax.text(0.0, 1.01, proc_label, transform=ax.transAxes,
                fontsize=22, va="bottom", ha="left")
        plt.legend(loc="upper left")
        plt.title(projection_title(fit.scenario_dict["total_lumi"],
                                   unit="ab", fmt="{:.1f}"), loc="right", fontsize=20)
        plt.xlabel(r"Rel. syst. uncert. in WW xsec [$\times 10^{-4}$]")
        plt.ylabel("Rel. increase in tot. uncert.")
        save_figure(outdir, _impact_pois_filename("xsecsyst", unit_pois))


# ---------------------------------------------------------------------------
# Generic 1-D constraint sigma sweep
# ---------------------------------------------------------------------------
def scan_constraint(fit, name, grid, *, axis_unit, axis_label, plot_filename_stem):
    """Sweep the Gaussian-prior width on ``fit._constraints[name]`` over
    ``grid``, record hesse uncertainties for each POI in
    ``card.POI_DISPLAY``, and plot one panel per unit-group.

    The constraint's sigma feeds only the chi² constraint term — not cov,
    not the morph matrix — so we mutate it in place, run a fresh local
    Minuit (warm-start) per grid point, and restore on exit.
    ``axis_unit`` rescales sigma into plot-space units (e.g. 1e3 for α_S
    in 10⁻³ ticks; 100 for Yukawa in %). One panel per unit-group
    saved to ``uncert_<pois>_vs_<plot_filename_stem>``."""
    start = list(fit.minuit.values)
    saved_sigma = fit._constraints[name]["sigma"]
    baseline = saved_sigma
    pois = fit.tracked_pois()
    raw = {poi: [] for poi in pois}
    try:
        for u in grid:
            fit._constraints[name]["sigma"] = u
            m = run_local_migrad(fit, start)
            for poi in pois:
                step = fit.parameters.step(poi)
                scale = fit.card.POI_DISPLAY[poi]["scale"]
                raw[poi].append(step * m.errors[fit._idx[poi]] * scale)
    finally:
        fit._constraints[name]["sigma"] = saved_sigma

    centrals = _centrals(fit, pois)
    impacts, baselines = {}, {}
    for poi in pois:
        scale = fit.card.POI_DISPLAY[poi]["scale"]
        nominal_stat = fit.last_fit_results[fit._idx[poi]].s * scale
        baselines[poi] = quadrature_subtract(nominal_stat, raw[poi][0])
        impacts[poi] = impact(raw[poi])

    for unit, unit_pois in _pois_by_unit(fit).items():
        plt.figure()
        for i, poi in enumerate(unit_pois):
            val = _maybe_relative(impacts[poi], fit, poi, centrals)
            base = _maybe_relative(baselines[poi], fit, poi, centrals)
            color, ls = _poi_line(i)
            mcolor, marker = _poi_marker(i)
            sym = poi_symbol(fit, poi)
            plt.plot(grid * axis_unit, val, color=color, linestyle=ls,
                     label=rf"Impact on ${sym}$", linewidth=2)
            plt.plot(baseline * axis_unit, base, marker=marker, color=mcolor,
                     linestyle="None", label=rf"Baseline ${sym}$", markersize=8)
        plt.legend(loc="upper left")
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel(axis_label)
        plt.ylabel(f"Impact on fitted parameter [{unit}]")
        process_annotation(fit.card, x=0.92, y=0.17)
        save_figure(fit.plot_dir, _impact_pois_filename(plot_filename_stem, unit_pois))


def scan_alphas(fit, *, hi=3e-4, step=1e-5):
    """``doAlphaSscans`` — sweep the externally-imposed alpha_s prior width."""
    grid = np.arange(1e-10, hi + step / 2, step)
    scan_constraint(fit, "alphas", grid,
                     axis_unit=1e3,
                     axis_label=r"Uncertainty in $\alpha_\mathrm{S} (m_\mathrm{Z}^2) [x10^3]$",
                     plot_filename_stem="alphas")


# ---------------------------------------------------------------------------
# Scale variation
# ---------------------------------------------------------------------------
def scan_scale_vars(fit):
    """``doScaleVars`` — sweep the renormalisation scale and read fitted shift.

    ``scale_var_scenario`` feeds only ``_xsec_base`` in ``_build_chi2_caches``.
    Mutate both on ``fit``, run a fresh local Minuit per scale, restore on exit.

    Produces one figure per non-``%`` unit-group of ``fit.tracked_pois()``.
    The WbWb yukawa-shift panel lives in
    :func:`process.wbwb.scans.scan_scale_vars_yukawa`.
    """
    pois = fit.tracked_pois()
    l_vars, shifts = sweep_scale_vars(fit, pois)

    for unit, unit_pois in _pois_by_unit(fit).items():
        if unit == "%":
            continue
        plt.figure()
        for i, poi in enumerate(unit_pois):
            scale = fit.card.POI_DISPLAY[poi]["scale"]
            color, ls = _poi_line(i)
            sym = poi_symbol(fit, poi)
            plt.plot(l_vars, np.array(shifts[poi]) * scale, color=color, linestyle=ls,
                     label=rf"Shift in fitted ${sym}$", linewidth=2)
        plt.plot(fit.mass_scale, 0, "ro", label="Starting point", markersize=8)
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        plt.xlabel(r"Renormalisation scale $\mu$ [GeV]")
        plt.ylabel(f"Shift in fitted parameter [{unit}]")
        process_annotation(fit.card, x=0.6, y=0.17, include_reference=True)
        save_figure(fit.plot_dir, _impact_pois_filename("scale", unit_pois))


# ---------------------------------------------------------------------------
# True-value scan
# ---------------------------------------------------------------------------
def scan_true_value(fit):
    """``doTrueValueScan`` — fit each pseudo-data template in ``INPUT_DIRS.pseudo``.

    Mutate ``scenario_dict[scan_list]`` + ``lumi_uncorr`` per iteration
    (baseline + coarse sub-scan) and feed each file's smeared template as
    ``create_scenario`` pseudodata. Use a fresh local Minuit; restore on exit.
    Produces one panel per POI in ``fit.tracked_pois()``.
    """
    indir = fit.card.INPUT_DIRS["pseudo"]
    pois = fit.tracked_pois()
    results_baseline = {poi: {} for poi in pois}
    results_coarse = {poi: {} for poi in pois}

    saved_scan_list = list(fit.scenario_dict["scan_list"])
    saved_lumi_uncorr = fit.lumi_uncorr
    start = np.zeros(len(fit.param_names))

    def _fit_once(pseudo):
        fit.rebuild_chi2_state(init_vars=True, pseudodata=pseudo)
        m = run_local_migrad(fit, start)
        return fit.results_from_minuit(m)

    pivot = fit.card.SCENARIO["true_value_pivot"]
    pivot_tag = f"_{pivot}"
    try:
        for fname in sorted(os.listdir(indir)):
            # Filename pattern: ..._{pivot}X.XX_... — the pivot value is what
            # we sweep over (e.g. "mass" for WbWb / WW threshold scans).
            after_pivot = fname.split(pivot_tag, 1)[1]
            pivot_val = after_pivot.split("_")[0]
            pseudo = fit.smear(fit.read_xsec(os.path.join(indir, fname)))

            fit.scenario_dict["scan_list"] = saved_scan_list
            fit.lumi_uncorr = saved_lumi_uncorr
            fr = _fit_once(pseudo)
            bias = fr[fit._idx[pivot]].n - float(pivot_val)
            pivot_unc = fr[fit._idx[pivot]].s
            if abs(bias) > pivot_unc * 0.7:
                continue
            for poi in pois:
                scale = fit.card.POI_DISPLAY[poi]["scale"]
                results_baseline[poi][pivot_val] = fr[fit._idx[poi]].s * scale

            cs = fit.card.SCENARIO["coarse_scan"]
            coarse_scan = [ecm_to_str(e) for e in
                           np.arange(cs["scan_min"], cs["scan_max"] + cs["scan_step"] / 2, cs["scan_step"])]
            fit.scenario_dict["scan_list"] = coarse_scan
            fit.lumi_uncorr = saved_lumi_uncorr * cs["lumi_factor"]
            fr = _fit_once(pseudo)
            for poi in pois:
                scale = fit.card.POI_DISPLAY[poi]["scale"]
                results_coarse[poi][pivot_val] = fr[fit._idx[poi]].s * scale
    finally:
        fit.scenario_dict["scan_list"] = saved_scan_list
        fit.lumi_uncorr = saved_lumi_uncorr
        fit.rebuild_chi2_state(init_vars=True)

    centrals = _centrals(fit, pois)
    pivot_vals = np.array([float(k) for k in results_baseline[pois[0]].keys()]) if pois else np.array([])
    # Sort the x-axis numerically: the dict keys arrive in lexical listdir order,
    # which only equals numeric order for fixed-width pivot strings — negatives or
    # differing decimal widths would draw a zig-zag connecting line.
    order = np.argsort(pivot_vals)
    pivot_vals = pivot_vals[order]
    sym_pivot = poi_symbol(fit, pivot)
    for poi in pois:
        disp = fit.card.POI_DISPLAY[poi]
        errs = _maybe_relative(np.array(list(results_baseline[poi].values())), fit, poi, centrals)[order]
        errs_coarse = _maybe_relative(np.array(list(results_coarse[poi].values())), fit, poi, centrals)[order]
        sym = poi_symbol(fit, poi)
        plt.plot(pivot_vals, errs, "b-", label=rf"Uncertainty in ${sym}$", linewidth=2)
        plt.plot(pivot_vals, errs_coarse, "g--",
                 label=rf"Uncertainty in ${sym}$ (coarse scan)", linewidth=2)
        plt.xlabel(rf"True value of ${sym_pivot}$ [GeV]")
        plt.ylabel(rf"Uncertainty in fitted ${sym}$ [{disp['unit']}]")
        plt.legend()
        plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
        process_annotation(fit.card, x=0.92, y=0.17)
        save_figure(fit.plot_dir, f"uncert_{poi}_vs_true_{pivot}")


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
            f"process.wbwb.xsec_calculator/ttThresholdScanISR.cpp, rebuild "
            f"(process.wbwb.xsec_calculator/compile_calc.sh), and regenerate the cross-section "
            f"templates in {fit.input_dir}.\n"
            f"Alternatively, reduce ``max_abs_shift_neg`` / ``max_abs_shift_pos`` "
            f"so that the shifted ecms stay inside the existing template range."
        )
    saved_scan_list = list(fit.scenario_dict["scan_list"])
    start = np.zeros(len(fit.param_names))
    pois = fit.tracked_pois()
    raw = {poi: [] for poi in pois}
    try:
        for shift in shifts:
            fit.scenario_dict["scan_list"] = [ecm_to_str(e) for e in scan_list + shift]
            fit.rebuild_chi2_state(init_vars=True)
            m = run_local_migrad(fit, start)
            fr = fit.results_from_minuit(m)
            for poi in pois:
                raw[poi].append(fr[fit._idx[poi]].s)
    finally:
        # Restore scenario tensors + chi2 caches in place; fit.minuit was
        # never touched.
        fit.scenario_dict["scan_list"] = saved_scan_list
        fit.rebuild_chi2_state(init_vars=True)

    plt.figure()
    for i, poi in enumerate(pois):
        nominal = fit.last_fit_results[fit._idx[poi]].s
        rel = (np.array(raw[poi]) / nominal - 1) * 100
        color, ls = _poi_line(i)
        sym = poi_symbol(fit, poi)
        plt.plot(shifts, rel, color=color, linestyle=ls, label=rf"${sym}$", linewidth=2)
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
    """POIs to include in the chi² profile scan — tracked POIs that
    are also scannable per the subclass hook."""
    return [p for p in fit.tracked_pois() if fit.is_scannable_poi(p)]


def scan_chi2(fit):
    """``doChi2Scans`` — 1D and 2D chi2 profile scans for the floating parameters.

    Builds a fresh :class:`iminuit.Minuit` per grid point on the shared
    ``fit.chi2`` callable instead of ``deepcopy(fit)`` — the FitCore state
    (cov factor, morph matrix, smeared templates) is read-only inside
    ``chi2``, so cloning the whole object per iteration is wasted work.
    """
    from framework.common.plots import param_axis_label
    labels = {p: param_axis_label(fit.card, p) for p in fit.param_names}
    keys = _scan_keys(fit)
    start = list(fit.minuit.values)  # warm-start every profile fit from the global best

    def profile_chi2(fixed_idx, fixed_val):
        m = _local_minuit(fit, start)
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
            # grid_chi2[i, j] = chi2(gA[i], gB[j]) (axis0=gA, axis1=gB), but
            # plt.contour(X=gA, Y=gB, Z) needs Z[row=Y=gB, col=X=gA] → pass the
            # transpose. Both axes are length 51 so matplotlib does not raise on
            # the un-transposed array; it just reflects the ellipse across the
            # diagonal and flips the apparent m_W–Γ_W correlation tilt.
            plt.contour(fit.value_from_param(gA, pa),
                        fit.value_from_param(gB, pb),
                        grid_chi2.T,
                        levels=[fit.minuit.fval + 1, fit.minuit.fval + 4],
                        colors=["#377eb8", "#4daf4a"], linewidths=2)
            plt.plot(fit.value_from_param(fit.minuit.values[ia_full], pa),
                     fit.value_from_param(fit.minuit.values[ib_full], pb),
                     "k*", label="Best fit value (input)", markersize=10)
            plt.plot([], [], color="#377eb8", label=r"$\Delta\chi^2=1$ ($1\sigma$ 1D proj.)")
            plt.plot([], [], color="#4daf4a", label=r"$\Delta\chi^2=4$ ($2\sigma$ 1D proj.)")
            plt.xlabel(labels.get(pa, pa))
            plt.ylabel(labels.get(pb, pb))
            plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
            process_annotation(fit.card, x=0.95, y=0.15, include_reference=True)
            plt.legend(loc="upper left")
            y_offset = -0.003 if pb == "width" and pa == "mass" else 0
            plt.ylim(fit.value_from_param(gB[0], pb) + y_offset,
                     fit.value_from_param(gB[-1], pb) + y_offset)
            # Round to enough digits that the 5 tick positions stay
            # distinct: a 2-decimal round collapses the few-MeV-wide m_W
            # range (in GeV) to a single value and crashes the locator.
            _xt = np.unique(np.round(np.linspace(fit.value_from_param(gA[0], pa),
                                                 fit.value_from_param(gA[-1], pa), 5), 4))
            if len(_xt) >= 2:
                plt.xticks(_xt)
            save_figure(fit.plot_dir, f"chi2_scan_{pa}_{pb}")
