#!/usr/bin/env python
"""Validate LUMI_MODE='nuisance' vs LUMI_MODE='cov' on WbWb and WW.

Both modes are physically equivalent at LO in the lumi priors: floating a
per-bin (and a fully-correlated) lumi nuisance with Gaussian prior σ_p =
PRIORS["lumi"]["uncorr/corr"] absorbs the same data-cov contribution that
the cov mode adds via Σ_ij ∝ σ_th σ_th × σ_p² . The two should produce
identical POI uncertainties up to Minuit precision.

For each process and each LUMI_MODE we capture:
  - Nominal fit: per-POI total hesse uncertainty.
  - Syst table: per-POI lumi_uncorr / lumi_corr quadrature contributions.
  - scan_lumi sweep: per-POI hesse unc at each grid point (uncorr & corr).

Side-by-side numerical comparison goes to stdout; a per-process plot of
the scan_lumi sweep (overlaying both modes) goes under the investigation
directory.
"""

import argparse
import copy
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from cards import wbwb_default, ww_default  # noqa: E402
from framework.common.scans import run_local_migrad  # noqa: E402
from framework.common.systematics import (  # noqa: E402
    estimate_systematic,
    systematic_list,
)

OUT_DIR = os.path.join(REPO_ROOT, "scripts", "investigations", "lumi_nuisance")

# ---------------------------------------------------------------------------
# Card hot-swap (no edits to the on-disk card)
# ---------------------------------------------------------------------------
def with_lumi_mode(card, mode):
    """Return a shallow-cloned card module with LUMI_MODE overridden."""
    new = types.ModuleType(f"{card.__name__}_{mode}")
    for k in dir(card):
        if k.startswith("_"):
            continue
        setattr(new, k, getattr(card, k))
    new.LUMI_MODE = mode
    # Card-level mutables that FitCore stores by reference (PRIORS,
    # SYSTEMATICS, INPUT_VAR, SCENARIO, ...) must be deep-copied so the
    # two modes can't interfere through shared dicts.
    for k in ("PRIORS", "SYSTEMATICS", "INPUT_VAR", "SCENARIO", "SCENARIO_TWOPOINTS"):
        if hasattr(new, k):
            setattr(new, k, copy.deepcopy(getattr(new, k)))
    return new


# ---------------------------------------------------------------------------
# Fit construction
# ---------------------------------------------------------------------------
def build_wbwb(card):
    from framework.process.wbwb.fit import WbWbFit
    from framework.process.wbwb.generator import WbWbGenerator
    gen = WbWbGenerator(order=card.ORDER, isr=True)
    fit = WbWbFit(card, gen, asimov=True, constrain_yukawa=True)
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
    )
    # Match doFit_wbwb.py --systTable default: BEC + BES on.
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    return fit


def build_ww(card):
    from framework.process.ww.fit import WWFit
    from framework.process.ww.generator import WWGenerator
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, asimov=True)
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
    )
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    return fit


PROCESSES = {
    "wbwb": (wbwb_default, build_wbwb),
    "ww":   (ww_default,   build_ww),
}


# ---------------------------------------------------------------------------
# Capture: nominal, syst table, scan_lumi sweep
# ---------------------------------------------------------------------------
def capture_nominal(fit):
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    return {poi: res[fit._idx[poi]].s * fit.card.POI_DISPLAY[poi]["scale"]
            for poi in fit.tracked_pois()}


def capture_syst_table(fit):
    """Run the full syst table, return {poi: {row: value_in_display_unit}}.

    Mirrors print_syst_table but skips the print / latex side effects.
    """
    syst = {poi: {} for poi in fit.tracked_pois()}
    try:
        for s in systematic_list(fit):
            estimate_systematic(fit, s, syst)
    finally:
        fit.reinitialise_to_nominal()
        fit.fit_parameters()
    # Drop "total" from each POI (we have it separately via capture_nominal)
    for poi in syst:
        syst[poi].pop("total", None)
    return syst


def sweep_lumi_mode_agnostic(fit, l_lumi, pois):
    """Sweep both correlation directions of the lumi unc — works in either
    mode by introspecting whether lumi is a binned nuisance.

    Returns ``{"uncorr": {poi: [hesse_unc_per_point]},
               "corr":   {poi: [...]}}`` in raw (unscaled) units.
    """
    start = np.zeros(len(fit.param_names))
    raw = {d: {p: [] for p in pois} for d in ("uncorr", "corr")}
    if "lumi" in fit._active_binned_nuisances:
        saved = dict(fit._nuisance_priors["lumi"])
        try:
            for direction in ("uncorr", "corr"):
                for lumi in l_lumi:
                    u = lumi if direction == "uncorr" else 1e-10
                    c = lumi if direction == "corr"   else 1e-10
                    fit.set_binned_nuisance_priors("lumi", uncorr=u, corr=c)
                    m = run_local_migrad(fit, start)
                    fr = fit.results_from_minuit(m)
                    for poi in pois:
                        raw[direction][poi].append(fr[fit._idx[poi]].s)
        finally:
            fit._nuisance_priors["lumi"] = saved
    else:
        saved_u, saved_c = fit.lumi_uncorr, fit.lumi_corr
        try:
            for direction in ("uncorr", "corr"):
                for lumi in l_lumi:
                    fit.lumi_uncorr = lumi if direction == "uncorr" else 0.0
                    fit.lumi_corr   = lumi if direction == "corr"   else 0.0
                    fit._build_cov()
                    m = run_local_migrad(fit, start)
                    fr = fit.results_from_minuit(m)
                    for poi in pois:
                        raw[direction][poi].append(fr[fit._idx[poi]].s)
        finally:
            fit.lumi_uncorr, fit.lumi_corr = saved_u, saved_c
            fit._build_cov()
    return raw


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _rel(a, b):
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _flag(rel, tol):
    return "  OK  " if rel < tol else "  FAIL"


def _scale(fit, poi):
    return fit.card.POI_DISPLAY[poi]["scale"]


def _unit(fit, poi):
    return fit.card.POI_DISPLAY[poi]["unit"]


def report_process(name, cov_fit, cov_data, nuis_fit, nuis_data, *, tol=0.02):
    """Print a side-by-side report for one process.

    ``tol`` is the relative-difference threshold above which a row is
    flagged FAIL. 2% is loose — Minuit hesse precision under a syst-table
    rerun is typically O(0.1%) but priors near OFF can wobble more.
    """
    cov_nom, cov_syst, cov_sweep = cov_data
    nuis_nom, nuis_syst, nuis_sweep = nuis_data
    pois = list(cov_nom.keys())
    print()
    print("=" * 90)
    print(f" PROCESS: {name}")
    print("=" * 90)

    # ---- Nominal total uncertainty ----
    print("\n[1] NOMINAL fit — total uncertainty per POI:")
    print(f"{'POI':<10} {'unit':<6} {'cov':>12} {'nuis':>12} {'rel diff':>10}  {'verdict':>7}")
    print("-" * 60)
    for poi in pois:
        a, b = cov_nom[poi], nuis_nom[poi]
        rel = _rel(a, b)
        print(f"{poi:<10} {_unit(cov_fit, poi):<6} {a:>12.4f} {b:>12.4f} {rel:>10.2%}  {_flag(rel, tol)}")

    # ---- Syst-table rows ----
    print("\n[2] SYST TABLE — per-POI lumi rows (display units):")
    rows = sorted(set(cov_syst[pois[0]].keys()) & set(nuis_syst[pois[0]].keys()))
    lumi_rows = [r for r in rows if r.startswith("lumi")]
    other_rows = [r for r in rows if not r.startswith("lumi")]
    for poi in pois:
        print(f"  POI: {poi}")
        for r in lumi_rows:
            a = cov_syst[poi].get(r, float("nan"))
            b = nuis_syst[poi].get(r, float("nan"))
            rel = _rel(a, b)
            print(f"    {r:<14} cov={a:>10.4f}  nuis={b:>10.4f}  rel={rel:>7.2%}  {_flag(rel, tol)}")
        # also show non-lumi rows for sanity (they should be unchanged)
        for r in other_rows:
            a = cov_syst[poi].get(r, float("nan"))
            b = nuis_syst[poi].get(r, float("nan"))
            rel = _rel(a, b)
            ok = _flag(rel, tol)
            print(f"    {r:<14} cov={a:>10.4f}  nuis={b:>10.4f}  rel={rel:>7.2%}  {ok}")

    # ---- Per-point scan_lumi ----
    print("\n[3] scan_lumi — per-point hesse uncertainty (raw, then display-scaled):")
    for direction in ("uncorr", "corr"):
        print(f"  direction={direction}")
        n_pts = len(cov_sweep[direction][pois[0]])
        for poi in pois:
            scale = _scale(cov_fit, poi)
            print(f"    {poi}  (×{scale}):")
            header = "    " + " idx" + " ".join(f"{i:>9d}" for i in range(n_pts))
            print(header)
            cov_pts  = np.array(cov_sweep[direction][poi])  * scale
            nuis_pts = np.array(nuis_sweep[direction][poi]) * scale
            print("    cov   " + " ".join(f"{v:>9.4f}" for v in cov_pts))
            print("    nuis  " + " ".join(f"{v:>9.4f}" for v in nuis_pts))
            rel_pts  = np.array([_rel(a, b) for a, b in zip(cov_pts, nuis_pts)])
            flags = [_flag(r, tol).strip() for r in rel_pts]
            print("    Δrel  " + " ".join(f"{r:>9.2%}" for r in rel_pts))
            print("    pass  " + " ".join(f"{f:>9s}" for f in flags))


def make_plot(name, cov_data, nuis_data, l_lumi, base_unc):
    _, _, cov_sweep = cov_data
    _, _, nuis_sweep = nuis_data
    pois = list(cov_sweep["uncorr"].keys())
    x = l_lumi * 100  # → percent
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, direction in zip(axes, ("uncorr", "corr")):
        for poi in pois:
            ax.plot(x, np.array(cov_sweep[direction][poi]) * 1000,
                    label=f"cov / {poi}", linewidth=2)
            ax.plot(x, np.array(nuis_sweep[direction][poi]) * 1000,
                    "--", label=f"nuis / {poi}", linewidth=2)
        ax.axvline(base_unc * 100, color="0.5", linestyle=":", label="card baseline")
        ax.set_xlabel(f"lumi {direction} uncert [%]")
        ax.set_ylabel("hesse unc on POI [MeV]")
        ax.set_title(f"{name} — {direction}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"sweep_compare_{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n[plot] {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one_process(name, *, points=7):
    base_card, build = PROCESSES[name]
    base = base_card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(0, 3, points) * base

    print(f"\n>>> building {name} in cov mode ...")
    cov_fit = build(with_lumi_mode(base_card, "cov"))
    cov_fit.fit_parameters()
    cov_nom  = capture_nominal(cov_fit)
    cov_syst = capture_syst_table(cov_fit)
    pois = list(cov_nom.keys())
    print(f">>> sweeping lumi in cov mode ...")
    cov_sweep = sweep_lumi_mode_agnostic(cov_fit, l_lumi, pois)
    cov_data = (cov_nom, cov_syst, cov_sweep)

    print(f"\n>>> building {name} in nuisance mode ...")
    nuis_fit = build(with_lumi_mode(base_card, "nuisance"))
    nuis_fit.fit_parameters()
    nuis_nom  = capture_nominal(nuis_fit)
    nuis_syst = capture_syst_table(nuis_fit)
    print(f">>> sweeping lumi in nuisance mode ...")
    nuis_sweep = sweep_lumi_mode_agnostic(nuis_fit, l_lumi, pois)
    nuis_data = (nuis_nom, nuis_syst, nuis_sweep)

    report_process(name, cov_fit, cov_data, nuis_fit, nuis_data)
    make_plot(name, cov_data, nuis_data, l_lumi, base)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", choices=("wbwb", "ww", "both"), default="both")
    ap.add_argument("--points", type=int, default=7,
                    help="grid points per direction in the lumi sweep")
    args = ap.parse_args()
    procs = ["wbwb", "ww"] if args.proc == "both" else [args.proc]
    for p in procs:
        run_one_process(p, points=args.points)


if __name__ == "__main__":
    main()
