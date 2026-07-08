#!/usr/bin/env python
"""Wall-time comparison: LUMI_MODE='cov' vs 'nuisance' for both processes.

Timed operations (typical wall-clock units users care about):
  - Nominal fit (single migrad + fit_results).
  - Full syst-table flow (one migrad per row, the heaviest production path).
  - Per-point lumi sweep (7 points × 2 directions × local migrad).

Each operation is repeated ``--repeats`` times and we report the mean
wall time plus the min as a noise floor. Cov is the reference; nuis is
expressed as +/- % vs cov.

The nuisance mode adds N+1 fit parameters (N=#scan bins) for the lumi
nuisance, so a slight slowdown is expected.
"""

import argparse
import contextlib
import copy
import io
import os
import sys
import time
import types

import matplotlib
matplotlib.use("Agg")  # noqa: E402
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


def card_with(card, mode):
    new = types.ModuleType(f"{card.__name__}_{mode}")
    for k in dir(card):
        if k.startswith("_"):
            continue
        setattr(new, k, getattr(card, k))
    for k in ("PRIORS", "SYSTEMATICS", "INPUT_VAR", "SCENARIO", "SCENARIO_TWOPOINTS"):
        if hasattr(new, k):
            setattr(new, k, copy.deepcopy(getattr(new, k)))
    new.LUMI_MODE = mode
    return new


def build_fit(proc, card):
    if proc == "ww":
        from framework.process.ww.fit import WWFit
        from framework.process.ww.generator import WWGenerator
        gen = WWGenerator.from_card(card)
        fit = WWFit(card, gen, asimov=True)
    else:
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
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.fit_parameters()
    with contextlib.redirect_stdout(io.StringIO()):
        fit.fit_results(printout=True)
    return fit


def time_nominal(proc, card, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit = build_fit(proc, card)
        # build_fit already runs fit_parameters() + fit_results(); we want the
        # fit cost itself, not the construction. Time a clean re-fit:
        t1 = time.perf_counter()
        fit.fit_parameters()
        with contextlib.redirect_stdout(io.StringIO()):
            fit.fit_results(printout=True)
        t2 = time.perf_counter()
        # (construction time, re-fit time)
        times.append((t1 - t0, t2 - t1))
    return times


def time_systable(fit, repeats):
    times = []
    for _ in range(repeats):
        syst = {poi: {} for poi in fit.tracked_pois()}
        t0 = time.perf_counter()
        try:
            for s in systematic_list(fit):
                estimate_systematic(fit, s, syst)
        finally:
            fit.reinitialise_to_nominal()
            fit.fit_parameters()
        times.append(time.perf_counter() - t0)
    return times


def time_sweep(fit, repeats, n_points=7):
    base = fit.card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(0, 3, n_points) * base
    pois = fit.tracked_pois()
    start = np.zeros(len(fit.param_names))
    is_nuis = "lumi" in fit._active_binned_nuisances
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if is_nuis:
            saved = dict(fit._nuisance_priors["lumi"])
            try:
                for direction in ("uncorr", "corr"):
                    for lumi in l_lumi:
                        u = lumi if direction == "uncorr" else 1e-10
                        c = lumi if direction == "corr"   else 1e-10
                        fit.set_binned_nuisance_priors("lumi", uncorr=u, corr=c)
                        m = run_local_migrad(fit, start)
                        fit.results_from_minuit(m)
            finally:
                fit._nuisance_priors["lumi"] = saved
        else:
            su, sc = fit.lumi_uncorr, fit.lumi_corr
            try:
                for direction in ("uncorr", "corr"):
                    for lumi in l_lumi:
                        fit.lumi_uncorr = lumi if direction == "uncorr" else 0.0
                        fit.lumi_corr   = lumi if direction == "corr"   else 0.0
                        fit._build_cov()
                        m = run_local_migrad(fit, start)
                        fit.results_from_minuit(m)
            finally:
                fit.lumi_uncorr, fit.lumi_corr = su, sc
                fit._build_cov()
        times.append(time.perf_counter() - t0)
    return times


def fmt(times):
    arr = np.array(times)
    return f"mean={arr.mean():7.3f}s  min={arr.min():7.3f}s  (n={len(arr)})"


def pct_delta(cov_times, nuis_times):
    a, b = np.mean(cov_times), np.mean(nuis_times)
    return (b - a) / a * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=("wbwb", "ww", "both"), default="both")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    procs = ["wbwb", "ww"] if args.proc == "both" else [args.proc]
    print(f"# repeats per measurement: {args.repeats}")

    for proc in procs:
        base = wbwb_default if proc == "wbwb" else ww_default
        print(f"\n================  {proc.upper()}  ================")

        cov_nom = time_nominal(proc, card_with(base, "cov"), args.repeats)
        nui_nom = time_nominal(proc, card_with(base, "nuisance"), args.repeats)
        cov_build = [t[0] for t in cov_nom]
        cov_refit = [t[1] for t in cov_nom]
        nui_build = [t[0] for t in nui_nom]
        nui_refit = [t[1] for t in nui_nom]
        print(f"  [build_fit + first migrad]  cov: {fmt(cov_build)}")
        print(f"  [build_fit + first migrad]  nui: {fmt(nui_build)}    Δ={pct_delta(cov_build, nui_build):+.1f}%")
        print(f"  [re-fit only]               cov: {fmt(cov_refit)}")
        print(f"  [re-fit only]               nui: {fmt(nui_refit)}    Δ={pct_delta(cov_refit, nui_refit):+.1f}%")

        fit_cov  = build_fit(proc, card_with(base, "cov"))
        fit_nui  = build_fit(proc, card_with(base, "nuisance"))

        cov_st = time_systable(fit_cov,  args.repeats)
        nui_st = time_systable(fit_nui, args.repeats)
        print(f"  [syst table]                cov: {fmt(cov_st)}")
        print(f"  [syst table]                nui: {fmt(nui_st)}    Δ={pct_delta(cov_st, nui_st):+.1f}%")

        cov_sw = time_sweep(fit_cov, args.repeats)
        nui_sw = time_sweep(fit_nui, args.repeats)
        print(f"  [scan_lumi sweep 14 pts]    cov: {fmt(cov_sw)}")
        print(f"  [scan_lumi sweep 14 pts]    nui: {fmt(nui_sw)}    Δ={pct_delta(cov_sw, nui_sw):+.1f}%")

        print(f"  param_names lengths:  cov={len(fit_cov.param_names)}  nui={len(fit_nui.param_names)}")


if __name__ == "__main__":
    main()
