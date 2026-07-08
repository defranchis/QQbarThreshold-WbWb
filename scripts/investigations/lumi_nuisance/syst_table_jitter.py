#!/usr/bin/env python
"""Quantify intrinsic syst-table-flow jitter, within a single mode.

For each (process, lumi_mode) we run the syst table twice from the same
starting fit and report row-by-row absolute deltas. If the BES_corr /
BEC_corr cov-vs-nuis "outliers" are dominated by quadrature-subtraction
noise at small absolute scales (rather than a real cov-vs-nuis physics
difference), the same-mode jitter should be of the same order as the
cross-mode delta.

A "real" cov-vs-nuis difference would be jitter << delta; if instead
jitter ~ delta on the same rows, the deltas are pure numerical noise.
"""

import argparse
import copy
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from cards import wbwb_default, ww_default  # noqa: E402
from framework.common.systematics import estimate_systematic, systematic_list  # noqa: E402


def card_with_mode(card, mode):
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
    return fit


def capture_syst(fit):
    syst = {poi: {} for poi in fit.tracked_pois()}
    try:
        for s in systematic_list(fit):
            estimate_systematic(fit, s, syst)
    finally:
        fit.reinitialise_to_nominal()
        fit.fit_parameters()
    for poi in syst:
        syst[poi].pop("total", None)
    return syst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=("wbwb", "ww", "both"), default="both")
    args = ap.parse_args()
    procs = ["wbwb", "ww"] if args.proc == "both" else [args.proc]

    for proc in procs:
        base_card = wbwb_default if proc == "wbwb" else ww_default
        for mode in ("cov", "nuisance"):
            fit = build_fit(proc, card_with_mode(base_card, mode))
            s1 = capture_syst(fit)
            s2 = capture_syst(fit)
            print(f"\n=== {proc} / {mode}: same-mode jitter (run 1 vs run 2) ===")
            for poi in s1:
                scale = fit.card.POI_DISPLAY[poi]["scale"]
                print(f"  POI {poi} ({fit.card.POI_DISPLAY[poi]['unit']}):")
                rows = sorted(set(s1[poi]) & set(s2[poi]))
                for r in rows:
                    a = s1[poi][r]
                    b = s2[poi][r]
                    abs_delta = abs(a - b)
                    rel = abs_delta / max(abs(a), abs(b), 1e-30)
                    flag = "" if rel < 0.02 or abs_delta < 1e-3 else "  <-- non-trivial wobble"
                    print(f"    {r:<14} run1={a:>10.5f}  run2={b:>10.5f}  abs_d={abs_delta:>8.5f}  rel={rel:>7.2%}{flag}")


if __name__ == "__main__":
    main()
