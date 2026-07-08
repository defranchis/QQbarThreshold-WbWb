#!/usr/bin/env python
"""Detailed cov-vs-nuisance comparison table.

Reuses the validate.py helpers (with_lumi_mode, build_wbwb / build_ww,
capture_nominal, capture_syst_table, sweep_lumi_mode_agnostic) and
emits one consolidated table per process, with explicit absolute-diff
columns and a noise-floor-aware verdict. The validate.py output flags
sub-0.05 MeV rows as FAIL when their relative diff exceeds 2% — those
are Minuit-restart noise, not real mode disagreement.

Verdict rules:
  - OK   if rel < 2 %
  - OK*  if rel >= 2 % but |Δ| < ``abs_tol`` MeV   (noise floor)
  - FAIL otherwise

Run from the WW_threshold/ directory:

    python3 -m scripts.investigations.lumi_nuisance.detailed_compare
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from scripts.investigations.lumi_nuisance.validate import (  # noqa: E402
    PROCESSES,
    capture_nominal,
    capture_syst_table,
    sweep_lumi_mode_agnostic,
    with_lumi_mode,
)


def _rel(a, b):
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def verdict(a, b, rel_tol=0.02, abs_tol=0.05):
    rel = _rel(a, b)
    if rel < rel_tol:
        return "OK"
    if abs(a - b) < abs_tol:
        return "OK*"
    return "FAIL"


def _scale(fit, poi):
    return fit.card.POI_DISPLAY[poi]["scale"]


def _unit(fit, poi):
    return fit.card.POI_DISPLAY[poi]["unit"]


def _hr(width):
    return "-" * width


def _row(cells, widths):
    return "  ".join(f"{c:>{w}}" if i else f"{c:<{w}}"
                     for i, (c, w) in enumerate(zip(cells, widths)))


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def print_nominal_table(name, cov_fit, cov_nom, nuis_nom):
    pois = list(cov_nom.keys())
    print(f"\n## {name.upper()} — NOMINAL totals (per-POI hesse uncertainty)\n")
    cols = ("POI", "unit", "cov", "nuis", "|Δ|", "rel %", "verdict")
    widths = (8, 6, 10, 10, 10, 9, 7)
    print(_row(cols, widths))
    print(_hr(sum(widths) + 2 * (len(widths) - 1)))
    for poi in pois:
        a, b = cov_nom[poi], nuis_nom[poi]
        print(_row(
            (poi, _unit(cov_fit, poi), fmt(a), fmt(b), fmt(abs(a - b)),
             f"{_rel(a, b) * 100:.2f}", verdict(a, b)),
            widths))


def print_syst_table(name, cov_fit, cov_syst, nuis_syst):
    pois = list(cov_syst.keys())
    print(f"\n## {name.upper()} — SYST-TABLE row-by-row (display units)\n")
    cols = ("POI", "row", "cov", "nuis", "|Δ|", "rel %", "verdict")
    widths = (8, 14, 10, 10, 10, 9, 7)
    print(_row(cols, widths))
    print(_hr(sum(widths) + 2 * (len(widths) - 1)))
    for poi in pois:
        rows = sorted(set(cov_syst[poi].keys()) & set(nuis_syst[poi].keys()))
        rows = (sorted(r for r in rows if r.startswith("lumi"))
                + sorted(r for r in rows if not r.startswith("lumi")))
        for r in rows:
            a = cov_syst[poi].get(r, float("nan"))
            b = nuis_syst[poi].get(r, float("nan"))
            print(_row(
                (poi, r, fmt(a), fmt(b), fmt(abs(a - b)),
                 f"{_rel(a, b) * 100:.2f}", verdict(a, b)),
                widths))


def print_sweep_table(name, cov_fit, cov_sweep, nuis_sweep, l_lumi, base):
    pois = list(cov_sweep["uncorr"].keys())
    print(f"\n## {name.upper()} — scan_lumi sweep (×baseline, display units)\n")
    cols = ("POI", "dir", "× base", "lumi %", "cov", "nuis", "|Δ|", "rel %", "verdict")
    widths = (8, 7, 7, 7, 10, 10, 10, 9, 7)
    print(_row(cols, widths))
    print(_hr(sum(widths) + 2 * (len(widths) - 1)))
    for direction in ("uncorr", "corr"):
        for poi in pois:
            scale = _scale(cov_fit, poi)
            cov_pts = np.array(cov_sweep[direction][poi]) * scale
            nuis_pts = np.array(nuis_sweep[direction][poi]) * scale
            multipliers = l_lumi / base if base > 0 else l_lumi
            for i, (mult, lumi, a, b) in enumerate(zip(multipliers, l_lumi, cov_pts, nuis_pts)):
                print(_row(
                    (poi, direction, f"{mult:.2f}", f"{lumi * 100:.3f}",
                     fmt(a), fmt(b), fmt(abs(a - b)),
                     f"{_rel(a, b) * 100:.2f}", verdict(a, b)),
                    widths))


def run_one_process(name, points=7):
    base_card, build = PROCESSES[name]
    base = base_card.PRIORS["lumi"]["uncorr"]
    l_lumi = np.linspace(0, 3, points) * base

    print(f"\n{'=' * 90}")
    print(f" {name.upper()}: detailed cov-vs-nuisance comparison "
          f"(baseline lumi prior = {base * 100:.2f}% uncorr)")
    print('=' * 90)

    cov_fit = build(with_lumi_mode(base_card, "cov"))
    cov_fit.fit_parameters()
    cov_nom = capture_nominal(cov_fit)
    cov_syst = capture_syst_table(cov_fit)
    pois = list(cov_nom.keys())
    cov_sweep = sweep_lumi_mode_agnostic(cov_fit, l_lumi, pois)

    nuis_fit = build(with_lumi_mode(base_card, "nuisance"))
    nuis_fit.fit_parameters()
    nuis_nom = capture_nominal(nuis_fit)
    nuis_syst = capture_syst_table(nuis_fit)
    nuis_sweep = sweep_lumi_mode_agnostic(nuis_fit, l_lumi, pois)

    print_nominal_table(name, cov_fit, cov_nom, nuis_nom)
    print_syst_table(name, cov_fit, cov_syst, nuis_syst)
    print_sweep_table(name, cov_fit, cov_sweep, nuis_sweep, l_lumi, base)

    n_total = 0
    n_ok = n_okstar = n_fail = 0
    for poi in pois:
        for r in set(cov_syst[poi]) & set(nuis_syst[poi]):
            n_total += 1
            v = verdict(cov_syst[poi][r], nuis_syst[poi][r])
            n_ok += v == "OK"
            n_okstar += v == "OK*"
            n_fail += v == "FAIL"
        for direction in ("uncorr", "corr"):
            scale = _scale(cov_fit, poi)
            for a, b in zip(np.array(cov_sweep[direction][poi]) * scale,
                            np.array(nuis_sweep[direction][poi]) * scale):
                n_total += 1
                v = verdict(a, b)
                n_ok += v == "OK"
                n_okstar += v == "OK*"
                n_fail += v == "FAIL"
    print(f"\n## {name.upper()} — summary across {n_total} comparisons: "
          f"OK={n_ok}  OK*={n_okstar} (noise floor)  FAIL={n_fail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", choices=("wbwb", "ww", "both"), default="both")
    ap.add_argument("--points", type=int, default=7)
    args = ap.parse_args()
    procs = ["wbwb", "ww"] if args.proc == "both" else [args.proc]
    for p in procs:
        run_one_process(p, points=args.points)


if __name__ == "__main__":
    main()
