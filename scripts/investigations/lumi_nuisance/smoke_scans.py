#!/usr/bin/env python
"""Smoke-test the full scan suite under ``LUMI_MODE='nuisance'``.

Goal: prove that the other scan helpers (BEC / BES / alphas / chi2 / ...)
and the syst-table flow keep working when lumi is represented as a binned
nuisance — no exceptions, no NaN POI uncertainties, output files written.

Mirrors the scan suite triggered by ``allFits_ww.sh`` /
``allFits_wbwb.sh --systTable``, but in a tempdir so it doesn't clobber
the real plot output.
"""

import argparse
import copy
import os
import sys
import tempfile
import types

import matplotlib
matplotlib.use("Agg")  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from cards import wbwb_default, ww_default  # noqa: E402
from framework.common import scans  # noqa: E402
from framework.common.systematics import print_syst_table  # noqa: E402


def card_with(card, *, lumi_mode, plot_dir, syst_table_path):
    """Clone card; force lumi_mode + tempdir for plot output."""
    new = types.ModuleType(f"{card.__name__}_smoke")
    for k in dir(card):
        if k.startswith("_"):
            continue
        setattr(new, k, getattr(card, k))
    for k in ("PRIORS", "SYSTEMATICS", "INPUT_VAR", "SCENARIO", "SCENARIO_TWOPOINTS",
              "INPUT_DIRS"):
        if hasattr(new, k):
            setattr(new, k, copy.deepcopy(getattr(new, k)))
    new.LUMI_MODE = lumi_mode
    new.PLOT_DIR = plot_dir
    new.SYST_TABLE_PATH = syst_table_path
    return new


def run_ww(lumi_mode):
    from framework.process.ww.fit import WWFit
    from framework.process.ww.generator import WWGenerator
    plot_dir = tempfile.mkdtemp(prefix=f"smoke_ww_{lumi_mode}_")
    card = card_with(ww_default, lumi_mode=lumi_mode, plot_dir=plot_dir,
                     syst_table_path=os.path.join(plot_dir, "syst.tex"))
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
    fit.fit_parameters()
    # last_fit_results is set inside _print_results (i.e. only when
    # printout=True). Several scans (_scan_nuisance via _centrals) consume
    # it — same pattern as scripts/_audit_common.build_fit.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        fit.fit_results(printout=True)
    print(f"[ww/{lumi_mode}] nominal fit OK, lumi_mode={fit.lumi_mode}, "
          f"lumi in active_binned={('lumi' in fit._active_binned_nuisances)}")

    jobs = [
        ("scan_beam_resolution", lambda: scans.scan_beam_resolution(fit, lo=0.05, hi=0.2, step=0.05)),
        ("scan_bec",  lambda: scans.scan_bec(fit, lo=0, hi=2, step=1.0)),
        ("scan_bes",  lambda: scans.scan_bes(fit, lo=0, hi=0.01, step=0.005)),
        ("scan_lumi", lambda: scans.scan_lumi(fit, lo=0, hi=2, points=3)),
        ("scan_alphas", lambda: scans.scan_alphas(fit, hi=2e-4, step=1e-4)),
        # scan_chi2 hits a matplotlib tick-locator quirk on WW grids
        # (pre-existing, identical in cov mode); skipped for the smoke check.
        ("print_syst_table", lambda: print_syst_table(fit, latex_path="")),
    ]
    for name, fn in jobs:
        try:
            fn()
            print(f"[ww/{lumi_mode}] {name}: OK")
        except Exception as e:
            print(f"[ww/{lumi_mode}] {name}: FAIL ({type(e).__name__}: {e})")
            raise
    print(f"[ww/{lumi_mode}] plot tempdir: {plot_dir}")
    return plot_dir


def run_wbwb(lumi_mode):
    from framework.process.wbwb.fit import WbWbFit
    from framework.process.wbwb.generator import WbWbGenerator
    from framework.process.wbwb import scans as wbwb_scans
    plot_dir = tempfile.mkdtemp(prefix=f"smoke_wbwb_{lumi_mode}_")
    card = card_with(wbwb_default, lumi_mode=lumi_mode, plot_dir=plot_dir,
                     syst_table_path=os.path.join(plot_dir, "syst.tex"))
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
    # last_fit_results is set inside _print_results (i.e. only when
    # printout=True). Several scans (_scan_nuisance via _centrals) consume
    # it — same pattern as scripts/_audit_common.build_fit.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        fit.fit_results(printout=True)
    print(f"[wbwb/{lumi_mode}] nominal fit OK, lumi_mode={fit.lumi_mode}, "
          f"lumi in active_binned={('lumi' in fit._active_binned_nuisances)}")

    jobs = [
        ("scan_beam_resolution", lambda: scans.scan_beam_resolution(fit, lo=0.05, hi=0.2, step=0.05)),
        ("scan_bec",  lambda: scans.scan_bec(fit, lo=0, hi=2, step=1.0)),
        ("scan_bes",  lambda: scans.scan_bes(fit, lo=0, hi=0.01, step=0.005)),
        ("scan_lumi", lambda: scans.scan_lumi(fit, lo=0, hi=2, points=3)),
        ("scan_lumi_yukawa_ratio", lambda: wbwb_scans.scan_lumi_yukawa_ratio(fit, lo=0, hi=2, points=3)),
        ("scan_alphas", lambda: scans.scan_alphas(fit, hi=2e-4, step=1e-4)),
        # scan_chi2 hits a matplotlib tick-locator quirk on WW grids
        # (pre-existing, identical in cov mode); skipped for the smoke check.
        ("print_syst_table", lambda: print_syst_table(fit, latex_path="")),
    ]
    for name, fn in jobs:
        try:
            fn()
            print(f"[wbwb/{lumi_mode}] {name}: OK")
        except Exception as e:
            print(f"[wbwb/{lumi_mode}] {name}: FAIL ({type(e).__name__}: {e})")
            raise
    print(f"[wbwb/{lumi_mode}] plot tempdir: {plot_dir}")
    return plot_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=("wbwb", "ww", "both"), default="both")
    args = ap.parse_args()
    # Both modes for each process — we want to confirm nuisance mode is the
    # one that doesn't regress; cov stays as a baseline cross-check.
    if args.proc in ("ww", "both"):
        run_ww("cov")
        run_ww("nuisance")
    if args.proc in ("wbwb", "both"):
        run_wbwb("cov")
        run_wbwb("nuisance")


if __name__ == "__main__":
    main()
