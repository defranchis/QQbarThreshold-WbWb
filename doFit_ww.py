#!/usr/bin/env python
"""Entry point for the WW threshold fit — PLACEHOLDER.

Mirrors :mod:`doFit_wbwb` but trims Yukawa-related options. Cannot do
anything useful until a real WW cross-section generator is plugged into
:class:`process.ww.generator.WWGenerator` and a corresponding set of
template files is produced (see ``INPUT_DIRS`` in ``cards/ww_default.py``).
"""

import argparse

from cards import ww_default as card
from common import scans
from common.plots import plot_fit_scenario, plot_parameter_variations
from common.systematics import print_syst_table
from process.ww.fit import WWFit
from process.ww.generator import WWGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="WW threshold fit (placeholder)")
    parser.add_argument("--pseudo", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lastecm", action="store_true")
    parser.add_argument("--sameNevts", action="store_true")
    parser.add_argument("--scaleVars", action="store_true")
    parser.add_argument("--BECnuisances", action="store_true")
    parser.add_argument("--BESnuisances", action="store_true")
    parser.add_argument("--inputDir", default=None)
    # Scans
    parser.add_argument("--LSscan", action="store_true")
    parser.add_argument("--BECscans", action="store_true")
    parser.add_argument("--BESscans", action="store_true")
    parser.add_argument("--lumiscans", action="store_true")
    parser.add_argument("--alphaSscan", action="store_true")
    parser.add_argument("--chi2scans", action="store_true")
    parser.add_argument("--systTable", action="store_true")
    parser.add_argument("--noPlots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    generator = WWGenerator.from_card(card)
    fit = WWFit(
        card,
        generator,
        input_dir=args.inputDir,
        asimov=not args.pseudo,
        read_scale_vars=args.scaleVars,
        mass_scheme=card.MASS_SCHEME,
        debug=args.debug,
    )

    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
        add_last_ecm=args.lastecm,
        same_evts=args.sameNevts,
    )

    if args.BECnuisances or args.BECscans or args.systTable:
        fit.add_binned_nuisance("BEC")
    if args.BESnuisances or args.BESscans or args.systTable:
        fit.add_binned_nuisance("BES")

    if not args.noPlots:
        plot_parameter_variations(fit)

    fit.fit_parameters()
    fit.fit_results()

    if not args.noPlots:
        plot_fit_scenario(fit)

    if args.LSscan:
        scans.scan_beam_resolution(fit)
    if args.BECscans:
        scans.scan_bec(fit)
    if args.BESscans:
        scans.scan_bes(fit)
    if args.lumiscans:
        scans.scan_lumi(fit)
    if args.alphaSscan:
        scans.scan_alphas(fit)
    if args.chi2scans:
        scans.scan_chi2(fit)
    if args.systTable:
        print_syst_table(fit)

    if not args.noPlots:
        import os
        from common.eos_publish import publish
        publish(card.PLOT_DIR, os.environ.get("WW_FIT_PUBSUB", "ww"))


if __name__ == "__main__":
    main()
