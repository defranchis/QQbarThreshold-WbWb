#!/usr/bin/env python
"""Entry point for the WbWb threshold fit.

Thin wrapper: load the steering card, build the generator and fit object,
optionally add nuisances, run the fit, print results, run any requested
scans / produce plots / build the systematic-uncertainty table.
"""

import argparse

from cards import wbwb_default as card
from common import scans
from common.parallel import run_parallel
from common.plots import plot_fit_scenario, plot_parameter_variations
from common.systematics import print_syst_table
from process.wbwb import scans as wbwb_scans
from process.wbwb.fit import WbWbFit
from process.wbwb.generator import WbWbGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="WbWb threshold fit")
    # Fit configuration
    parser.add_argument("--pseudo", action="store_true", help="use pseudo-experiment toy data (default: Asimov)")
    parser.add_argument("--legacyPseudoRng", action="store_true",
                        help="(--pseudo only) restore the pre-fix RNG behaviour where every "
                             "create_scenario call re-seeds the global MT19937 to 42 — same noise every call; "
                             "kept for byte-reproducing prior pseudo runs only")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--SMwidth", action="store_true", help="constrain width to the SM/QCD prediction")
    parser.add_argument("--fitYukawa", action="store_true", help="float Yukawa (default: constrained)")
    parser.add_argument("--addSw2", action="store_true", help="add sw2 nuisance")
    parser.add_argument("--lastecm", action="store_true", help="include the above-threshold point for Yukawa")
    parser.add_argument("--sameNevts", action="store_true", help="distribute lumi to keep N_events per point constant")
    parser.add_argument("--scaleVars", action="store_true", help="read scale-variation templates")
    parser.add_argument("--BECnuisances", action="store_true", help="add BEC nuisances")
    parser.add_argument("--BESnuisances", action="store_true", help="add BES nuisances")
    parser.add_argument("--oneS", action="store_true", help="use the 1S mass scheme")
    parser.add_argument("--twopoints", action="store_true", help="use the coarse two-point scenario")
    parser.add_argument("--inputDir", default=None, help="override INPUT_DIRS.nominal")
    # Scans & outputs
    parser.add_argument("--LSscan", action="store_true", help="beam-energy resolution scan")
    parser.add_argument("--BECscans", action="store_true", help="BEC nuisance prior scan")
    parser.add_argument("--BESscans", action="store_true", help="BES nuisance prior scan")
    parser.add_argument("--lumiscans", action="store_true", help="luminosity uncertainty scan")
    parser.add_argument("--alphaSscan", action="store_true", help="alpha_s prior scan (also runs the Yukawa-constraint scan if Yukawa is constrained)")
    parser.add_argument("--yukawaThScan", action="store_true", help="Yukawa theory-shift scan (needs --lastecm)")
    parser.add_argument("--widthscan", action="store_true", help="width-scan (only with --SMwidth)")
    parser.add_argument("--chi2scans", action="store_true", help="1D + 2D chi2 profile scans")
    parser.add_argument("--truevaluescan", action="store_true", help="iterate over the pseudo-data templates")
    parser.add_argument("--scaleVarsScan", action="store_true", help="renormalisation-scale variation scan")
    parser.add_argument("--shiftScan", action="store_true",
                        help="scan the ecm shift of the whole scan grid (needs templates outside the original ecm range)")
    parser.add_argument("--systTable", action="store_true", help="produce the systematic-uncertainty table")
    parser.add_argument("--noPlots", action="store_true", help="skip the diagnostic plots")
    parser.add_argument("--parallel", type=int, default=6, metavar="N",
                        help="run the requested scans in parallel with up to N worker processes "
                             "(default: 6; pass --parallel 1 to force sequential; systTable always "
                             "runs sequentially after scans complete)")
    return parser.parse_args()


def _check_args(args):
    if (args.BECscans or args.BESscans or args.BECnuisances or args.BESnuisances) and args.scaleVars:
        raise ValueError("BEC/BES scan is incompatible with scale variations")
    if args.alphaSscan and args.lastecm:
        raise ValueError("alpha_s scan is incompatible with last-ecm point")
    if args.yukawaThScan and not args.lastecm:
        raise ValueError("Yukawa theory-shift scan requires --lastecm")
    if args.lastecm and not args.fitYukawa:
        raise ValueError("--lastecm is incompatible with the Yukawa constraint; pass --fitYukawa to float Yukawa")


def main():
    args = parse_args()
    _check_args(args)

    generator = WbWbGenerator(order=card.ORDER, isr=True)
    fit = WbWbFit(
        card,
        generator,
        input_dir=args.inputDir,
        sm_width=args.SMwidth,
        asimov=not args.pseudo,
        constrain_yukawa=not args.fitYukawa,
        read_scale_vars=args.scaleVars,
        mass_scheme="1S" if args.oneS else card.MASS_SCHEME,
        shift_scan=args.shiftScan,
        legacy_pseudo_rng=args.legacyPseudoRng,
        debug=args.debug,
    )

    scenario = card.SCENARIO_TWOPOINTS if args.twopoints else card.SCENARIO
    fit.init_scenario(
        scan_min=scenario["scan_min"],
        scan_max=scenario["scan_max"],
        scan_step=scenario["scan_step"],
        total_lumi=scenario["total_lumi"],
        last_lumi=scenario["last_lumi"],
        add_last_ecm=args.lastecm,
        same_evts=args.sameNevts,
    )

    if args.BECnuisances or args.BECscans or args.systTable:
        fit.add_binned_nuisance("BEC")
    if args.BESnuisances or args.BESscans or args.systTable:
        fit.add_binned_nuisance("BES")
    if args.addSw2:
        fit.add_global_nuisance("sw2")

    if not args.noPlots:
        plot_parameter_variations(fit)

    fit.fit_parameters()
    fit.fit_results()

    if not args.noPlots:
        plot_fit_scenario(fit)

    # ---- scans ------------------------------------------------------------
    # Build the requested scans as zero-arg callables capturing fit by
    # closure. They neither mutate fit nor depend on each other (see
    # /tmp/audit_scans.py), so the order is irrelevant and they're
    # safe to run in parallel.
    scan_jobs = []
    if args.LSscan:
        scan_jobs.append(lambda: scans.scan_beam_resolution(fit))
    if args.scaleVarsScan:
        scan_jobs.append(lambda: scans.scan_scale_vars(fit))
        scan_jobs.append(lambda: wbwb_scans.scan_scale_vars_yukawa(fit))
    if args.BECscans:
        scan_jobs.append(lambda: scans.scan_bec(fit))
    if args.BESscans:
        scan_jobs.append(lambda: scans.scan_bes(fit))
    if args.lumiscans:
        scan_jobs.append(lambda: scans.scan_lumi(fit))
        scan_jobs.append(lambda: wbwb_scans.scan_lumi_yukawa_ratio(fit))
    if args.alphaSscan:
        scan_jobs.append(lambda: scans.scan_alphas(fit))
        if not args.fitYukawa:
            scan_jobs.append(lambda: wbwb_scans.scan_yukawa_constraint(fit))
    if args.yukawaThScan:
        scan_jobs.append(lambda: wbwb_scans.scan_yukawa_theory(fit))
    if args.widthscan:
        scan_jobs.append(lambda: wbwb_scans.scan_width(fit))
    if args.truevaluescan:
        scan_jobs.append(lambda: scans.scan_true_value(fit))
    if args.chi2scans:
        scan_jobs.append(lambda: scans.scan_chi2(fit))
    if args.shiftScan:
        scan_jobs.append(lambda: scans.scan_shift(fit))

    if args.parallel > 1 and len(scan_jobs) > 1:
        run_parallel(scan_jobs, max_workers=args.parallel)
    else:
        for job in scan_jobs:
            job()

    if args.systTable:
        print_syst_table(fit)


if __name__ == "__main__":
    main()
