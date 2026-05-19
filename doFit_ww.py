#!/usr/bin/env python
"""Entry point for the WW threshold fit — PLACEHOLDER.

Mirrors :mod:`doFit_wbwb` but trims Yukawa-related options. Cannot do
anything useful until a real WW cross-section generator is plugged into
:class:`process.ww.generator.WWGenerator` and a corresponding set of
template files is produced (see ``INPUT_DIRS`` in ``cards/ww_default.py``).
"""

import argparse
import json
import os
from datetime import datetime, timezone

from cards import ww_default as card
from framework.common import scans
from framework.common.plots import (
    plot_fit_scenario, plot_parameter_variations, set_active_chain_label,
)
from framework.common.systematics import print_syst_table
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator


def dump_fit_metadata(fit, args):
    """Write a ``fit_metadata.json`` next to the fit plots describing the
    chain that produced the input templates, the fit-time flags, and the
    timestamp. ``fit.template_metadata()`` returns the ``# key: value``
    preamble read from one of the input CSVs (chain summary, ISR scheme,
    etc.); we merge it with the fit-side run knobs.
    """
    payload = {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir":        fit.input_dir,
        "plot_dir":         fit.plot_dir,
        "template":         fit.template_metadata(),
        "fit": {
            "asimov":          fit.asimov,
            "scan_min":        card.SCENARIO["scan_min"],
            "scan_max":        card.SCENARIO["scan_max"],
            "scan_step":       card.SCENARIO["scan_step"],
            "total_lumi":      fit.scenario_dict["total_lumi"],
            "scale_vars":      args.scaleVars,
            "lastecm":         args.lastecm,
            "BECnuisances":    args.BECnuisances,
            "BESnuisances":    args.BESnuisances,
        },
    }
    os.makedirs(fit.plot_dir, exist_ok=True)
    out = os.path.join(fit.plot_dir, "fit_metadata.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[metadata] wrote {out}")
    chain = payload["template"].get("chain")
    if chain:
        print(f"[metadata] template chain: {chain}")


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
    parser.add_argument("--parallel", type=int, default=6, metavar="N",
                        help="run the requested scans in parallel with up to N worker "
                             "processes (default: 6; pass --parallel 1 to force sequential)")
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

    dump_fit_metadata(fit, args)
    # Make the chain summary from the actual templates show up at the
    # bottom of every saved plot. ``None`` (= no header) disables it.
    set_active_chain_label(fit.template_metadata().get("chain"))

    if not args.noPlots:
        plot_parameter_variations(fit)

    fit.fit_parameters()
    fit.fit_results()

    if not args.noPlots:
        plot_fit_scenario(fit)

    scan_jobs = []
    if args.LSscan:
        scan_jobs.append(lambda: scans.scan_beam_resolution(fit))
    if args.BECscans:
        scan_jobs.append(lambda: scans.scan_bec(fit))
    if args.BESscans:
        scan_jobs.append(lambda: scans.scan_bes(fit))
    if args.lumiscans:
        scan_jobs.append(lambda: scans.scan_lumi(fit))
    if args.alphaSscan:
        scan_jobs.append(lambda: scans.scan_alphas(fit))
    if args.chi2scans:
        scan_jobs.append(lambda: scans.scan_chi2(fit))

    if args.parallel > 1 and len(scan_jobs) > 1:
        scans.run_parallel(scan_jobs, max_workers=args.parallel)
    else:
        for job in scan_jobs:
            job()

    if args.systTable:
        print_syst_table(fit)

    if not args.noPlots:
        from framework.common.eos_publish import publish
        publish(card.PLOT_DIR, os.environ.get("WW_FIT_PUBSUB", "ww/plots"))


if __name__ == "__main__":
    main()
