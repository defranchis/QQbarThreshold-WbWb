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
    plot_fit_input_azzurri_overlay, plot_fit_input_ratios,
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
    # --- Theory-uncertainty ladder (standalone mode) -----------------------
    parser.add_argument("--theoryLadder", action="store_true",
                        help="run the incremental-piece theory-uncertainty Asimov "
                             "ladder (LO→+NLO→+NNLO→+δ_QCD, ×{LL,NLL} ISR) against "
                             "the production chain as truth, then exit. Fit uses "
                             "m_W/Γ_W + stat + correlated lumi only; see "
                             "framework/process/ww/theory_ladder.py.")
    parser.add_argument("--ladderISR", choices=["both", "LL", "NLL"], default="both",
                        help="which ISR leg(s) of the ladder to run (default both)")
    parser.add_argument("--ladderWorkers", type=int, default=48, metavar="N",
                        help="process-pool size for ladder template generation "
                             "(default 48 = the fcc-ironic core cap; the inner "
                             "per-√s eMELA loop is forced serial so workers map "
                             "1:1 to cores without oversubscription)")
    parser.add_argument("--ladderOut", default="plots/theory_ladder", metavar="STEM",
                        help="output stem for the ladder table (.txt + .csv)")
    parser.add_argument("--ladderKeep", action="store_true",
                        help="keep the temporary ladder template directory")
    parser.add_argument("--ladderNoSchemeVar", action="store_true",
                        help="skip the ISR scheme-variation block (α-renormalisation "
                             "ALPMZ/ALGMU/α(0) + ξ stability) that otherwise rides on "
                             "top of the perturbative ladder")
    parser.add_argument("--parallel", type=int, default=6, metavar="N",
                        help="run the requested scans in parallel with up to N worker "
                             "processes (default: 6; pass --parallel 1 to force sequential)")
    return parser.parse_args()


def _check_template_freshness(fit, generator):
    """Abort if the templates' stored ``template_fingerprint`` disagrees with
    what the live card+generator would produce — i.e. the card's physics
    parameters no longer correspond to the calculation the templates were
    generated from. Two failure modes are caught, both fail *closed*:

      * **mismatch** — a fingerprint key present in the template header but
        with a value different from the live card (e.g. ``m_t`` edited in the
        card without regenerating templates);
      * **missing** — a fingerprint key the live card expects but the template
        header lacks, i.e. the template predates that input. Left unchecked,
        such an input could differ silently, so it is treated as stale.

    Templates with NO fingerprint at all (empty header — pre-metadata runs)
    are skipped, matching the documented back-compat behaviour. The
    ``v not in (None, "")`` guard mirrors ``compose_header``, which omits
    empty values, so a legitimately-blank fingerprint field is not reported
    as missing."""
    metadata = fit.template_metadata()
    if not metadata:
        return
    fingerprint = generator.template_fingerprint()
    mismatched = [
        f"  {k}: template={metadata[k]!r}  card={v!r}"
        for k, v in fingerprint.items()
        if k in metadata and metadata[k] != v
    ]
    missing = [
        f"  {k}: (absent from template header)  card={v!r}"
        for k, v in fingerprint.items()
        if k not in metadata and v not in (None, "")
    ]
    if mismatched or missing:
        lines = []
        if mismatched:
            lines.append("Input(s) that changed since the templates were generated:")
            lines.extend(mismatched)
        if missing:
            lines.append("Input(s) the templates predate (not in their fingerprint):")
            lines.extend(missing)
        raise SystemExit(
            "\n[stale templates] The card's parameters no longer match the\n"
            "calculation the input templates were generated from:\n"
            + "\n".join(lines) + "\n"
            "→ regenerate templates: `python compute_xsec_ww.py`\n"
            "→ then re-run this script. Refusing to fit on outdated templates."
        )


def main():
    args = parse_args()

    if args.theoryLadder:
        from framework.process.ww.theory_ladder import run_theory_ladder
        run_theory_ladder(isr=args.ladderISR, workers=args.ladderWorkers,
                          out=args.ladderOut, keep=args.ladderKeep,
                          scheme_var=not args.ladderNoSchemeVar)
        return

    generator = WWGenerator.from_card(card)
    fit = WWFit(
        card,
        generator,
        input_dir=args.inputDir,
        asimov=not args.pseudo,
        read_scale_vars=args.scaleVars,
        mass_scheme=getattr(card, "MASS_SCHEME", "OS"),
        debug=args.debug,
    )
    _check_template_freshness(fit, generator)

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
        plot_fit_input_ratios(fit)
        plot_fit_input_azzurri_overlay(fit)

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
