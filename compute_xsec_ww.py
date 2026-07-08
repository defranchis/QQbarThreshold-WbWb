"""Batch driver for the WW threshold template generation.

Analogue of ``compute_xsec_wbwb.py`` (WbWb / QQbar_threshold), but
calling :class:`process.ww.generator.WWGenerator` instead of the C++
xsec_calc. Generates:

  * nominal templates in ``card.INPUT_DIRS["nominal"]`` for every
    parameter-variation tag (``nominal``, ``pseudodata``, ``<param>_var``).
  * BEC-variation templates in ``card.INPUT_DIRS["BEC"]/scan_{p,m}<var>/``
    with the ECM grid uniformly shifted by ±``INPUT_VAR["BEC"]`` MeV.

A template is (re)generated only if it is **missing** or its stored
fingerprint header no longer matches the live card (``ensure_scan``);
unchanged templates are reused, so re-running after an unrelated card edit
is cheap. Pass ``--force`` to regenerate unconditionally.

With the NLL-eMELA default a single fine-grid template is ~10 min on one
core (the LL+exp analytic chain is ~ms), so the reuse check matters: never
rebuild an expensive set you already have.
"""

from __future__ import annotations

import argparse
import os
import time

from cards import ww_default as card
from framework.common.fit_core import bec_var_dir
from framework.common.parameters import Parameters
from framework.process.ww.xsec_calculator.eft_xsec import BFSCorrections
from framework.process.ww.generator import WWGenerator


def _generate_set(generator: WWGenerator, params: Parameters, *,
                  mass_scale: float, width_scale: float, mass_scheme: str,
                  outdir: str, ecm_shift_MeV: float = 0.0,
                  tags: list[str] | None = None, force: bool = False,
                  verbose: bool = True) -> None:
    for tag in (tags if tags is not None else params.tags):
        vals = params.values(tag)
        path, regen = generator.ensure_scan(
            vals,
            mass_scale=mass_scale, width_scale=width_scale,
            mass_scheme=mass_scheme, outdir=outdir,
            ecm_shift_MeV=ecm_shift_MeV, force=force,
        )
        if verbose:
            print(f"  {tag:14s} [{'gen  ' if regen else 'reuse'}] → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=card.INPUT_DIRS["nominal"],
                    help=f"nominal output directory (default: {card.INPUT_DIRS['nominal']})")
    ap.add_argument("--bec-outdir", default=card.INPUT_DIRS["BEC"],
                    help=f"BEC-variation output directory (default: {card.INPUT_DIRS['BEC']})")
    ap.add_argument("--bec-vars-MeV", nargs="*", type=float,
                    default=[card.INPUT_VAR["BEC"]],
                    help="absolute BEC shifts in MeV (each generates scan_p{v} and scan_m{v}); "
                         "default = card's INPUT_VAR['BEC'] so the generator follows the same "
                         "step the fit consumes (single source of truth); "
                         "set to empty list to skip")
    ap.add_argument("--no-bec", action="store_true", help="skip BEC-variation templates")
    ap.add_argument("--force", action="store_true",
                    help="regenerate every template even if an up-to-date one "
                         "(matching fingerprint header) is already on disk; "
                         "default reuses fresh templates and only (re)builds "
                         "missing or stale ones")
    ap.add_argument("--only-tag", default=None,
                    help="restrict to a single PARAMETERS tag (e.g. 'nominal', 'mass_var'); "
                         "use with --only-bec-shift for per-job HTCondor fan-out")
    ap.add_argument("--only-bec-shift", type=float, default=None,
                    help="emit ONLY the set at this BEC shift in MeV. 0 → the nominal "
                         "set (no BEC subdir); ±v → the matching BEC/scan_{p,m}|v| subdir. "
                         "Overrides --bec-vars-MeV.")
    ap.add_argument("--diagnostic-bfs-coulomb-nlo",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="DIAGNOSTIC ONLY — toggle the standalone BFS NLO "
                         "Coulomb α² subleading piece (eq. 62 of arXiv:0707.0773) "
                         "via BFSCorrections. With include_NLO_hard_decay=True "
                         "(the production default) this double-counts; use only "
                         "when reproducing isolated BFS paper plots. CLI value "
                         "overrides card's NLO_CONFIG['diagnostic_bfs_coulomb_nlo']. "
                         "NOTE: this flag is NOT encoded in the template filename or "
                         "fingerprint header, so a diagnostic run writes to — and the "
                         "freshness check would later reuse — the SAME path as a "
                         "production template. Always pair it with --force or a "
                         "dedicated --outdir to avoid aliasing/clobbering production "
                         "templates.")
    args = ap.parse_args()

    params = Parameters(card.PARAMETERS, scale_vars=[],
                        cross_terms=getattr(card, "CROSS_TERMS", ()))
    tags = None
    if args.only_tag is not None:
        if args.only_tag not in params.tags:
            raise SystemExit(f"--only-tag {args.only_tag!r} not in {params.tags}")
        tags = [args.only_tag]
    # CLI override of the card's diagnostic_bfs_coulomb_nlo flag.
    bfs = (BFSCorrections(enabled_coulomb_NLO=args.diagnostic_bfs_coulomb_nlo)
           if args.diagnostic_bfs_coulomb_nlo is not None else None)
    generator = WWGenerator.from_card(card, bfs=bfs)
    print(f"[{generator.describe()}]")
    # ``mass_scale`` / ``width_scale`` / ``mass_scheme`` are scaffolding
    # used only to label template files (the WW chain has no μ-renormalisation
    # scale and no alternate mass scheme). Defaults match the legacy
    # WW_NNLO_...scaleM80.0_scaleW80.0 filename convention.
    scales = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0, "vars": []})
    mass_scale = scales["mass"]
    width_scale = scales["width"]
    mass_scheme = getattr(card, "MASS_SCHEME", "OS")

    t0 = time.time()

    if args.only_bec_shift is not None:
        shift = args.only_bec_shift
        if shift == 0.0:
            print(f"\n[ nominal ]  outdir = {args.outdir}")
            _generate_set(generator, params,
                          mass_scale=mass_scale, width_scale=width_scale,
                          mass_scheme=mass_scheme, outdir=args.outdir,
                          tags=tags, force=args.force)
        else:
            subdir = os.path.join(args.bec_outdir, bec_var_dir(shift))
            print(f"\n[ BEC {shift:+.0f} MeV ]  outdir = {subdir}")
            _generate_set(generator, params,
                          mass_scale=mass_scale, width_scale=width_scale,
                          mass_scheme=mass_scheme, outdir=subdir,
                          ecm_shift_MeV=shift, tags=tags, force=args.force)
    else:
        print(f"\n[ nominal ]  outdir = {args.outdir}")
        _generate_set(generator, params,
                      mass_scale=mass_scale, width_scale=width_scale,
                      mass_scheme=mass_scheme, outdir=args.outdir,
                      tags=tags, force=args.force)

        if not args.no_bec:
            for var in args.bec_vars_MeV:
                for shift in (+var, -var):
                    subdir = os.path.join(args.bec_outdir, bec_var_dir(shift))
                    print(f"\n[ BEC {shift:+.0f} MeV ]  outdir = {subdir}")
                    _generate_set(generator, params,
                                  mass_scale=mass_scale, width_scale=width_scale,
                                  mass_scheme=mass_scheme, outdir=subdir,
                                  ecm_shift_MeV=shift, tags=tags, force=args.force)

    print(f"\nDone in {time.time() - t0:.2f} s.")


if __name__ == "__main__":
    main()
