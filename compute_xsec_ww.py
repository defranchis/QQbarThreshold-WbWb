"""Batch driver for the WW threshold template generation.

Analogue of ``compute_xsec_parallel.py`` (WbWb / QQbar_threshold), but
calling :class:`process.ww.generator.WWGenerator` instead of the C++
xsec_calc. Generates:

  * nominal templates in ``card.INPUT_DIRS["nominal"]`` for every
    parameter-variation tag (``nominal``, ``pseudodata``, ``<param>_var``).
  * BEC-variation templates in ``card.INPUT_DIRS["BEC"]/scan_{p,m}<var>/``
    with the ECM grid uniformly shifted by ±``INPUT_VAR["BEC"]`` MeV.

The WW LO+Coulomb+LL-ISR generator is Python-only and vectorised — fast
enough that we don't bother with multiprocessing here (each template
~50 ms).
"""

from __future__ import annotations

import argparse
import os
import time

from cards import ww_default as card
from common.parameters import Parameters
from process.ww.eft_xsec import BFSCorrections
from process.ww.generator import WWGenerator


def _generate_set(generator: WWGenerator, params: Parameters, *,
                  mass_scale: float, width_scale: float, mass_scheme: str,
                  outdir: str, ecm_shift_MeV: float = 0.0,
                  verbose: bool = True) -> None:
    for tag in params.tags:
        vals = params.values(tag)
        path = generator.do_scan(
            vals,
            mass_scale=mass_scale, width_scale=width_scale,
            mass_scheme=mass_scheme, outdir=outdir,
            ecm_shift_MeV=ecm_shift_MeV,
        )
        if verbose:
            print(f"  {tag:14s} → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=card.INPUT_DIRS["nominal"],
                    help=f"nominal output directory (default: {card.INPUT_DIRS['nominal']})")
    ap.add_argument("--bec-outdir", default=card.INPUT_DIRS["BEC"],
                    help=f"BEC-variation output directory (default: {card.INPUT_DIRS['BEC']})")
    ap.add_argument("--bec-vars-MeV", nargs="*", type=float, default=[10.0, 30.0],
                    help="absolute BEC shifts in MeV (each generates scan_p{v} and scan_m{v}); "
                         "set to empty list to skip")
    ap.add_argument("--no-bec", action="store_true", help="skip BEC-variation templates")
    ap.add_argument("--bfs-coulomb-nlo", action="store_true",
                    help="enable BFS NLO Coulomb correction (eq. 62 of arXiv:0707.0773); "
                         "OFF by default — without it the templates are LO+Coulomb (FKM)+ISR.")
    args = ap.parse_args()

    params = Parameters(card.PARAMETERS, scale_vars=[])
    bfs = BFSCorrections(enabled_coulomb_NLO=args.bfs_coulomb_nlo)

    # Pick up theory inputs + NLO config from the card so each run records
    # an explicit configuration (see cards/ww_default.py THEORY_INPUTS and
    # NLO_CONFIG). Fall back to LO+Coulomb+ISR-only if the card has no NLO
    # block (back-compat with older cards).
    theory = getattr(card, "THEORY_INPUTS", {})
    nlo_cfg = getattr(card, "NLO_CONFIG", {})
    alpha_s = float(theory.get("alpha_s_MW", 0.1199))
    generator = WWGenerator(
        order=card.ORDER, bfs=bfs,
        include_NLO_hard_decay=bool(nlo_cfg.get("include_NLO_hard_decay", False)),
        apply_delta_QCD=bool(nlo_cfg.get("apply_delta_QCD", False)),
        br_convention=str(nlo_cfg.get("br_convention", "pdg-constant")),
        alpha_s=alpha_s,
    )
    if args.bfs_coulomb_nlo:
        print("[BFS] NLO Coulomb correction ENABLED (eq. 62 of arXiv:0707.0773)")
    if generator.include_NLO_hard_decay:
        print(f"[BFS] NLO hard+soft+collinear + EW-decay ENABLED")
    if generator.apply_delta_QCD:
        print(f"[BFS] delta_QCD(alpha_s={alpha_s}) ENABLED — multiplicative QCD on σ")
    print(f"[BFS] br_convention = {generator.br_convention}")
    mass_scale = card.RENORM_SCALES["mass"]
    width_scale = card.RENORM_SCALES["width"]
    mass_scheme = card.MASS_SCHEME

    t0 = time.time()

    print(f"\n[ nominal ]  outdir = {args.outdir}")
    _generate_set(generator, params,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme, outdir=args.outdir)

    if not args.no_bec:
        for var in args.bec_vars_MeV:
            for sign, sub in (("+", "p"), ("-", "m")):
                shift = (+var if sign == "+" else -var)
                subdir = os.path.join(args.bec_outdir, f"scan_{sub}{int(var)}")
                print(f"\n[ BEC {sign}{var:.0f} MeV ]  outdir = {subdir}")
                _generate_set(generator, params,
                              mass_scale=mass_scale, width_scale=width_scale,
                              mass_scheme=mass_scheme, outdir=subdir,
                              ecm_shift_MeV=shift)

    print(f"\nDone in {time.time() - t0:.2f} s.")


if __name__ == "__main__":
    main()
