#!/usr/bin/env python3
"""Diagnostic for the lumi promotion gate: is the lumi↔2-D cross-fit Δm_W the
2-D's n_quad RIPPLE (→0 as n_quad→∞, lumi = converged 2-D, FAITHFUL) or a genuine
lumi shape error (stays finite)?

Cross-fit truth=lumi (smooth, = converged 2-D) vs morph=2-D at increasing n_quad,
both on the SAME eMELA-grid radiator.  If |Δm_W| shrinks toward 0 with n_quad, the
luminosity IS the ripple-free 2-D and the ~MeV at n_quad=128 is the ripple the
luminosity removes (the production-impact number, NOT a faithfulness failure).

Run:  source setup.sh && WW_INDEP_NJOBS=16 PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/crossfit_lumi_nquad.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import isr_beta
import scripts.investigations.nll_isr.crossfit_lumi_gate as G

NQUADS = [128, 256, 512, 1024]


def cfg_2d(n_quad):
    return isr_beta.ISRConfig(
        nll=True, alpha=isr_beta.ALPHA_MZ_EMELA, emela_fac_scheme="DELTA",
        emela_ren_scheme="ALPMZ", emela_grid=G.PROD_GRID, lumi=False, n_quad=n_quad)


def main():
    print("lumi ↔ 2-D cross-fit vs 2-D n_quad (same grid radiator; truth=lumi smooth)")
    print("  if |Δm_W| → 0 with n_quad ⇒ lumi = converged 2-D (faithful); the n_quad=128")
    print("  value is the production RIPPLE the luminosity removes.\n")

    cfg_lumi = G._cfg(True)
    print("[gen] luminosity morph ...", flush=True)
    d_lumi = G.gen_full_templates(cfg_lumi, "lumi")
    truth_lumi = G.fresh_fit(cfg_lumi, d_lumi).template("nominal")

    built = {}
    for nq in NQUADS:                       # build each 2-D morph ONCE
        print(f"[gen] 2-D morph n_quad={nq} ...", flush=True)
        built[nq] = (cfg_2d(nq), G.gen_full_templates(cfg_2d(nq), f"twoD_{nq}"))

    print(f"\n{'mode':24s} {'n_quad':>6s} {'dm_W':>8s} {'dG_W':>8s}   [MeV]")
    for shape_only in (True, False):
        mode = "shape-only" if shape_only else "cov-lumi"
        for nq in NQUADS:
            cfg, d_2d = built[nq]
            # truth=lumi (smooth) ↔ morph=2-D(n_quad): the real-fit-like direction
            r = G.crossfit(truth_lumi, cfg, d_2d, shape_only)
            print(f"{('lumi↔2D '+mode):24s} {nq:6d} {r['dmW']:8.3f} {r['dgW']:8.3f}"
                  f"{'' if r['valid'] else '  !INVALID'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
