#!/usr/bin/env python3
"""Confirm WHERE the NLL 'alpha-hypersensitivity' lives (step-1 follow-up).

scheme_alpha_scan.py showed the DELTA NLL-KERNEL pull is alpha-stable
(swing ~0.1 MeV).  So the reported 6.9-MeV swing of the FULL LL->NLL pull must
live in the alpha-VALUE piece (LL at alpha(M_Z) vs LL at production alpha_Gmu),
NOT the NLL kernel or the factorisation scheme.  This decomposes the DELTA full
pull at BOTH alpha values to prove it:

    full(alpha)   = crossfit( truth = NLL(DELTA, alpha),   morph = LL(alpha_Gmu) )
    aval(alpha)   = crossfit( truth = LL(alpha),           morph = LL(alpha_Gmu) )
    kernel(alpha) = crossfit( truth = NLL(DELTA, alpha),   morph = LL(alpha)     )
                    (full ~ aval + kernel)

per-piece swing = piece(moca) - piece(alpmz).  Expectation: the swing concentrates
in `aval`; `kernel` is flat.

Run after build_scheme_scan_grids.py + scheme_alpha_scan.py (reuses their grids
and templates).

Run:  PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/confirm_alpha_decomp.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import isr_beta
import scheme_alpha_scan as S  # noqa: E402  (same dir; reuse helpers)

# add the production alpha_Gmu LL morph
GMU = isr_beta.ALPHA_GMU


def gmu_cfg():
    return isr_beta.ISRConfig(scheme="LO_beta", alpha=GMU)


def main():
    for atag, a in S.ALPHAS.items():
        print(f"alpha[{atag}] = {a:.8e}  (1/{1.0/a:.3f})")
    print(f"alpha[gmu]   = {GMU:.8e}  (1/{1.0/GMU:.3f})   [production LL morph]\n")

    # templates: LL(gmu) morph + LL(alpha) truths + NLL(DELTA,alpha) truths
    gmu_dir = S._gen_templates(gmu_cfg(), "LL_gmu")
    ll_dirs = {t: S._gen_templates(S.ll_cfg(t), f"LL_{t}") for t in S.ALPHAS}
    ll_truth, nll_truth = {}, {}
    for t in S.ALPHAS:
        ll_truth[t] = S._fresh_fit(S.ll_cfg(t), ll_dirs[t]).template("nominal")
        d = S._gen_templates(S.nll_cfg("DELTA", t), f"NLL_DELTA_{t}")
        nll_truth[t] = S._fresh_fit(S.nll_cfg("DELTA", t), d).template("nominal")

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"=== DELTA pull decomposition, {mode} ===")
        print(f"{'piece':10s} {'alpmz':>9s} {'moca':>9s} {'swing':>9s}   [MeV, dm_W]")
        pieces = {}
        # aval: truth=LL(alpha), morph=LL(gmu)
        for t in S.ALPHAS:
            r = S._crossfit(ll_truth[t], gmu_cfg(), gmu_dir, shape_only)
            pieces[("aval", t)] = r["bias_mW"]
        # kernel: truth=NLL(DELTA,alpha), morph=LL(alpha)
        for t in S.ALPHAS:
            r = S._crossfit(nll_truth[t], S.ll_cfg(t), ll_dirs[t], shape_only)
            pieces[("kernel", t)] = r["bias_mW"]
        # full: truth=NLL(DELTA,alpha), morph=LL(gmu)
        for t in S.ALPHAS:
            r = S._crossfit(nll_truth[t], gmu_cfg(), gmu_dir, shape_only)
            pieces[("full", t)] = r["bias_mW"]
        for name in ("aval", "kernel", "full"):
            a, m = pieces[(name, "alpmz")], pieces[(name, "moca")]
            print(f"{name:10s} {a:9.2f} {m:9.2f} {m-a:9.2f}")
        # consistency: full ~ aval + kernel
        for t in S.ALPHAS:
            s = pieces[("aval", t)] + pieces[("kernel", t)]
            print(f"  check aval+kernel[{t}] = {s:7.2f} vs full = "
                  f"{pieces[('full', t)]:7.2f}")
        print()


if __name__ == "__main__":
    main()
