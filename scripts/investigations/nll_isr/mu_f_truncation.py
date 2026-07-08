#!/usr/bin/env python3
"""Step-5: re-derive the ISR theory systematic on the STABLE DELTA-NLL baseline.

Background.  The OLD ISR systematic (report sec:val-isr-cross) is +-22.4 MeV =
17.0 (analytic beta^3 LL+exp vs eMELA's full DGLAP LL) + 5.4 (eMELA LL -> NLL).
That is the bias of NOT running NLL -- the 17.0 piece is the *approximation
error of the analytic beta^3 formula*, which simply vanishes once eMELA's NLL
DGLAP ePDF IS the central value.  On the NLL baseline the residual theory
uncertainty is instead:

    (A) NNLL+ truncation  -- missing higher logarithmic orders
    (B) residual factorisation scheme (DELTA matched; O(alpha^2) residual)
    (C) alpha(M_Z) input  -- a SEPARATE profiled nuisance (aem_isr), ~0.005 MeV

This script measures all three on the DELTA / PDG-alpha / grid baseline and
prints the assembled budget.

(A) NNLL truncation -- two handles:
  (A1) mu_F / xi factorisation-scale variation: crossfit( truth = NLL(mu_F=xi.sqrt(s)),
       morph = NLL(mu_F=sqrt(s)) ) for xi in {0.5, 2.0}.  CAVEAT: in the indep
       chain sigma_hat_NLO is mu_F-INDEPENDENT (pdf_set=none, no IS mass
       factorisation), so the ePDF's mu_F-running is UNCOMPENSATED -> this is a
       CONSERVATIVE UPPER BOUND (it is the full LL scale dependence, not the
       truncation order; the report's "probes DGLAP numerics" statement, now
       measured).
  (A2) order-step bracket: the last computed correction itself = the NLL-KERNEL
       pull |dm_W| = crossfit( truth = NLL(DELTA, PDG), morph = LL(PDG) ).  This
       is the honest "size of the highest included order" estimate of the next
       missing one.

(B) residual scheme = the NLL-kernel alpha-swing (DELTA pull at PDG vs MoCaNLO
    alpha): an unmatched O(alpha) term would scale with alpha; the tiny residual
    swing bounds the leftover O(alpha^2)/scheme ambiguity.  (From scheme_alpha_scan.)

Uses the grids built by build_scheme_scan_grids.py + the templates left by
scheme_alpha_scan.py in /tmp/ww_nll_scheme_scan/.  The fit scan is 157-163 GeV,
so mu_F = xi.sqrt(s) in [0.5,2]*[157,163] = [78,326] GeV stays inside the grid
Q-knot range [75,350] -- no direct eMELA needed.

Run (after build_scheme_scan_grids.py + scheme_alpha_scan.py):
  source setup.sh
  PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/mu_f_truncation.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.process.ww.indep import isr_beta
import scheme_alpha_scan as S  # reuse _card_2poi/_gen_templates/_fresh_fit/_crossfit

XIS = [0.5, 2.0]            # mu_F = xi * sqrt(s)
SCHEME = "DELTA"
ABASE = "alpmz"            # PDG alpha(M_Z) = production baseline


def nll_cfg_muF(scheme, atag, xi):
    """DELTA NLL grid cfg with mu_F = xi*sqrt(s) (the grid guard checks only
    alpha/scheme, NOT mu_F, so xi-variation reuses the same .npz)."""
    return isr_beta.ISRConfig(
        nll=True, alpha=S.ALPHAS[atag], emela_fac_scheme=scheme,
        emela_ren_scheme="ALPMZ", emela_grid=S.grid_path(scheme, atag),
        mu_F_factor=xi)


def main():
    a = S.ALPHAS[ABASE]
    print(f"baseline: {SCHEME} NLL, alpha = {a:.8e} (1/{1.0/a:.3f}), grid mu_F=sqrt(s)\n")

    morph_cfg = S.nll_cfg(SCHEME, ABASE)                 # mu_F_factor = 1.0
    morph_dir = os.path.join(S.BASE, f"NLL_{SCHEME}_{ABASE}")
    if not os.path.isdir(morph_dir):
        sys.exit(f"missing morph templates {morph_dir}; run scheme_alpha_scan.py first")

    # --- (A1) mu_F / xi scale variation ------------------------------------
    # truth nominals at mu_F = xi*sqrt(s); morph = mu_F = sqrt(s) NLL set.
    truth_xi = {}
    for xi in XIS:
        d = S._gen_templates(nll_cfg_muF(SCHEME, ABASE, xi), f"NLL_{SCHEME}_{ABASE}_xi{xi}")
        truth_xi[xi] = S._fresh_fit(nll_cfg_muF(SCHEME, ABASE, xi), d).template("nominal")
    # xi=1 self-consistency: truth = morph's own nominal -> must be ~0.
    truth_xi[1.0] = S._fresh_fit(morph_cfg, morph_dir).template("nominal")

    muF = {}
    for shape_only in (True, False):
        mode = "shape-only" if shape_only else "cov-lumi"
        muF[mode] = {}
        for xi in [0.5, 1.0, 2.0]:
            r = S._crossfit(truth_xi[xi], morph_cfg, morph_dir, shape_only)
            muF[mode][xi] = r["bias_mW"]

    # --- (A2) order-step bracket + (B) residual scheme (existing templates) -
    # kernel pull = crossfit(truth=NLL(DELTA,alpha), morph=LL(alpha)) at both alphas.
    kernel = {}
    for atag in S.ALPHAS:
        td = os.path.join(S.BASE, f"NLL_{SCHEME}_{atag}")
        ld = os.path.join(S.BASE, f"LL_{atag}")
        if not (os.path.isdir(td) and os.path.isdir(ld)):
            kernel[atag] = None
            continue
        truth = S._fresh_fit(S.nll_cfg(SCHEME, atag), td).template("nominal")
        kernel[atag] = S._crossfit(truth, S.ll_cfg(atag), ld, True)["bias_mW"]

    # ---------------------------- report -----------------------------------
    print("=== (A1) mu_F / xi factorisation-scale variation ===")
    print("    truth NLL @ mu_F=xi.sqrt(s)  vs  morph NLL @ mu_F=sqrt(s)")
    print(f"    {'xi':>5s} {'shape-only':>12s} {'cov-lumi':>12s}   [MeV, dm_W]")
    for xi in [0.5, 1.0, 2.0]:
        print(f"    {xi:5.2f} {muF['shape-only'][xi]:12.2f} {muF['cov-lumi'][xi]:12.2f}")
    for mode in ("shape-only", "cov-lumi"):
        lo, hi = muF[mode][0.5], muF[mode][2.0]
        env = max(abs(lo), abs(hi))
        half = abs(hi - lo) / 2.0
        print(f"    {mode:11s} envelope max|.| = {env:6.2f}   half-spread = {half:6.2f} MeV")
    print("    NOTE: uncompensated (sigma_hat is mu_F-independent) -> CONSERVATIVE")
    print("          UPPER BOUND on the scale ambiguity, not the truncation order.\n")

    print("=== (A2) order-step bracket  +  (B) residual scheme ===")
    print(f"    {'alpha':7s} {'NLL-kernel pull':>16s}   [MeV, dm_W, shape-only]")
    for atag in S.ALPHAS:
        v = kernel[atag]
        print(f"    {atag:7s} {('n/a' if v is None else f'{v:+.2f}'):>16s}")
    if kernel.get("alpmz") is not None:
        print(f"    (A2) |last included order| = |kernel @PDG| = {abs(kernel['alpmz']):.2f} MeV")
    if kernel.get("alpmz") is not None and kernel.get("moca") is not None:
        sw = abs(kernel["moca"] - kernel["alpmz"])
        print(f"    (B)  residual scheme = alpha-swing = {sw:.2f} MeV "
              f"(O(alpha) term would not be flat)")
    print()

    # ---------------------- assembled budget -------------------------------
    a2 = abs(kernel["alpmz"]) if kernel.get("alpmz") is not None else float("nan")
    b = (abs(kernel["moca"] - kernel["alpmz"])
         if kernel.get("alpmz") is not None and kernel.get("moca") is not None
         else float("nan"))
    c = 0.005  # aem_isr profiled nuisance (separate input unc), report par:alpha-em
    quad = (a2**2 + b**2 + c**2) ** 0.5
    a1 = max(abs(muF["shape-only"][0.5]), abs(muF["shape-only"][2.0]))
    print("=== NEW ISR THEORY SYSTEMATIC on the DELTA-NLL baseline ===")
    print(f"  (A) NNLL truncation [order-step, central]     {a2:6.2f} MeV")
    print(f"  (B) residual factorisation scheme             {b:6.2f} MeV")
    print(f"  (C) alpha(M_Z) input (separate nuisance)      {c:6.3f} MeV")
    print(f"  ------------------------------------------------------")
    print(f"  quadrature (A+B+C)                            {quad:6.2f} MeV")
    print(f"  [conservative upper bound via mu_F envelope:  {a1:6.2f} MeV]")
    print(f"\n  vs OLD +-22.4 MeV (17.0 analytic-LL-trunc + 5.4 NLL-bracket):")
    print(f"  the 17.0 was the beta^3-formula approximation error (gone on the")
    print(f"  NLL baseline); the 5.4 NLL bracket is now the *included* correction.")


if __name__ == "__main__":
    main()
