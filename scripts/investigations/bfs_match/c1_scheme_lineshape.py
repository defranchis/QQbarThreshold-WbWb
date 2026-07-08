"""Decisive, extrapolation-free check: does the eMELA NLL-DELTA radiator's scheme
match the collinear (β=2L, no −1) or the beta (β=2L−1) leading log?

eMELA documents (eMELA.hh) two LL conventions:
  collinear (LLPDF idx 0): β = α/π·log(μ²/μ₀²)        — the 1911.12040 / DELTA LL
  beta      (LLPDF idx 1): β = α/π·(log(μ²/μ₀²) − 1)  — the code's current C₁
The analytic DELTA fixed-order O(α) ePDF = collinear's O(α).  isr_beta exposes both
as convolve_2leg(scheme="LO_collinear") and (scheme="LO_beta"); the eMELA NLL-DELTA
radiator is convolve_2leg(nll=True, DELTA).

Logic: the genuine NLL effect (radiator LL→NLL in the SAME scheme) is ~0.26 % at 161
(memory).  So the SCHEME-MATCHED LL should differ from NLL-DELTA by ~that pure-NLL
amount; the SCHEME-MISMATCHED LL differs by NLL + the scheme −1 (~1.6 % at line-shape
level).  Whichever LL scheme sits CLOSER to NLL-DELTA identifies its scheme — and
hence whether the matching C₁ should be collinear/DELTA (2L) or beta (2L−1).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold"
sys.path.insert(0, REPO)

from framework.process.ww.indep import isr_beta as ib
from framework.process.ww.indep import partonic_grid as pg

ALPHA_MZ = 1.0 / 128.943


def main():
    grids = pg.load_grids(scheme_alpha="gf", lepton_cut=None)
    keys = sorted(grids)
    chosen = next((k for k in keys if "nominal" in k[1].lower()), keys[len(keys)//2])
    born = grids[chosen].born_fn()
    print(f"channel/varpoint = {chosen}")

    sqrt_s = np.array([159.0, 161.0, 162.5])
    bv = born(sqrt_s)

    cfg_beta = ib.ISRConfig(scheme="LO_beta", alpha=ALPHA_MZ, nll=False)
    cfg_coll = ib.ISRConfig(scheme="LO_collinear", alpha=ALPHA_MZ, nll=False)
    cfg_nll = ib.ISRConfig(scheme="LO_beta", alpha=ALPHA_MZ, nll=True,
                           emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ")

    L_beta = ib.convolve_2leg(sqrt_s, born, cfg_beta)
    L_coll = ib.convolve_2leg(sqrt_s, born, cfg_coll)
    L_nll = ib.convolve_2leg(sqrt_s, born, cfg_nll)

    # exact analytic O(α) overlaps (C₁) for the two LL schemes
    C1_beta = ib.oalpha_isr_subtraction(sqrt_s, born, cfg_beta)
    C1_coll = ib.oalpha_isr_subtraction(sqrt_s, born, cfg_coll)

    print(f"\n  {'√s':>6s} {'L_beta':>10s} {'L_coll':>10s} {'L_NLL':>10s}"
          f"  {'coll/beta-1':>11s} {'NLL/beta-1':>11s} {'NLL/coll-1':>11s}  [%]")
    for i, s in enumerate(sqrt_s):
        rcb = (L_coll[i]/L_beta[i] - 1) * 100
        rnb = (L_nll[i]/L_beta[i] - 1) * 100
        rnc = (L_nll[i]/L_coll[i] - 1) * 100
        print(f"  {s:6.2f} {L_beta[i]:10.5f} {L_coll[i]:10.5f} {L_nll[i]:10.5f}"
              f"  {rcb:+11.4f} {rnb:+11.4f} {rnc:+11.4f}")

    print("\n  DECISION — the SCHEME-MATCHED LL sits closest to NLL-DELTA (≈ pure-NLL,")
    print("  ~0.26%@161); the mismatched one is off by the extra scheme −1 (~1.6%).")
    for i, s in enumerate(sqrt_s):
        dnc = abs(L_nll[i]/L_coll[i] - 1) * 100
        dnb = abs(L_nll[i]/L_beta[i] - 1) * 100
        which = "COLLINEAR (=DELTA → C₁ should be 2L)" if dnc < dnb else \
                "BETA (→ C₁ stays 2L−1, no change)"
        print(f"    √s={s:6.2f}:  |NLL−coll|={dnc:.4f}%  |NLL−beta|={dnb:.4f}%"
              f"   → NLL-DELTA tracks {which}")

    # cross-matching residuals: (L_NLL − C₁) − Born, for each C₁ choice
    print(f"\n  matched-with-Born residual (L_NLL − C₁ − σ̂_Born)/σ̂_Born  [%]"
          f"  — the O(α)+ leftover after subtraction:")
    print(f"    {'√s':>6s} {'C₁=beta(2L−1)':>14s} {'C₁=coll(2L=Δ)':>14s}")
    for i, s in enumerate(sqrt_s):
        rb = ((L_nll[i] - C1_beta[i]) - bv[i]) / bv[i] * 100
        rc = ((L_nll[i] - C1_coll[i]) - bv[i]) / bv[i] * 100
        print(f"    {s:6.2f} {rb:14.4f} {rc:14.4f}")
    print("  (these still contain genuine resummed higher-order ISR; the point is the")
    print("   DIFFERENCE between the two C₁ columns = the scheme term being tested.)")


if __name__ == "__main__":
    main()
