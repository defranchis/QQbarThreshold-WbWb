"""Decisive O(α) test: is the exact DELTA C₁ the one that cancels the eMELA NLL
radiator's O(α) ISR, or is it the code's β-LL C₁?

Strategy — scale the coupling α → ε·α and read off the O(α) piece (∝ ε):
  PART A (ePDF level):  extract eMELA's O(α) DELTA ePDF coefficient and compare to
                        the analytic prediction  Γ^[1],Δ(x) = (1+x²)/(1−x)·log(Q²/m²)
                        [Frixione 1909.03886 G1sol2 + 2105.06688 Kdelz: the −2log(1-x)−1
                        cancels].  Convention: D(x,Q) = δ(1-x) + (α/2π)Γ^[1](x) + O(α²).
  PART B (convolved):   the radiator's O(α) ISR on a real σ̂_Born is
                        a₁ = lim_{ε→0} [convolve_2leg_ε(σ̂_Born) − σ̂_Born]/ε.
                        The matching subtraction must equal a₁.  C₁ is exactly linear
                        in α (∝ ε), so compare a₁ to C₁^β and C₁^Δ = C₁^β·2L/(2L−1).
                        If a₁ = C₁^Δ, the matched residual (conv − C₁^Δ − Born) is
                        O(ε²) and C₁^β leaves an O(ε) residual = the predicted gap.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold"
sys.path.insert(0, REPO)

from framework.process.ww.xsec_calculator import emela_wrapper as emela
from framework.process.ww.indep import isr_beta as ib
from framework.process.ww.indep import partonic_grid as pg

ALPHA_MZ = 1.0 / 128.943
M_E = ib.M_E
PI = math.pi
TWO_PI = 2.0 * math.pi


def part_A():
    print("=" * 78)
    print("PART A — eMELA O(α) DELTA ePDF  vs  analytic 2L·(1+x²)/(1−x)")
    print("=" * 78)
    Q = 161.0
    twoL = 2.0 * math.log(Q / M_E)            # = log(Q²/m_e²)
    xs = [0.60, 0.70, 0.80, 0.90, 0.95]
    eps_list = [0.5, 0.25, 0.125, 0.0625]
    print(f"  Q={Q} GeV, 2L=log(Q²/m²)={twoL:.4f}\n")
    print(f"  {'x':>6s}  {'analytic Γ^[1]':>14s}   "
          + "  ".join(f"ε={e:g}" for e in eps_list) + "   (extracted Γ^[1])")
    for x in xs:
        analytic = (1.0 + x * x) / (1.0 - x) * twoL
        row = []
        for eps in eps_list:
            a = eps * ALPHA_MZ
            emela.initialize(pert_order="NLL", fac_scheme="DELTA",
                             ren_scheme="ALPMZ", alpha=a)
            xD = emela.code_pdf(x, 1.0 - x, Q)        # = x·D(x,Q)
            D_minus_delta = xD / x                    # D at x<1 (no δ here)
            gamma1 = D_minus_delta * TWO_PI / a       # strip (α/2π)·ε
            row.append(gamma1)
        cells = "  ".join(f"{g:11.3f}" for g in row)
        ratio = row[-1] / analytic
        print(f"  {x:6.2f}  {analytic:14.3f}   {cells}   ratio(ε→0)/an={ratio:.4f}")
    print("\n  → extracted Γ^[1] should converge (as ε→0) to the analytic DELTA value.")


def part_B():
    print("\n" + "=" * 78)
    print("PART B — radiator O(α) ISR  vs  C₁^β  and  C₁^Δ  (convolved on σ̂_Born)")
    print("=" * 78)
    grids = pg.load_grids(scheme_alpha="gf", lepton_cut=None)
    keys = sorted(grids)
    chosen = next((k for k in keys if "nominal" in k[1].lower()), keys[len(keys)//2])
    born = grids[chosen].born_fn()
    print(f"  channel/varpoint = {chosen}")

    sqrt_s = np.array([159.0, 161.0, 162.5])
    born_val = born(sqrt_s)
    eps_list = [1.0, 0.5, 0.25, 0.125]

    # collect per-ε arrays
    conv = {}
    c1b = {}
    c1d = {}
    for eps in eps_list:
        cfg = ib.ISRConfig(scheme="LO_beta", alpha=eps * ALPHA_MZ, nll=True,
                           emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ")
        conv[eps] = ib.convolve_2leg(sqrt_s, born, cfg)
        c1b[eps] = ib.oalpha_isr_subtraction(sqrt_s, born, cfg)
        twoL = np.array([2.0 * math.log(cfg.mu_F(float(s)) / cfg.m_e) for s in sqrt_s])
        c1d[eps] = c1b[eps] * twoL / (twoL - 1.0)

    for i, s in enumerate(sqrt_s):
        print(f"\n  √s = {s} GeV   (σ̂_Born = {born_val[i]:.4f} fb)")
        print(f"    {'ε':>6s} {'(conv−B)/ε':>11s} {'C₁^β/ε':>10s} {'C₁^Δ/ε':>10s}"
              f" {'resβ/ε':>11s} {'resΔ/ε':>11s} {'resΔ/ε²':>11s}")
        a1_seq = []
        for eps in eps_list:
            isr = conv[eps][i] - born_val[i]
            rb = (conv[eps][i] - c1b[eps][i]) - born_val[i]
            rd = (conv[eps][i] - c1d[eps][i]) - born_val[i]
            a1_seq.append(isr / eps)
            print(f"    {eps:6.3f} {isr/eps:11.5f} {c1b[eps][i]/eps:10.5f} "
                  f"{c1d[eps][i]/eps:10.5f} {rb/eps:11.5f} {rd/eps:11.5f} "
                  f"{rd/eps**2:11.5f}")
        # Richardson: a1 ≈ 2 f(ε/2) − f(ε) using the two smallest ε
        f_small, f_smaller = a1_seq[-2], a1_seq[-1]
        a1 = 2.0 * f_smaller - f_small
        C1b = c1b[eps_list[-1]][i] / eps_list[-1]
        C1d = c1d[eps_list[-1]][i] / eps_list[-1]
        print(f"    → radiator O(α)  a₁ (Richardson) = {a1:11.5f}")
        print(f"      C₁^β = {C1b:11.5f}  (a₁−C₁^β = {a1-C1b:+.5f}, "
              f"{(a1-C1b)/a1*100:+.2f}%)")
        print(f"      C₁^Δ = {C1d:11.5f}  (a₁−C₁^Δ = {a1-C1d:+.5f}, "
              f"{(a1-C1d)/a1*100:+.2f}%)")
        verdict = "DELTA" if abs(a1 - C1d) < abs(a1 - C1b) else "BETA"
        print(f"      VERDICT: radiator O(α) matches  C₁^{verdict}")


if __name__ == "__main__":
    part_A()
    part_B()
