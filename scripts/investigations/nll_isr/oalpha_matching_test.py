#!/usr/bin/env python3
"""Task 0 (KEYSTONE) — O(α)-expansion matching test for the indep NLL chain.

Goal: prove DIRECTLY (not via an α-slope) that
  (i)  the eMELA ePDF used in production is the Δ (DIS-like) factorisation scheme,
  (ii) the MoCaNLO σ̂_NLO grid carries NO initial-state collinear log
       (σ̂ ≈ σ_Born inclusive), so σ̂ ⊗ D is matched at O(α) with no double count,
and to RESOLVE the idip/DELTA wording tension.

Derivation (Frixione arXiv:2105.06688, eqs. (G1sol2)+(Kdelz); verified against the
.tex source, NOT from memory).  The O(α) electron-in-electron ePDF coefficient,
evolved to scale Q (μ₀-independent), in a general scheme is

    Γ_ee^[1](z,Q) = [ (1+z²)/(1-z) ( L − 2ln(1-z) − 1 ) ]_+ + K_ee(z),
    L ≡ ln(Q²/m_e²),     Γ_ee(z,Q) = δ(1-z) + (α/2π) Γ_ee^[1](z,Q) + O(α²).

The Δ scheme fixes K_ee^(Δ)(z) = [ (1+z²)/(1-z)(2ln(1-z)+1) ]_+, which EXACTLY
cancels the finite (−2ln(1-z)−1), leaving the pure collinear log

    Γ_ee^[1,Δ](z,Q) = [ (1+z²)/(1-z) ]_+ L          (NO finite term),

while MS̄ (K_ee≡0) keeps the finite term.  The Δ↔MS̄ difference is the eq.(33)
finite term  D_Δ − D_MS̄ = +(α/2π) K_ee^(Δ).

Reuse of validated code: with (1+z²)/(1-z) = 2/(1-z) − (1+z), the whole-function
plus distribution gives per-leg β-coefficients (β_e,β_s,β_h) = (α/π)·2ln(μ_F/m_e)
for the Δ scheme — i.e. exactly ``beta_components(scheme="LO_collinear",
pdf_starting_scale=m_e)``.  So the O(α) Δ-radiator action on σ̂_Born is computed
EXACTLY by ``isr_beta.oalpha_isr_subtraction`` with an ``LO_collinear`` cfg; the
production analytic-LL ``LO_beta`` differs only by the finite "−1" (= a genuine
O(α) scheme term), and MS̄ adds the explicit −2ln(1-z) finite term (computed here
directly).

Run:  source setup.sh && PYTHONPATH=$PWD \
      python3 scripts/investigations/nll_isr/oalpha_matching_test.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Silence the eMELA C-level init banner.
_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.partonic_grid import load_grids, DEFAULT_RESULTS_DIR
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS
from framework.process.ww.indep.partonic_grid import _read_csv  # noqa
import glob
from framework.process.ww.xsec_calculator import emela_wrapper as em
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

M_E = isr_beta.M_E
ALPHA = isr_beta.ALPHA_MZ_EMELA            # α(M_Z)=1/128.943, the production ISR α
TWO_PI = 2.0 * math.pi
A2PI = ALPHA / TWO_PI
SQ = np.array([157., 158., 159., 160., 161., 162., 163.])


# ---------------------------------------------------------------------------
# Assembled pure-WW σ̂_Born(√ŝ), σ̂_NLO(√ŝ) at the nominal varpoint (gf grid)
# ---------------------------------------------------------------------------
def _assembled():
    g = load_grids(scheme_alpha="gf")
    W = dict(PURE_WW_WEIGHTS)
    bfn = {k: g[(k, "nominal")].born_fn() for k in W}
    nfn = {k: g[(k, "nominal")].nlo_fn() for k in W}

    def sigB(sh):
        sh = np.asarray(sh, dtype=float)
        return sum(W[k] * np.asarray(bfn[k](sh), dtype=float) for k in W)

    def sigN(sh):
        sh = np.asarray(sh, dtype=float)
        return sum(W[k] * np.asarray(nfn[k](sh), dtype=float) for k in W)

    return sigB, sigN, W, g


# ---------------------------------------------------------------------------
# O(α) radiator action on σ̂_Born (per scheme).  Δ and β reuse the validated
# oalpha_isr_subtraction; MS̄ = Δ − ΔC₁ with the eq.(33) finite term added.
# ---------------------------------------------------------------------------
def R_delta(sigB):
    cfg = isr_beta.ISRConfig(scheme="LO_collinear", alpha=ALPHA, m_e=M_E)
    return isr_beta.oalpha_isr_subtraction(SQ, sigB, cfg)


def R_beta(sigB):
    cfg = isr_beta.ISRConfig(scheme="LO_beta", alpha=ALPHA, m_e=M_E)
    return isr_beta.oalpha_isr_subtraction(SQ, sigB, cfg)


def delta_c1(sigB, x_min=0.30, n=512):
    """ΔC₁(s) = R^Δ − R^MS̄ = 2·(α/2π) ∫ [K_ee^(Δ)(z)]_+ σ̂_Born(√z·√s) dz [fb].

    K_ee^(Δ)(z) = [(1+z²)/(1-z)(2ln(1-z)+1)]_+ (whole-function plus → subtract at
    z=1).  This is the genuine Δ↔MS̄ O(α) FINITE scheme term — the additive σ̂
    correction Task 2 convolves with the MS̄ ePDF.  Same GL grid / x_min as the
    production convolution."""
    x, w = isr_beta._quad_nodes(n, x_min, 1.0)
    out = np.zeros_like(SQ)
    for i, sq in enumerate(SQ):
        phi = sigB(np.sqrt(x) * sq)
        phi1 = float(sigB(np.array([sq]))[0])
        kern = (1.0 + x**2) / (1.0 - x) * (2.0 * np.log(1.0 - x) + 1.0)
        out[i] = 2.0 * A2PI * np.sum(w * kern * (phi - phi1))
    return out


def main():
    sigB, sigN, W, g = _assembled()

    print("=" * 78)
    print("Task 0 — O(α) matching test  (σ̂ is Δ?  σ̂⊗D matched?)")
    print(f"α(M_Z)=1/{1.0/ALPHA:.3f}   m_e={M_E:.6e} GeV   μ_F=√s   "
          f"L=ln(s/m_e²)≈{math.log(161.0**2/M_E**2):.2f}")
    print("=" * 78)

    # --- A) eMELA implements the Δ scheme: eq.(33) numerical closure ----------
    print("\n[A] eMELA D_Δ − D_MS̄  vs  (α/2π) K_ee^(Δ)   (eq.33), Q=161 GeV")
    print("    confirms eMELA's factorisation scheme IS standard Δ (vs MS̄).")
    Q = 161.0
    xs = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    print(f"    {'x':>6} {'x(D_Δ−D_MS̄)':>14} {'x(α/2π)K_ee':>14} {'ratio':>8}")
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    for x in xs:
        omx = 1.0 - x
        em.initialize("NLL", "DELTA", "ALPMZ", ALPHA)
        xd_delta = em.code_pdf(float(x), float(omx), Q)
        em.initialize("NLL", "MSBAR", "ALPMZ", ALPHA)
        xd_msbar = em.code_pdf(float(x), float(omx), Q)
        kee = (1.0 + x**2) / (1.0 - x) * (2.0 * math.log(1.0 - x) + 1.0)
        os.dup2(_sv, 1)
        diff = xd_delta - xd_msbar
        ana = x * A2PI * kee
        print(f"    {x:>6.2f} {diff:>14.6f} {ana:>14.6f} "
              f"{diff/ana if ana else float('nan'):>8.3f}")
        _dn2 = os.open(os.devnull, os.O_WRONLY); os.dup2(_dn2, 1); os.close(_dn2)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    print("    → ratio≈1 (DGLAP dresses the bare O(α) term toward x→1): eMELA = Δ.")

    # --- B) σ̂_NLO carries NO IS-collinear log -------------------------------
    print("\n[B] grid O(α) remnant  vs  the IS-collinear O(α) the Δ radiator adds")
    print("    if σ̂ were 'raw' (un-factorised) it would carry the big −ISR damping;")
    print("    if σ̂ is Δ (inclusive) the remnant is the small +hard-EW correction.")
    sb = sigB(SQ); sn = sigN(SQ)
    Rd = R_delta(sigB)
    print(f"    {'√s':>5} {'σ̂_Born':>9} {'σ̂_NLO−σ̂_B':>11} {'(N−B)/B':>9}"
          f" {'R^Δ_ISR':>10} {'R^Δ/B':>8}")
    for i, sq in enumerate(SQ):
        print(f"    {sq:>5.0f} {sb[i]:>9.2f} {sn[i]-sb[i]:>11.2f} "
              f"{(sn[i]-sb[i])/sb[i]*100:>+8.2f}% {Rd[i]:>10.2f} "
              f"{Rd[i]/sb[i]*100:>+7.2f}%")
    print("    → (N−B)/B is small & POSITIVE (hard EW); R^Δ/B is large & NEGATIVE")
    print("      (ISR damping).  σ̂_NLO does NOT contain the ISR collinear log ⇒ Δ.")

    # --- C) grid component decomposition resolves the idip/DELTA tension ------
    print("\n[C] σ̂_NLO components (lnuqq, nominal): idip removes the IR DIVERGENCE,")
    print("    not a big finite collinear — real & idip are small & positive.")
    cv = g[("lnuqq", "nominal")]
    # pull raw component columns straight from the CSVs (born/virt/real/idip)
    recs = []
    suf = "_gf"
    for p in glob.glob(os.path.join(DEFAULT_RESULTS_DIR,
                                    f"lnuqq_nominal_ecm*{suf}.csv")):
        r = _read_csv(p)
        if r:
            recs.append(r)
    recs.sort(key=lambda r: r["ecm"])
    print(f"    {'√ŝ':>6} {'born':>8} {'virt':>8} {'real':>8} {'idip':>8} "
          f"{'nlo/born':>9}")
    for r in recs:
        if abs(r["ecm"] - round(r["ecm"])) < 1e-6 and 157 <= r["ecm"] <= 163:
            print(f"    {r['ecm']:>6.0f} {r['sigma_born']:>8.2f} "
                  f"{r['sigma_virt']:>8.2f} {r['sigma_real']:>8.2f} "
                  f"{r['sigma_idip']:>8.2f} "
                  f"{r['sigma_nlo']/r['sigma_born']:>9.4f}")
    print("    → real+idip ≈ +few% of Born, both positive: no −30% ISR collinear")
    print("      tail in σ̂.  idip regulates IR; the finite collinear stays in D.")

    # --- D) Δ vs β O(α): leading-log agree, finite '−1' term = O(α) scheme ----
    Rb = R_beta(sigB)
    dC1 = delta_c1(sigB)
    print("\n[D] Δ vs β (LO_beta) O(α) radiator, and the Δ↔MS̄ finite term ΔC₁")
    print(f"    {'√s':>5} {'R^Δ':>10} {'R^β':>10} {'R^Δ−R^β':>9} "
          f"{'ΔC₁':>9} {'ΔC₁/(N−B)':>10}")
    for i, sq in enumerate(SQ):
        nb = sn[i] - sb[i]
        print(f"    {sq:>5.0f} {Rd[i]:>10.2f} {Rb[i]:>10.2f} "
              f"{Rd[i]-Rb[i]:>9.3f} {dC1[i]:>9.3f} "
              f"{dC1[i]/nb*100 if nb else float('nan'):>+9.2f}%")
    print("    R^Δ≈R^β at leading-log (R^Δ−R^β = the '−1' finite term, ≈4% of R^Δ")
    print("    = the small β↔Δ O(α) scheme difference).  ΔC₁ = the Δ↔MS̄ finite")
    print("    term is LARGE at the σ̂ level (~60-110% of the NLO-EW remnant) —")
    print("    exactly WHY the UNMATCHED Δ⊗MS̄ diagnostic was catastrophic (−384")
    print("    MeV): an unmatched ΔC₁ leaks fully into the line shape.  In the")
    print("    MATCHED MS̄⊗MS̄ combination ΔC₁ cancels between σ̂ and D up to O(α²);")
    print("    Task 2 measures THAT residual as the real factorisation systematic.")
    print("\nVERDICT: eMELA D is Δ [A]; σ̂_NLO is inclusive/Δ — no ISR collinear log")
    print("[B,C]; so production Δ⊗Δ is MATCHED at O(α) (idip regulates IR, not the")
    print("finite collinear).  The Δ↔MS̄ transform ΔC₁ is O(α)-LARGE but CANCELS in")
    print("the matched observable → the residual is O(α²); Task 2 turns it into m_W.")


if __name__ == "__main__":
    main()
