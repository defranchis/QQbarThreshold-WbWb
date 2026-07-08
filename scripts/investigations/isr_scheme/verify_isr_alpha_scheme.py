#!/usr/bin/env python3
"""Independent cross-section-level verification of the ISR α-renormalisation
scheme dependence of σ_obs (and hence the fitted m_W).

Context: a theory-ladder study found that varying the ISR QED-coupling
renormalisation scheme (ALPMZ → ALGMU, ALPMZ → α(0)) shifts the fitted
m_W by a surprisingly large amount (+3.5 MeV pure-shape / +36 MeV under a
realistic luminosity prior for ALGMU). This script verifies, at the
σ level, that the dependence is GENUINE (not a plumbing/quadrature bug)
and decomposes it into normalisation vs shape.

All heavy compute (eMELA NLL 2-leg convolution) — run on ironic, not lxplus.

Checks:
  1. σ-level shift, normalisation vs shape (R(√s) grid, mean & slope).
  2. Confirm it is purely ISR (partonic σ̂ identical across schemes).
  3. Plumbing: ALPMZ reproduces production default; eMELA banner α matches;
     α-VALUE vs scheme-STRING separation.
  4. Quadrature robustness (n_quad 128 vs 192).
  5. β_e linearity in α.
"""
from __future__ import annotations

import numpy as np

from framework.process.ww.xsec_calculator.isr import (
    sigma_observed_munuqq,
    beta_ISR,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    alpha_Gmu, ALPHA_EM_0, M_W_BFS_REF,
    sigma_partonic_munuqq,
)

MW = 80.379
GW = 2.085
GRID = np.array([157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0])

ALPHA_ALPMZ = 1.0 / 128.943
ALPHA_ALGMU = alpha_Gmu(M_W_BFS_REF)
ALPHA_A0    = ALPHA_EM_0

SCHEMES = {
    "ALPMZ": ("ALPMZ", ALPHA_ALPMZ),
    "ALGMU": ("ALGMU", ALPHA_ALGMU),
    "ALPHA0": ("FIXED", ALPHA_A0),
}


def sigma_nll(ren_scheme, alpha, n_quad=128):
    """Observed σ via eMELA NLL 2-leg convolution for an explicit scheme/α."""
    return sigma_observed_munuqq(
        GRID, mW=MW, gammaW=GW,
        isr_nll=True,
        isr_emela_ren_scheme=ren_scheme,
        alpha_em_isr=alpha,
        n_quad=n_quad,
    )


def decompose(R):
    """Return (norm_shift_pct, slope_per_GeV_pct, spread_pct)."""
    mean = float(np.mean(R))
    norm = (mean - 1.0) * 100.0
    # linear slope of R vs √s (units: fractional / GeV), expressed in %/GeV
    slope = np.polyfit(GRID, R, 1)[0] * 100.0
    spread = (float(np.max(R)) - float(np.min(R))) * 100.0
    return norm, slope, spread


def main():
    print("=" * 78)
    print("ISR α-renormalisation-scheme verification (σ level)")
    print("=" * 78)
    print(f"  m_W={MW}  Γ_W={GW}  isr_nll=True  grid={list(GRID)}")
    print(f"  α(ALPMZ) = {ALPHA_ALPMZ:.8f}  (1/{1/ALPHA_ALPMZ:.4f})")
    print(f"  α(ALGMU) = {ALPHA_ALGMU:.8f}  (1/{1/ALPHA_ALGMU:.4f})  "
          f"Δα/α = {(ALPHA_ALGMU/ALPHA_ALPMZ-1)*100:+.3f}%")
    print(f"  α(α(0))  = {ALPHA_A0:.8f}  (1/{1/ALPHA_A0:.4f})  "
          f"Δα/α = {(ALPHA_A0/ALPHA_ALPMZ-1)*100:+.3f}%")

    # ------------------------------------------------------------------
    # 5. β_e linearity (cheap, do first)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[5] β_e(s=161^2) linearity in α")
    print("-" * 78)
    s161 = 161.0 ** 2
    for name, (_, a) in SCHEMES.items():
        b = beta_ISR(s161, alpha_em=a)
        print(f"  {name:7s} α={a:.8f}  β_e={b:.8f}  β/α={b/a:.6f}")
    print("  → β/α must be identical across schemes (β linear in α).")

    # ------------------------------------------------------------------
    # 2. Partonic invariance (no ISR kwargs — must be scheme-blind)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[2] Partonic (no-ISR) σ̂ — must be IDENTICAL across schemes")
    print("-" * 78)
    sig_part = sigma_partonic_munuqq(GRID ** 2, MW, GW)
    print(f"  σ̂(√s) [pb]: {np.array2string(np.asarray(sig_part), precision=6)}")
    print("  (partonic σ̂ takes no ISR α kwarg → structurally scheme-invariant)")

    # ------------------------------------------------------------------
    # 1. σ-level shift + 3a. plumbing reproduction of production default
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[1] σ_obs(√s) per scheme  +  [3a] ALPMZ == production default")
    print("-" * 78)
    sig = {}
    for name, (ren, a) in SCHEMES.items():
        sig[name] = np.asarray(sigma_nll(ren, a))
    # production default: no scheme/alpha override (card default ALPMZ)
    sig_prod = np.asarray(sigma_observed_munuqq(
        GRID, mW=MW, gammaW=GW, isr_nll=True))

    print(f"  {'√s':>6} {'ALPMZ':>11} {'ALGMU':>11} {'ALPHA0':>11} "
          f"{'prod(def)':>11}")
    for i, sq in enumerate(GRID):
        print(f"  {sq:6.1f} {sig['ALPMZ'][i]:11.6f} {sig['ALGMU'][i]:11.6f} "
              f"{sig['ALPHA0'][i]:11.6f} {sig_prod[i]:11.6f}")

    rel_prod = np.max(np.abs(sig['ALPMZ'] / sig_prod - 1.0))
    print(f"\n  [3a] max|ALPMZ/prod - 1| = {rel_prod:.2e} "
          f"(must be ~0: card default is ALPMZ)")

    # ------------------------------------------------------------------
    # 1 (cont). R(√s) ratios + norm/shape decomposition
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[1] R(√s) = σ_scheme / σ_ALPMZ   and norm vs shape decomposition")
    print("-" * 78)
    print(f"  {'√s':>6} {'R(ALGMU)':>12} {'R(ALPHA0)':>12}")
    R_algmu = sig['ALGMU'] / sig['ALPMZ']
    R_a0    = sig['ALPHA0'] / sig['ALPMZ']
    for i, sq in enumerate(GRID):
        print(f"  {sq:6.1f} {R_algmu[i]:12.8f} {R_a0[i]:12.8f}")

    for label, R, dadalpha in (
            ("ALGMU", R_algmu, (ALPHA_ALGMU/ALPHA_ALPMZ-1)),
            ("ALPHA0", R_a0,   (ALPHA_A0/ALPHA_ALPMZ-1))):
        norm, slope, spread = decompose(R)
        print(f"\n  {label}:  norm shift (mean R-1) = {norm:+.4f}%   "
              f"slope = {slope:+.5f} %/GeV   spread = {spread:.4f}%")
        print(f"           Δα/α = {dadalpha*100:+.3f}%  → "
              f"norm/(Δα/α) = {norm/(dadalpha*100):+.3f} "
              f"(fraction of Δα/α showing up as rate)")

    # ------------------------------------------------------------------
    # 3c. α-VALUE vs scheme-STRING: ALPMZ-string with ALGMU α value
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[3c] α-VALUE vs scheme-STRING separation")
    print("-" * 78)
    # Same ren_scheme STRING (ALPMZ) but ALGMU's α VALUE:
    sig_str = np.asarray(sigma_nll("ALPMZ", ALPHA_ALGMU))
    R_str = sig_str / sig['ALPMZ']     # vs ALPMZ-string + ALPMZ-α
    R_both = R_algmu                    # ALGMU-string + ALGMU-α  (from above)
    print("  Compare: change α VALUE only (ALPMZ string, α=ALGMU)  vs")
    print("           change BOTH (ALGMU string + ALGMU α).")
    print(f"  {'√s':>6} {'R(α-only)':>12} {'R(both)':>12} {'diff':>12}")
    for i, sq in enumerate(GRID):
        print(f"  {sq:6.1f} {R_str[i]:12.8f} {R_both[i]:12.8f} "
              f"{R_str[i]-R_both[i]:12.2e}")
    n_only, _, _ = decompose(R_str)
    n_both, _, _ = decompose(R_both)
    print(f"\n  norm(α-only) = {n_only:+.4f}%   norm(both) = {n_both:+.4f}%")
    print("  → If effect tracks α VALUE, R(α-only) ≈ R(both) and the scheme")
    print("    STRING alone (DGLAP renorm) contributes the residual.")

    # ------------------------------------------------------------------
    # 4. Quadrature robustness (128 vs 192)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("[4] Quadrature robustness: R(√s) at n_quad=128 vs 192")
    print("-" * 78)
    sig_alpmz_192 = np.asarray(sigma_nll("ALPMZ", ALPHA_ALPMZ, n_quad=192))
    sig_algmu_192 = np.asarray(sigma_nll("ALGMU", ALPHA_ALGMU, n_quad=192))
    R_algmu_192 = sig_algmu_192 / sig_alpmz_192
    print(f"  {'√s':>6} {'R@128':>12} {'R@192':>12} {'|Δ|':>12}")
    for i, sq in enumerate(GRID):
        print(f"  {sq:6.1f} {R_algmu[i]:12.8f} {R_algmu_192[i]:12.8f} "
              f"{abs(R_algmu[i]-R_algmu_192[i]):12.2e}")
    print(f"\n  max|ΔR| (128 vs 192) = "
          f"{np.max(np.abs(R_algmu - R_algmu_192)):.2e}  "
          f"(must be < 1e-4 → physics, not quadrature)")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
