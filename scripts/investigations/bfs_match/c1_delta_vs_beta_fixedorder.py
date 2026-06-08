"""Fixed-order check: what is the EXACT O(α) C₁ the DELTA radiator needs, and how
does it differ from the code's β-scheme (LO_beta) C₁?

Source (e-print, verified):
  Frixione 1909.03886 Eq. G1sol2 — O(α) ePDF (units α/2π) in a generic scheme:
     Γ_ee^[1](z,μ²) = [ (1+z²)/(1-z)( log(μ²/m²) − 2log(1-z) − 1 ) ]_+ + K_ee(z)
  Frixione 2105.06688 Eq. Kdelz — DELTA choice:
     K_ee^(Δ)(z)     = [ (1+z²)/(1-z)( 2log(1-z) + 1 ) ]_+
  ⇒ DELTA O(α) ePDF  = (α/2π) [ (1+z²)/(1-z) ]_+ · log(μ²/m²)     (the −2log(1-z)−1 cancel)

Code's β C₁ kernel (isr_beta.oalpha_isr_subtraction, LO_beta, all β=(α/π)(2L−1),
L=log(μ_F/m_e)):
     D₁^β(x) = β [ 1/(1-x) ]_+ + (3/4)β δ(1-x) − (β/2)(1+x)
             = (α/2π)(2L−1) [ (1+x²)/(1-x) ]_+        (algebra; 2L = log(μ_F²/m²))

So the code's C₁ and the exact DELTA C₁ share the SAME kernel [(1+x²)/(1-x)]_+ and
differ ONLY by the scalar prefactor:  (2L−1)  vs  (2L).
  ⇒ exact DELTA C₁ = code C₁ × 2L/(2L−1)   — a near-uniform rescale, NOT a λ₁ soft
    constant and NOT an independent x-shape.

This script confirms that numerically: (1) the code's oalpha_isr_subtraction equals
(α/2π)(2L−1)∫[(1+x²)/(1-x)]_+ σ̂_Born; (2) the DELTA correction is the +1/(2L−1)
rescale; (3) it is smooth in √s (normalisation-like) → bounds the m_W impact.
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

M_E = ib.M_E
ALPHA = 1.0 / 128.943
PI = math.pi


def delta_c1(sqrt_s, born_fn, cfg):
    """Exact DELTA-scheme O(α) C₁: same as the code's oalpha_isr_subtraction but
    with the β prefactor (2L−1)→(2L) (the DELTA scheme drops the '−1' finite term).
    Implemented by scaling the kernel pieces: build C₁ with a cfg whose β carries 2L."""
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)
    x_nodes, x_w = ib._quad_nodes(cfg.n_quad, cfg.x_min, 1.0)
    log_omx_min = math.log(1.0 - cfg.x_min)
    for idx, sq in enumerate(sqrt_s_arr):
        mu_F = cfg.mu_F(float(sq))
        twoL = 2.0 * math.log(mu_F / cfg.m_e)          # = log(mu_F^2/m_e^2)
        b_delta = (cfg.resolved_alpha() / PI) * twoL   # β with NO −1  (DELTA)
        s_full = float(sq)
        sig_s = float(np.asarray(born_fn(np.array([s_full])), dtype=float)[0])
        sqrt_shat = np.sqrt(x_nodes) * s_full
        sig_x = np.asarray(born_fn(sqrt_shat), dtype=float)
        plus_int = np.sum(x_w * (sig_x - sig_s) / (1.0 - x_nodes))
        hard_int = np.sum(x_w * (1.0 + x_nodes) * sig_x)
        out[idx] = 2.0 * (0.75 * b_delta * sig_s
                          + b_delta * sig_s * log_omx_min
                          + b_delta * plus_int
                          - 0.5 * b_delta * hard_int)
    return out


def main():
    grids = pg.load_grids(scheme_alpha="gf", lepton_cut=None)
    keys = sorted(grids)
    chosen = next((k for k in keys if "nominal" in k[1].lower()), keys[len(keys)//2])
    grid = grids[chosen]
    born = grid.born_fn()
    print(f"channel/varpoint = {chosen};  √ŝ grid [{grid.ecm.min():.1f},{grid.ecm.max():.1f}]")

    cfg = ib.ISRConfig(alpha=ALPHA, nll=True, emela_fac_scheme="DELTA",
                       emela_ren_scheme="ALPMZ")

    sqrt_s = np.array([157.0, 159.0, 161.0, 162.5, 163.0])
    c1_beta = ib.oalpha_isr_subtraction(sqrt_s, born, cfg)   # code's β C₁
    c1_delta = delta_c1(sqrt_s, born, cfg)                   # exact DELTA C₁
    born_vals = born(sqrt_s)

    print("\n  √s     L=ln(μ/m_e)   2L/(2L−1)    C₁_β [fb]    C₁_Δ [fb]   "
          "ΔC₁/C₁_β    ΔC₁/σ̂_Born")
    for i, s in enumerate(sqrt_s):
        L = math.log(s / M_E)
        predicted = (2*L) / (2*L - 1.0)
        rel = c1_delta[i]/c1_beta[i] - 1.0 if c1_beta[i] else float("nan")
        dC1_over_born = (c1_delta[i]-c1_beta[i])/born_vals[i] if born_vals[i] else float("nan")
        print(f"  {s:6.2f}   {L:8.4f}    {predicted:.5f}    {c1_beta[i]:9.4f}   "
              f"{c1_delta[i]:9.4f}   {rel:+.5f}    {dC1_over_born:+.3e}")

    print("\n  → If ΔC₁/C₁_β tracks 2L/(2L−1)−1 ≈ +4.1%% and is flat in √s, the exact")
    print("    DELTA C₁ is a near-uniform rescale of a small, smooth subtraction —")
    print("    normalisation-like, NOT a λ₁ soft constant and NOT a steep m_W shape.")
    print("  ΔC₁/σ̂_Born is the size of the per-√s shift relative to Born; its")
    print("    √s-VARIATION (not its mean) is what can bias m_W.")
    span = (c1_delta - c1_beta) / born_vals
    print(f"\n  ΔC₁/σ̂_Born:  mean {span.mean():+.3e}   "
          f"spread(max−min) {span.max()-span.min():.3e}")


if __name__ == "__main__":
    main()
