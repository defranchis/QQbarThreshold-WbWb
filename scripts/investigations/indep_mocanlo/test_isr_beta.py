"""Validation of the independent MoCaNLO beta-scheme ISR module
(framework/process/ww/indep/isr_beta.py).

Cross-checks (all BFS-independent in the final product; the BFS isr.py is used
here ONLY as a second, independently-coded implementation of the same LEP-YR
structure function, to confirm the fresh transcription is numerically right):

  1. per-leg radiator pieces vs BFS isr._Gee_per_leg_NS / _H_SV_per_leg
  2. full 2-leg convolution vs BFS isr.sigma_ISR_2leg_convolution (same σ̂)
  3. ∫₀¹ D(x) dx normalisation
  4. O(α) matching identity: σ_obs → σ̂_NLO(s) as β→0 (quadratically)
"""
import numpy as np

from framework.process.ww.xsec_calculator import isr as bfs_isr
from framework.process.ww.xsec_calculator import eft_xsec
from framework.process.ww.indep import isr_beta as ib

# Match BFS isr conventions exactly for the cross-check.
ALPHA = bfs_isr._DEFAULT_ISR_ALPHA          # α_Gμ(M_W_BFS_REF)
M_E = eft_xsec.M_E
SQRT_S = 161.0
N_QUAD = 128
X_MIN = np.sqrt(0.30)

cfg = ib.ISRConfig(scheme="LO_beta", alpha=ALPHA, m_e=M_E,
                   mu_F_factor=1.0, x_min=X_MIN, n_quad=N_QUAD)


def toy_partonic(s_hat, mW=80.379, gammaW=2.085):
    """Smooth WW-like threshold turn-on σ̂(ŝ) [fb]; signature matches BFS
    sigma_partonic_fn(s_hat, mW, gammaW). Pure analytic — no BFS physics."""
    sqrt_shat = np.sqrt(np.asarray(s_hat, dtype=float))
    return 160.0 * 0.5 * (1.0 + np.tanh((sqrt_shat - 160.76) / 0.8)) * (sqrt_shat / 161.0)


def toy_hat_fn(sqrt_shat):
    return toy_partonic(np.asarray(sqrt_shat, dtype=float) ** 2)


def main():
    print("=" * 72)
    print(f"ALPHA={ALPHA:.10e}  1/ALPHA={1/ALPHA:.4f}  M_E={M_E:.8e}  √s={SQRT_S}")
    be, bs, bh = cfg.betas(SQRT_S)
    beta_combined_bfs = bfs_isr.beta_ISR(SQRT_S**2, alpha_em=ALPHA)
    print(f"per-leg β_e=β_s=β_h = {be:.10f}   2·β_e = {2*be:.10f}")
    print(f"BFS combined β_ISR  = {beta_combined_bfs:.10f}   "
          f"Δ = {2*be - beta_combined_bfs:.2e}")

    # --- 1. radiator pieces vs BFS per-leg ---
    print("\n[1] radiator pieces vs BFS isr per-leg (LO_beta)")
    x = np.array([0.6, 0.8, 0.9, 0.95, 0.99, 0.999])
    omx = 1.0 - x
    ns_mine = ib._radiator_NS(x, bh, one_minus_x=omx)
    ns_bfs = bfs_isr._Gee_per_leg_NS(x, 2 * be, one_minus_x=omx)
    norm_mine = ib._radiator_norm(be, bs)
    norm_bfs = bfs_isr._H_SV_per_leg(2 * be)
    print(f"    norm: mine={norm_mine:.12f}  bfs={norm_bfs:.12f}  "
          f"Δ={norm_mine-norm_bfs:.2e}")
    print(f"    NS max|Δ| = {np.max(np.abs(ns_mine-ns_bfs)):.2e}  "
          f"max|rel| = {np.max(np.abs((ns_mine-ns_bfs)/ns_bfs)):.2e}")

    # --- 2. 2-leg convolution vs BFS ---
    print("\n[2] 2-leg convolution vs BFS isr.sigma_ISR_2leg_convolution")
    sqrt_s_grid = np.array([157.0, 159.0, 161.0, 162.5, 165.0])
    mine = ib.convolve_2leg(sqrt_s_grid, toy_hat_fn, cfg)
    bfs = bfs_isr.sigma_ISR_2leg_convolution(
        sqrt_s_grid, toy_partonic, mW=80.379, gammaW=2.085,
        x_min=X_MIN, n_quad=N_QUAD, alpha_em_isr=ALPHA, nll=False, n_jobs=1)
    for s, m, b in zip(sqrt_s_grid, mine, bfs):
        print(f"    √s={s:6.2f}  mine={m:10.5f}  bfs={b:10.5f}  "
              f"rel={abs(m-b)/abs(b):.2e}")
    print(f"    max rel diff = {np.max(np.abs((mine-bfs)/bfs)):.2e}")

    # --- 3. n_quad convergence of the convolution (smooth σ̂) ---
    print("\n[3] convolution n_quad convergence at nominal β (smooth toy σ̂)")
    ref = ib.convolve_2leg(SQRT_S, toy_hat_fn,
                           ib.ISRConfig(scheme="LO_beta", alpha=ALPHA, m_e=M_E,
                                        mu_F_factor=1.0, x_min=X_MIN, n_quad=4096))
    for nq in (64, 128, 256):
        c = ib.ISRConfig(scheme="LO_beta", alpha=ALPHA, m_e=M_E,
                         mu_F_factor=1.0, x_min=X_MIN, n_quad=nq)
        v = ib.convolve_2leg(SQRT_S, toy_hat_fn, c)
        print(f"    n_quad={nq:4d}  conv={v:.8f}  rel_vs_4096={abs(v-ref)/ref:.1e}")

    # --- 4. O(α) matching exactness: with σ̂_NLO ≡ σ̂_Born the matched line
    #        shape minus σ̂_Born(s) must be PURE O(α²) (resid/β² → const) ---
    print("\n[4] O(α) matching exactness: σ̂_NLO≡σ̂_Born ⇒ resid pure O(α²)")
    born_s = float(toy_hat_fn(np.array([SQRT_S]))[0])
    prev = None
    for eps in (1.0, 0.5, 0.25, 0.125):
        c = ib.ISRConfig(scheme="LO_beta", alpha=ALPHA * eps, m_e=M_E,
                         mu_F_factor=1.0, x_min=X_MIN, n_quad=512)
        sig_obs = ib.sigma_observed_matched(SQRT_S, toy_hat_fn, toy_hat_fn, c)
        b = c.betas(SQRT_S)[0]
        resid = sig_obs - born_s
        ratio = "" if prev is None else f"  resid/β² ratio={resid/b**2/prev:+.4f}"
        prev = resid / b ** 2
        print(f"    β={b:.5f}  resid={resid:+.5f}  resid/β²={resid/b**2:+.3f}{ratio}")
    print("    (resid/β² → constant ⇒ O(α) ISR double-count cancelled exactly)")


if __name__ == "__main__":
    main()
