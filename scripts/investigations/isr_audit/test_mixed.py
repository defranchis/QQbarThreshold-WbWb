"""Test MIXED scheme: β_exp = β_S = β = (2α/π)(L-1), β_H = η = (2α/π)L (no -1).
The difference from BETA is the (1+x) coefficient and the β² bracket use η instead of β,
giving slightly more cross section."""
import numpy as np
from scipy.special import gamma as gamma_fn
from process.ww.isr import (
    sigma_ISR_2leg_convolution, beta_ISR, _quad_nodes,
)
from process.ww.eft_xsec import alpha_Gmu, M_E
EULER_GAMMA = 0.5772156649015329

# Override the per-leg NS function to use β_H = η = (2α/π)L
import process.ww.isr as isr_mod

def _Gee_per_leg_NS_MIXED(x, beta_exp, beta_H, one_minus_x=None):
    """MIXED scheme: η in non-singular pieces, β in soft-singular kernel."""
    x = np.asarray(x, dtype=float)
    if one_minus_x is None:
        one_minus_x = np.maximum(1.0 - x, 1e-300)
    one_minus_x = np.asarray(one_minus_x, dtype=float)
    one_minus_x = np.maximum(one_minus_x, 1e-300)
    x_safe = np.maximum(x, 1e-300)
    log1mx = np.log(one_minus_x)
    logx = np.log(x_safe)
    NS_1 = -(beta_H / 4.0) * (1.0 + x)
    NS_2 = -(beta_H ** 2 / 32.0) * (
        (1.0 + 3.0 * x * x) / one_minus_x * logx
        + 4.0 * (1.0 + x) * log1mx
        + 5.0 + x
    )
    out = NS_1 + NS_2
    out = np.where(x > 0.0, out, 0.0)
    if np.ndim(x) == 0:
        return float(out)
    return out

def sigma_2leg_MIXED(sqrt_s, sigma_partonic_fn, mW, gammaW, x_min=0.316, n_quad=32, **kw):
    """2-leg with MIXED scheme: β in kernel, η in NS pieces."""
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)
    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s)  # = (2α/π)(L-1) per leg
        L = np.log(s/M_E**2)
        eta = (2 * alpha_Gmu(mW) / np.pi) * L  # = (2α/π)L per leg (no -1)
        H_sv = np.exp(0.5 * beta * (0.75 - EULER_GAMMA)) / gamma_fn(1.0 + beta/2.0)
        half_b = beta / 2.0
        u_max = (1.0 - x_min) ** half_b
        u, w = _quad_nodes(n_quad, 0.0, u_max)
        one_minus_x = u ** (1.0 / half_b)
        x_vals = 1.0 - one_minus_x
        with np.errstate(over="ignore", invalid="ignore"):
            jac_NS = np.where(u > 1e-300, u**(1.0/half_b - 1.0)/half_b, 0.0)
        NS_vals = _Gee_per_leg_NS_MIXED(x_vals, beta, eta, one_minus_x=one_minus_x)
        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        s_hat = X1 * X2 * s
        sigma_hat = np.asarray(sigma_partonic_fn(s_hat.ravel(), mW, gammaW, **kw),
                              dtype=float).reshape(s_hat.shape)
        W1, W2 = np.meshgrid(w, w, indexing="ij")
        NS_jac = NS_vals * jac_NS
        NS1g, NS2g = np.meshgrid(NS_jac, NS_jac, indexing="ij")
        integrand = (H_sv*H_sv + H_sv*NS2g + NS1g*H_sv + NS1g*NS2g) * sigma_hat
        out[idx] = np.sum(W1*W2*integrand)
    if np.ndim(sqrt_s) == 0:
        return float(out[0])
    return out

from process.ww.eft_xsec import sigma_partonic_munuqq

BFS = {158: 45.64, 161: 108.60, 164: 219.7, 167: 310.2, 170: 378.4}
print("=== BETA vs MIXED scheme vs BFS Table 3 ===")
print(f"{'√s':>5}  {'BFS T3':>9}  {'BETA':>10}  {'MIXED':>10}  {'BETA/BFS':>10}  {'MIXED/BFS':>10}")
for sq in [158.0, 161.0, 164.0, 167.0, 170.0]:
    bfs_val = BFS[int(sq)]
    sig_beta = sigma_ISR_2leg_convolution(np.array([sq]), sigma_partonic_munuqq,
        mW=80.377, gammaW=2.09201, x_min=0.316, n_quad=32,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True)[0] * 1e3
    sig_mixed = sigma_2leg_MIXED(np.array([sq]), sigma_partonic_munuqq,
        mW=80.377, gammaW=2.09201, x_min=0.316, n_quad=32,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True)[0] * 1e3
    print(f"  {sq:4.0f}  {bfs_val:8.3f}  {sig_beta:9.3f}  {sig_mixed:9.3f}   {sig_beta/bfs_val:.5f}   {sig_mixed/bfs_val:.5f}")
