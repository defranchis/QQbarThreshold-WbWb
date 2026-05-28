"""Validation of eMELA NLL ISR integration.

Three tests:

1. Per-leg PDF comparison: eMELA LLPDF(beta=1) vs our _Gee_per_leg analytic
   at several x values.  Checks that the ctypes binding and scheme match.

2. 2-leg convolution closure: sigma_ISR_2leg_convolution(nll=False) vs the
   same integral driven by eMELA LLPDF(1).  Numerically equivalent → checks
   that the new per-leg code path gives the right quadrature.

3. NLL correction table: σ_NLL / σ_LL − 1 across 157–165 GeV, compared to
   the expected ~1% level from BCFS arXiv:1911.12040 (Section 8).

Usage:
  cd <WW_threshold root>
  source setup.sh
  python scripts/investigations/nll_isr/validate_emela_isr.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import os

# Suppress the eMELA banner on initialization (redirects C-level stdout)
_devnull_fd: int | None = None
_saved_fd:   int | None = None

def _stdout_off():
    global _devnull_fd, _saved_fd
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_fd   = os.dup(1)
    os.dup2(_devnull_fd, 1)

def _stdout_on():
    if _saved_fd is not None:
        os.dup2(_saved_fd, 1)
        os.close(_saved_fd)
        os.close(_devnull_fd)

_stdout_off()
from framework.process.ww.xsec_calculator import emela_wrapper as em
from framework.process.ww.xsec_calculator.isr import (
    beta_ISR, _H_SV_per_leg, _Gee_per_leg_NS, _endpoint_substitution,
    sigma_ISR_2leg_convolution, _DEFAULT_ISR_ALPHA, _SAFE_FLOOR,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq, M_W_DEFAULT, GAMMA_W_DEFAULT,
)
em.initialize("NLL", "DELTA", "ALGMU", _DEFAULT_ISR_ALPHA)
_stdout_on()

SEP = "=" * 72

# ---------------------------------------------------------------------------
# Test 1: per-leg PDF comparison at selected x values
# ---------------------------------------------------------------------------

print(SEP)
print("TEST 1  Per-leg PDF: eMELA LLPDF(beta=1) vs analytic at √s=161 GeV")
print(SEP)

sq = 161.0
s  = sq**2
beta = beta_ISR(s, alpha_em=_DEFAULT_ISR_ALPHA)
H_sv = _H_SV_per_leg(beta)

print(f"  β = {beta:.6f}   H_SV_per_leg = {H_sv:.8f}\n")
print(f"  {'x':>8}  {'omx':>10}  {'eMELA_LL [xD]':>15}  {'analytic_SV+NS [xD]':>20}  {'rel diff [%]':>14}")

# x values from far (0.5) to near-1 (0.999); avoid x=1
x_test = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
for x in x_test:
    omx = 1.0 - x
    xD_emela = em.ll_pdf(1, x, omx, sq)         # x * D_LL_beta(x, Q)
    # analytic: (beta/2)(1-x)^(beta/2-1)*H_sv + NS  → multiply by x
    sv_part  = (beta / 2.0) * omx**(beta / 2.0 - 1.0) * H_sv
    ns_part  = _Gee_per_leg_NS(x, beta, one_minus_x=omx)
    analytic = x * (sv_part + ns_part)
    rel = (xD_emela / analytic - 1.0) * 100.0 if abs(analytic) > 0 else float("nan")
    print(f"  {x:>8.4f}  {omx:>10.1e}  {xD_emela:>15.8f}  {analytic:>20.8f}  {rel:>+14.4f}")

print()

# ---------------------------------------------------------------------------
# Test 2: 2-leg convolution — eMELA LL per-leg vs isr.py LL+exp
# ---------------------------------------------------------------------------

print(SEP)
print("TEST 2  2-leg σ_obs: eMELA-LL per-leg vs isr.py LL+exp (should agree)")
print(SEP)

def sigma_2leg_emela_ll(sqrt_s_arr, x_min=0.55, n_quad=32):
    """2-leg convolution using eMELA LLPDF(beta=1) for the per-leg weight."""
    out = np.zeros(len(sqrt_s_arr))
    for k, sq in enumerate(sqrt_s_arr):
        s = sq**2
        beta = beta_ISR(s, alpha_em=_DEFAULT_ISR_ALPHA)
        u, w, x_vals, one_minus_x, jac_NS = _endpoint_substitution(
            beta / 2.0, x_min, n_quad)
        # Per-leg in u-space: D_LL(x)*|dx/du| = (xD_LL / x) * jac_NS
        H_sv = _H_SV_per_leg(beta)
        per_leg = np.empty_like(x_vals)
        for i in range(len(x_vals)):
            omx_i = float(one_minus_x[i])
            if omx_i < 1e-15:
                per_leg[i] = H_sv
            else:
                xD = em.ll_pdf(1, float(x_vals[i]), omx_i, sq)
                per_leg[i] = xD / float(x_vals[i]) * float(jac_NS[i])
        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        sigma_hat = np.asarray(
            sigma_partonic_munuqq(
                (X1 * X2 * s).ravel(), M_W_DEFAULT, GAMMA_W_DEFAULT),
            dtype=float,
        ).reshape(X1.shape)
        wt = w * per_leg
        out[k] = np.einsum("i,j,ij->", wt, wt, sigma_hat)
    return out

sq_grid = np.array([158., 159., 160., 161., 162., 163., 165.])
sig_ll_isr  = sigma_ISR_2leg_convolution(sq_grid, sigma_partonic_munuqq, nll=False)
sig_ll_emela = sigma_2leg_emela_ll(sq_grid)

print(f"  {'√s':>6}  {'isr.py LL [pb]':>14}  {'eMELA LL [pb]':>14}  {'diff [%]':>10}")
for sq, a, b in zip(sq_grid, sig_ll_isr, sig_ll_emela):
    d = (b/a - 1)*100 if a > 0 else float("nan")
    print(f"  {sq:>6.1f}  {a:>14.6f}  {b:>14.6f}  {d:>+10.4f}")

max_diff = np.max(np.abs(sig_ll_emela / sig_ll_isr - 1)) * 100
print(f"\n  Max |diff| = {max_diff:.4f} %  (target < 0.5 %)")
print()

# ---------------------------------------------------------------------------
# Test 3: NLL correction table across WW threshold
# ---------------------------------------------------------------------------

print(SEP)
print("TEST 3  NLL / LL correction: σ_NLL / σ_LL − 1  (expect +0.5% to +2%)")
print("        (BCFS arXiv:1911.12040 §8: NLL adds a positive correction to")
print("         the Sudakov exponent, slightly restoring σ vs LL suppression)")
print(SEP)

sq_fine = np.array([157., 158., 159., 160., 161., 162., 163., 165., 170.])
sig_ll_f  = sigma_ISR_2leg_convolution(sq_fine, sigma_partonic_munuqq, nll=False)
sig_nll_f = sigma_ISR_2leg_convolution(sq_fine, sigma_partonic_munuqq, nll=True)

print(f"  {'√s':>6}  {'σ_LL [pb]':>12}  {'σ_NLL [pb]':>12}  {'NLL/LL−1 [%]':>14}")
for sq, sll, snll in zip(sq_fine, sig_ll_f, sig_nll_f):
    r = (snll/sll - 1)*100 if sll > 0 else float("nan")
    print(f"  {sq:>6.1f}  {sll:>12.6f}  {snll:>12.6f}  {r:>+14.4f}")

print()

# ---------------------------------------------------------------------------
# Test 4: α_em_isr scheme variation (ALGMU vs ALPMZ)
# ---------------------------------------------------------------------------

print(SEP)
print("TEST 4  Scheme variation: ALGMU (α_Gμ≈1/132.1) vs ALPMZ (α(MZ)≈1/128.9)")
print("        difference = alpha_em_isr nuisance in PARAM_UNC")
print(SEP)

ALPHA_MZ = 1.0 / 128.9

_stdout_off()
em.initialize("NLL", "DELTA", "ALPMZ", ALPHA_MZ)   # reinit for ALPMZ
_stdout_on()

sig_nll_alpmz = sigma_ISR_2leg_convolution(
    sq_fine, sigma_partonic_munuqq, nll=True,
    alpha_em_isr=ALPHA_MZ)

# Restore ALGMU for subsequent use
_stdout_off()
em.initialize("NLL", "DELTA", "ALGMU", _DEFAULT_ISR_ALPHA)
_stdout_on()

print(f"  {'√s':>6}  {'ALGMU [pb]':>12}  {'ALPMZ [pb]':>12}  {'ALPMZ/ALGMU−1 [%]':>18}")
for sq, a, b in zip(sq_fine, sig_nll_f, sig_nll_alpmz):
    d = (b/a - 1)*100 if a > 0 else float("nan")
    print(f"  {sq:>6.1f}  {a:>12.6f}  {b:>12.6f}  {d:>+18.4f}")

print()
print("All tests complete.")
