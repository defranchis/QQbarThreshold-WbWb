"""Convergence study for sigma_ISR_2leg_convolution.

Quantifies the impact of (n_quad, x_min) on σ_obs for both LL (analytic
β-scheme) and NLL (eMELA CodePdf) paths.

Reference σ for the convergence ratio is the highest-resolution run
(n_quad_ref, x_min_ref).  Metrics reported:
  - max relative deviation across the √s grid
  - second-difference noise (relevant for the m_W fit)
  - approximate Δm_W via local slope dσ/dm_W

All scenarios use Born-level partonic σ (no NLO/NNLO/δ_QCD/anchor) to
isolate the ISR-quadrature behaviour from physics-side dependencies.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np

# Suppress eMELA banner before import
_dev = os.open(os.devnull, os.O_WRONLY)
_sav = os.dup(1)
os.dup2(_dev, 1)
from framework.process.ww.xsec_calculator import emela_wrapper as em
from framework.process.ww.xsec_calculator.isr import (
    sigma_ISR_2leg_convolution, _DEFAULT_ISR_ALPHA,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq, M_W_DEFAULT, GAMMA_W_DEFAULT,
)
em.initialize("NLL", "DELTA", "ALGMU", _DEFAULT_ISR_ALPHA)
os.dup2(_sav, 1); os.close(_sav); os.close(_dev)

MW = M_W_DEFAULT
GW = GAMMA_W_DEFAULT
BORN_KW = dict(
    include_NLO_hard_decay=False,
    include_BFS_NNLO=False,
    apply_delta_QCD=False,
    apply_whizard_anchor=False,
    br_convention="pdg-constant",
)

# √s grid — coarse for the convergence comparisons themselves, fine for
# the second-difference noise diagnostic.
SQ_COARSE = np.array([157.0, 158.0, 161.0, 162.5, 163.0, 165.0, 170.0])
SQ_FINE   = np.arange(160.0, 163.05, 0.10)        # 31 points, 100 MeV step
# Two points around the fit minimum, for ∂σ/∂m_W estimation via finite diff
DELTA_MW  = 0.001   # 1 MeV
SQ_SLOPE  = np.array([162.0])

REF_NQUAD = 256
REF_XMIN  = float(np.sqrt(0.02))   # z_min = 0.02 → tighter than production

def run(sqgrid, n_quad, x_min, *, nll):
    return sigma_ISR_2leg_convolution(
        sqgrid, sigma_partonic_munuqq,
        mW=MW, gammaW=GW, x_min=x_min, n_quad=n_quad,
        nll=nll, n_jobs=12,
        emela_ren_scheme="ALGMU",  # match module-level em.initialize above
        **BORN_KW,
    )

def slope_dsigma_dmw(n_quad, x_min, *, nll):
    """∂σ/∂m_W at √s=162 GeV via central difference, in pb/GeV."""
    kw = dict(
        sigma_partonic_fn=sigma_partonic_munuqq,
        gammaW=GW, x_min=x_min, n_quad=n_quad,
        nll=nll, n_jobs=1,
        emela_ren_scheme="ALGMU",  # match module-level em.initialize above
        **BORN_KW,
    )
    sig_p = sigma_ISR_2leg_convolution(SQ_SLOPE, mW=MW + DELTA_MW, **kw)
    sig_m = sigma_ISR_2leg_convolution(SQ_SLOPE, mW=MW - DELTA_MW, **kw)
    return float((sig_p[0] - sig_m[0]) / (2 * DELTA_MW))

def second_difference_noise(sig):
    """L∞ norm of (σ_i - 2 σ_i+1 + σ_i+2) / σ_i+1.  Sensitive to
    quadrature jitter that survives the smooth physics."""
    s = np.asarray(sig)
    d2 = s[:-2] - 2.0 * s[1:-1] + s[2:]
    return float(np.max(np.abs(d2 / s[1:-1])))

print("=" * 78)
print("Convergence study: sigma_ISR_2leg_convolution")
print(f"  ref grid: n_quad={REF_NQUAD}, x_min=√0.02≈{REF_XMIN:.4f}")
print("=" * 78)

for mode in ("LL", "NLL"):
    nll = (mode == "NLL")
    print(f"\n--- {mode} ---")
    t0 = time.time()
    sig_ref_coarse = run(SQ_COARSE, REF_NQUAD, REF_XMIN, nll=nll)
    sig_ref_fine   = run(SQ_FINE,   REF_NQUAD, REF_XMIN, nll=nll)
    slope_ref      = slope_dsigma_dmw(REF_NQUAD, REF_XMIN, nll=nll)
    print(f"  reference computed in {time.time()-t0:5.1f} s   "
          f"dσ/dm_W @ 162 GeV = {slope_ref*1e3:+.4f} fb/MeV")

    # --- n_quad scan at x_min = production (√0.10) ---
    print("\n  n_quad scan (x_min = √0.10 = 0.3162, production):")
    print(f"    {'n_quad':>7}  {'max|Δσ/σ|':>11}  {'2nd-diff':>11}  "
          f"{'Δm_W [keV]':>12}  {'wall [s]':>9}")
    for nq in (32, 48, 64, 96, 128, 192, 256):
        t0 = time.time()
        sig_c = run(SQ_COARSE, nq, float(np.sqrt(0.10)), nll=nll)
        sig_f = run(SQ_FINE,   nq, float(np.sqrt(0.10)), nll=nll)
        slope = slope_dsigma_dmw(nq, float(np.sqrt(0.10)), nll=nll)
        wall = time.time() - t0
        max_rel = np.max(np.abs(sig_c / sig_ref_coarse - 1))
        d2_noise = second_difference_noise(sig_f)
        # Δm_W estimate: how far off would m_W be if σ shifts by
        # (sig_at_this_nq - sig_ref) at √s=162.5?  Use ref slope.
        idx_162 = np.argmin(np.abs(SQ_COARSE - 162.5))
        dsig = sig_c[idx_162] - sig_ref_coarse[idx_162]
        dmw_keV = abs(dsig / slope_ref) * 1e6   # GeV → keV
        print(f"    {nq:>7d}  {max_rel:11.2e}  {d2_noise:11.2e}  "
              f"{dmw_keV:12.2f}  {wall:9.2f}")

    # --- x_min scan at n_quad = production (128) ---
    print("\n  x_min scan (n_quad = 128, production):")
    print(f"    {'z_min':>7}  {'x_min':>7}  {'max|Δσ/σ|':>11}  "
          f"{'Δm_W [keV]':>12}  {'wall [s]':>9}")
    for zmin in (0.30, 0.20, 0.10, 0.05, 0.02, 0.01):
        xmin = float(np.sqrt(zmin))
        t0 = time.time()
        sig_c = run(SQ_COARSE, 128, xmin, nll=nll)
        wall = time.time() - t0
        max_rel = np.max(np.abs(sig_c / sig_ref_coarse - 1))
        idx_162 = np.argmin(np.abs(SQ_COARSE - 162.5))
        dsig = sig_c[idx_162] - sig_ref_coarse[idx_162]
        dmw_keV = abs(dsig / slope_ref) * 1e6
        print(f"    {zmin:7.3f}  {xmin:7.4f}  {max_rel:11.2e}  "
              f"{dmw_keV:12.2f}  {wall:9.2f}")

print("\n" + "=" * 78)
print("Done.")
