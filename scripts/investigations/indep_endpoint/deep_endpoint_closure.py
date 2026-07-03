"""Indep-chain deep-endpoint closure: is the omx<1e-15 norm_nll substitution
hiding the same genuine NLL soft enhancement the BFS chain fixed in 9a8862b?

Background. The BFS-side 2026-07-02 review established that eMELA's code_pdf
is healthy to omx <= 1e-60 and that its 'drift' beyond the analytic norm_nll
constant is the GENUINE NLL soft exponent (integrand ~ omx^(-4.3e-4), i.e.
u^(-0.007) in the BFS u=(1-x)^(beta/2) variable); replacing it with the LL+exp
constant cost -0.19% per-leg radiator mass / -0.45% sigma_obs. The indep side
(isr_beta._per_leg_emela_nll / _per_leg_grid_nll and the isr_lumi norm_nll
V->0 anchor) STILL carries that substitution, justified by a 2026-06-03
analysis that read the same drift as an eMELA x->1 artifact.

This script measures, on the production indep configuration
(NLL, alpha(M_Z)=1/128.943, DELTA/ALPMZ, mu_F = sqrt(s)):

 (a) the per-leg radiator MASS at sqrt(s)=161: grid+anchor path vs direct
     eMELA+anchor vs direct eMELA with NO substitution (every node queried,
     omx passed explicitly — the BFS-fix construction), at n_quad=128/256;
     plus the node-by-node code_pdf/norm_nll drift down to omx~1e-66.
 (b) the resulting sigma_obs shift on the indep chain (2-D direct-eMELA
     convolution of the MoCaNLO pure-WW sigma-hat) at 157.5/161/162.5.

The in-process radiator cache is cleared between variants and the DISK cache
is disabled for this process (so patched radiators can never poison the
shared production cache).

Usage: WW_ISR_RADIATOR_CACHE= PYTHONPATH=.:$PYTHONPATH \
       python3 scripts/investigations/indep_endpoint/deep_endpoint_closure.py

2026-07-03 POSTSCRIPT: the fix proposed by this measurement is now APPLIED
(_per_leg_emela_nll queries code_pdf at every node; the grid path continues
log-linearly below the deep grid edge; isr_lumi absorbs the drift into its
Jacobi weights).  Re-run as the ACCEPTANCE TEST: part (a) grid/direct/TRUE
must agree (TRUE per-leg mass 0.999642(1) at 161/n128, grid vs direct ≤1 ppm)
and part (b) shifts must collapse to ~0 (the patched TRUE builder is now
identical to production).
"""

import math
import os
import sys

os.environ["WW_ISR_RADIATOR_CACHE"] = ""      # never write patched radiators

import numpy as np

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep import isr_emela_grid as EG
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS
from framework.process.ww.indep.generator_mocanlo import PROD_EMELA_GRID
from framework.process.ww.indep.partonic_grid import load_grids
from framework.process.ww.xsec_calculator import emela_wrapper as _emela

ALPHA = isr_beta.ALPHA_MZ_EMELA


def cfg_direct(n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", n_quad=n_quad)


def cfg_grid(n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ",
                              emela_grid=PROD_EMELA_GRID, n_quad=n_quad)


def _per_leg_emela_nll_TRUE(cfg, be, norm, x_vals, one_minus_x, jac_NS, sqrt_s):
    """The BFS-fix construction: query code_pdf at EVERY node, omx explicit,
    no analytic substitution anywhere."""
    alpha = cfg.resolved_alpha()
    _emela.initialize(pert_order="NLL", fac_scheme=cfg.emela_fac_scheme,
                      ren_scheme=cfg.emela_ren_scheme, alpha=alpha)
    Q = cfg.mu_F(sqrt_s)
    per_leg = np.empty_like(x_vals)
    for i in range(len(x_vals)):
        omx_i = float(one_minus_x[i])
        x_i = float(x_vals[i])
        x_q = x_i if x_i < 1.0 else 1.0     # x underflows to 1.0 deep in
        per_leg[i] = _emela.code_pdf(x_q, omx_i, Q) / max(x_q, 1e-300) \
            * float(jac_NS[i])
    return per_leg


def perleg_mass(cfg, sq, mode):
    """Per-leg radiator mass = sum(w * per_leg) for one variant."""
    be, bs, _bh = cfg.betas(sq)
    norm = isr_beta._radiator_norm(be, bs)
    u, w, x_vals, omx, jac = isr_beta._endpoint_grid(be, cfg.x_min, cfg.n_quad)
    if mode == "grid":
        pl = isr_beta._per_leg_grid_nll(cfg, be, norm, x_vals, omx, jac, sq)
    elif mode == "direct":
        pl = isr_beta._per_leg_emela_nll(cfg, be, norm, x_vals, omx, jac, sq)
    elif mode == "true":
        pl = _per_leg_emela_nll_TRUE(cfg, be, norm, x_vals, omx, jac, sq)
    else:
        raise ValueError(mode)
    return float(np.sum(w * pl))


def part_a():
    sq = 161.0
    print("=" * 78)
    print("(a) per-leg radiator mass at sqrt(s)=161, production cfg")
    print("=" * 78)
    for n in (128, 256):
        mg = perleg_mass(cfg_grid(n), sq, "grid")
        md = perleg_mass(cfg_direct(n), sq, "direct")
        mt = perleg_mass(cfg_direct(n), sq, "true")
        print(f"  n={n:3d}  grid+anchor={mg:.6f}  direct+anchor={md:.6f}  "
              f"TRUE(no subst)={mt:.6f}")
        print(f"         grid vs direct: {1e6*(mg/md-1):+8.2f} ppm   "
              f"anchor vs TRUE: {1e2*(md/mt-1):+7.4f} %")
    # node-level drift: code_pdf/(norm_nll) at deep omx
    cfg = cfg_direct(128)
    be, bs, _bh = cfg.betas(sq)
    norm = isr_beta._radiator_norm(be, bs)
    norm_nll = isr_beta._norm_nll_endpoint(be, norm, cfg.resolved_alpha())
    _emela.initialize(pert_order="NLL", fac_scheme="DELTA",
                      ren_scheme="ALPMZ", alpha=cfg.resolved_alpha())
    print(f"\n  node-level drift vs the anchored constant norm_nll={norm_nll:.6f}:")
    for omx in (1e-9, 1e-12, 1e-15, 1e-20, 1e-30, 1e-45, 1e-60, 1e-66):
        u = omx ** be
        jac = u ** (1.0 / be - 1.0) / be
        val = _emela.code_pdf(1.0, omx, sq) / 1.0 * jac
        # expected genuine NLL drift ~ omx^(-4.3e-4) (BFS-side reading)
        print(f"    omx={omx:7.0e}  code/norm_nll = {val/norm_nll:8.5f}")


def sigma_obs(cfg, sq_arr, sigma_fn):
    isr_beta._RADIATOR_CACHE.clear()
    return np.array([isr_beta.convolve_2leg(float(s), sigma_fn, cfg)
                     for s in np.atleast_1d(sq_arr)])


def part_b():
    print("\n" + "=" * 78)
    print("(b) sigma_obs shift, indep 2-D direct-eMELA chain, pure-WW MoCaNLO")
    print("=" * 78)
    grids = load_grids()
    fns = {ch: grids[(ch, "nominal")].nlo_fn() for ch in PURE_WW_WEIGHTS}

    def sig_hat(sqrt_shat):
        tot = None
        for ch, wt in PURE_WW_WEIGHTS.items():
            v = wt * np.asarray(fns[ch](sqrt_shat), dtype=float)
            tot = v if tot is None else tot + v
        return tot

    sq = np.array([157.5, 161.0, 162.5])
    base = sigma_obs(cfg_direct(128), sq, sig_hat)
    # patch in the TRUE per-leg builder, clear cache, recompute
    orig = isr_beta._per_leg_emela_nll
    try:
        isr_beta._per_leg_emela_nll = _per_leg_emela_nll_TRUE
        true = sigma_obs(cfg_direct(128), sq, sig_hat)
    finally:
        isr_beta._per_leg_emela_nll = orig
        isr_beta._RADIATOR_CACHE.clear()
    for s, b, t in zip(sq, base, true):
        print(f"  sqrt_s={s:6.1f}  anchor={b:.6f} fb  TRUE={t:.6f} fb  "
              f"shift={(t/b-1)*100:+.4f} %")


if __name__ == "__main__":
    part_a()
    part_b()
    sys.exit(0)
