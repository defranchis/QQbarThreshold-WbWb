#!/usr/bin/env python3
"""Faithful luminosity ISR convolution + the diagnosis of the prototype residual
(HANDOFF_lumi_convolution_2026-06-17.md).  CONCLUSION (validated below): the 1D
luminosity reformulation IS faithful to the 2D ``convolve_2leg`` to < 0.05 % and is
SMOOTHER than the 2D; the prototype's ~0.1 % residual was (a) the 2D's OWN
Gauss-Legendre non-convergence and (b) under-resolved lumi quadratures, NOT a
physics/convention error.

WHAT THE 2D INTEGRATES.  ``convolve_2leg`` = INT INT D(x1)D(x2) sig_hat dx1 dx2 with
the per-leg radiator D(x) = x*D / x from the eMELA NLL ePDF grid for
omx=1-x >= OMX_FLOOR(1e-15), and the analytic soft+virtual constant ``norm_nll``
for omx < 1e-15.  Per the eMELA README, eMELA's CodePdf is RELIABLE for all
omx >~ 1e-16 (numerical above 1-x~1e-8, its own analytic asymptotics below), so the
grid IS the genuine NLL ePDF down to the endpoint -- the measured "+0.72 % at
omx=1e-15 vs norm_nll" is real NLL structure, NOT an artefact.  ``norm_nll`` is only
the patch for omx<1e-15, where x=1-omx underflows to 1.0 in float64 and CodePdf
breaks.  => the FAITHFUL per-leg density is the eMELA GRID (soft_mode='grid'); the
``norm_nll``-floor variant (soft_mode='nll') DISCARDS real eMELA values in
[1e-15,1e-8] and is LESS faithful (~-0.14 %).

THE LUMINOSITY.  sig_obs(s) = INT L(z) sig_hat(sqrt(z)*sqrt(s)) dz, z=x1*x2; in
V=-ln z the luminosity is the radiator self-convolution L_V(V)=INT rho(v)rho(V-v)dv,
rho(v)=x*D, factored as L_V = V^(2be-1) * L-tilde(V) with L-tilde smooth and
integrated by Gauss-Jacobi(be-1,be-1).  The outer V-integral uses a single
Gauss-Jacobi(2be-1,0) panel on [0,V_top], V_top=2 ln(sqrt(s)/156): the sig_hat
grid-edge step at sqrt(shat)=156 is the integration LIMIT (never interior), so the
line shape has NO Gauss-Legendre ripple -- this is WHY the lumi is smoother than the
2D, whose plain GL straddles that edge.

CONVERGENCE (validated, lnuqq gf-NLL vs ripple-free 2D mean):
  * smoothness max|2nd-diff| = 1.35e-3, < 2D-truth 1.42e-3 and << 2D-128 7.08e-3;
  * grid-soft lumi -> 2D mean as n_jac->400 & n_out->384 (mean resid ~+35 ppm);
    n_jac controls how deep into the soft grid the self-conv samples (the 2D reaches
    omx~1e-15), n_out the outer V-integral.  Residual shape ~200 ppm is at the
    2D's OWN ripple floor (the 2D is only defined to ~tens of ppm even at n_quad=4096).

``soft_mode='grid'`` (default) = FAITHFUL (eMELA grid);
``soft_mode='nll'``  = norm_nll floor below omx_trust (the LESS-faithful variant,
kept to quantify the deep-soft sensitivity).

2026-07-03 UPDATE (deep-endpoint fix, mirror of BFS 9a8862b): the production
chain no longer substitutes norm_nll anywhere — the genuine NLL soft drift
rho~ ~ v^(-delta) (delta≈4e-4) is absorbed into the Gauss-Jacobi weights
(beta' = beta_e - delta, rho^ = rho~ * v^delta log-flat at v->0), because the
inner nodes only reach omx~3e-8 and a polynomial rule cannot carry the drift
below its deepest node.  FaithfulLumi mirrors that construction (soft_mode=
'grid'), so it remains the independent same-physics harness for
validate_lumi_production.py; the numbers quoted ABOVE in this docstring are
the historical pre-fix ones.

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/lumi_faithful.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from scipy.interpolate import UnivariateSpline  # noqa: E402
from scipy.special import roots_jacobi  # noqa: E402
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep import isr_emela_grid as EG  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
from framework.process.ww.xsec_calculator.isr import LAMBDA1_NF0  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

ALPHA = isr_beta.ALPHA_MZ_EMELA
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
SIGMA_GRID_LO = 156.0


def prod_nll(n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID,
                              n_quad=n_quad)


def _jac01(n, a, b):
    x, w = roots_jacobi(n, b, a)
    return 0.5 * (x + 1.0), w / 2.0 ** (a + b + 1)


def norm_nll_of(cfg, Q):
    be, bs, bh = cfg.betas(Q)
    norm = isr_beta._radiator_norm(be, bs)
    return be, norm * math.exp(be * (cfg.resolved_alpha() / math.pi)
                               * (LAMBDA1_NF0 / 4.0))


class FaithfulLumi:
    """L-tilde(V) self-convolution at one Q, via Gauss-Jacobi(be-1,be-1).

    Per-leg rho~(v) = x*D * v^(1-be):
      * soft_mode='grid' (DEFAULT, FAITHFUL): rho~ = grid.xfxQ * v^(1-be) for all
        sampled omx -- the genuine eMELA NLL ePDF, IDENTICAL to what the 2D
        integrates (eMELA's CodePdf is reliable down to omx~1e-16; see module
        docstring).  Converging the self-conv (n_jac) samples the soft grid as deep
        as the 2D reaches; -> matches the 2D to ~tens of ppm.
      * soft_mode='nll': floors rho~ to the analytic soft+virtual norm_nll*be for
        omx < omx_trust.  LESS faithful (discards real eMELA grid values in the
        reliable soft region); kept only to quantify the deep-soft sensitivity
        (~-0.14 % vs grid).  Sensitive to omx_trust by design.
    """

    def __init__(self, cfg, Q, soft_mode="grid", n_jac=256, omx_trust=1e-6):
        self.cfg = cfg
        self.Q = float(Q)
        self.grid = EG.load_grid(cfg.emela_grid)
        self.be, self.norm_nll = norm_nll_of(cfg, self.Q)
        self.soft_mode = soft_mode
        self.omx_trust = omx_trust
        self.rt_soft = self.norm_nll * self.be          # analytic soft+virtual rho~
        # 2026-07-03 endpoint fix: absorb the genuine NLL soft drift v^(-delta)
        # into the Jacobi weight (mirror of production isr_lumi._rho_hat_factory).
        self.delta = (self.be - 1.0) - self.grid.deep_slope(self.Q)
        self.be_eff = self.be - self.delta
        self._t, self._wj = _jac01(n_jac, self.be_eff - 1.0, self.be_eff - 1.0)

    def rho_tilde(self, v):
        """rho^(v) = x*D(v) * v^(1-be+delta) — drift-absorbed, log-flat at v->0
        (soft_mode='nll' floors it to the analytic norm_nll*be legacy variant)."""
        v = np.asarray(v, float)
        x = np.exp(-v)
        omx = -np.expm1(-v)
        v_c = np.maximum(v, 1e-300)
        xD = self.grid.xfxQ(x, np.maximum(omx, 1e-300), self.Q)
        rt = xD * v_c ** (1.0 - self.be + self.delta)
        if self.soft_mode == "nll":
            # Legacy substitution expressed in the hatted variables:
            # rho~ -> norm_nll*be  <=>  rho^ -> norm_nll*be * v^delta.
            rt = np.where(omx < self.omx_trust,
                          self.rt_soft * v_c ** self.delta, rt)
        return rt

    def Ltilde(self, V):
        """L-tilde(V) = INT_0^1 t^(be-1)(1-t)^(be-1) rho~(Vt) rho~(V(1-t)) dt."""
        t = self._t
        return float(np.sum(self._wj * self.rho_tilde(V * t)
                            * self.rho_tilde(V * (1.0 - t))))

    def Ltilde_vec(self, Varr):
        Varr = np.atleast_1d(np.asarray(Varr, float))
        t = self._t                                       # (n_jac,)
        Vt = np.outer(Varr, t)                            # (nV, n_jac)
        rt1 = self.rho_tilde(Vt.ravel()).reshape(Vt.shape)
        rt2 = self.rho_tilde((Varr[:, None] * (1.0 - t)).ravel()).reshape(Vt.shape)
        return (rt1 * rt2) @ self._wj


def sigma_obs(sq, sigma_fn, lumi: FaithfulLumi, n_out=64):
    """sigma_obs(sqrt(s)) = INT_0^{V_top} V^(2be'-1) L^(V) sig_hat(e^{-V/2} sqrt(s)) dV."""
    be_eff = lumi.be_eff
    p = 2.0 * be_eff - 1.0
    tj, wj = _jac01(n_out, p, 0.0)                 # INT_0^1 tau^(2be'-1)
    V_top = 2.0 * math.log(sq / SIGMA_GRID_LO)
    V = V_top * tj
    Lt = lumi.Ltilde_vec(V)
    integ = Lt * np.asarray(sigma_fn(np.exp(-V / 2.0) * sq))
    return V_top ** (2.0 * be_eff) * np.sum(wj * integ)


def line_shape(cfg, SQ, soft_mode="nll", n_out=64, n_jac=160, omx_trust=1e-6):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        nlo = grids[(ch, "nominal")].nlo_fn()
        obs = np.array([
            sigma_obs(float(sq), nlo,
                      FaithfulLumi(cfg, float(sq), soft_mode=soft_mode,
                                   n_jac=n_jac, omx_trust=omx_trust),
                      n_out=n_out)
            for sq in SQ])
        tot = tot + wt * obs
    return tot * FB_TO_PB


def line_shape_channel(cfg, SQ, ch="lnuqq", soft_mode="nll", n_out=64,
                       n_jac=160, omx_trust=1e-6):
    grids = load_grids(scheme_alpha="gf")
    nlo = grids[(ch, "nominal")].nlo_fn()
    return np.array([
        sigma_obs(float(sq), nlo,
                  FaithfulLumi(cfg, float(sq), soft_mode=soft_mode,
                               n_jac=n_jac, omx_trust=omx_trust),
                  n_out=n_out)
        for sq in SQ])


def ripple_mean_2leg(cfg_fn, SQ, ch="lnuqq", ns=range(1024, 2049, 128)):
    grids = load_grids(scheme_alpha="gf")
    nlo = grids[(ch, "nominal")].nlo_fn()
    vals = np.array([isr_beta.convolve_2leg(SQ, nlo, cfg_fn(n)) for n in ns])
    return vals.mean(0), vals.std(0)


def main():
    SQ = np.array([157.5, 158.0, 158.5, 159.0, 159.5, 160.0, 160.5, 161.0, 161.5,
                   162.0, 162.5])
    ch = "lnuqq"
    cfg = prod_nll(128)
    print("Faithful (soft-floored) luminosity  (gf NLL, channel=%s) [fb]\n" % ch)

    # ripple-free 2D truth: high-n mean (sem ~ tens of ppm)
    mean, std = ripple_mean_2leg(prod_nll, SQ, ch=ch, ns=range(2048, 4097, 128))
    sem = std / math.sqrt(len(range(2048, 4097, 128)))
    print("2D truth = mean(convolve_2leg, n_quad 2048..4096):")
    print("  sigma:", " ".join("%9.5f" % x for x in mean))
    print("  sem  :", " ".join("%8.0fppm" % (s / m * 1e6) for s, m in zip(sem, mean)))

    print("\n--- outer-integral (n_out) convergence, soft_mode=nll, omx_trust=1e-6 ---")
    for n_out in (32, 64, 128, 256):
        obs = line_shape_channel(cfg, SQ, ch=ch, soft_mode="nll", n_out=n_out)
        d = (obs / mean - 1.0) * 1e6
        print("  n_out=%3d  d_vs_2D(ppm) max|%.0f| mean%+.0f" %
              (n_out, np.max(np.abs(d)), np.mean(d)))

    print("\n--- omx_trust insensitivity (n_out=192, soft_mode=nll) ---")
    for ot in (1e-5, 1e-6, 1e-7, 1e-8):
        obs = line_shape_channel(cfg, SQ, ch=ch, soft_mode="nll", n_out=192, omx_trust=ot)
        d = (obs / mean - 1.0) * 1e6
        print("  omx_trust=%.0e  d_vs_2D(ppm) max|%.0f| mean%+.0f" %
              (ot, np.max(np.abs(d)), np.mean(d)))

    print("\n--- faithful (nll) vs grid-soft, d & normalised SHAPE vs 2D (n_out=192) ---")
    i0 = int(np.argmin(np.abs(SQ - 160.5)))
    for soft_mode in ("grid", "nll"):
        obs = line_shape_channel(cfg, SQ, ch=ch, soft_mode=soft_mode, n_out=192)
        d = (obs / mean - 1.0) * 1e6
        sh = ((obs / obs[i0]) / (mean / mean[i0]) - 1.0) * 1e6
        print(" soft=%-4s" % soft_mode)
        print("   d_vs_2D(ppm) :", " ".join("%+7.0f" % x for x in d))
        print("   SHAPE  (ppm) :", " ".join("%+7.0f" % x for x in sh),
              " max|shape|=%.0f ppm" % np.max(np.abs(sh)))


if __name__ == "__main__":
    main()
