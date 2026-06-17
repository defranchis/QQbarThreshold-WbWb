#!/usr/bin/env python3
"""Prototype: luminosity-function form of the ISR convolution (the efficient one).

σ_obs(s) = ∫∫ D(x₁)D(x₂) σ̂(√(x₁x₂)√s) dx₁dx₂  is collapsed onto z=x₁x₂:
    σ_obs(s) = ∫ L(z;μ_F) σ̂(√z·√s) dz,   L(z) = ∫ (dx/x) D(x) D(z/x),
the two-leg LUMINOSITY (radiator self-convolution).  Work in V = −ln z (so the
self-convolution is additive): with the per-leg density ρ(v)=x·D(x) (=eMELA's xD),
    L(z) dz = L_V(V) dV,   L_V(V) = ∫₀^V ρ(v) ρ(V−v) dv .
ρ(v) ~ v^{β_e−1} at the soft end, so the self-convolution has a Beta-type
double-soft endpoint, integrated EXACTLY by Gauss–Jacobi(β_e−1, β_e−1):
    L_V(V) = V^{2β_e−1} · L̃(V),   L̃(V) smooth.
Then
    σ_obs(√s) = ∫₀^{V_hi} V^{2β_e−1} L̃(V) σ̂(e^{−V/2}√s) dV,
with the V→0 weight handled by Gauss–Jacobi(0, 2β_e−1) on the first panel and a
panel split at the σ̂ turn-on V_kink = 2 ln(√s/2m_W) (and the σ̂ spline knots),
so the σ̂ kink never sits inside a panel ⇒ smooth line shape at low node count.

EFFICIENCY: L̃(V) (ALL the radiator / eMELA-ePDF work) is precomputed ONCE on a
V-grid; per √s only σ̂ is evaluated (n_out nodes), on a kink-split grid.  μ_F=√s
drifts ~0.5% across the scan, so L̃ is built at the central √s (the μ_F-drift
error is quantified below; a few-point μ_F interpolation removes it if needed).

Validates against ``convolve_2leg`` at n_quad=512 (the converged 2D truth).

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/luminosity_prototype.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from scipy.interpolate import UnivariateSpline  # noqa: E402
from scipy.special import roots_jacobi  # noqa: E402
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep import isr_emela_grid as _eg  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

ALPHA = isr_beta.ALPHA_MZ_EMELA
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
MW = 80.379
SHAT_BREAKS = (2.0 * MW, 157.0, 158.0, 160.0, 161.0, 162.0)
SIGMA_GRID_LO = 156.0     # σ̂ grid bottom (σ̂≡0 below) → caps the V-range


def prod_nll(n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID,
                              n_quad=n_quad)


def _jac01(n, a, b):
    """Nodes t∈[0,1] and weights for ∫₀¹ t^a (1−t)^b f(t) dt  (Gauss–Jacobi)."""
    x, w = roots_jacobi(n, b, a)          # scipy weight (1−x)^b (1+x)^a
    return 0.5 * (x + 1.0), w / 2.0 ** (a + b + 1)


def _rho_tilde_factory(cfg, Q):
    """ρ̃(v) = ρ(v)·v^{1−β_e}, with ρ(v)=x·D(x)=eMELA xD(x,Q), x=e^{−v}.  Smooth
    (the v^{β_e−1} soft singularity divided out)."""
    grid = _eg.load_grid(cfg.emela_grid)
    be, bs, bh = cfg.betas(Q)

    def rho_tilde(v):
        v = np.asarray(v, float)
        x = np.exp(-v)
        omx = -np.expm1(-v)                          # 1−x, accurate for small v
        omx = np.clip(omx, _eg.OMX_FLOOR, None)
        xD = grid.xfxQ(x, omx, Q)                    # = ρ(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = np.where(v > 0, xD * v ** (1.0 - be), 0.0)
        return np.nan_to_num(rt)
    return rho_tilde, be


def build_Ltilde(cfg, Q, V_hi=0.16, n_V=240, n_jac=64):
    """Precompute L̃(V) on a V-grid (denser near 0) by Gauss–Jacobi double-soft
    self-convolution.  Returns (L̃-spline, β_e)."""
    rho_tilde, be = _rho_tilde_factory(cfg, Q)
    t, wj = _jac01(n_jac, be - 1.0, be - 1.0)        # ∫₀¹ t^{β−1}(1−t)^{β−1}
    # V-grid concentrated near 0 (the soft peak); power spacing.
    u = np.linspace(0.0, 1.0, n_V)
    Vg = V_hi * u ** 2.5
    Vg[0] = Vg[1] * 0.25                              # avoid exact 0
    Lt = np.empty_like(Vg)
    for i, V in enumerate(Vg):
        Lt[i] = np.sum(wj * rho_tilde(V * t) * rho_tilde(V * (1.0 - t)))
    spl = UnivariateSpline(Vg, Lt, k=3, s=0, ext="const")
    return spl, be


def sigma_obs_lumi(sqrt_s_arr, sigma_fn, Ltilde, be, n_out=24):
    """σ_obs(√s)=∫ V^{2β−1} L̃(V) σ̂(e^{−V/2}√s) dV, panel-split at the σ̂ kink/knots.
    Returns (σ_obs, σ̂-eval count)."""
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s_arr, float))
    out = np.empty_like(sqrt_s_arr)
    nev = 0
    p = 2.0 * be - 1.0
    tj, wj = _jac01(n_out, p, 0.0)                    # ∫₀¹ τ^{2β−1} (first panel)
    xg, wg = np.polynomial.legendre.leggauss(n_out)   # plain GL for V>0 panels
    for idx, sq in enumerate(sqrt_s_arr):
        V_top = 2.0 * np.log(sq / SIGMA_GRID_LO)      # σ̂≡0 above this V
        # SINGLE Gauss-Jacobi panel [0,V_top]: V→0 soft handled exactly by the
        # τ^{2β−1} rule, σ̂ grid-edge step at √ŝ=156 is the limit V_top.  σ̂ is
        # off-shell-smooth through 2m_W, so NO V_kink split (it would inject a
        # panel-transition kink at √s=2m_W).
        breaks = [0.0, V_top]
        acc = 0.0
        for a, b in zip(breaks[:-1], breaks[1:]):
            if a <= 0.0:                              # first panel: V^{2β−1} weight
                V = b * tj
                integ = Ltilde(V) * np.asarray(sigma_fn(np.exp(-V / 2.0) * sq))
                acc += b ** (2.0 * be) * np.sum(wj * integ)
            else:                                     # interior: smooth integrand
                V = 0.5 * (b - a) * xg + 0.5 * (b + a)
                integ = (V ** p * Ltilde(V)
                         * np.asarray(sigma_fn(np.exp(-V / 2.0) * sq)))
                acc += 0.5 * (b - a) * np.sum(wg * integ)
            nev += n_out
        out[idx] = acc
    return out, nev


# ---------------------------------------------------------------------------
def line_shape_2leg(cfg, SQ):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        tot = tot + wt * isr_beta.convolve_2leg(SQ, grids[(ch, "nominal")].nlo_fn(), cfg)
    return tot * FB_TO_PB, len(SQ) * cfg.n_quad ** 2


def line_shape_lumi(cfg, SQ, build_once, n_out):
    grids = load_grids(scheme_alpha="gf")
    Q_ref = float(0.5 * (SQ[0] + SQ[-1]))
    tot = np.zeros_like(SQ)
    nev = 0
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        nlo = grids[(ch, "nominal")].nlo_fn()
        if build_once:
            Lt, be = build_Ltilde(cfg, Q_ref)
            obs, n = sigma_obs_lumi(SQ, nlo, Lt, be, n_out=n_out)
        else:                                         # faithful: L at μ_F=√s each
            obs = np.empty_like(SQ); n = 0
            for i, sq in enumerate(SQ):
                Lt, be = build_Ltilde(cfg, float(sq))
                o, k = sigma_obs_lumi([sq], nlo, Lt, be, n_out=n_out)
                obs[i] = o[0]; n += k
        tot = tot + wt * obs
        nev += n
    return tot * FB_TO_PB, nev


def smooth_metric(c, i0):
    cn = c / c[i0]
    d2 = np.abs(np.diff(cn, 2))
    return d2.max()


def main():
    SQ = np.linspace(157.5, 161.5, 81)
    i0 = int(np.argmin(np.abs(SQ - 161.0)))
    print("Luminosity-function convolution prototype vs 2D truth "
          "(gf NLL pure-WW)\n")

    t = time.time()
    truth, nev_t = line_shape_2leg(prod_nll(512), SQ)
    print(f"[truth] 2D n_quad=512        : {time.time()-t:5.1f}s  "
          f"σ̂-evals={nev_t:,}  max|2nd-diff|={smooth_metric(truth,i0):.2e}")
    p128, nev_p = line_shape_2leg(prod_nll(128), SQ)
    print(f"[2D-128] production          : Δσ={np.max(np.abs(p128/truth-1))*100:.3f}%"
          f"  σ̂-evals={nev_p:,}  max|2nd-diff|={smooth_metric(p128,i0):.2e}")

    print("\n[lumi] L̃ rebuilt per √s (faithful) and built ONCE @ central √s "
          "(efficient):")
    for build_once in (False, True):
        for n_out in (24, 48):
            t = time.time()
            obs, nev = line_shape_lumi(prod_nll(128), SQ, build_once, n_out)
            dt = time.time() - t
            tag = "once " if build_once else "per√s"
            print(f"  L̃={tag} n_out={n_out:2d}: Δσ={np.max(np.abs(obs/truth-1))*100:.3f}%"
                  f"  max|2nd-diff|={smooth_metric(obs,i0):.2e}  σ̂-evals={nev:,}"
                  f"  ({dt:.1f}s)")
    print("\n  target: match truth's accuracy+smoothness at σ̂-evals ≪ 2D, with the")
    print("  radiator built ONCE.  μ_F-drift = (once − per√s) Δσ gap.")


if __name__ == "__main__":
    main()
