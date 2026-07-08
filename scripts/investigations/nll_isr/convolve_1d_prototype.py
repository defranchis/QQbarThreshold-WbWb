#!/usr/bin/env python3
"""Prototype: 1D-collapse of the two-leg ISR convolution (efficiency + smoothness).

The production convolution is a 2D Gauss-Legendre quadrature,
    σ_obs(s) = ∫∫ D(x₁)D(x₂) σ̂(√(x₁x₂)·√s) dx₁dx₂   →   Σ_ij W_i W_j σ̂(√(x_i x_j)√s),
O(n_quad²) σ̂ evals.  The visible line-shape ripple is a GL artefact: σ̂ has a steep
turn-on near √ŝ=2m_W, and the nodes √(x_i x_j) beat against it as √s sweeps.

This prototype uses the EXACT factorisation of the two-leg integral into two
single-leg passes (both radiators at the collision scale μ_F=√s, as in the 2D form):
    σ_obs(s) = C[ C[σ̂] ](√s),     C[f](E) = ∫ D(x) f(√x·E) dx.
  PASS 1 (inner):  g(E) = C[σ̂](E) on an E-grid, with the x-integration SPLIT into
                   GL panels at the σ̂ turn-on / spline-knot images x=(√ŝ_break/E)²,
                   so each panel integrand is smooth → g(E) is smooth and accurate.
  PASS 2 (outer):  σ_obs(√s) = C[ĝ](√s), ĝ a cubic interpolant of the smooth g —
                   the single radiator already smeared σ̂'s kink, so no split / low
                   node count suffices and there is no ripple.
Cost: O(m_E · n_panel) σ̂ evals (1D), vs O(n_quad²) (2D).  The per-leg radiator
weight reuses ``isr_beta._per_leg_grid_nll`` verbatim, so it is faithful to the
production NLL/eMELA-grid radiator (only the QUADRATURE is reorganised).

Validation: against ``convolve_2leg`` at n_quad=512 (the converged 2D truth).

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/convolve_1d_prototype.py
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
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

ALPHA = isr_beta.ALPHA_MZ_EMELA
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
MW = 80.379                       # σ̂ turn-on at √ŝ = 2 m_W
# σ̂ √ŝ breakpoints to land on GL panel boundaries (turn-on + the steep-region
# spline knots) so each inner panel sees a smooth σ̂.
SHAT_BREAKS = (2.0 * MW, 157.0, 158.0, 160.0, 161.0, 162.0)


def prod_nll(n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID,
                              n_quad=n_quad)


def _panel_grid(be, x_min, x_breaks, n_panel):
    """GL nodes/weights on each x-panel, via the u=(1−x)^β_e substitution per panel.
    x_breaks: interior x boundaries.  Returns (w, x, omx, jac_NS) concatenated."""
    xb = sorted({x_min, 1.0} | {b for b in x_breaks if x_min < b < 1.0})
    W, X, OMX, JAC = [], [], [], []
    for a, b in zip(xb[:-1], xb[1:]):
        ua, ub = (1.0 - a) ** be, (1.0 - b) ** be      # ua > ub  (a < b)
        u, w = isr_beta._quad_nodes(n_panel, ub, ua)
        omx = u ** (1.0 / be)
        with np.errstate(over="ignore", invalid="ignore"):
            jac = np.where(u > 1e-300, u ** (1.0 / be - 1.0) / be, 0.0)
        W.append(w); X.append(1.0 - omx); OMX.append(omx); JAC.append(jac)
    return (np.concatenate(W), np.concatenate(X),
            np.concatenate(OMX), np.concatenate(JAC))


def _single_leg(sigma_fn, E_arr, scale, cfg, n_panel):
    """g(E) = ∫ D(x) σ̂(√x·E) dx for each E, radiator at fixed scale √s=`scale`,
    x-integration split at the σ̂ √ŝ-breaks mapped to x=(√ŝ_break/E)²."""
    be, bs, bh = cfg.betas(scale)
    norm = isr_beta._radiator_norm(be, bs)
    g = np.empty(len(E_arr))
    nev = 0
    for k, E in enumerate(E_arr):
        breaks = [(sb / E) ** 2 for sb in SHAT_BREAKS]
        w, x, omx, jac = _panel_grid(be, cfg.x_min, breaks, n_panel)
        per = isr_beta._per_leg_grid_nll(cfg, be, norm, x, omx, jac, scale)
        g[k] = np.sum(w * per * np.asarray(sigma_fn(np.sqrt(x) * E)))
        nev += x.size
    return g, nev


def convolve_1d(sqrt_s_arr, sigma_fn, cfg, n_panel=24, m_E=96):
    """σ_obs via the iterated single-leg (1D) form.  Inner pass (σ̂→g) is
    kink-split; outer pass convolves the SMOOTH g with the standard single-panel
    weights (a smooth integrand needs no split).  σ̂ is evaluated only inside.
    Returns (σ_obs, σ̂-eval count)."""
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s_arr, float))
    out = np.empty_like(sqrt_s_arr)
    total = 0
    for idx, sq in enumerate(sqrt_s_arr):
        # PASS 1 (inner): σ̂ → g on a UNIFORM E-grid spanning the outer-node range
        # [√(x_min)·√s, √s], kink-split so g is smooth and well-sampled everywhere.
        E_grid = np.linspace(float(np.sqrt(cfg.x_min) * sq), float(sq), m_E)
        g, nev = _single_leg(sigma_fn, E_grid, float(sq), cfg, n_panel)
        total += nev
        g_spl = UnivariateSpline(E_grid, g, k=3, s=0, ext="zeros")
        # PASS 2 (outer): convolve the smooth g, ALSO kink-split at g's turn-on so
        # the outer GL does not beat against the (smeared but real) rise.  Evaluates
        # g_spl (cheap), not σ̂ — no σ̂ cost added.
        obs, _ = _single_leg(g_spl, np.array([sq]), float(sq), cfg, n_panel)
        out[idx] = obs[0]
    return out, total


def _count_2leg(cfg, n_sqrts):
    return n_sqrts * cfg.n_quad ** 2


def line_shape(method, cfg, SQ, **kw):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    nev = 0
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        nlo = grids[(ch, "nominal")].nlo_fn()
        if method == "2leg":
            tot = tot + wt * isr_beta.convolve_2leg(SQ, nlo, cfg)
            nev += _count_2leg(cfg, len(SQ))
        else:
            obs, n = convolve_1d(SQ, nlo, cfg, **kw)
            tot = tot + wt * obs
            nev += n
    return tot * FB_TO_PB, nev


def smooth_metric(c, i0):
    cn = c / c[i0]
    d2 = np.abs(np.diff(cn, 2))
    return d2.max(), np.median(d2)


def main():
    SQ = np.linspace(157.5, 161.5, 81)
    i0 = int(np.argmin(np.abs(SQ - 161.0)))
    print("1D-collapse prototype vs 2D convolve_2leg (gf NLL pure-WW line shape)\n")

    # truth: converged 2D
    t = time.time()
    truth, nev_t = line_shape("2leg", prod_nll(512), SQ)
    print(f"[truth] 2D n_quad=512 : {time.time()-t:5.1f}s  σ̂-evals={nev_t:,}")
    mt, _ = smooth_metric(truth, i0)

    # production 2D (jittery reference)
    p128, nev_p = line_shape("2leg", prod_nll(128), SQ)
    mp, _ = smooth_metric(p128, i0)
    dp = np.max(np.abs(p128 / truth - 1)) * 100
    print(f"[2D-128] production    : max|2nd-diff|={mp:.2e}  "
          f"Δσ vs truth={dp:.4f}%  σ̂-evals={nev_p:,}")

    # 1D collapse, a few settings
    print()
    for n_panel, m_E in [(24, 96), (24, 192), (32, 256), (48, 384), (64, 512)]:
        t = time.time()
        c1d, nev = line_shape("1d", prod_nll(128), SQ, n_panel=n_panel, m_E=m_E)
        dt = time.time() - t
        m1, _ = smooth_metric(c1d, i0)
        d = np.max(np.abs(c1d / truth - 1)) * 100
        print(f"[1D] n_panel={n_panel:2d} m_E={m_E:3d}: max|2nd-diff|={m1:.2e}  "
              f"Δσ vs truth={d:.4f}%  σ̂-evals={nev:,}  ({dt:.1f}s)")

    print(f"\n  (truth max|2nd-diff|={mt:.2e}; the 1D target is to match truth's")
    print("   accuracy AND smoothness at far fewer σ̂ evals than 2D-512.)")


if __name__ == "__main__":
    main()
