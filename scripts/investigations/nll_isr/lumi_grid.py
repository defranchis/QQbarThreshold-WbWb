#!/usr/bin/env python3
"""Luminosity grid L̃(V; μ_F): precomputed radiator self-convolution for the
efficient ISR convolution (production-grade form of luminosity_prototype.py).

The two-leg ISR convolution collapses to σ_obs(s)=∫ L(z) σ̂(√z·√s) dz, z=x₁x₂.
In V=−ln z the luminosity L is the radiator self-convolution; factoring the
soft V→0 behaviour, L(V;μ_F)=V^{2β_e−1}·L̃(V;μ_F) with L̃ smooth.  L̃ depends ONLY
on the ISR radiator (eMELA ePDF grid) and μ_F — NOT on σ̂, the channel, the
varpoint or √s except through μ_F=√s.  So we build L̃ ONCE on a (V, μ_F) grid
(mirroring isr_emela_grid for the per-leg ePDF) and reuse it for every channel /
varpoint / scan point.  Per-√s evaluation is then just: look up L̃(·;√s) + the
cheap kink-split σ̂ integral (~24 nodes) — no radiator work at runtime.

Provenance (α, fac/ren scheme, eMELA grid) is baked into the .npz meta and must
match the consuming cfg, exactly like isr_emela_grid.

Build + benchmark:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/lumi_grid.py
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
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
import scripts.investigations.nll_isr.luminosity_prototype as LP  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

MW = LP.MW
SHAT_BREAKS = LP.SHAT_BREAKS
SIGMA_GRID_LO = LP.SIGMA_GRID_LO


# ---------------------------------------------------------------------------
# Build / persist / load
# ---------------------------------------------------------------------------
def _V_knots(V_hi, n_V):
    u = np.linspace(0.0, 1.0, n_V)
    V = V_hi * u ** 2.5            # concentrated near the soft peak V→0
    V[0] = V[1] * 0.25
    return V


def build_lumi_grid(cfg, V_hi=0.18, n_V=160, n_jac=128,
                    muF_lo=155.0, muF_hi=165.0, n_muF=11):
    """Compute L̃(V; μ_F) on a (V, μ_F) grid by Gauss–Jacobi double-soft
    self-convolution of the eMELA per-leg ePDF (one column per μ_F knot)."""
    V_knots = _V_knots(V_hi, n_V)
    muF_knots = np.linspace(muF_lo, muF_hi, n_muF)
    table = np.empty((n_V, n_muF))
    for j, mu in enumerate(muF_knots):
        rho_tilde, be = LP._rho_tilde_factory(cfg, float(mu))
        t, wj = LP._jac01(n_jac, be - 1.0, be - 1.0)
        for i, V in enumerate(V_knots):
            table[i, j] = np.sum(wj * rho_tilde(V * t) * rho_tilde(V * (1.0 - t)))
    meta = dict(scheme=cfg.scheme, alpha=float(cfg.resolved_alpha()),
                m_e=float(cfg.m_e), mu_F_factor=float(cfg.mu_F_factor),
                mu_F_abs=float(cfg.mu_F_abs), x_min=float(cfg.x_min),
                fac_scheme=cfg.emela_fac_scheme, ren_scheme=cfg.emela_ren_scheme,
                emela_grid=cfg.emela_grid)
    return LumiGrid(V_knots, muF_knots, table, meta)


class LumiGrid:
    def __init__(self, V_knots, muF_knots, table, meta):
        self.V_knots = np.asarray(V_knots, float)
        self.muF_knots = np.asarray(muF_knots, float)
        self.table = np.asarray(table, float)
        self.meta = dict(meta)
        ky = min(3, len(self.muF_knots) - 1)
        self._spl = RectBivariateSpline(self.V_knots, np.log(self.muF_knots),
                                        self.table, kx=3, ky=ky)

    def write(self, path):
        np.savez(path, V_knots=self.V_knots, muF_knots=self.muF_knots,
                 table=self.table, meta_keys=np.array(list(self.meta.keys())),
                 meta_vals=np.array([str(v) for v in self.meta.values()]))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        meta = {}
        for k, v in zip(d["meta_keys"], d["meta_vals"]):
            s = str(v)
            try:
                meta[str(k)] = float(s)
            except ValueError:
                meta[str(k)] = s
        return LumiGrid(d["V_knots"], d["muF_knots"], d["table"], meta)

    def muF(self, sqrt_s):
        return (self.meta["mu_F_abs"] if self.meta["mu_F_abs"] > 0
                else self.meta["mu_F_factor"] * sqrt_s)

    def beta_e(self, sqrt_s):
        return isr_beta.beta_components(
            self.muF(sqrt_s), scheme=self.meta["scheme"],
            alpha=self.meta["alpha"], m_e=self.meta["m_e"])[0]

    def ltilde(self, V, sqrt_s):
        """L̃(V; μ_F=μ_F(√s)) — vectorised in V."""
        V = np.atleast_1d(np.asarray(V, float))
        return self._spl(V, np.log(self.muF(sqrt_s)))[:, 0]


def check_provenance(grid, cfg):
    a = cfg.resolved_alpha()
    bad = (abs(grid.meta["alpha"] - a) > 1e-9 * a
           or grid.meta["fac_scheme"] != cfg.emela_fac_scheme
           or grid.meta["ren_scheme"] != cfg.emela_ren_scheme)
    if bad:
        raise ValueError(f"lumi grid provenance mismatch: {grid.meta} vs cfg "
                         f"(α={a}, {cfg.emela_fac_scheme}/{cfg.emela_ren_scheme})")


# ---------------------------------------------------------------------------
# Evaluation: σ_obs(√s) = ∫ V^{2β−1} L̃(V;√s) σ̂(e^{−V/2}√s) dV  (kink-split)
# ---------------------------------------------------------------------------
def sigma_obs_grid(sqrt_s_arr, sigma_fn, grid, n_out=24):
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s_arr, float))
    out = np.empty_like(sqrt_s_arr)
    nev = 0
    xg, wg = np.polynomial.legendre.leggauss(n_out)
    for idx, sq in enumerate(sqrt_s_arr):
        be = grid.beta_e(float(sq))
        p = 2.0 * be - 1.0
        tj, wj = LP._jac01(n_out, p, 0.0)              # ∫₀¹ τ^{2β−1}  (first panel)
        V_top = 2.0 * np.log(sq / SIGMA_GRID_LO)
        # SINGLE Gauss-Jacobi panel [0,V_top]: the only integrand features are the
        # V→0 soft weight (handled EXACTLY by the τ^{2β−1} Jacobi rule) and the σ̂
        # grid-edge step at √ŝ=156 (= the integration LIMIT V_top, never interior).
        # MoCaNLO σ̂ is off-shell-smooth through 2m_W (no real turn-on kink), so a
        # V_kink split is unnecessary AND harmful — it injects a panel-transition
        # kink at √s=2m_W.  Single panel → smooth to ~1e-5 (≪ the 2D truth).
        breaks = [0.0, V_top]
        acc = 0.0
        for a, b in zip(breaks[:-1], breaks[1:]):
            if a <= 0.0:
                V = b * tj
                integ = grid.ltilde(V, sq) * np.asarray(sigma_fn(np.exp(-V / 2) * sq))
                acc += b ** (2.0 * be) * np.sum(wj * integ)
            else:
                V = 0.5 * (b - a) * xg + 0.5 * (b + a)
                integ = (V ** p * grid.ltilde(V, sq)
                         * np.asarray(sigma_fn(np.exp(-V / 2) * sq)))
                acc += 0.5 * (b - a) * np.sum(wg * integ)
            nev += n_out
        out[idx] = acc
    return out, nev


def line_shape_grid(grid, SQ, n_out=24):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    nev = 0
    for ch, wt in dict(PURE_WW_WEIGHTS).items():       # L̃ shared across channels
        obs, n = sigma_obs_grid(SQ, grids[(ch, "nominal")].nlo_fn(), grid, n_out)
        tot = tot + wt * obs
        nev += n
    return tot * FB_TO_PB, nev


# ---------------------------------------------------------------------------
def main():
    SQ = np.linspace(157.5, 161.5, 81)
    i0 = int(np.argmin(np.abs(SQ - 160.0)))
    cfg = LP.prod_nll(128)
    sm = lambda c: np.abs(np.diff(c / c[i0], 2)).max()  # noqa: E731

    print("Luminosity-grid build + per-√s evaluation vs the 2D quadrature\n")
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    truth, _ = LP.line_shape_2leg(LP.prod_nll(1024), SQ)
    p128, nev128 = LP.line_shape_2leg(LP.prod_nll(128), SQ)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    print(f"[truth ] 2D n_quad=1024 (reference)")
    print(f"[2D-128] production: Δσ_core={_core(p128, truth, SQ):.3f}%  "
          f"max|2nd-diff|={sm(p128):.2e}")

    # BUILD (one-off, channel/varpoint/√s-independent)
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    t = time.time()
    grid = build_lumi_grid(cfg)
    t_build = time.time() - t
    path = "/tmp/ww_lumi_grid_delta_alpmz.npz"
    grid.write(path)
    grid = LumiGrid.load(path)               # round-trip through disk
    check_provenance(grid, cfg)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    print(f"\n[lumi-grid] build (ONCE): {t_build:.1f}s  "
          f"file={os.path.getsize(path)/1024:.0f} kB  "
          f"({grid.table.shape[0]}×{grid.table.shape[1]} V×μ_F)")

    # EVALUATE (the per-√s hot path)
    for n_out in (16, 24, 48):
        _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
        t = time.time()
        obs, nev = line_shape_grid(grid, SQ, n_out=n_out)
        dt = time.time() - t
        os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
        print(f"[lumi-grid] eval n_out={n_out:2d}: Δσ_core={_core(obs, truth, SQ):.3f}%"
              f"  max|2nd-diff|={sm(obs):.2e}  σ̂-evals={nev:,}  ({dt*1e3:.0f} ms)")
    print("\n  L̃ is built ONCE (radiator self-conv, channel/√s-independent) then")
    print("  reused for every channel & scan point — per-√s cost is σ̂ lookups only.")


def _core(c, ref, SQ):
    core = (SQ >= 158.5) & (SQ <= 161.0)
    ic = int(np.argmin(np.abs(SQ - 160.0)))
    s = (c / c[ic]) / (ref / ref[ic]) - 1.0
    return np.max(np.abs(s[core])) * 100


if __name__ == "__main__":
    main()
