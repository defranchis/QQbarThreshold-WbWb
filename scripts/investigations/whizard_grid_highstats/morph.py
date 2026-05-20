"""Shared primitives for the WHIZARD-grid morphing scheme.

Operational predictor:

    σ_pred(√s, m_W, Γ_W) = σ_nom(√s)
                         × R_m(√s, m_W)            # quadratic in m_W
                         × R_Γ(√s, Γ_W)            # quadratic in Γ_W
                         × [1 + β(√s)·Δm·ΔΓ]      # bilinear cross

Each of the 8 √s-dependent quantities (σ_nom, σ_nom_m, σ_nom_g, 3 coef_m,
3 coef_g, β) is fitted independently at every grid √s and then cubic-
spline-interpolated along the √s axis. The Γ_W axis is restricted to
the 5 uniformly-spaced PDG values for the morph fit — the BFS-reference
duplicates (2.04483, 2.09201) crowd the LSQ and aren't operationally
useful here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Nominal anchor point and uniform Γ_W subset used for the morph fit
MW0 = 80.379
GW0 = 2.085
GW_UNIFORM = [2.045, 2.065, 2.085, 2.105, 2.125]


def load_grid(path: Path) -> pd.DataFrame:
    """Read a 5-column WHIZARD grid CSV (sqrts, sigma, err, mW, Γ_W) in fb/GeV."""
    return pd.read_csv(path, comment="#", sep=r"\s+", header=None,
                       names=["sqrts", "sigma_fb", "err_fb", "mW", "gammaW"])


def filter_uniform_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the BFS-reference Γ_W duplicates; keep only the 5 PDG-uniform values."""
    return df[np.isin(np.round(df.gammaW, 5), GW_UNIFORM)].copy()


def fit_morph_at_sqrts(slice_df: pd.DataFrame, *,
                       mw0: float = MW0, gw0: float = GW0) -> dict | None:
    """At one √s slice, fit the 8 morph numbers (3 R_m coefs + 3 R_Γ coefs +
    σ_nom + β). Returns None if the slice lacks the required nominal axes.

    R_m and R_Γ are *un-normalised* quadratic polynomials in m_W / Γ_W; the
    morph function evaluates them and divides by their value at the nominal
    to form the relative morph factor.
    """
    sl_m = slice_df[np.isclose(slice_df.gammaW, gw0)].sort_values("mW")
    sl_g = slice_df[np.isclose(slice_df.mW,    mw0)].sort_values("gammaW")
    nom  = slice_df[(np.isclose(slice_df.mW, mw0))
                    & (np.isclose(slice_df.gammaW, gw0))]
    if len(sl_m) < 3 or len(sl_g) < 3 or len(nom) == 0:
        return None
    coef_m = np.polyfit(sl_m.mW.values,    sl_m.sigma_fb.values, 2)
    coef_g = np.polyfit(sl_g.gammaW.values, sl_g.sigma_fb.values, 2)
    sigma_nom_m = np.polyval(coef_m, mw0)
    sigma_nom_g = np.polyval(coef_g, gw0)
    sigma_nom   = float(nom.sigma_fb.iloc[0])

    # β: bilinear cross-term coefficient fitted from doubly-off-axis residuals
    off = slice_df[(~np.isclose(slice_df.mW, mw0))
                    & (~np.isclose(slice_df.gammaW, gw0))]
    if len(off) == 0:
        return None
    Rm = np.polyval(coef_m, off.mW.values)    / sigma_nom_m
    Rg = np.polyval(coef_g, off.gammaW.values) / sigma_nom_g
    sigma_pred_no_cross = sigma_nom * Rm * Rg
    dm = off.mW.values    - mw0
    dG = off.gammaW.values - gw0
    A = sigma_pred_no_cross * dm * dG
    b = off.sigma_fb.values - sigma_pred_no_cross
    w = 1.0 / off.err_fb.values**2
    beta = np.sum(w * A * b) / np.sum(w * A * A) if np.sum(w * A * A) > 0 else 0.0

    return {
        "sigma_nom":   sigma_nom,
        "sigma_nom_m": sigma_nom_m,
        "sigma_nom_g": sigma_nom_g,
        "coef_m":      coef_m,
        "coef_g":      coef_g,
        "beta":        beta,
    }


def build_splines(sqrts_axis: np.ndarray, morphs: list[dict]) -> dict:
    """Cubic-spline each of the 8 morph numbers along √s. Returns a dict
    keyed by name (sigma_nom, sigma_nom_m, sigma_nom_g, beta, coef_m_{0..2},
    coef_g_{0..2}); each value is a scipy CubicSpline."""
    kw = dict(bc_type="natural", extrapolate=True)
    spl = {
        "sigma_nom":   CubicSpline(sqrts_axis, [m["sigma_nom"]   for m in morphs], **kw),
        "sigma_nom_m": CubicSpline(sqrts_axis, [m["sigma_nom_m"] for m in morphs], **kw),
        "sigma_nom_g": CubicSpline(sqrts_axis, [m["sigma_nom_g"] for m in morphs], **kw),
        "beta":        CubicSpline(sqrts_axis, [m["beta"]        for m in morphs], **kw),
    }
    cm = np.array([m["coef_m"] for m in morphs])
    cg = np.array([m["coef_g"] for m in morphs])
    for k in range(3):
        spl[f"coef_m_{k}"] = CubicSpline(sqrts_axis, cm[:, k], **kw)
        spl[f"coef_g_{k}"] = CubicSpline(sqrts_axis, cg[:, k], **kw)
    return spl


def sigma_morph(s, mW: float, gammaW: float, *, splines: dict,
                mw0: float = MW0, gw0: float = GW0):
    """Evaluate the full morphing prediction at (s, m_W, Γ_W). s may be
    scalar or array; mW, gammaW scalar."""
    coef_m = np.array([splines["coef_m_0"](s),
                       splines["coef_m_1"](s),
                       splines["coef_m_2"](s)])
    coef_g = np.array([splines["coef_g_0"](s),
                       splines["coef_g_1"](s),
                       splines["coef_g_2"](s)])
    Rm = np.polyval(coef_m, mW)     / splines["sigma_nom_m"](s)
    Rg = np.polyval(coef_g, gammaW) / splines["sigma_nom_g"](s)
    cross = 1.0 + splines["beta"](s) * (mW - mw0) * (gammaW - gw0)
    return splines["sigma_nom"](s) * Rm * Rg * cross


def build_morph_from_grid(grid_csv: Path,
                           extra_csv: Path | None = None) -> tuple[np.ndarray, dict]:
    """Convenience builder: load grid (+ optional second CSV), fit morphs at
    every √s, return (sqrts_axis, splines)."""
    df = filter_uniform_gw(load_grid(grid_csv))
    if extra_csv is not None and extra_csv.exists():
        df_extra = filter_uniform_gw(load_grid(extra_csv))
        df = pd.concat([df, df_extra], ignore_index=True)
    sqrts_grid = np.array(sorted(df.sqrts.unique()))
    morphs, kept = [], []
    for s in sqrts_grid:
        m = fit_morph_at_sqrts(df[np.isclose(df.sqrts, s, atol=1e-3)])
        if m is not None:
            morphs.append(m)
            kept.append(s)
    sqrts_axis = np.array(kept)
    return sqrts_axis, build_splines(sqrts_axis, morphs), morphs
