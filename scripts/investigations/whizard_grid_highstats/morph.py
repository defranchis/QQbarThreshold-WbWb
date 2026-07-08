"""Shared primitives for the WHIZARD-grid morphing scheme.

Operational predictor:

    σ_pred(√s, m_W, Γ_W) = σ_nom(√s)
                         × R_m(√s, m_W)            # quadratic in m_W
                         × R_Γ(√s, Γ_W)            # quadratic in Γ_W
                         × [1 + β(√s)·Δm·ΔΓ]      # bilinear cross

Each of the 8 √s-dependent quantities (σ_nom, 3 coef_m, 3 coef_g, β) is
fitted independently at every grid √s and then interpolated along the √s
axis with a natural cubic spline. R_m / R_Γ are normalised by the
coefficient polynomial evaluated at the nominal point, so σ_pred is
exact at (m_W₀, Γ_W₀).

An interpolating spline would thread per-√s Monte-Carlo fluctuations
and propagate them as a ripple. To prevent that the grid is *denoised
along √s before the morph fits it* (see ``denoise_grid``): the slowly-
varying ratios σ/σ_nom are χ²-smoothed in √s, and σ_nom is denoised by
averaging across the (m_W, Γ_W) plane (no √s-smoothing of the steep
line shape). So every morph input is already smooth in √s and the
cubic spline threads clean points. Denoising never touches the
anti-correlated R_m / R_Γ polynomial coefficients directly — smoothing
those independently breaks the error cancellation that makes their
combination accurate (a GCV smoothing spline on the coefficients was
tried for exactly this and inflated the √s LOO residual ~3×).

The Γ_W axis is restricted to the 5 uniformly-spaced PDG values for the
morph fit — the BFS-reference duplicates (2.04483, 2.09201) crowd the
LSQ and aren't operationally useful here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline

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


WHIZARD_WORK = Path(__file__).resolve().parents[4] / "whizard" / "work"
GRID_FINE_CSV       = WHIZARD_WORK / "grid_fine" / "grid.csv"
GRID_HIGHSTATS_CSV  = WHIZARD_WORK / "grid_highstats" / "grid.csv"
GRID_VALIDATE_CSV   = WHIZARD_WORK / "grid_validate" / "grid.csv"
GRID_VALIDATE_FINE_CSV = WHIZARD_WORK / "grid_validate_fine" / "grid.csv"


def load_operational_grid() -> pd.DataFrame:
    """Return the input grid the operational morph is built on:
    ``grid_fine`` (0.1-GeV √s in [155, 165] over 9 m_W × 7 Γ_W, ~0.008 % MC)
    plus the outer 0.5-GeV wings from ``grid_highstats`` (√s ∈ {154, 154.5,
    165.5, …, 172}) as spline support beyond the fine window.

    The returned frame keeps all 7 Γ_W rows (including the two BFS-reference
    duplicates) so external closure scripts can compare at those exact
    values; ``build_morph_from_grid`` calls ``filter_uniform_gw`` internally
    before fitting, so the BFS-ref rows do not enter the morph fit.
    """
    fine = load_grid(GRID_FINE_CSV)
    hs = load_grid(GRID_HIGHSTATS_CSV)
    fine_sqrts = set(np.round(fine.sqrts.unique(), 4))
    wings = hs[~np.round(hs.sqrts, 4).isin(fine_sqrts)].copy()
    return pd.concat([fine, wings], ignore_index=True)


def denoise_grid(df: pd.DataFrame, *, chi2_per_dof: float = 1.0) -> pd.DataFrame:
    """Denoise the grid along √s before the morph fits it — in three
    passes, so that *no* morph input is an interpolating spline through
    raw per-√s MC fluctuations.

    Pass 1 — ratios. The √s ripple the morph would propagate lives in
    the ratios R(√s) = σ(√s, m_W, Γ_W) / σ(√s, m_W⁰, Γ_W⁰): these are
    slowly varying — the steep threshold rise cancels — so they can be
    smoothed without bias, unlike the steep absolute σ. Each ratio is
    replaced by a weighted penalised spline

        UnivariateSpline(√s, R, w = 1/δR_MC, s = chi2_per_dof · N)

    whose target ``s`` sets the fit to χ² ≈ N — within the MC error
    bars, not through every point.

    Pass 2 — σ_nom. The nominal line shape is steep, so it cannot be
    √s-smoothed without risking the threshold knee. Instead it is
    denoised *across the (m_W, Γ_W) plane*: every grid curve divided by
    its own (now smooth) ratio is an independent estimate of σ_nom, so
    the mean over the plane beats the per-curve MC noise down by
    ≈√N_curves with no √s-smoothing at all.

    Pass 3 — rebuild σ = R_smooth · σ_nom_clean.

    Smoothing acts only on physical, slowly-varying quantities (ratios)
    and on a plane average — never on the anti-correlated R_m / R_Γ
    polynomial coefficients. Returns a copy of ``df`` with ``sigma_fb``
    replaced by the denoised values; ``err_fb`` is left intact.
    """
    out = df.copy()
    nom = df[(np.isclose(df.mW, MW0)) & (np.isclose(df.gammaW, GW0))]
    nom = nom.sort_values("sqrts")
    x_nom, snom_raw = nom.sqrts.values, nom.sigma_fb.values

    # pass 1 — denoise each ratio R(√s) = σ / σ_nom_raw
    curves = []  # (df-index, √s, R_smooth)
    for _, g in df.groupby(["mW", "gammaW"], sort=False):
        g = g.sort_values("sqrts")
        x, y, e = g.sqrts.values, g.sigma_fb.values, g.err_fb.values
        if len(x) < 5:
            continue
        snom_here = np.interp(x, x_nom, snom_raw)
        ratio, ratio_err = y / snom_here, e / snom_here
        spl = UnivariateSpline(x, ratio, w=1.0 / ratio_err,
                               s=chi2_per_dof * len(x))
        curves.append((g.index, x, y, spl(x)))

    # pass 2 — σ_nom from the plane average of σ_raw / R_smooth
    est: dict[float, list] = {}
    for _, x, y, Rs in curves:
        for xi, yi, ri in zip(x, y, Rs):
            if ri > 0:
                est.setdefault(round(float(xi), 4), []).append(yi / ri)
    snom_clean = {k: float(np.mean(v)) for k, v in est.items()}

    # pass 3 — rebuild σ = R_smooth · σ_nom_clean
    for index, x, _, Rs in curves:
        sc = np.array([snom_clean[round(float(xi), 4)] for xi in x])
        out.loc[index, "sigma_fb"] = Rs * sc
    return out


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
    waa = float(np.sum(w * A * A))
    if waa > 0:
        beta = float(np.sum(w * A * b) / waa)
        beta_err = float(1.0 / np.sqrt(waa))
    else:
        beta = 0.0
        beta_err = float("inf")

    return {
        "sigma_nom": sigma_nom,
        "coef_m":    coef_m,
        "coef_g":    coef_g,
        "beta":      beta,
        "beta_err":  beta_err,
    }


def build_splines(sqrts_axis: np.ndarray, morphs: list[dict], *,
                  denoise_beta: bool = True,
                  beta_chi2_per_dof: float = 1.0) -> dict:
    """Interpolating natural cubic spline of each of the 8 morph numbers
    along √s. Returns a dict keyed by name (sigma_nom, beta,
    coef_m_{0..2}, coef_g_{0..2}); each value is a scipy CubicSpline.

    Interpolation is safe for σ_nom and the R_m/R_Γ coefficients because
    the per-√s MC noise is removed upstream by ``denoise_grid`` — those
    morph numbers are smooth in √s, so the spline threads clean points
    without rippling. The three R_m / R_Γ coefficients must not be
    smoothed directly: they are strongly anti-correlated and
    ``polyval(coef, ·)`` is accurate only because their errors cancel, a
    cancellation interpolation preserves (it commutes with ``polyval``)
    but independent coefficient smoothing destroys.

    β is a scalar (not part of an anti-correlated coefficient group), so
    a χ²-targeted weighted spline can be applied to its per-√s series
    directly without disturbing the other morph quantities. When
    ``denoise_beta`` is True (default) and a per-√s ``beta_err`` is
    available from :func:`fit_morph_at_sqrts`, β is replaced by a
    weighted ``UnivariateSpline`` evaluated at the same nodes before the
    cubic spline is built. Set ``beta_chi2_per_dof`` to control the
    target χ² (=1.0 means the spline is consistent with the β error
    bars).
    """
    kw = dict(bc_type="natural", extrapolate=True)

    beta_vals = np.array([m["beta"] for m in morphs], dtype=float)
    if denoise_beta and all("beta_err" in m and np.isfinite(m["beta_err"])
                            for m in morphs):
        beta_err = np.array([m["beta_err"] for m in morphs], dtype=float)
        # χ²-targeted weighted spline through (sqrts, β) ± beta_err.
        # Same recipe as ``denoise_grid`` uses on the σ/σ_nom ratios.
        beta_spl = UnivariateSpline(sqrts_axis, beta_vals, w=1.0 / beta_err,
                                    s=beta_chi2_per_dof * len(sqrts_axis))
        beta_smooth = beta_spl(sqrts_axis)
    else:
        beta_smooth = beta_vals

    spl = {
        "sigma_nom": CubicSpline(sqrts_axis, [m["sigma_nom"] for m in morphs], **kw),
        "beta":      CubicSpline(sqrts_axis, beta_smooth, **kw),
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
    # Normalise by the coefficient polynomial at the nominal point so
    # R_m(s, m_W₀) = R_Γ(s, Γ_W₀) = 1 exactly under smoothing.
    Rm = np.polyval(coef_m, mW)     / np.polyval(coef_m, mw0)
    Rg = np.polyval(coef_g, gammaW) / np.polyval(coef_g, gw0)
    cross = 1.0 + splines["beta"](s) * (mW - mw0) * (gammaW - gw0)
    return splines["sigma_nom"](s) * Rm * Rg * cross


def build_morph_from_grid(grid_csv: Path,
                           extra_csv: Path | None = None, *,
                           denoise: bool = True) -> tuple[np.ndarray, dict, list]:
    """Convenience builder: load grid (+ optional second CSV), denoise it
    along √s, fit morphs at every √s, return (sqrts_axis, splines, morphs).

    ``denoise=True`` (default) runs :func:`denoise_grid` so the per-√s
    morph numbers are smooth and the √s spline does not propagate MC
    noise; pass ``denoise=False`` to morph the raw grid (e.g. to expose
    the noise in a diagnostic)."""
    df = filter_uniform_gw(load_grid(grid_csv))
    if extra_csv is not None and extra_csv.exists():
        df_extra = filter_uniform_gw(load_grid(extra_csv))
        df = pd.concat([df, df_extra], ignore_index=True)
    return build_morph_from_df(df, denoise=denoise)


def build_morph_from_df(df: pd.DataFrame, *,
                        denoise: bool = True) -> tuple[np.ndarray, dict, list]:
    """Same as :func:`build_morph_from_grid` but takes an in-memory frame
    (already restricted to uniform Γ_W). Lets callers assemble custom input
    grids — combined campaigns, thinned LOO subsets — without round-tripping
    through CSVs.

    The ``denoise`` toggle controls both the upstream σ-ratio denoising
    (:func:`denoise_grid`) and the χ²-targeted β-spline smoothing inside
    :func:`build_splines`, so ``denoise=False`` reproduces the raw
    per-√s morph faithfully (for raw-vs-denoised diagnostic plots).
    """
    if denoise:
        df = denoise_grid(df)
    sqrts_grid = np.array(sorted(df.sqrts.unique()))
    morphs, kept = [], []
    for s in sqrts_grid:
        m = fit_morph_at_sqrts(df[np.isclose(df.sqrts, s, atol=1e-3)])
        if m is not None:
            morphs.append(m)
            kept.append(s)
    sqrts_axis = np.array(kept)
    return sqrts_axis, build_splines(sqrts_axis, morphs,
                                     denoise_beta=denoise), morphs


def build_operational_morph(*, denoise: bool = True
                            ) -> tuple[np.ndarray, dict, list]:
    """One-shot builder for the operational morph: loads the combined
    grid_fine + 0.5-GeV wings input, restricts to uniform Γ_W, denoises,
    fits at every √s, returns (sqrts_axis, splines, morphs).

    Single source of truth for the input grid used by every plot script
    under this directory."""
    df = filter_uniform_gw(load_operational_grid())
    return build_morph_from_df(df, denoise=denoise)
