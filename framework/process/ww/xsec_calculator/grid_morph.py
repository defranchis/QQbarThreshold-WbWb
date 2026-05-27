"""WHIZARD-grid morphing predictor — framework module.

Promoted from ``scripts/investigations/whizard_grid_highstats/morph.py``.
Provides the production-quality morphing predictor used by
:func:`~bfs_eft.whizard_anchor_factor` when
``whizard_anchor_source="morph"``.

Operational predictor
---------------------

    σ_pred(√s, m_W, Γ_W) = σ_nom(√s)
                         × R_m(√s, m_W)            # quadratic in m_W
                         × R_Γ(√s, Γ_W)            # quadratic in Γ_W
                         × [1 + β(√s)·Δm·ΔΓ]      # bilinear cross

Each of the 8 √s-dependent quantities (σ_nom, 3 coef_m, 3 coef_g, β) is
fitted independently at every grid √s slice and then interpolated along
√s with a natural cubic spline.  The input grid is *denoised along √s*
before fitting so the spline threads smooth points rather than
per-point MC fluctuations (see :func:`denoise_grid`).

Input grid
----------
``grid_fine`` (6363 pts, 0.1-GeV √s step, 9 m_W × 7 Γ_W, ~0.008 % MC)
+ outer 0.5-GeV wings from ``grid_highstats`` for √s-spline support
beyond [155, 165] GeV.  Located at
``$QQBAR_THRESHOLD_ROOT/whizard/work/{grid_fine,grid_highstats}/grid.csv``
where ``QQBAR_THRESHOLD_ROOT`` is resolved as
``Path(__file__).resolve().parents[5]``.

Validation (2026-05-26)
-----------------------
* Per-√s quadratic RMS residual: 0.019 % (m_W), 0.017 % (Γ_W).
* 4D √s LOO (interior): max 0.088 %, median 0.005 %.
* 1-MeV held-out (grid_validate): max 0.038 %, median 0.009-0.018 %.
* Sub-MeV held-out (grid_validate_fine): max 0.020 %, median 0.004 %.
* BFS-table closure: T1/T2 max 0.149 % (WHIZARD 3.1.5 vs BFS-era gap).

Public interface
----------------
:func:`whizard_sigma_morph` — drop-in replacement for
:func:`~whizard_grid.whizard_sigma`; takes ``s`` (GeV²), returns σ [fb].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline

# ---------------------------------------------------------------------------
# Nominal anchor point and Γ_W subset
# ---------------------------------------------------------------------------
MW0 = 80.379
GW0 = 2.085
GW_UNIFORM = [2.045, 2.065, 2.085, 2.105, 2.125]

# ---------------------------------------------------------------------------
# Grid paths
# ---------------------------------------------------------------------------
# framework/process/ww/xsec_calculator/grid_morph.py
#   parents[4] = WW_threshold/
#   parents[5] = QQbar_threshold/  ← shared whizard/ directory lives here
_QQBAR_ROOT    = Path(__file__).resolve().parents[5]
_WHIZARD_WORK  = _QQBAR_ROOT / "whizard" / "work"

GRID_FINE_CSV          = _WHIZARD_WORK / "grid_fine"       / "grid.csv"
GRID_HIGHSTATS_CSV     = _WHIZARD_WORK / "grid_highstats"  / "grid.csv"
GRID_VALIDATE_CSV      = _WHIZARD_WORK / "grid_validate"   / "grid.csv"
GRID_VALIDATE_FINE_CSV = _WHIZARD_WORK / "grid_validate_fine" / "grid.csv"


# ---------------------------------------------------------------------------
# Grid I/O helpers
# ---------------------------------------------------------------------------

def load_grid(path: Path) -> pd.DataFrame:
    """Read a 5-column WHIZARD grid CSV (√s  σ  err  m_W  Γ_W) in fb."""
    return pd.read_csv(path, comment="#", sep=r"\s+", header=None,
                       names=["sqrts", "sigma_fb", "err_fb", "mW", "gammaW"])


def filter_uniform_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop BFS-reference Γ_W duplicates; keep only the 5 PDG-uniform values."""
    return df[np.isin(np.round(df.gammaW, 5), GW_UNIFORM)].copy()


def load_operational_grid() -> pd.DataFrame:
    """Combined grid_fine (0.1-GeV √s step) + outer wings from grid_highstats.

    Returns the full frame including the two BFS-reference Γ_W rows so
    external closure scripts can compare at those exact values.
    :func:`build_morph_from_df` calls :func:`filter_uniform_gw` internally
    before fitting, so the BFS-ref rows do not enter the morph fit.
    """
    fine = load_grid(GRID_FINE_CSV)
    hs   = load_grid(GRID_HIGHSTATS_CSV)
    fine_sqrts = set(np.round(fine.sqrts.unique(), 4))
    wings = hs[~np.round(hs.sqrts, 4).isin(fine_sqrts)].copy()
    return pd.concat([fine, wings], ignore_index=True)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise_grid(df: pd.DataFrame, *, chi2_per_dof: float = 1.0) -> pd.DataFrame:
    """Denoise the grid along √s in three passes.

    Pass 1 — ratios R(√s) = σ(√s, m_W, Γ_W) / σ_nom are χ²-smoothed
    (weighted ``UnivariateSpline`` targeting χ² ≈ N within MC error bars).

    Pass 2 — σ_nom is denoised by averaging σ_raw / R_smooth over the
    (m_W, Γ_W) plane (no √s-smoothing of the steep line shape).

    Pass 3 — σ = R_smooth · σ_nom_clean.

    The R_m / R_Γ polynomial coefficients are **never smoothed directly**
    to preserve the anti-correlated error cancellation in ``polyval``.
    """
    out = df.copy()
    nom = df[(np.isclose(df.mW, MW0)) & (np.isclose(df.gammaW, GW0))]
    nom = nom.sort_values("sqrts")
    x_nom, snom_raw = nom.sqrts.values, nom.sigma_fb.values

    curves = []
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

    est: dict[float, list] = {}
    for _, x, y, Rs in curves:
        for xi, yi, ri in zip(x, y, Rs):
            if ri > 0:
                est.setdefault(round(float(xi), 4), []).append(yi / ri)
    snom_clean = {k: float(np.mean(v)) for k, v in est.items()}

    for index, x, _, Rs in curves:
        sc = np.array([snom_clean[round(float(xi), 4)] for xi in x])
        out.loc[index, "sigma_fb"] = Rs * sc
    return out


# ---------------------------------------------------------------------------
# Per-√s slice fit
# ---------------------------------------------------------------------------

def fit_morph_at_sqrts(slice_df: pd.DataFrame, *,
                       mw0: float = MW0, gw0: float = GW0) -> dict | None:
    """Fit the 8 morph numbers at one √s slice.

    Returns a dict with keys ``sigma_nom``, ``coef_m`` (array[3]),
    ``coef_g`` (array[3]), ``beta``, ``beta_err``, or ``None`` if the
    slice lacks the required nominal axes.
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

    off = slice_df[(~np.isclose(slice_df.mW, mw0))
                    & (~np.isclose(slice_df.gammaW, gw0))]
    if len(off) == 0:
        return None
    Rm = np.polyval(coef_m, off.mW.values)    / sigma_nom_m
    Rg = np.polyval(coef_g, off.gammaW.values) / sigma_nom_g
    sigma_pred_no_cross = sigma_nom * Rm * Rg
    dm = off.mW.values    - mw0
    dG = off.gammaW.values - gw0
    A  = sigma_pred_no_cross * dm * dG
    b  = off.sigma_fb.values - sigma_pred_no_cross
    w  = 1.0 / off.err_fb.values**2
    waa = float(np.sum(w * A * A))
    if waa > 0:
        beta     = float(np.sum(w * A * b) / waa)
        beta_err = float(1.0 / np.sqrt(waa))
    else:
        beta     = 0.0
        beta_err = float("inf")

    return {"sigma_nom": sigma_nom, "coef_m": coef_m, "coef_g": coef_g,
            "beta": beta, "beta_err": beta_err}


# ---------------------------------------------------------------------------
# √s-spline builder
# ---------------------------------------------------------------------------

def build_splines(sqrts_axis: np.ndarray, morphs: list[dict], *,
                  denoise_beta: bool = True,
                  beta_chi2_per_dof: float = 1.0) -> dict:
    """Natural cubic spline of each of the 8 morph numbers along √s.

    R_m / R_Γ coefficients must not be smoothed independently (they are
    anti-correlated; ``polyval`` accuracy relies on their joint behaviour).
    β is a scalar and admits a χ²-targeted weighted spline directly.
    """
    kw = dict(bc_type="natural", extrapolate=True)

    beta_vals = np.array([m["beta"] for m in morphs], dtype=float)
    if denoise_beta and all("beta_err" in m and np.isfinite(m["beta_err"])
                            for m in morphs):
        beta_err = np.array([m["beta_err"] for m in morphs], dtype=float)
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


# ---------------------------------------------------------------------------
# Morph evaluator
# ---------------------------------------------------------------------------

def sigma_morph(sqrts, mW: float, gammaW: float, *, splines: dict,
                mw0: float = MW0, gw0: float = GW0):
    """Morphing prediction at (√s, m_W, Γ_W); √s may be scalar or array.

    .. note::
        The argument is ``sqrts`` (GeV), not ``s`` (GeV²).  The public
        :func:`whizard_sigma_morph` accepts ``s`` (GeV²) and converts.
    """
    coef_m = np.array([splines["coef_m_0"](sqrts),
                       splines["coef_m_1"](sqrts),
                       splines["coef_m_2"](sqrts)])
    coef_g = np.array([splines["coef_g_0"](sqrts),
                       splines["coef_g_1"](sqrts),
                       splines["coef_g_2"](sqrts)])
    Rm    = np.polyval(coef_m, mW)     / np.polyval(coef_m, mw0)
    Rg    = np.polyval(coef_g, gammaW) / np.polyval(coef_g, gw0)
    cross = 1.0 + splines["beta"](sqrts) * (mW - mw0) * (gammaW - gw0)
    return splines["sigma_nom"](sqrts) * Rm * Rg * cross


# ---------------------------------------------------------------------------
# Grid → splines builders
# ---------------------------------------------------------------------------

def build_morph_from_df(df: pd.DataFrame, *,
                        denoise: bool = True) -> tuple[np.ndarray, dict, list]:
    """Build morph splines from an in-memory frame (uniform Γ_W subset).

    Returns ``(sqrts_axis, splines, morphs)``.  ``denoise=False`` keeps
    raw per-√s values — useful for diagnostic raw-vs-denoised comparisons.
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
    """One-shot builder: grid_fine + 0.5-GeV wings → denoised morph splines.

    This is the single source of truth for the input grid used in
    production.  The result is cached by :func:`_get_morph_splines`.
    """
    df = filter_uniform_gw(load_operational_grid())
    return build_morph_from_df(df, denoise=denoise)


# ---------------------------------------------------------------------------
# Cached production interface
# ---------------------------------------------------------------------------

_MORPH_CACHE: dict[str, dict] = {}


def _get_morph_splines(grid_path=None) -> dict:
    """Return cached morph splines; build on first call (~3 s).

    ``grid_path=None`` uses the operational grid (grid_fine + wings).
    Pass an explicit ``Path`` to build from a custom grid (e.g. for
    diagnostic or validation comparisons).
    """
    key = "operational" if grid_path is None else str(grid_path)
    if key not in _MORPH_CACHE:
        if grid_path is None:
            _, splines, _ = build_operational_morph()
        else:
            df = filter_uniform_gw(load_grid(Path(grid_path)))
            _, splines, _ = build_morph_from_df(df)
        _MORPH_CACHE[key] = splines
    return _MORPH_CACHE[key]


def whizard_sigma_morph(s, mW: float, gammaW: float, *,
                        grid_path=None):
    """Morphing-predictor estimate of σ_WHIZARD(e⁺e⁻ → μ⁻ν̄_μ ud̄) in fb.

    Drop-in replacement for :func:`~whizard_grid.whizard_sigma`.

    Parameters
    ----------
    s : array-like
        Partonic CM energy² in GeV².  Converted to √s internally.
    mW, gammaW : float
        W mass and width in GeV.
    grid_path : path-like, optional
        Override the default operational grid.  Splines are cached per path.

    Returns
    -------
    sigma : float or ndarray
        Same shape as ``s``, in fb.
    """
    splines = _get_morph_splines(grid_path)
    sqrts   = np.sqrt(np.asarray(s, dtype=float))
    result  = sigma_morph(sqrts, mW, gammaW, splines=splines)
    if np.ndim(s) == 0:
        return float(result)
    return result
