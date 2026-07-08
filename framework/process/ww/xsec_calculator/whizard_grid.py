"""WHIZARD anchor grid — 3D interpolator over (√s, m_W, Γ_W).

Backs ``whizard_anchor_factor`` in :mod:`bfs_eft`. Reads the channel-specific
4f Born cross section σ(e⁺e⁻ → μ⁻ν̄_μ ud̄) [fb] from a rectangular grid CSV
and returns interpolated values. The grid file is produced by the pipeline
under ``WW_threshold/whizard/`` — see ``whizard/README.md`` for the
end-to-end reproducer recipe.

The grid path defaults to ``../whizard/work/grid/grid.csv`` relative to the
WW_threshold repo root; callers may override the grid path per-call via
``whizard_sigma(grid_path=...)``.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# WW_threshold/framework/process/ww/xsec_calculator/whizard_grid.py
#   parents[4]        = WW_threshold/      (the repo root)
#   parents[4].parent = QQbar_threshold/   (the parent of the repo)
# whizard/ sits one level ABOVE the repo, as a sibling of WW_threshold/.
_DEFAULT_GRID = (Path(__file__).resolve().parents[4].parent
                 / "whizard" / "work" / "grid" / "grid.csv")


def _build_interpolator(path: Path) -> RegularGridInterpolator:
    """Load grid.csv (5 columns: √s σ err m_W Γ_W) → 3D linear interpolator
    over (√s [GeV], m_W [GeV], Γ_W [GeV]) returning σ [fb].

    Linear (not cubic) because the 0.5 GeV √s spacing is dense enough that
    the residual interpolation error is well below the MC stat (≈0.05 %)
    of any individual grid point.
    """
    data = np.loadtxt(path, comments="#")
    sqrts = np.array(sorted({round(v, 2) for v in data[:, 0]}))
    mWs   = np.array(sorted({round(v, 5) for v in data[:, 3]}))
    gWs   = np.array(sorted({round(v, 5) for v in data[:, 4]}))

    sigma = np.full((len(sqrts), len(mWs), len(gWs)), np.nan)
    sq_idx = {v: i for i, v in enumerate(sqrts)}
    mw_idx = {v: i for i, v in enumerate(mWs)}
    gw_idx = {v: i for i, v in enumerate(gWs)}
    for row in data:
        sigma[sq_idx[round(row[0], 2)],
              mw_idx[round(row[3], 5)],
              gw_idx[round(row[4], 5)]] = row[1]

    if np.isnan(sigma).any():
        n_nan = int(np.isnan(sigma).sum())
        raise ValueError(
            f"WHIZARD grid at {path} is not rectangular: {n_nan} missing "
            f"points (expected {sigma.size}). Re-run whizard/fixup.py."
        )

    return RegularGridInterpolator(
        (sqrts, mWs, gWs), sigma,
        method="linear", bounds_error=False, fill_value=None,
    )


_INTERP_CACHE = {}


def _get_interpolator(path: Path) -> RegularGridInterpolator:
    """Cached per-path loader (loading is ~5 ms; cache keeps it amortised)."""
    key = str(path)
    if key not in _INTERP_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"WHIZARD grid not found at {path}. Generate it with "
                f"WW_threshold/whizard/ — see whizard/README.md."
            )
        _INTERP_CACHE[key] = _build_interpolator(path)
    return _INTERP_CACHE[key]


def whizard_sigma(s, mW: float, gammaW: float, *, grid_path: Optional[Path] = None):
    """Interpolated WHIZARD 4f Born σ(e⁺e⁻ → μ⁻ν̄_μ ud̄) in fb.

    Parameters
    ----------
    s : array-like
        Partonic CM energy² in GeV².
    mW, gammaW : float
        W mass and width in GeV.
    grid_path : Path, optional
        Override the default grid file (``../whizard/work/grid/grid.csv``).

    Returns
    -------
    sigma : array-like
        Same shape as ``s``. Trilinear interpolation; extrapolates past the
        grid edges (with fill_value=None) — caller's responsibility to keep
        inputs within physical range.
    """
    interp = _get_interpolator(Path(grid_path) if grid_path else _DEFAULT_GRID)
    sqrts = np.sqrt(np.asarray(s, dtype=float))
    mW_arr = np.full_like(sqrts, float(mW))
    gW_arr = np.full_like(sqrts, float(gammaW))
    points = np.stack([sqrts.ravel(), mW_arr.ravel(), gW_arr.ravel()], axis=-1)
    sigma = interp(points).reshape(sqrts.shape)
    if np.ndim(s) == 0:
        return float(sigma)
    return sigma
