"""Factorized (BFS-style) morph for the independent MoCaNLO line shapes.

    σ(√s; m_W, Γ_W) = σ_nom(√s)
                    · R_m(√s, Δm)            # quadratic ratio in Δm
                    · R_Γ(√s, ΔΓ)            # quadratic ratio in ΔΓ
                    · [1 + β(√s)·Δm·ΔΓ]      # bilinear cross

This mirrors the BFS production morph
(:mod:`framework.process.ww.xsec_calculator.grid_morph`) but operates on the
independent varpoint line shapes — one assembled σ(√s) array per (Δm, ΔΓ)
varpoint — rather than a WHIZARD grid CSV, so the independent calculation
shares no code with the BFS chain.

Why factorized rather than the earlier additive Taylor lstsq
(``c0+c1Δm+c2ΔΓ+c3Δm²+c4ΔΓ²+c5ΔmΔΓ``):

* the steep √s line-shape lives entirely in ``σ_nom``; the responses ``R_m``,
  ``R_Γ`` are dimensionless ratios ≈ 1 that vary slowly with √s, so the fit is
  better conditioned and the nominal is reproduced **exactly**
  (``σ_nom · 1 · 1 · 1`` at the reference);
* the additive cross term ``c5·ΔmΔΓ`` has to absorb both the implicit product
  cross ``R_m·R_Γ`` *and* the genuine non-separability β; the factorized form
  isolates β cleanly on the diagonal varpoints.

Δm, ΔΓ are in **MeV** — the varpoint design-matrix coordinates
(``VarPoint.dmW_MeV`` / ``dgW_MeV``).  The on-axis varpoints (one of Δm, ΔΓ
zero) pin ``coef_m`` / ``coef_g``; the off-axis (cross) varpoints pin β.  Each
morph number is returned on the input √s grid — already √s-smooth from the σ̂
``UnivariateSpline`` upstream — with an optional weighted-spline denoise of β
(the noisiest, fitted from only the few cross points).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _polyval_cols(coef: np.ndarray, x: float) -> np.ndarray:
    """Evaluate, at scalar ``x``, a stack of polynomials stored column-wise.

    ``coef`` has shape ``(deg+1, n_s)`` with the highest power first (the
    :func:`numpy.polyfit` convention); returns shape ``(n_s,)``.
    """
    out = np.zeros(coef.shape[1], dtype=float)
    for c in coef:                     # highest power first → Horner
        out = out * x + c
    return out


@dataclass
class FactorizedMorph:
    """Per-√s factorized morph numbers + the multiplicative evaluator."""
    sqrt_s: np.ndarray         # (n_s,)
    sigma_nom: np.ndarray      # (n_s,)   nominal line shape (reproduced exactly)
    coef_m: np.ndarray         # (deg_m+1, n_s)  quadratic in Δm [MeV]
    coef_g: np.ndarray         # (deg_g+1, n_s)  quadratic in ΔΓ [MeV]
    beta: np.ndarray           # (n_s,)   bilinear cross coefficient

    def evaluate(self, dmW_MeV: float, dgW_MeV: float) -> np.ndarray:
        """σ(√s) [same units as the input line shapes] at (Δm, ΔΓ) in MeV."""
        dm, dw = float(dmW_MeV), float(dgW_MeV)
        Rm = _polyval_cols(self.coef_m, dm) / _polyval_cols(self.coef_m, 0.0)
        Rg = _polyval_cols(self.coef_g, dw) / _polyval_cols(self.coef_g, 0.0)
        cross = 1.0 + self.beta * dm * dw
        return self.sigma_nom * Rm * Rg * cross


def _smooth_beta(sqrt_s: np.ndarray, beta: np.ndarray,
                 chi2_per_dof: float = 1.0) -> np.ndarray:
    """χ²-targeted natural smoothing of β(√s) (the noisiest morph number).

    β has no per-point MC error here (it is fit from already-denoised line
    shapes), so use a unit-weight :class:`~scipy.interpolate.UnivariateSpline`
    with ``s = chi2·N`` — mirrors ``grid_morph.build_splines(denoise_beta=True)``.
    """
    from scipy.interpolate import UnivariateSpline
    order = min(3, len(sqrt_s) - 1)
    if order < 1:
        return beta
    spl = UnivariateSpline(sqrt_s, beta, k=order, s=chi2_per_dof * len(sqrt_s))
    return spl(sqrt_s)


def fit_factorized(coords: dict[str, tuple[float, float]],
                   lineshapes: dict[str, np.ndarray],
                   sqrt_s: np.ndarray, *,
                   deg_m: int = 2, deg_g: int = 2,
                   denoise_beta: bool = False) -> FactorizedMorph:
    """Build a :class:`FactorizedMorph` from the varpoint line shapes.

    Parameters
    ----------
    coords
        ``varpoint_key -> (Δm_MeV, ΔΓ_MeV)`` for every key in ``lineshapes``.
    lineshapes
        ``varpoint_key -> σ(√s)`` array (assembled total, any units), all on the
        same ``sqrt_s`` grid.
    deg_m, deg_g
        Polynomial degree of the m_W / Γ_W ratio fits (default quadratic, as BFS).
    denoise_beta
        χ²-smooth β(√s) along √s (default off — the line shapes are already
        √s-smooth, so β is too; enable if the cross varpoints are noisy).
    """
    keys = list(lineshapes)
    dm = np.array([coords[k][0] for k in keys], dtype=float)
    dw = np.array([coords[k][1] for k in keys], dtype=float)
    Y = np.array([np.asarray(lineshapes[k], dtype=float) for k in keys])  # (n_vp, n_s)
    sqrt_s = np.asarray(sqrt_s, dtype=float)

    nom = (dm == 0.0) & (dw == 0.0)
    if int(nom.sum()) != 1:
        raise ValueError("factorized morph needs exactly one nominal varpoint "
                         f"(Δm=ΔΓ=0); found {int(nom.sum())}")
    sigma_nom = Y[nom][0]

    m_axis = dw == 0.0                          # includes nominal
    g_axis = dm == 0.0
    if int(m_axis.sum()) < deg_m + 1 or int(g_axis.sum()) < deg_g + 1:
        raise ValueError(
            f"insufficient on-axis varpoints: {int(m_axis.sum())} on the m-axis / "
            f"{int(g_axis.sum())} on the Γ-axis, need ≥{deg_m + 1}/{deg_g + 1} for "
            f"the degree-{deg_m}/{deg_g} ratio fits")
    coef_m = np.polyfit(dm[m_axis], Y[m_axis], deg_m)   # (deg_m+1, n_s)
    coef_g = np.polyfit(dw[g_axis], Y[g_axis], deg_g)

    # β from the off-axis (cross) varpoints: σ ≈ σ_nom·R_m·R_Γ·(1+β·Δm·ΔΓ).
    # Solve the residual b = σ − σ_pred,no-cross against A = σ_pred,no-cross·Δm·ΔΓ
    # per √s (least squares over the cross points).
    cross = (dm != 0.0) & (dw != 0.0)
    if not cross.any():
        beta = np.zeros_like(sigma_nom)
    else:
        denom_m = _polyval_cols(coef_m, 0.0)
        denom_g = _polyval_cols(coef_g, 0.0)
        Rm_c = np.array([_polyval_cols(coef_m, x) for x in dm[cross]]) / denom_m
        Rg_c = np.array([_polyval_cols(coef_g, x) for x in dw[cross]]) / denom_g
        pred = sigma_nom[None, :] * Rm_c * Rg_c          # (n_cross, n_s)
        A = pred * (dm[cross] * dw[cross])[:, None]       # (n_cross, n_s)
        b = Y[cross] - pred
        waa = np.sum(A * A, axis=0)
        beta = np.divide(np.sum(A * b, axis=0), waa,
                         out=np.zeros_like(waa), where=waa > 0)

    if denoise_beta:
        beta = _smooth_beta(sqrt_s, beta)

    return FactorizedMorph(sqrt_s, sigma_nom, coef_m, coef_g, beta)
