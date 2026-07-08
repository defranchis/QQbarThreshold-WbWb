"""Luminosity (1-D) form of the two-leg eMELA-NLL ISR convolution — BFS chain.

Self-contained port of the independent chain's ``indep/isr_lumi.py`` to the
BFS-EFT partonic cross section.  The production BFS observable is the two-leg
double convolution
``σ_obs(s) = ∫∫ D(x₁) D(x₂) σ̂(√(x₁x₂)·√s) dx₁ dx₂``
(``isr.sigma_ISR_2leg_convolution``), a tensor Gauss-Legendre rule over the
(x₁, x₂) mesh.  The *edge-aware* variant already makes σ̂'s support floor an
integration LIMIT on both legs, so the BFS 2-leg line shape is smooth to ≲3 ppm
(much smoother than the MoCaNLO 2-D, whose σ̂ is a MC grid with a hard step).

This module offers the SAME collapse the independent chain uses — onto the
single luminosity variable ``z = x₁ x₂``::

    σ_obs(s) = ∫ L(z; μ_F) σ̂(√z·√s) dz ,   L(z) = ∫ (dx/x) D(x) D(z/x) ,

the two-leg LUMINOSITY = the per-leg radiator self-convolution.  In
``V = −ln z`` the self-convolution is additive and its soft ``V→0`` behaviour
factorises::

    L(z) dz = L_V(V) dV ,   L_V(V) = V^{2β′−1} · L̂(V; μ_F) ,   L̂ smooth,

with per-leg density ρ(v) = x·D (eMELA's ``code_pdf`` = x·D) and ``L̂`` the
Beta-type double-soft self-convolution
``L̂(V) = ∫₀¹ t^{β′−1}(1−t)^{β′−1} ρ̂(Vt) ρ̂(V(1−t)) dt`` integrated EXACTLY by
Gauss-Jacobi(β′−1, β′−1), where ``ρ̂(v) = x·D·v^{1−β_e+δ}`` is smooth AND
log-flat at v→0, and ``β′ = β_e − δ``.  δ (≈4e-4, the genuine NLL soft-drift
exponent) is measured from eMELA's own deep-endpoint log-log slope and absorbed
into the Jacobi weight — leaving it in the integrand loses ~0.4 % of σ_obs
(measured on the independent chain), so it must be divided out.

The outer V-integral uses a SINGLE Gauss-Jacobi(2β′−1, 0) panel on
``[0, V_top]`` with ``V_top = 2 ln(√s / SIGMA_FLOOR)``, ``SIGMA_FLOOR`` = the
BFS σ̂ support floor (``eft_xsec._SQRTS_BFS_FLOOR`` = 149 GeV, σ̂ ≡ 0 below): the
σ̂ support edge is the integration *limit*, never an interior node.

Independence / provenance
-------------------------
This is a BFS-chain module: it shares NO code with ``indep/`` — ρ comes from the
same ``emela_wrapper.code_pdf`` the BFS 2-leg path already queries (the eMELA
library and the λ₁(N_F=0) NLL constant are shared with the independent chain BY
DESIGN, as documented in ``isr.py``), and the σ̂ is the BFS partonic function.
It is therefore an independent quadrature confirmation of the edge-aware 2-leg
line shape, not a dependency on the MoCaNLO chain.

OPT-IN.  ``isr.sigma_observed_munuqq(..., isr_lumi=True)`` routes the NLL 2-leg
convolution here; the production default stays the edge-aware 2-leg tensor rule.
Validated to reproduce it to ≲30 ppm shape across the 157–163 GeV scan window
(``scripts/investigations/nll_isr/validate_lumi_bfs.py``).
"""
from __future__ import annotations

import math

import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import (
    M_E, ALPHA_MZ_PDG, M_W_BFS_REF, alpha_Gmu, _SQRTS_BFS_FLOOR,
)
from framework.process.ww.xsec_calculator.isr import (
    beta_ISR, _resolve_isr_alpha, LAMBDA1_NF0,
)

#: σ̂ support floor (σ̂ ≡ 0 below): caps V_top = 2 ln(√s/SIGMA_FLOOR) so the σ̂
#: support edge is always the outer integration LIMIT, never interior.  Bound to
#: the BFS partonic floor so it tracks any change there (no hand-synced literal).
SIGMA_FLOOR = _SQRTS_BFS_FLOOR

#: Production quadrature for the luminosity convolution (mirrors the independent
#: chain's LUMI_N_OUT / LUMI_N_JAC).  ``N_JAC`` (inner self-conv) deepest node
#: reaches omx≈3e-8, with the genuine NLL soft drift v^{−δ} absorbed into the
#: Jacobi weight so the deep endpoint is exact regardless of n_jac; ``N_OUT``
#: (outer panel) has the line SHAPE converged to ≲20 ppm.  Both are a ONE-TIME
#: per-√s cost (σ̂-independent, cached).
N_OUT = 192
N_JAC = 400

#: eMELA reference sampling for the per-leg ρ log-log spline (built once per √s
#: from direct ``code_pdf`` queries; deep enough that the Jacobi nodes stay
#: interior).  Log-spaced in omx = 1−x from OMX_HI down to OMX_LO.
RHO_N_REF = 256
OMX_HI = 0.5
OMX_LO = 1e-40


def _default_isr_alpha() -> float:
    """α_Gμ at the BFS reference m_W — the BFS-prescription ISR α default."""
    return alpha_Gmu(M_W_BFS_REF)


def _jac01(n: int, a: float, b: float):
    """Nodes t∈[0,1] and weights for ∫₀¹ t^a (1−t)^b f(t) dt (Gauss-Jacobi)."""
    from scipy.special import roots_jacobi
    x, w = roots_jacobi(n, b, a)              # scipy weight (1−x)^b (1+x)^a
    return 0.5 * (x + 1.0), w / 2.0 ** (a + b + 1.0)


# ---------------------------------------------------------------------------
# Per-leg density ρ(v) = x·D from direct eMELA (cached log-log spline per √s)
# ---------------------------------------------------------------------------

_RHO_CACHE: dict = {}


def _rho_spline(Q: float, alpha_a: float, emela_pert_order: str,
                emela_fac_scheme: str, emela_ren_scheme: str):
    """(ln-ln spline of x·D, deep_slope) at scale Q, from direct eMELA queries.

    Samples ``code_pdf(x, omx, Q)`` at ``RHO_N_REF`` log-spaced omx knots and
    fits a natural cubic spline of ``ln(x·D)`` vs ``ln omx`` — the variables in
    which the resummed soft structure function is log-linear at the deep edge
    (so extrapolation below the deepest knot is exact).  ``deep_slope`` =
    d ln(x·D)/d ln(omx) from the two deepest knots (= β_e−1−δ).  Cached on
    (Q, α, scheme) — σ̂-independent, reused across every σ̂ variation."""
    key = (round(float(Q), 9), round(float(alpha_a), 15),
           emela_pert_order, emela_fac_scheme, emela_ren_scheme)
    cached = _RHO_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.interpolate import CubicSpline
    from . import emela_wrapper as _emela
    _emela.initialize(pert_order=emela_pert_order, fac_scheme=emela_fac_scheme,
                      ren_scheme=emela_ren_scheme, alpha=alpha_a)
    omx = np.logspace(math.log10(OMX_HI), math.log10(OMX_LO), RHO_N_REF)
    ln_omx = np.log(omx)
    ln_xD = np.empty_like(omx)
    for i, om in enumerate(omx):
        x = 1.0 - om if om < 1.0 else 0.0
        ln_xD[i] = math.log(max(_emela.code_pdf(x, float(om), Q), 1e-300))
    order = np.argsort(ln_omx)               # ascending ln omx
    ln_omx, ln_xD = ln_omx[order], ln_xD[order]
    spline = CubicSpline(ln_omx, ln_xD, bc_type="natural")
    deep_slope = float((ln_xD[1] - ln_xD[0]) / (ln_omx[1] - ln_omx[0]))
    t_lo, val_lo = float(ln_omx[0]), float(ln_xD[0])
    out = (spline, t_lo, val_lo, deep_slope)
    _RHO_CACHE[key] = out
    return out


def _rho_hat_factory(sq: float, *, isr_scale_factor: float, alpha_a: float,
                     emela_pert_order: str, emela_fac_scheme: str,
                     emela_ren_scheme: str):
    """(ρ̂, β′) for one √s.  ρ̂(v) = x·D(x,Q)·v^{1−β_e+δ}, x = e^{−v}; smooth and
    LOG-FLAT at v→0 so Gauss-Jacobi(β′−1, β′−1) integrates the self-convolution
    exactly.  β_e is the per-leg BFS ISR exponent (β_ISR/2); δ = β_e−1−deep_slope
    the genuine NLL soft drift, read off eMELA's deep edge."""
    s = float(sq) * float(sq)
    Q = float(sq) * isr_scale_factor
    beta = beta_ISR(s, alpha_em=alpha_a, isr_scale_factor=isr_scale_factor)
    be = beta / 2.0                                        # per-leg β_e
    spline, t_lo, val_lo, deep_slope = _rho_spline(
        Q, alpha_a, emela_pert_order, emela_fac_scheme, emela_ren_scheme)
    delta = (be - 1.0) - deep_slope
    ex = 1.0 - be + delta

    def rho_hat(v):
        v = np.asarray(v, dtype=float)
        v_c = np.maximum(v, 1e-300)                        # v=0 is measure-zero
        omx = -np.expm1(-v_c)                              # 1−x, accurate small v
        t = np.log(np.maximum(omx, 1e-300))
        ln_xD = np.where(t < t_lo, val_lo + deep_slope * (t - t_lo),
                         spline(np.clip(t, t_lo, spline.x[-1])))
        return np.exp(ln_xD) * v_c ** ex

    return rho_hat, be - delta


def _ltilde_at(rho_hat, be_eff: float, V: np.ndarray, n_jac: int) -> np.ndarray:
    """L̂(V) = ∫₀¹ t^{β′−1}(1−t)^{β′−1} ρ̂(Vt) ρ̂(V(1−t)) dt, vectorised over V,
    by Gauss-Jacobi(β′−1, β′−1).  V=0 → ρ̂≡ρ̂(0) → L̂ = ρ̂(0)²·B(β′,β′) (exact)."""
    t, wj = _jac01(n_jac, be_eff - 1.0, be_eff - 1.0)
    V = np.atleast_1d(np.asarray(V, dtype=float))
    Vt = np.outer(V, t)
    rt1 = rho_hat(Vt.ravel()).reshape(Vt.shape)
    rt2 = rho_hat((V[:, None] * (1.0 - t)).ravel()).reshape(Vt.shape)
    return (rt1 * rt2) @ wj


# ---------------------------------------------------------------------------
# Per-√s luminosity setup (σ̂-independent outer-node weights), cached
# ---------------------------------------------------------------------------

_LUMI_CACHE: dict = {}


def _lumi_setup(sqrt_s_arr: np.ndarray, *, isr_scale_factor: float,
                alpha_a: float, emela_pert_order: str, emela_fac_scheme: str,
                emela_ren_scheme: str, n_out: int, n_jac: int):
    """List of (shat_arg, outer_w) per √s such that
    ``σ_obs(√s) = Σ outer_w · σ̂(shat_arg)`` — the σ̂-independent luminosity
    weights (outer Gauss-Jacobi panel × the directly-computed L̂ at its nodes)."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    key = ((a.shape, a.tobytes()), round(isr_scale_factor, 12),
           round(alpha_a, 15), emela_pert_order, emela_fac_scheme,
           emela_ren_scheme, int(n_out), int(n_jac))
    cached = _LUMI_CACHE.get(key)
    if cached is not None:
        return cached
    setups = []
    for sq in a:
        sq = float(sq)
        if sq <= SIGMA_FLOOR:                              # √s ≤ σ̂ floor → no support
            setups.append((np.array([sq]), np.array([0.0])))
            continue
        rho_hat, be_eff = _rho_hat_factory(
            sq, isr_scale_factor=isr_scale_factor, alpha_a=alpha_a,
            emela_pert_order=emela_pert_order, emela_fac_scheme=emela_fac_scheme,
            emela_ren_scheme=emela_ren_scheme)
        V_top = 2.0 * math.log(sq / SIGMA_FLOOR)
        tj, wj = _jac01(n_out, 2.0 * be_eff - 1.0, 0.0)    # ∫₀¹ τ^{2β′−1}
        V = V_top * tj
        Lt = _ltilde_at(rho_hat, be_eff, V, n_jac)
        outer_w = V_top ** (2.0 * be_eff) * wj * Lt        # σ̂-independent weight
        shat_arg = np.exp(-V / 2.0) * sq                   # √ŝ = √z·√s
        setups.append((shat_arg, outer_w))
    _LUMI_CACHE[key] = setups
    return setups


def sigma_obs(sqrt_s, sigma_hat_fn, *, alpha_em_isr: float | None = None,
              isr_scale_factor: float = 1.0, emela_pert_order: str = "NLL",
              emela_fac_scheme: str = "DELTA", emela_ren_scheme: str = "ALPMZ",
              n_out: int = N_OUT, n_jac: int = N_JAC):
    """σ_obs(√s) via the luminosity (single Gauss-Jacobi outer panel; L̂ by direct
    self-conv of the eMELA per-leg density ρ̂, cached per √s).

    Drop-in for the NLL branch of ``isr.sigma_ISR_2leg_convolution``:
    ``sigma_hat_fn(sqrt_shat)`` returns σ̂ for an array of √ŝ [GeV] (same units as
    the BFS partonic σ̂).  Vectorised in ``sqrt_s``.  ``alpha_em_isr`` selects the
    α in β_e and the eMELA scale (None → ALPMZ pairs α(M_Z); else α_Gμ(m_W_BFS))."""
    alpha_a = _resolve_isr_alpha(alpha_em_isr, emela_ren_scheme)
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.empty_like(sqrt_s_arr)
    setups = _lumi_setup(
        sqrt_s_arr, isr_scale_factor=isr_scale_factor, alpha_a=alpha_a,
        emela_pert_order=emela_pert_order, emela_fac_scheme=emela_fac_scheme,
        emela_ren_scheme=emela_ren_scheme, n_out=n_out, n_jac=n_jac)
    for idx in range(len(sqrt_s_arr)):
        shat_arg, outer_w = setups[idx]
        out[idx] = float(np.sum(outer_w
                                * np.asarray(sigma_hat_fn(shat_arg), dtype=float)))
    return float(out[0]) if np.ndim(sqrt_s) == 0 else out
