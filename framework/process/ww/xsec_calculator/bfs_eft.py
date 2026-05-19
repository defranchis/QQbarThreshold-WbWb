"""BFS unstable-particle EFT cross section for e+e- → W+W- near threshold.

Closed-form leading-order results from Beneke, Falgari, Schwinn, Signer,
Zanderighi, *"Four-fermion production near the W pair production
threshold"*, [arXiv:0707.0773](https://arxiv.org/abs/0707.0773).

Conventions match the paper. The natural BFS quantity is σ_LR^(0), the
LR-helicity cross section for the *specific* channel e+e- → μ⁻ν̄_μ ud̄;
helpers below convert to unpolarised totals.

Born-level EFT contributions implemented here:

  • σ_LR^(0)             doubly-resonant LO,                eq. (17)
  • σ_LR^(1), σ_RL^(1)   NLO subleading potential,          eq. (33)
  • σ_LR^(1/2), σ_RL^(1/2)
                         hard non-resonant h1-h3 + h4-h7,   eq. (37) + (38)
                         + appendix A C^f_{h_i,h} K^f_{h_i} coefficients
  • σ_LR^(3/2),a, σ_RL^(3/2),a
                         energy-dependent N^{3/2}LO h1-h3,  eq. (39) + (40)

NLO loop corrections (section 4 of the paper):
  • hard production C_p,LR^(1,fin), eq. (52) + HSC bracket (eq. finalcross)
  • NLO Coulomb (eq. 62)
  • NLO decay EW (eq. Gamma1ewFS / eq. 60)
  • δ_QCD multiplier (eq. delta_qcd / section 6.1)

Omitted / negligible:
  • σ^(3/2),b (eq. 41) — paper estimate ≲ 2 fb energy-independent. Empirical
    Scenario A closure to 4-5 digits at all BFS Table 1 √s confirms this
    piece contributes < 0.01 fb in practice — much smaller than 2 fb.
  • Im[c_p,LR^(1,fin)] — paper §4.2 around eq. (53): "...we take the real
    part of the matching coefficients C_p,LR^(1) and C_p,RL^(1)" for the
    flavour-specific cross section.
  • C_p,RL^(1) NLO — paper §4.1: no helicity interference, drops at NLO.

The sum σ^(0) + σ^(1) + σ^(1/2) + σ^(3/2),a is what the paper labels
``EFT(N^{3/2}LO)`` in Table 1 — Scenario A closure with this code is
0.9999-1.0000 (4-5 digits) at all six BFS Table 1 energies.

Auxiliary kinematic functions, eq. (32):

  • ξ(s) = −3 M_W² (s − 2 M_Z² s_W²) / (s (s − M_Z²))
  • χ(s) = −6 M_W² M_Z² s_W²        / (s (s − M_Z²))

All cross sections returned in pb. Vectorised in ``s``.
"""

from __future__ import annotations

import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import (
    ALPHA_EM_0,                         # not used for these LO formulae; kept for parity
    ALPHA_S_MW_DEFAULT,
    GEV_M2_TO_PB,
    M_W_DEFAULT, GAMMA_W_DEFAULT, M_Z, M_W_BFS_REF,
    alpha_Gmu, sin2_thetaW_OS,
)


def gamma_W_LO(mW: float) -> float:
    """LO theoretical W width Γ_W^(0) = 3 α(m_W) m_W / (4 s_W²), eq. (19)
    of arXiv:0707.0773. Used in the BR-factor convention for the BFS
    formulae (see ``_BR_correction`` below).
    """
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    return 3.0 * alpha * mW / (4.0 * sW2)


def _BR_correction(mW: float, gammaW: float) -> float:
    """Replace the LO BR factor 1/27 in the BFS formulae (eqs. 17, 33, 37,
    39) with Γ_μν^(0) Γ_ud̄^(0) / Γ_W² = (Γ_W^(0)/Γ_W)² × 1/27 when the
    propagator uses an NLO+ width Γ_W ≠ Γ_W^(0). See section 6.1 of
    arXiv:0707.0773 (paragraph after eq. 82). Without this correction
    the BFS prediction is ~4–5 % too large when Γ_W is the physical
    (PDG / NLO) width rather than the LO theoretical value.
    """
    gW0 = gamma_W_LO(mW)
    return (gW0 / gammaW) ** 2


# --- coefficients from the paper -------------------------------------------
# eq. (38) — h1, h2, h3 N^{1/2}LO hard non-resonant
_K_H1 = -2.35493
_K_H2 = +3.86286
_K_H3 = +1.88122
# eq. (40) — h1, h2, h3 N^{3/2}LO,a energy-dependent
_K_H1_A = -5.87912
_K_H2_A = -19.15095
_K_H3_A = -6.18662

# BFS appendix h4-h7 single-resonant coefficients (lines 3116-3148 of source).
# Per-fermion K factors; the same f flavor labels (u, d, μ, ν_μ) used for the
# four final-state fermions in the specific channel μ⁻ν̄_μ u d̄.
_K_H4 = {"u": -0.266477, "nu_mu": -0.266477, "d": +0.190394, "mu": +0.190394}
_K_H5 = {"u": +0.455244, "nu_mu": +0.455244, "d": -0.455244, "mu": -0.455244}
_K_H6 = {"u": +0.0804075, "d": +0.0804075, "mu": +0.0804075, "nu_mu": +0.0804075}
_K_H7 = {"u": +0.0213082, "d": +0.0213082, "mu": +0.0213082, "nu_mu": +0.0213082}

# Fermion electric charges (in units of the positron charge) and weak isospin.
_Q_F = {"u": 2.0/3.0, "d": -1.0/3.0, "mu": -1.0, "nu_mu": 0.0}
_I3_F = {"u": +0.5, "d": -0.5, "mu": -0.5, "nu_mu": +0.5}
# SU(2) doublet partner mapping (used in h7 "barred" couplings).
_PARTNER = {"u": "d", "d": "u", "mu": "nu_mu", "nu_mu": "mu"}


def _ew_couplings(mW: float):
    """Return (sW, cW, Q_e, C_e_L, C_e_R, C_f_L per flavor) needed for
    the BFS h4-h7 single-resonant coefficients (BFS appendix line 3119).

    C_f^L = (I_W,f^3 − sW² Q_f) / (sW c_w)
    C_e^L = same with f = e (Q = -1, I^3 = -1/2)
    C_e^R = −(sW/c_w) Q_e = +sW/c_w     (since Q_e = -1)
    """
    sW2 = sin2_thetaW_OS(mW)
    sW = np.sqrt(sW2)
    cW = np.sqrt(1.0 - sW2)
    Q_e = -1.0
    I3_e = -0.5
    C_e_L = (I3_e - sW2 * Q_e) / (sW * cW)
    C_e_R = -(sW / cW) * Q_e
    C_f_L = {f: (_I3_F[f] - sW2 * _Q_F[f]) / (sW * cW) for f in _Q_F}
    return sW, cW, sW2, Q_e, C_e_L, C_e_R, C_f_L


def _C_h4_LR(s, mW: float, f: str):
    """C^f_{h4, LR} = 3 M_W² sW² × [-Q_f/s + C_e^L C_f^L / (s - M_Z²)]."""
    sW, cW, sW2, Q_e, C_e_L, C_e_R, C_f_L = _ew_couplings(mW)
    s_arr = np.asarray(s, dtype=float)
    return 3.0 * mW**2 * sW2 * (
        -_Q_F[f] / s_arr + C_e_L * C_f_L[f] / (s_arr - M_Z**2)
    )


def _C_h5(s, mW: float, h: str, f: str):
    """h ∈ {'LR', 'RL'}; both contribute.
       C^f_{h5,h} = 9 M_W^4 sW^4 ×
           [-Q_f/s² + C_e^h C_f^L / (s(s-M_Z²)) + (c_w/sW) Q_f C_e^h / (s(s-M_Z²))
            - (c_w/sW) (C_e^h)² C_f^L / (s-M_Z²)²]
    """
    sW, cW, sW2, Q_e, C_e_L, C_e_R, C_f_L = _ew_couplings(mW)
    C_e_h = C_e_L if h == "LR" else C_e_R
    s_arr = np.asarray(s, dtype=float)
    sm = s_arr - M_Z**2
    return 9.0 * mW**4 * sW2**2 * (
        -_Q_F[f] / s_arr**2
        + C_e_h * C_f_L[f] / (s_arr * sm)
        + (cW / sW) * _Q_F[f] * C_e_h / (s_arr * sm)
        - (cW / sW) * C_e_h**2 * C_f_L[f] / sm**2
    )


def _C_h6(s, mW: float, h: str, f: str):
    """C^f_{h6,h} = 9 M_W^4 sW^4 × [-Q_f/s + C_e^h C_f^L / (s-M_Z²)]²"""
    sW, cW, sW2, Q_e, C_e_L, C_e_R, C_f_L = _ew_couplings(mW)
    C_e_h = C_e_L if h == "LR" else C_e_R
    s_arr = np.asarray(s, dtype=float)
    sm = s_arr - M_Z**2
    inner = -_Q_F[f] / s_arr + C_e_h * C_f_L[f] / sm
    return 9.0 * mW**4 * sW2**2 * inner ** 2


def _C_h7(s, mW: float, h: str, f: str):
    """C^f_{h7,h} = 9 M_W^4 sW^4 × [
         Q_f Q̄_f/s² − Q_f C_e^h C̄_f^L/(s(s-M_Z²)) − Q̄_f C_e^h C_f^L/(s(s-M_Z²))
         + (C_e^h)² C_f^L C̄_f^L / (s-M_Z²)²
       ]
    where the "bar" denotes the SU(2) doublet partner."""
    sW, cW, sW2, Q_e, C_e_L, C_e_R, C_f_L = _ew_couplings(mW)
    C_e_h = C_e_L if h == "LR" else C_e_R
    f_bar = _PARTNER[f]
    Q_f, Q_fbar = _Q_F[f], _Q_F[f_bar]
    Cf_L, Cfbar_L = C_f_L[f], C_f_L[f_bar]
    s_arr = np.asarray(s, dtype=float)
    sm = s_arr - M_Z**2
    return 9.0 * mW**4 * sW2**2 * (
        Q_f * Q_fbar / s_arr**2
        - Q_f * C_e_h * Cfbar_L / (s_arr * sm)
        - Q_fbar * C_e_h * Cf_L / (s_arr * sm)
        + C_e_h**2 * Cf_L * Cfbar_L / sm**2
    )


def _sigma_half_h47_LR(s, mW: float):
    """h4-h7 contribution to σ_LR^(1/2) at the level of the BFS bracket
    [K_h1 + K_h2 ξ + K_h3 ξ²] of eq. (treehard). Returns the contribution
    to be added to the existing K_h1+K_h2·ξ+K_h3·ξ² bracket.

    Per BFS source line 3112: "Only the configuration e_L^- e_R^+
    contributes to the cut diagram h4". So h4 enters σ_LR only.
    h5, h6, h7 contribute to both helicities.
    """
    contrib = 0.0
    for f in _Q_F:
        contrib = contrib + _C_h4_LR(s, mW, f) * _K_H4[f]
        contrib = contrib + _C_h5(s, mW, "LR", f) * _K_H5[f]
        contrib = contrib + _C_h6(s, mW, "LR", f) * _K_H6[f]
        contrib = contrib + _C_h7(s, mW, "LR", f) * _K_H7[f]
    return contrib


def _sigma_half_h57_RL(s, mW: float):
    """h5, h6, h7 contribution to σ_RL^(1/2). h4 only contributes to LR."""
    contrib = 0.0
    for f in _Q_F:
        contrib = contrib + _C_h5(s, mW, "RL", f) * _K_H5[f]
        contrib = contrib + _C_h6(s, mW, "RL", f) * _K_H6[f]
        contrib = contrib + _C_h7(s, mW, "RL", f) * _K_H7[f]
    return contrib

# ALPHA_S_MW_DEFAULT is imported from process.ww.xsec_calculator.eft_xsec (single source of
# truth). Used by ``delta_QCD_factor`` (BFS eq. delta_qcd).


# ---------------------------------------------------------------------------
# Whizard-anchor correction — BFS section 6.2 prescription
# ---------------------------------------------------------------------------
# BFS replaces the EFT Born by the Whizard exact 4f Born in their published
# Table 4 numbers (paper line 2647-2656). We do the same: multiply the
# BFS-EFT N^(3/2)LO Born by
#
#     f(s, m_W, Γ_W) = σ_Whizard / σ_EFT-N3/2
#
# Two sources are kept side-by-side and selectable via the ``source`` arg
# of :func:`whizard_anchor_factor`:
#
#   "spline" — BFS arXiv:0707.0773 Tables 1+2: cubic spline in
#              δ = √s − 2 m_W, linear interp in Γ_W between the two
#              reference points {2.04483, 2.09201}. Smooth in (s, m_W, Γ_W)
#              with no MC stat noise, but only two Γ_W anchor points.
#
#   "grid"   — 1295-point WHIZARD 3.1.5 grid (5 m_W × 7 Γ_W × 37 √s)
#              produced by ``WW_threshold/whizard/``, trilinear interp in
#              (s, m_W, Γ_W). Full (m_W, Γ_W) coverage with no
#              extrapolation, but carries ~0.05-0.2 % per-point MC stat
#              noise that propagates into variation templates.
#
# Both implementations are BR-aware: when called with
# ``apply_BR_correction=False`` the σ_EFT denominator is brought to the
# same BR convention as the caller's σ_LR/σ_RL, avoiding the (Γ^(0)/Γ_W)²
# factor mismatch.

# ----- "spline" source: BFS Tables 1+2 ratios --------------------------------
# Data taken directly from BFS arXiv:0707.0773:
#   Table 1 (LO width): m_W = 80.377, Γ_W = Γ_W^(0)(80.377) = 2.04483 GeV
#   Table 2 (NLO+QCD width): m_W = 80.379, Γ_W = 2.09201 GeV
# Both at: M_Z = 91.188, m_t = 174.2, M_H = 115, G_μ = 1.16637e-5.
#
# δ = √s − 2 m_W is essentially the same at each √s for Tables 1 and 2 (the
# 2 MeV m_W difference is negligible compared to the 3 GeV δ spacing). For
# the spline we use the Table 1 δ values as the abscissa. The BFS Tables
# use the per-component BR correction (eq. 83), so the spline's "native"
# denominator is the BR-corrected σ_EFT.

_BFS_TABLE_1_SQRTS = [155.0, 158.0, 161.0, 164.0, 167.0, 170.0]
_BFS_TABLE_1_MW    = 80.377
_BFS_TABLE_1_GW    = 2.04483
_BFS_TABLE_1_EFT   = [31.30, 62.50, 160.89, 318.80, 429.70, 505.40]   # fb, EFT N^(3/2)LO
_BFS_TABLE_1_WHIZ  = [34.43, 63.39, 160.62, 318.30, 428.60, 505.10]   # fb, Whizard 4f Born

_BFS_TABLE_2_SQRTS = [155.0, 158.0, 161.0, 164.0, 167.0, 170.0]
_BFS_TABLE_2_MW    = 80.379
_BFS_TABLE_2_GW    = 2.09201
_BFS_TABLE_2_EFT   = [30.54,  60.83, 154.44, 303.70, 409.30, 481.70]
_BFS_TABLE_2_WHIZ  = [33.58,  61.67, 154.19, 303.00, 408.80, 481.70]


def _build_whizard_anchor_splines():
    """Cubic splines of σ_Whiz(δ) at the two BFS reference Γ_W values.
    Abscissa is δ = √s − 2 m_W from the Tables' (constant) m_W; natural BC.

    We spline σ_Whiz (the Whizard column), NOT the σ_Whiz/σ_EFT ratio: the
    σ_EFT denominator is recomputed at the actual (s, m_W, Γ_W) with the
    caller's BR convention so the σ^(0):σ^(1/2):σ^(1)_pot:σ^(3/2),a mix
    is handled correctly (the ratio splines would bake in the BFS BR
    convention, miscorrecting the σ^(1/2) piece).
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        CubicSpline = None

    delta_T1 = np.array([s - 2.0 * _BFS_TABLE_1_MW for s in _BFS_TABLE_1_SQRTS])
    delta_T2 = np.array([s - 2.0 * _BFS_TABLE_2_MW for s in _BFS_TABLE_2_SQRTS])
    whiz_T1 = np.array(_BFS_TABLE_1_WHIZ, dtype=float)
    whiz_T2 = np.array(_BFS_TABLE_2_WHIZ, dtype=float)

    if CubicSpline is not None:
        s1 = CubicSpline(delta_T1, whiz_T1, bc_type="natural", extrapolate=True)
        s2 = CubicSpline(delta_T2, whiz_T2, bc_type="natural", extrapolate=True)
        return s1, s2
    return (lambda d: np.interp(d, delta_T1, whiz_T1),
            lambda d: np.interp(d, delta_T2, whiz_T2))


_WHIZ_SPLINE_T1, _WHIZ_SPLINE_T2 = _build_whizard_anchor_splines()


def _bfs_table_whizard_sigma_fb(s, mW: float, gammaW: float):
    """σ_Whiz(s, m_W, Γ_W) [fb, specific channel μνud̄] reconstructed from
    BFS Tables 1+2 by cubic spline in δ = √s − 2 m_W and linear interp /
    extrapolation in Γ_W between the two reference points.
    """
    s_arr = np.asarray(s, dtype=float)
    delta = np.sqrt(s_arr) - 2.0 * mW
    w1 = np.asarray(_WHIZ_SPLINE_T1(delta), dtype=float)
    w2 = np.asarray(_WHIZ_SPLINE_T2(delta), dtype=float)
    alpha = (gammaW - _BFS_TABLE_1_GW) / (_BFS_TABLE_2_GW - _BFS_TABLE_1_GW)
    return (1.0 - alpha) * w1 + alpha * w2


# WHIZARD's specific-channel σ has an implicit BR² squeeze: it computes σ
# from a Lagrangian whose partial widths are SM-LO functions of m_W and the
# EW couplings (independent of the `gw` parameter). As we step gw alone, the
# propagator's Γ_W changes but Γ_partial stays at SM-LO, so the implicit BR
# = Γ_partial / Γ_W shrinks. For the PDG-constant BR convention we want
# σ_observed = σ_WW × BR_PDG (Azzurri picture: Γ_W is a pure propagator
# parameter, decays held at their PDG-measured values, *m_W-independent*).
# Strip the implicit squeeze with (Γ_W / _GAMMA_W_LO_REF)² — a FIXED
# reference, not Γ_W^(0)(running m_W). The running Γ_W^(0)(m_W) ∝ m_W³
# would inject a spurious m_W slope ~6 ΔmW/m_W into the anchor: in
# PDG-constant BR the decay sector knows nothing about the fit m_W. The
# reference is gamma_W_LO(M_W_BFS_REF = 80.377 GeV) = 2.04485 GeV — the
# same reference point BFS uses for Tables 1+2.
_GAMMA_W_LO_REF = gamma_W_LO(M_W_BFS_REF)


def _br_strip_factor(gammaW: float, *, apply_BR_correction: bool):
    if apply_BR_correction:
        return 1.0
    return (gammaW / _GAMMA_W_LO_REF) ** 2


def whizard_anchor_factor_spline(s, mW: float = M_W_DEFAULT,
                                 gammaW: float = GAMMA_W_DEFAULT,
                                 *, apply_BR_correction: bool = True):
    """f(s, m_W, Γ_W) bringing the BFS-EFT N^(3/2)LO Born to the Whizard
    4f Born. σ_Whiz from BFS Tables 1+2 (spline in δ, linear in Γ_W).

    When ``apply_BR_correction=False`` (PDG-constant chain), σ_Whiz is
    multiplied by (Γ_W/Γ_W^(0))² so the anchor produces σ_WW (pure
    W-pair, BR-independent). Otherwise σ_Whiz is used as-is so the
    anchor produces σ_specific (with WHIZARD's implicit BR² squeeze
    intact) for the BFS-EFT per-component chain.
    """
    sigma_whiz_fb = _bfs_table_whizard_sigma_fb(s, mW, gammaW)
    sigma_whiz_fb = sigma_whiz_fb * _br_strip_factor(
        gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR, sigma_RL = _accumulate_born_orders(
        s, mW, gammaW, "N3/2LO", apply_BR_correction=apply_BR_correction)
    sigma_EFT_fb = (sigma_LR + sigma_RL) / 4.0 * 1000.0   # pb → fb, unpolarised specific
    f = sigma_whiz_fb / sigma_EFT_fb

    if np.ndim(s) == 0:
        return float(f)
    return np.asarray(f, dtype=float)


def whizard_anchor_factor_grid(s, mW: float = M_W_DEFAULT,
                               gammaW: float = GAMMA_W_DEFAULT,
                               *, apply_BR_correction: bool = True):
    """f(s, m_W, Γ_W) bringing the BFS-EFT N^(3/2)LO Born to the Whizard
    4f Born. σ_Whiz from the 1295-pt WHIZARD scan (trilinear in (s, m_W,
    Γ_W)). BR² stripped iff ``apply_BR_correction=False`` — see
    :func:`whizard_anchor_factor_spline` docstring for the rationale.
    """
    from framework.process.ww.xsec_calculator.whizard_grid import whizard_sigma

    sigma_LR, sigma_RL = _accumulate_born_orders(
        s, mW, gammaW, "N3/2LO", apply_BR_correction=apply_BR_correction)
    sigma_EFT_fb = (sigma_LR + sigma_RL) / 4.0 * 1000.0   # pb → fb, unpolarised specific
    sigma_whiz_fb = whizard_sigma(s, mW, gammaW) * _br_strip_factor(
        gammaW, apply_BR_correction=apply_BR_correction)
    f = sigma_whiz_fb / sigma_EFT_fb

    if np.ndim(s) == 0:
        return float(f)
    return np.asarray(f, dtype=float)


WHIZARD_ANCHOR_SOURCES = ("spline", "grid")


def whizard_anchor_factor(s, mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          *, apply_BR_correction: bool = True,
                          source: str = "grid"):
    """Dispatch to the spline (BFS Tables) or grid (WHIZARD scan) anchor.
    Both are BR-aware: pass the same ``apply_BR_correction`` you used for
    the σ_LR/σ_RL that the factor is multiplied with.
    """
    if source == "spline":
        return whizard_anchor_factor_spline(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    if source == "grid":
        return whizard_anchor_factor_grid(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    raise ValueError(f"whizard_anchor source must be one of "
                     f"{WHIZARD_ANCHOR_SOURCES}; got {source!r}")


def _xi_chi(s, mW: float):
    """Photon/Z propagator-induced shape functions, eq. (32)."""
    s_arr = np.asarray(s, dtype=float)
    sW2 = sin2_thetaW_OS(mW)
    denom = s_arr * (s_arr - M_Z ** 2)
    xi = -3.0 * mW ** 2 * (s_arr - 2.0 * M_Z ** 2 * sW2) / denom
    chi = -6.0 * mW ** 2 * M_Z ** 2 * sW2 / denom
    return xi, chi


def _im_minus_sqrt(z):
    """Im[-√z] with the principal branch of √.

    For z in any quadrant of the complex plane, np.sqrt returns the
    principal square root (Re ≥ 0, or Im > 0 on the cut). Then -√z is
    its negative; we take the imaginary part. This is the kernel of
    eq. (17), with z = -(E + iΓ_W)/M_W.
    """
    return (-np.sqrt(np.asarray(z, dtype=complex))).imag


def sigma_LR0_specific_pb(s, mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          apply_BR_correction: bool = True):
    """σ_LR^(0) for the specific channel e+e- → μ⁻ν̄_μ ud̄, eq. (17).

    Units: pb. The 1/27 LO branching factor BR(W→μν̄) × BR(W→ud̄) is
    contained in the formula and is replaced by (Γ_W^(0)/Γ_W)² × 1/27
    when ``apply_BR_correction=True`` (section 6.1 of the paper). Set
    ``apply_BR_correction=False`` to reproduce Table 1 of the paper
    (which uses Γ_W = Γ_W^(0) so the correction is trivially 1).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW
    arg = -(E + 1j * gammaW) / mW
    pref = (4.0 * np.pi * alpha ** 2) / (27.0 * sW2 ** 2 * s_arr)
    val = pref * _im_minus_sqrt(arg) * GEV_M2_TO_PB
    if apply_BR_correction:
        val = val * _BR_correction(mW, gammaW)
    return val


def sigma_LR_RL_NLO_potential_specific_pb(s, mW: float = M_W_DEFAULT,
                                          gammaW: float = GAMMA_W_DEFAULT,
                                          gammaW_NLO: float = 0.0,
                                          apply_BR_correction: bool = True):
    """NLO Born expansion from the potential region, eq. (33).

    ``gammaW_NLO`` is Γ_W^(1) — the electroweak NLO correction to the W
    width. Per the paper, set this to 0 for Born comparison; only non-
    zero when used in the full NLO calculation including loops.

    Returns ``(σ_LR^(1), σ_RL^(1))`` in pb, for the specific channel.

    Formula:
        σ_LR^(1) = (4πα²)/(27 s_W^4 s) × {
            (11/6 + 2ξ² + 38 ξ / 9) × Im[ (-(E+iΓ_W^(0))/M_W)^(3/2) ]
            + Im[ (3E/(8M_W) + 17 i Γ_W^(0)/(8M_W)) × √(-(E+iΓ_W^(0))/M_W)
                  − (Γ_W^(0)² / (8 M_W²) − i Γ_W^(1)/(2 M_W))
                    × √(-M_W/(E+iΓ_W^(0))) ]
        }
        σ_RL^(1) = (8πα²)/(27 s_W^4 s) × χ²(s) × Im[(-(E+iΓ_W^(0))/M_W)^(3/2)]
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW
    xi, chi = _xi_chi(s_arr, mW)

    z = (E + 1j * gammaW) / mW                         # the kinematic ratio
    minus_z = -z                                       # = -(E+iΓ_W^(0))/M_W

    # Im[(-z)^(3/2)] — principal branch
    minus_z_arr = np.asarray(minus_z, dtype=complex)
    pow_32 = np.sqrt(minus_z_arr) ** 3                 # |.|^(3/2) with principal arg
    im_pow_32 = pow_32.imag

    # √(-z)
    sqrt_minus_z = np.sqrt(minus_z_arr)

    # √(-1/z) = √(-M_W/(E+iΓ_W^(0)))
    sqrt_inv = np.sqrt(np.asarray(-mW / (E + 1j * gammaW), dtype=complex))

    pref = (4.0 * np.pi * alpha ** 2) / (27.0 * sW2 ** 2 * s_arr)

    # Term 1: (11/6 + 2ξ² + 38ξ/9) × Im[(-z)^(3/2)]
    t1 = (11.0 / 6.0 + 2.0 * xi ** 2 + (38.0 / 9.0) * xi) * im_pow_32

    # Term 2: (3E/(8M_W) + 17 iΓ_W/(8 M_W)) × √(-z)
    coef_A = (3.0 * E / (8.0 * mW)) + (17.0j * gammaW / (8.0 * mW))
    term2 = coef_A * sqrt_minus_z

    # Term 3: − (Γ_W²/(8 M_W²) − i Γ_W^(1)/(2 M_W)) × √(-1/z)
    coef_B = (gammaW ** 2 / (8.0 * mW ** 2)) - (1j * gammaW_NLO / (2.0 * mW))
    term3 = -coef_B * sqrt_inv

    t23 = (term2 + term3).imag

    sigma_LR_1 = pref * (t1 + t23)
    sigma_RL_1 = (8.0 * np.pi * alpha ** 2) / (27.0 * sW2 ** 2 * s_arr) * chi ** 2 * im_pow_32

    out_LR = sigma_LR_1 * GEV_M2_TO_PB
    out_RL = sigma_RL_1 * GEV_M2_TO_PB
    if apply_BR_correction:
        c = _BR_correction(mW, gammaW)
        out_LR = out_LR * c
        out_RL = out_RL * c
    return out_LR, out_RL


def sigma_LR_RL_half_specific_pb(s, mW: float = M_W_DEFAULT,
                                 gammaW: float = GAMMA_W_DEFAULT,
                                 apply_BR_correction: bool = True):
    """N^{1/2}LO non-resonant pieces, eq. (37) with h1-h3 only (h4-h7 are
    <0.5% per the paper text after eq. (40)).

    Unlike σ^(0), σ^(1)_pot, σ^(3/2),a (whose BR correction is squared
    because both W's are in cut effective-theory propagators), the
    four-electron production-decay operator that gives σ^(1/2) has
    ONE W in narrow-width approximation and the OTHER in the
    production-decay operator at LO. Paper eq. (83) and surrounding
    text says we get a single prefactor Γ_partial^(0)/Γ_W rather than
    a squared one. Hence ``apply_BR_correction`` here uses the linear
    Γ_W^(0)/Γ_W factor (= √_BR_correction(...)).

    Returns ``(σ_LR^{1/2}, σ_RL^{1/2})`` in pb, for the specific channel.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    xi, chi = _xi_chi(s_arr, mW)
    pref = (4.0 * alpha ** 3) / (27.0 * sW2 ** 3 * s_arr)
    sigma_LR_half = pref * (_K_H1 + _K_H2 * xi + _K_H3 * xi ** 2
                            + _sigma_half_h47_LR(s_arr, mW))
    sigma_RL_half = pref * (_K_H3 * chi ** 2
                            + _sigma_half_h57_RL(s_arr, mW))
    sigma_LR_half = sigma_LR_half * GEV_M2_TO_PB
    sigma_RL_half = sigma_RL_half * GEV_M2_TO_PB
    if apply_BR_correction:
        c_lin = gamma_W_LO(mW) / gammaW   # linear (not squared)
        sigma_LR_half = sigma_LR_half * c_lin
        sigma_RL_half = sigma_RL_half * c_lin
    return sigma_LR_half, sigma_RL_half


def delta_sigma_Coulomb_NLO_specific_pb(s, mW: float = M_W_DEFAULT,
                                       gammaW: float = GAMMA_W_DEFAULT,
                                       apply_BR_correction: bool = True,
                                       subleading_only: bool = False):
    """NLO Coulomb correction Δσ_Coulomb^(1), eq. (62) of arXiv:0707.0773.

    Closed form, IR-finite:

        Δσ_Coulomb^(1) = (4πα²)/(27 s_W^4 s) × Im[
            -(α/2) ln(-(E + iΓ_W^(0))/M_W)        ← term 1: N^(1/2)LO
                                                    one-photon Coulomb,
                                                    ~5 % at threshold
            + (α² π²/12) × √(-M_W/(E + iΓ_W^(0))) ← term 2: NLO
                                                    two-photon, ~0.2 %
        ]

    Cross-checked numerically against FKM 1995 hep-ph/9507422 eq. (21)
    at threshold: their X/2 = 5.21 % and X²/6 = 0.17 % match terms 1 and
    2 here to within 1 %.

    **Overlap warning**: term 1 is the *same physics* as the leading
    α/v piece of the Fadin-Khoze-Martin K_C used in ``coulomb_K_factor``
    (Bardin-Riemann eq. 9 form, ~7 % at threshold — larger than 5 %
    because it uses off-shell momentum vs. FKM's on-shell limit). If
    both K_C and ``subleading_only=False`` are applied to the same σ,
    the leading Coulomb is double-counted at threshold (~5 %).

    Use ``subleading_only=True`` to return only term 2 (the BFS NLO
    two-photon Coulomb correction beyond the leading-α/v piece) — this
    is safe to combine with K_C.

    Returns the absolute Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    With ``apply_BR_correction=True`` (default), the (Γ_W^(0)/Γ_W)² BR
    correction is applied (two-cut-propagator piece).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW

    z = np.asarray(-(E + 1j * gammaW) / mW, dtype=complex)

    # term 1: N^(1/2)LO one-photon Coulomb (~5% at threshold, overlaps K_C)
    term1 = -(alpha / 2.0) * np.log(z)
    # term 2: NLO two-photon (~0.2% at threshold, K_C-safe subleading)
    term2 = (alpha ** 2 * np.pi ** 2 / 12.0) * np.sqrt(1.0 / z)

    pref = (4.0 * np.pi * alpha ** 2) / (27.0 * sW2 ** 2 * s_arr)
    if subleading_only:
        bracket = term2
    else:
        bracket = term1 + term2
    delta_sigma = pref * bracket.imag * GEV_M2_TO_PB

    if apply_BR_correction:
        delta_sigma = delta_sigma * _BR_correction(mW, gammaW)
    return delta_sigma


def sigma_LR_RL_three_half_a_specific_pb(s, mW: float = M_W_DEFAULT,
                                         gammaW: float = GAMMA_W_DEFAULT,
                                         apply_BR_correction: bool = True):
    """Energy-dependent N^{3/2}LO,a, eq. (39).

    Returns ``(σ_LR^{3/2,a}, σ_RL^{3/2,a})`` in pb, for the specific channel.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW
    xi, chi = _xi_chi(s_arr, mW)
    pref = (4.0 * alpha ** 3 * E) / (27.0 * sW2 ** 3 * s_arr * mW)
    sigma_LR_32a = pref * (_K_H1_A + _K_H2_A * xi + _K_H3_A * xi ** 2)
    sigma_RL_32a = pref * (_K_H3_A * chi ** 2)
    sigma_LR_32a = sigma_LR_32a * GEV_M2_TO_PB
    sigma_RL_32a = sigma_RL_32a * GEV_M2_TO_PB
    if apply_BR_correction:
        c = _BR_correction(mW, gammaW)
        sigma_LR_32a = sigma_LR_32a * c
        sigma_RL_32a = sigma_RL_32a * c
    return sigma_LR_32a, sigma_RL_32a


def _accumulate_born_orders(s, mW: float, gammaW: float, order: str,
                            *, apply_BR_correction: bool):
    """Sum the BFS Born expansion at the requested truncation, returning
    ``(σ_LR, σ_RL)`` in pb for the specific channel. Shared between
    ``sigma_BFS_LO_total_WW_pb`` and ``sigma_BFS_specific_munuud_pb``."""
    sigma_LR = sigma_LR0_specific_pb(s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_RL = np.zeros_like(np.asarray(sigma_LR, dtype=float))
    if order in ("N1/2LO", "NLO", "N3/2LO"):
        s_LR, s_RL = sigma_LR_RL_half_specific_pb(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR
        sigma_RL = sigma_RL + s_RL
    if order in ("NLO", "N3/2LO"):
        s_LR, s_RL = sigma_LR_RL_NLO_potential_specific_pb(
            s, mW, gammaW, gammaW_NLO=0.0, apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR
        sigma_RL = sigma_RL + s_RL
    if order == "N3/2LO":
        s_LR, s_RL = sigma_LR_RL_three_half_a_specific_pb(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR
        sigma_RL = sigma_RL + s_RL
    return sigma_LR, sigma_RL


def _add_nlo_loops_to_LR(sigma_LR, s, mW, gammaW, *, apply_BR_correction):
    """BFS NLO loops (eq. finalcross): HSC + EW decay + Coulomb_NLO. All
    additive to σ_LR; σ_RL has no NLO contribution since LO σ_RL = 0.
    """
    sigma_LR = sigma_LR + delta_sigma_NLO_hard_softcoll_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_NLO_decay_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_Coulomb_NLO_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction,
        subleading_only=False)
    return sigma_LR


def _add_nnlo_to_LR(sigma_LR, s, mW, gammaW, *, apply_BR_correction):
    """BFS dominant NNLO σ̂^(3/2)_LR (eq. 49 of arXiv:0807.0102): the sum
    of five pieces — C×[S+H], NLO-C, C×decay, C×res, C3. Total impact on
    m_W is ~3 MeV (~5 MeV without ISR convolution per BFS sec. 6.4). All
    additive to σ_LR; σ_RL has no contribution at this order.

    Defined later in this module — the function calls are dispatched after
    the NNLO helpers are introduced. ``include_BFS_NNLO=True`` on the
    public assembly functions activates this addition.
    """
    sigma_LR = sigma_LR + delta_sigma_NNLO_C_soft_hard_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_NNLO_C_decay_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_NNLO_C_residue_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR + delta_sigma_NNLO_triple_Coulomb_specific_pb(
        s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    return sigma_LR


def sigma_BFS_LO_total_WW_pb(s, mW: float = M_W_DEFAULT,
                             gammaW: float = GAMMA_W_DEFAULT,
                             order: str = "N3/2LO",
                             apply_BR_correction: bool = False,
                             include_NLO_hard_decay: bool = False,
                             include_BFS_NNLO: bool = False,
                             apply_delta_QCD: bool = False,
                             alpha_s: float = ALPHA_S_MW_DEFAULT,
                             apply_whizard_anchor: bool = False,
                             whizard_anchor_source: str = "grid"):
    """Total σ_WW = σ(e+e- → W+W-) at BFS LO_EFT, unpolarised initial state,
    summed over ALL 4-fermion final states.

    Each σ_LR^(...) carries the BFS 1/27 specific-channel BR factor. We
    strip it by × 27 and average over the four initial helicities by ÷4.
    The result is σ(e+e- → W+W-) — the total pair-production cross
    section, with no BR multiplication.

    IMPORTANT: the BR-correction factor (Γ_W^(0)/Γ_W)² from BFS section
    6.1 should NOT be applied to "total σ_WW". It applies inside the
    σ_LR formula when interpreting it as a *specific-channel* cross
    section with the correct BR weighting at NLO Γ_W. For total σ_WW we
    strip the 1/27 by ×27 and the correction would double-count — the
    total pair-production cross section has no BR dependence.

    Default is therefore ``apply_BR_correction=False``. Set True only to
    diagnose the σ_LR^(0..3/2) specific-channel values via this entry
    point. Paper Table-1 / Table-2 self-tests use the specific-channel
    functions directly with their own ``apply_BR_correction`` flag.

    Parameters
    ----------
    s : float or array
        Partonic CM energy² in GeV².
    mW, gammaW : float
        W mass and physical width in GeV. The width enters via the
        complex-velocity propagator only (no BR-correction multiplication
        with the default).
    order : {"LO", "N1/2LO", "NLO", "N3/2LO"}
        Truncation of the BFS Born expansion.
    apply_BR_correction : bool
        Default False — DO NOT use for total σ_WW. See module docstring.

    Returns
    -------
    σ_total_WW in pb (same shape as ``s``).
    """
    sigma_LR, sigma_RL = _accumulate_born_orders(
        s, mW, gammaW, order, apply_BR_correction=apply_BR_correction)

    # Whizard anchor on the Born sum only (NOT on the NLO loops below).
    # apply_BR_correction is forwarded so the σ_EFT denominator matches the
    # σ_LR/σ_RL convention we just computed (avoids (Γ^(0)/Γ_W)² mismatch).
    if apply_whizard_anchor:
        f = whizard_anchor_factor(s, mW, gammaW,
                                  apply_BR_correction=apply_BR_correction,
                                  source=whizard_anchor_source)
        sigma_LR = sigma_LR * f
        sigma_RL = sigma_RL * f

    if include_NLO_hard_decay:
        sigma_LR = _add_nlo_loops_to_LR(
            sigma_LR, s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    if include_BFS_NNLO:
        sigma_LR = _add_nnlo_to_LR(
            sigma_LR, s, mW, gammaW, apply_BR_correction=apply_BR_correction)

    sigma_total_WW = (sigma_LR + sigma_RL) * 27.0 / 4.0
    if apply_delta_QCD:
        sigma_total_WW = sigma_total_WW * delta_QCD_factor(alpha_s)

    if np.ndim(s) == 0:
        return float(sigma_total_WW)
    return sigma_total_WW


# ---------------------------------------------------------------------------
# NLO hard+soft+collinear (eq. finalcross of arXiv:0707.0773)
# ---------------------------------------------------------------------------

# Real part of the hard matching coefficient c_p,LR^(1,fin) from BFS line 1797:
#   c_p,LR^(1,fin) = -10.076 + 0.205 i
# Computed at BFS reference parameters (m_W = 80.377, M_Z = 91.188, m_t = 174.2,
# M_H = 115 GeV). The dependence on m_W in our fit range (80.0-80.7 GeV) is
# sub-percent on Re(c); we treat it as a constant here. The imaginary part
# does not contribute to the flavour-specific cross section (BFS section 4.2,
# discussion around eq. ImAC) — only Re enters.
_C_P_LR_1_FIN_RE = -10.076


def delta_sigma_NLO_hard_softcoll_specific_pb(s, mW: float = M_W_DEFAULT,
                                              gammaW: float = GAMMA_W_DEFAULT,
                                              apply_BR_correction: bool = True):
    """NLO hard + soft + collinear correction to σ_LR^specific (μ⁻ν̄_μ ud̄).

    Bracket term of eq. (eq:finalcross) of arXiv:0707.0773:

        Δσ̂_LR^(1,HSC)(s) = (4 α³)/(27 s_W^4 s) × Im{
            -√(z) × [ 2 ln(4 z) + Re(c_p,LR^(1,fin)) + π²/4 + 1/2 ]
        }

    with z = -(E + i Γ_W)/M_W, E = √s − 2 m_W. This is the residue of the
    HARD (eq. hardsigma) + SOFT (sec. 4.3) + COLLINEAR (sec. 4.4-4.5)
    corrections after all 1/ε² and 1/ε poles cancel against each other and
    against the conventional-scheme conversion of the LL ePDFs.

    Combined with Δσ_Coulomb^(1) (eq. 62, ``delta_sigma_Coulomb_NLO_specific_pb``)
    and Δσ_decay^(1) (eq. 49, ``delta_sigma_NLO_decay_specific_pb``) reproduces
    the full NLO ``σ̂_LR_conv^(1)`` of BFS eq. (finalcross), the NLO correction
    to the partonic σ_LR in the conventional ISR scheme.

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄. With
    ``apply_BR_correction=True`` (default), the (Γ_W^(0)/Γ_W)² factor of BFS
    section 6.1 is applied (same convention as for σ_LR^(0)).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW
    z = np.asarray(-(E + 1j * gammaW) / mW, dtype=complex)

    sqrt_z = np.sqrt(z)                              # principal branch
    ln_4z = np.log(4.0 * z)                          # complex log, principal branch
    bracket = (2.0 * ln_4z
               + _C_P_LR_1_FIN_RE
               + np.pi ** 2 / 4.0
               + 0.5)
    integrand = (-sqrt_z * bracket).imag
    pref = (4.0 * alpha ** 3) / (27.0 * sW2 ** 2 * s_arr)
    val = pref * integrand * GEV_M2_TO_PB
    if apply_BR_correction:
        val = val * _BR_correction(mW, gammaW)
    return val


# ---------------------------------------------------------------------------
# NLO decay correction (eq. delta-decay / 60 of arXiv:0707.0773 — EW only)
# ---------------------------------------------------------------------------

def delta_QCD_factor(alpha_s: float = ALPHA_S_MW_DEFAULT) -> float:
    """Universal QCD correction to hadronic partial widths, BFS eq. (delta_qcd):

        δ_QCD(α_s) = 1 + α_s/π + 1.409 (α_s/π)²

    α_s is α_s(M_W) in MS-bar. BFS section 6.1 (lines 2622-2637) explains that
    multiplying the entire NLO electroweak cross section by δ_QCD reproduces
    the QCD running of hadronic partial widths to NNLO precision.

    At α_s = 0.1199 this evaluates to 1.04025 (≈ +4.0 % multiplicative).
    Differential: ∂δ_QCD/∂α_s = 1/π + 2 × 1.409 × α_s/π² = +0.351 at the
    reference α_s — gives a ~0.1 % shift on σ per 1 % shift on α_s.
    """
    x = alpha_s / np.pi
    return 1.0 + x + 1.409 * x * x


# BFS eq. (Gamma1ewFS) explicit formula for the EW one-loop correction to the
# W partial width into a single lepton or quark doublet:
#
#   Γ_W,l/h^(1,ew) / Γ_W,l/h^(0) =
#     (α/(2π)) × [ 2·Re(c_d,l/h^(1,fin))
#                  + 101/12 + (19/2)·Q_f·Q̄_f
#                  - 7π²/12 - (π²/6)·Q_f·Q̄_f ]
#
# c_d,l/h^(1,fin) are the finite parts of the leptonic and hadronic decay
# matching coefficients computed in BFS appendix at the reference inputs
# m_W = 80.377, M_Z = 91.188, m_t = 174.2 GeV, M_H = 115 GeV (paper line 1920):
_C_D_L_1_FIN_RE = -2.709          # leptonic
_C_D_H_1_FIN_RE = -2.034          # hadronic
# Charge factors (BFS line 1916): leptonic has Q_f = -1, Q̄_f = 0 → product 0.
# Hadronic: Q_f = 2/3, Q̄_f = -1/3 → product -2/9.
_Q_PROD_L = 0.0
_Q_PROD_H = -2.0 / 9.0


def _delta_W_ew_partial(c_d_fin_re: float, q_prod: float, alpha: float) -> float:
    """Γ_x^(1,ew) / Γ_x^(0) per BFS eq. (Gamma1ewFS)."""
    bracket = (2.0 * c_d_fin_re
               + 101.0 / 12.0
               + (19.0 / 2.0) * q_prod
               - 7.0 * np.pi ** 2 / 12.0
               - (np.pi ** 2 / 6.0) * q_prod)
    return alpha / (2.0 * np.pi) * bracket


def delta_decay_EW_relative(mW: float = M_W_DEFAULT) -> float:
    """δ_decay^(1,ew) = Γ_l^(1,ew)/Γ_l^(0) + Γ_h^(1,ew)/Γ_h^(0)  (BFS eq. delta-decay).

    Computed from BFS eq. (Gamma1ewFS) with α = α_Gμ(m_W). The c_d,l/h^(1,fin)
    are kept at their BFS reference values (sub-percent m_W variation neglected).
    At m_W = 80.377 GeV: returns ≈ −0.0071 (−0.71 %).
    """
    alpha = alpha_Gmu(mW)
    delta_l = _delta_W_ew_partial(_C_D_L_1_FIN_RE, _Q_PROD_L, alpha)
    delta_h = _delta_W_ew_partial(_C_D_H_1_FIN_RE, _Q_PROD_H, alpha)
    return delta_l + delta_h


def delta_sigma_NLO_decay_specific_pb(s, mW: float = M_W_DEFAULT,
                                      gammaW: float = GAMMA_W_DEFAULT,
                                      apply_BR_correction: bool = True):
    """NLO **electroweak-only** decay-side correction (BFS eq. delta-decay):

        Δσ_decay^(1,ew) = ( Γ_l^(1,ew)/Γ_l^(0)
                            + Γ_h^(1,ew)/Γ_h^(0) ) × σ^(0)

    with Γ_x^(1,ew)/Γ_x^(0) given by BFS eq. (Gamma1ewFS) — explicit α/(2π)
    times a kinematic factor depending on the W → final-state charges and
    the finite decay-matching coefficients c_d,l/h^(1,fin) (BFS appendix).

    The QCD content of the decay correction is intentionally NOT included
    here; it is captured by the multiplicative ``delta_QCD_factor(α_s)``
    (BFS eq. delta_qcd / section 6.1 lines 2622-2637). Including both
    would double-count.

    At BFS reference parameters this is ≈ −0.71 % × σ^(0). Returns Δσ in
    pb. The (m_W, Γ_W) dependence is captured through α_Gμ(m_W) and
    sigma_LR0_specific_pb(s, mW, gammaW); the m_W dependence of
    c_d,l/h^(1,fin) themselves is sub-percent and is neglected.
    """
    sigma_LR0 = sigma_LR0_specific_pb(s, mW, gammaW,
                                       apply_BR_correction=apply_BR_correction)
    return delta_decay_EW_relative(mW) * sigma_LR0


# ---------------------------------------------------------------------------
# BFS NNLO (arXiv:0807.0102) — dominant N^{3/2}LO_EFT corrections
# ---------------------------------------------------------------------------
#
# The full NNLO correction of eq. (49) of arXiv:0807.0102 is the sum
#
#   σ̂^(3/2)_LR = Δσ^(C×[S+H])_LR + Δσ^(NLO-C)_LR + Δσ^(C×decay)_LR
#              + Δσ^(C×res)_LR    + Δσ^(C3)_LR.
#
# Two of these have closed-form analytic expressions in the paper
# (eqs. 11 and 48); the remaining three require sections 3.2–3.7 and are
# implemented elsewhere. Combined NNLO shifts m_W by ~3 MeV per BFS sec. 6
# (~5 MeV without ISR convolution).
#
# Paper convention used in Table 1: Δσ^i = Δσ^i_LR / 4 (helicity-averaged).
# The functions here return σ_LR (specific-channel, μ⁻ν̄_μ ud̄) in pb, in
# line with the existing NLO loop functions; the /4 is applied at assembly.

_ZETA_3 = 1.2020569031595942   # Apéry's constant ζ(3)


def _delta_sigma_C1_LR_specific_pb(s, mW: float, gammaW: float):
    """Single-Coulomb exchange piece Δσ^(C1)_LR — the α-order term of
    eq. (10) of arXiv:0807.0102:

        Δσ^(C1)_LR = -(2π α_ew² α / 27 s) × Im[ ln(-ℰ_W/M_W) ]

    Equivalent to the ``term1`` piece of ``delta_sigma_Coulomb_NLO_specific_pb``
    (eq. 62 of arXiv:0707.0773), without the two-photon ``term2``. Returned
    in pb for the specific channel, **without** BR-correction (callers apply
    it).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    E = np.sqrt(s_arr) - 2.0 * mW
    z = -(E + 1j * gammaW) / mW
    pref = -(2.0 * np.pi * alpha ** 2 * alpha) / (27.0 * sW2 ** 2 * s_arr)
    return pref * np.log(z).imag * GEV_M2_TO_PB


def delta_sigma_NNLO_triple_Coulomb_specific_pb(s, mW: float = M_W_DEFAULT,
                                                gammaW: float = GAMMA_W_DEFAULT,
                                                apply_BR_correction: bool = True):
    """Triple-Coulomb exchange Δσ^(C3)_LR, eq. (11) of arXiv:0807.0102:

        Δσ^(C3)_LR = (π α_ew² / 27 s) × α³ ζ(3) × Im[ −M_W / ℰ_W ]

    where ℰ_W = (√s − 2 M_W) + i Γ_W. Closed form, IR-finite. Numerically
    tiny: ≲ 0.01 fb (helicity-averaged) over the entire scan window.

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    E = np.sqrt(s_arr) - 2.0 * mW
    cE = E + 1j * gammaW                            # ℰ_W
    pref = (np.pi * alpha ** 2) / (27.0 * sW2 ** 2 * s_arr)   # π α_ew²/(27 s)
    delta = pref * (alpha ** 3) * _ZETA_3 * (-mW / cE).imag * GEV_M2_TO_PB
    if apply_BR_correction:
        delta = delta * _BR_correction(mW, gammaW)
    return delta


def delta_sigma_NNLO_C_residue_specific_pb(s, mW: float = M_W_DEFAULT,
                                           gammaW: float = GAMMA_W_DEFAULT,
                                           apply_BR_correction: bool = True):
    """Interference of residue correction and single-Coulomb exchange,
    Δσ^(C×res)_LR, eq. (48) of arXiv:0807.0102:

        Δσ^(C×res)_LR = (4π α_ew² α / 27 s) × (Γ_W / M_W)
                        × ln[ 2 |ℰ_W| (Re ℰ_W + |ℰ_W|) / Γ_W² ]

    with ℰ_W = (√s − 2 M_W) + i Γ_W. The Γ_W in the prefactor and inside
    the logarithm is the full on-shell width (see paper sec. 3.8, text
    after eq. 46). Dominant NNLO piece above threshold (~1 fb at 170 GeV).

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    E = np.sqrt(s_arr) - 2.0 * mW
    cE = E + 1j * gammaW                            # ℰ_W
    absE = np.abs(cE)
    log_arg = 2.0 * absE * (cE.real + absE) / (gammaW ** 2)
    pref = (4.0 * np.pi * alpha ** 2 * alpha) / (27.0 * sW2 ** 2 * s_arr)
    delta = pref * (gammaW / mW) * np.log(log_arg) * GEV_M2_TO_PB
    if apply_BR_correction:
        delta = delta * _BR_correction(mW, gammaW)
    return delta


def delta_sigma_NNLO_C_decay_specific_pb(s, mW: float = M_W_DEFAULT,
                                         gammaW: float = GAMMA_W_DEFAULT,
                                         apply_BR_correction: bool = True):
    """Interference of decay correction and single-Coulomb exchange,
    Δσ^(C×decay)_LR, eq. (40) of arXiv:0807.0102:

        Δσ^(C×decay)_LR = ( Γ_μν^(1,ew)/Γ_μν^(0) + Γ_ud^(1,ew)/Γ_ud^(0) )
                          × Δσ^(C1)_LR

    The bracket is the same EW partial-width correction factor used by
    ``delta_sigma_NLO_decay_specific_pb`` (BFS eq. 60 of arXiv:0707.0773
    via ``delta_decay_EW_relative``). Δσ^(C1)_LR is the α-order single-
    Coulomb piece of eq. (10).

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    """
    sigma_C1 = _delta_sigma_C1_LR_specific_pb(s, mW, gammaW)
    delta = delta_decay_EW_relative(mW) * sigma_C1
    if apply_BR_correction:
        delta = delta * _BR_correction(mW, gammaW)
    return delta


# Σ_f C_f Q_f² for the photon vacuum-polarisation fermion-bubble sum,
# top excluded (integrated out as a hard mode). u,c: 3×(2/3)²=4/3 each;
# d,s,b: 3×(1/3)²=1/3 each; e,μ,τ: 1×1²=1 each. Total = 20/3.
_SUM_CF_QF2_NO_TOP = 20.0 / 3.0

# α(M_Z)→G_μ conversion factor for the hard matching coefficient, BFS
# eq. (39). Paper text after eq. (39): "for the same input parameters
# as used for the hard-matching coefficient below (coeff1loop) the
# numerical value is δ_{α(M_Z)→G_μ} = 4.103 α". Same BFS reference
# parameters as ``_C_P_LR_1_FIN_RE`` above; m_W dependence is sub-percent.
_DELTA_ALPHA_MZ_TO_GMU_COEFF = 4.103


def delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb(
        s, mW: float = M_W_DEFAULT, gammaW: float = GAMMA_W_DEFAULT,
        apply_BR_correction: bool = True):
    """NLO corrections to the Coulomb potential Δσ^(NLO-C)_LR — eq. (39)
    of arXiv:0807.0102 (the ``bubble-gmu`` result):

        Δσ^(NLO-C)_LR = Δσ^(NLO-C)_LR|_α(M_Z) + δ_{α(M_Z)→G_μ} × Δσ^(C1)_LR

    with the α(M_Z)-scheme piece

        Δσ^(NLO-C)_LR|_α(M_Z) = -(α_ew² α² / 81 s) × (Σ_f C_f Q_f²)
            × { 4 ln(2 M_W / M_Z) Im[ln(−ℰ_W/M_W)]
                + Im[ln²(−ℰ_W/M_W)] }

    encoding the running of α via photon-bubble insertions on the
    Coulomb photon (semi-soft + hard, sec. 3.6 + appendix A). The
    α(M_Z)→G_μ conversion adds a finite shift proportional to the
    single-Coulomb piece Δσ^(C1)_LR. The conversion piece is the larger
    of the two at threshold and gives a net positive correction.

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    E = np.sqrt(s_arr) - 2.0 * mW
    z = -(E + 1j * gammaW) / mW
    lz = np.log(z)

    alpha_ew_sq = (alpha / sW2) ** 2
    pref_aMZ = -alpha_ew_sq * alpha ** 2 / (81.0 * s_arr)
    bracket = (4.0 * np.log(2.0 * mW / M_Z) * lz.imag
               + (lz ** 2).imag)
    delta_aMZ = pref_aMZ * _SUM_CF_QF2_NO_TOP * bracket * GEV_M2_TO_PB

    # α(M_Z)→G_μ conversion: add δ × Δσ^(C1)_LR
    delta_conv = (_DELTA_ALPHA_MZ_TO_GMU_COEFF * alpha
                  * _delta_sigma_C1_LR_specific_pb(s, mW, gammaW))

    total = delta_aMZ + delta_conv
    if apply_BR_correction:
        total = total * _BR_correction(mW, gammaW)
    return total


def delta_sigma_NNLO_C_soft_hard_specific_pb(
        s, mW: float = M_W_DEFAULT, gammaW: float = GAMMA_W_DEFAULT,
        apply_BR_correction: bool = True):
    """Interference of single-Coulomb with soft and hard corrections,
    Δσ̂^(C×[S+H])_LR — eq. (34) of arXiv:0807.0102 (the ISR-subtracted
    ``sigma-three-half`` partonic correction to be convoluted with the
    electron structure functions):

        Δσ̂^(C×[S+H])_LR = -(α_ew² α² / 27 s) × {
            (9 + π²/2 + 2 Re c_p,LR^(1,fin)) × Im[ ln(−ℰ_W/M_W) ]
            + 2 × Im[ ln²(−ℰ_W/M_W) ]
        }

    The hard matching coefficient ``Re c_p,LR^(1,fin) = -10.076`` is the
    SAME constant used in the NLO HSC term (``_C_P_LR_1_FIN_RE``); the
    BFS NNLO paper inherits it unchanged from arXiv:0707.0773. After the
    ISR subtraction described in paper section 3.4–3.5 (the `hat`
    superscript), all logs of m_e are absorbed into the structure
    functions and the partonic result is finite without an electron-mass
    scale dependence.

    Returns Δσ in pb for the specific channel μ⁻ν̄_μ ud̄.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    E = np.sqrt(s_arr) - 2.0 * mW
    z = -(E + 1j * gammaW) / mW
    lz = np.log(z)

    alpha_ew_sq = (alpha / sW2) ** 2
    pref = -alpha_ew_sq * alpha ** 2 / (27.0 * s_arr)
    coef = 9.0 + np.pi ** 2 / 2.0 + 2.0 * _C_P_LR_1_FIN_RE
    bracket = coef * lz.imag + 2.0 * (lz ** 2).imag
    delta = pref * bracket * GEV_M2_TO_PB
    if apply_BR_correction:
        delta = delta * _BR_correction(mW, gammaW)
    return delta


def sigma_BFS_specific_munuud_pb(s, mW: float = M_W_DEFAULT,
                                 gammaW: float = GAMMA_W_DEFAULT,
                                 order: str = "N3/2LO",
                                 include_NLO_hard_decay: bool = False,
                                 include_BFS_NNLO: bool = False,
                                 apply_delta_QCD: bool = False,
                                 alpha_s: float = ALPHA_S_MW_DEFAULT,
                                 apply_whizard_anchor: bool = False,
                                 whizard_anchor_source: str = "grid"):
    """σ(e+e- → μ⁻ν̄_μ ud̄) at BFS LO_EFT, unpolarised initial state, with
    the BR correction (BFS section 6.1, eq. 83) applied PER COMPONENT:

      * σ^(0), σ^(1)_pot, σ^(3/2),a:  (Γ_W^(0)/Γ_W)²   (two cut propagators)
      * σ^(1/2):                       Γ_W^(0)/Γ_W      (one cut propagator)

    Returned value is the specific-channel (one quark generation, one W
    charge) unpolarised cross section in pb. Inclusive μν qq̄ (both
    charges × ud̄+cs̄ summed) is 4× this value at LO.

    This is the entry point used by ``sigma_partonic_munuqq`` so that the
    BR factor's correct (m_W, Γ_W) dependence is preserved in the fit.

    ``include_NLO_hard_decay`` adds the NLO hard+soft+collinear piece
    (eq. finalcross bracket × √) and the decay correction (eq. 49) on
    top of the σ_LR Born expansion. This is the bulk of the BFS NLO
    physical-σ correction; the NLO Coulomb (eq. 62) is still handled
    separately via ``coulomb_K_factor`` / ``BFSCorrections`` so users can
    choose K_C vs eq. 62 without double-counting at LO.

    Parameters mirror ``sigma_BFS_LO_total_WW_pb``.
    """
    sigma_LR, sigma_RL = _accumulate_born_orders(
        s, mW, gammaW, order, apply_BR_correction=True)   # LINEAR BR per eq. 83

    # BFS section 6.2: replace the EFT N^(3/2)LO Born by the Whizard 4f Born
    # via the multiplicative anchor. Applied to the BORN SUM only — NOT to
    # the NLO loop corrections, which are added on top per BFS
    # eq. (finalcross). The σ_LR above use apply_BR_correction=True (linear
    # BR per eq. 83), so the anchor's σ_EFT denominator must match.
    if apply_whizard_anchor:
        f = whizard_anchor_factor(s, mW, gammaW,
                                  apply_BR_correction=True,
                                  source=whizard_anchor_source)
        sigma_LR = sigma_LR * f
        sigma_RL = sigma_RL * f

    if include_NLO_hard_decay:
        sigma_LR = _add_nlo_loops_to_LR(
            sigma_LR, s, mW, gammaW, apply_BR_correction=True)
    if include_BFS_NNLO:
        sigma_LR = _add_nnlo_to_LR(
            sigma_LR, s, mW, gammaW, apply_BR_correction=True)

    # Unpolarised specific = (σ_LR + σ_RL) / 4
    sigma_specific = (sigma_LR + sigma_RL) / 4.0
    if apply_delta_QCD:
        sigma_specific = sigma_specific * delta_QCD_factor(alpha_s)

    if np.ndim(s) == 0:
        return float(sigma_specific)
    return sigma_specific


# ---------------------------------------------------------------------------
# Validation against Table 2 of arXiv:0707.0773
# ---------------------------------------------------------------------------

# Table 2 of the paper: σ(e+e- → μ⁻ν̄_μ ud̄) in fb, with the NLO width Γ_W
# of eq. (81) = 2.09201 GeV resummed in the propagator. Columns are the
# successive EFT truncations and the Whizard exact Born.
TABLE_2 = {
    "sqrts_GeV": np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0]),
    "eft_NLO_fb": np.array([42.25, 65.99, 154.02, 298.6, 400.3, 469.4]),  # N3/2LO+NLO
    "eft_N32LO_fb": np.array([30.54, 60.83, 154.44, 303.7, 409.3, 481.7]),
    "exact_Born_fb": np.array([33.58, 61.67, 154.19, 303.0, 408.8, 481.7]),
}


def _self_test_specific_channel(mW: float, gammaW: float,
                                apply_BR_correction: bool, label: str,
                                paper_col_fb, whizard_fb, sqrts_GeV):
    """Reproduce the N^{3/2}LO column of a table of arXiv:0707.0773 by
    summing all the Born-expansion pieces in the specific channel.
    """
    print(f"\n{label}  (m_W={mW}, Γ_W={gammaW}, "
          f"BR_corr={apply_BR_correction})")
    print(f"{'√s':>6}  {'σ_BFS N32LO':>13}  {'σ paper N32LO':>15}  "
          f"{'σ Whizard Born':>16}  {'mine/Whizard':>14}")
    s = sqrts_GeV ** 2
    sLR0 = sigma_LR0_specific_pb(s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sLR12, sRL12 = sigma_LR_RL_half_specific_pb(s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sLR_NLO, sRL_NLO = sigma_LR_RL_NLO_potential_specific_pb(
        s, mW, gammaW, gammaW_NLO=0.0, apply_BR_correction=apply_BR_correction)
    sLR32a, sRL32a = sigma_LR_RL_three_half_a_specific_pb(s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_specific = (sLR0 + sLR12 + sLR_NLO + sLR32a + sRL12 + sRL_NLO + sRL32a) / 4.0
    sigma_specific_fb = sigma_specific * 1e3
    for i in range(len(sqrts_GeV)):
        print(f"  {sqrts_GeV[i]:5.1f}  {sigma_specific_fb[i]:11.2f} fb  "
              f"{paper_col_fb[i]:13.2f} fb  {whizard_fb[i]:14.2f} fb  "
              f"{sigma_specific_fb[i]/whizard_fb[i]:12.3f}")


TABLE_1 = {
    "sqrts_GeV": np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0]),
    "eft_NLO_fb":    np.array([43.28, 67.78, 160.45, 313.5, 420.4, 492.9]),
    "eft_N32LO_fb":  np.array([31.30, 62.50, 160.89, 318.8, 429.7, 505.4]),
    "exact_Born_fb": np.array([34.43, 63.39, 160.62, 318.3, 428.6, 505.1]),
}


if __name__ == "__main__":
    print("=" * 80)
    print("BFS LO_EFT (N^{3/2}LO) — round-trips vs Tables 1 & 2 of arXiv:0707.0773")
    print("=" * 80)
    # Table 1: paper inputs m_W=80.377 (pole), Γ_W = Γ_W^(0) = 2.04483 GeV
    # → BR correction (Γ_W^(0)/Γ_W)² = 1 (trivial)
    _self_test_specific_channel(
        mW=80.377, gammaW=2.04483, apply_BR_correction=False,
        label="Table 1 inputs (LO width, no BR correction):",
        paper_col_fb=TABLE_1["eft_N32LO_fb"],
        whizard_fb=TABLE_1["exact_Born_fb"],
        sqrts_GeV=TABLE_1["sqrts_GeV"],
    )
    # Table 2: paper inputs m_W=80.379, Γ_W^NLO = 2.09201 GeV (NLO + QCD)
    # → BR correction = (Γ_W^(0)(80.379) / 2.09201)² ≈ 0.955
    _self_test_specific_channel(
        mW=80.379, gammaW=2.09201, apply_BR_correction=True,
        label="Table 2 inputs (NLO+QCD width, with BR correction):",
        paper_col_fb=TABLE_2["eft_N32LO_fb"],
        whizard_fb=TABLE_2["exact_Born_fb"],
        sqrts_GeV=TABLE_2["sqrts_GeV"],
    )
