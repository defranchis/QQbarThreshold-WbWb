"""σ(e+e- → μν qq̄) near the WW threshold for FCC-ee — partonic level.

Default partonic cross section (= what the production fit templates use):
BFS-EFT N^(3/2)LO Born + NLO loops + δ_QCD + Whizard 4f Born anchor,
with finite-Γ_W complex-velocity smoothing. Above √s = 170 GeV a
RACOONWW CC03 calibration spline is used (only the 240 GeV "last_ecm"
reference point if --lastecm is enabled).

Pipeline (``sigma_partonic_munuqq``):

    σ̂_partonic(ŝ; m_W, Γ_W)
        = ( BFS N^(3/2)LO Born + Δσ_HSC + Δσ_Coul^NLO + Δσ_decay^EW )
          × f_Whizard(δ, Γ_W)        ← BFS sec. 6.2 anchor
          × δ_QCD(α_s)
          × K_Coulomb(ŝ; m_W, Γ_W)   ← Fadin-Khoze-Martin + Bardin-Riemann α²
          × BR_channel

Channels (set via ``channel`` arg of :func:`sigma_partonic_munuqq`):

    "inclusive"  (default)
        Inclusive μν qq̄ summed over both W charges:
            BR = 2 × BR(W→μν) × BR(W→hadrons)  (PDG-constant)
        This is what the FCC-ee threshold-scan m_W analysis sees.

    "munuud"
        Specific μ⁻ν̄_μ + (W⁺→ud̄ or cs̄) channel only:
            BR = BR(W→μν) × BR(W→ud̄/cs̄ summed)
        Useful for benchmarking against published BFS tables.

ISR convolution is in ``isr.py``. Validation against BFS Tables 1+2+3+4
in ``scripts/validate_bfs_nlo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# SM constants
# ---------------------------------------------------------------------------
M_W_DEFAULT     = 80.3692
GAMMA_W_DEFAULT = 2.085
M_Z             = 91.1876
M_E             = 0.5109989461e-3
G_F             = 1.1663787e-5
ALPHA_EM_0      = 1.0 / 137.035999084
GEV_M2_TO_PB    = 3.8937937217e8
PB_TO_FB        = 1000.0   # picobarn → femtobarn

# ---------------------------------------------------------------------------
# WW-chain defaults (single source of truth; imported by bfs_eft, isr,
# generator, compute_xsec_ww, cards). The "BFS reference" values are the
# inputs at which BFS arXiv:0707.0773 evaluated c_p,LR^(1,fin) = -10.076.
# Treating m_t / M_H as constants at those reference values is justified
# at the < 0.001 % level on σ_NLO (Scenario H of validate_bfs_nlo.py).
# ---------------------------------------------------------------------------
ALPHA_S_MW_DEFAULT = 0.1199    # α_s(M_W) in MS-bar, BFS reference
M_W_BFS_REF        = 80.377    # m_W at which BFS Tables / c_fin are tabulated
M_T_DEFAULT        = 174.2     # m_t (pole), BFS Table 4 input
M_H_DEFAULT        = 115.0     # M_H, BFS Table 4 input

# PDG branching ratios — primitives live in the steering card (single source
# of truth); the combinations below are derived once at module load.
from cards.ww_default import BR_W_MUNU, BR_W_HAD, BR_W_UD  # noqa: E402

# 2 × BR(W→μν) × BR(W→had): factor 2 because either W can be the muonic one.
BR_INCLUSIVE_MUNUQQ = 2.0 * BR_W_MUNU * BR_W_HAD
# BR(W→μν) × BR(W→ud̄/cs̄): legacy single-specific-channel convention from
# the BFS tables.
BR_MUNUUD          = BR_W_MUNU * BR_W_UD


def sin2_thetaW_OS(mW: float = M_W_DEFAULT) -> float:
    return 1.0 - (mW / M_Z) ** 2


def alpha_Gmu(mW: float = M_W_DEFAULT) -> float:
    sW2 = sin2_thetaW_OS(mW)
    return np.sqrt(2.0) * G_F * mW * mW * sW2 / np.pi


# ---------------------------------------------------------------------------
# Complex threshold velocity
# ---------------------------------------------------------------------------

def beta_complex(s, mW: float, gammaW: float):
    """
    Complex velocity β_M = √(1 - 4(m_W² - i m_W Γ_W)/s).
    Sign convention: Im[β_M] > 0. Re[β_M] is real-positive everywhere,
    including below threshold (EFT smoothing).

    Vectorised: accepts scalar or array ``s``; returns same shape (complex).
    """
    mW2c = mW * mW - 1j * mW * gammaW
    b2 = 1.0 - 4.0 * mW2c / s
    b = np.sqrt(np.asarray(b2, dtype=complex))
    b = np.where(b.imag < 0, -b, b)
    if np.ndim(s) == 0:
        return complex(b)
    return b


# ---------------------------------------------------------------------------
# σ̂_WW^Born -- on-shell, calibrated against RACOONWW
# ---------------------------------------------------------------------------
# LEP2/LEP-energy Born CC03 reference values (RACOONWW; values for
# m_W = 80.379 GeV, m_Z = 91.1876 GeV, in the G_μ scheme).
#
# 161.33–500 GeV: RACOONWW Born values (≈0.3% accurate).
# Below 161.33 GeV: BFS LO_EFT N^(3/2)LO Born expansion (eq. 17 + 33 + 37 +
#   39 of arXiv:0707.0773), matched multiplicatively to RACOONWW at 161.33 GeV
#   for C⁰ continuity. The BFS expansion agrees with the full 4f Born
#   (Whizard) to 0.1 % at threshold and ~3 % at 155 GeV per Table 1 of the
#   paper — vastly better than the previous power-law extrapolation.
# ---------------------------------------------------------------------------
_REF_SQRTS = np.array([
    161.33, 165.0, 170.0, 172.13, 175.0, 180.0, 182.66, 185.0,      # real RACOONWW
    188.63, 195.0, 200.0, 207.0, 220.0, 240.0, 280.0, 320.0, 365.0, 500.0,
])
_REF_SIGMA_BORN = np.array([
    3.69, 8.62, 11.50, 12.43, 13.55, 15.50, 16.62, 16.95,
    17.21, 17.50, 17.59, 17.50, 17.00, 16.50, 14.50, 12.80, 11.20, 7.50,
])

# Cross-section model regions:
#   √s < 149 GeV         : σ = 0 (ξ(s), χ(s) have spurious Z-pole far below
#                          BFS validity; ISR convolution would sample it)
#   149 ≤ √s ≤ 150 GeV   : BFS LO_EFT × smoothstep ramp (0 → 1) — softens
#                          the lower floor so the ISR convolution kernel
#                          doesn't pick up quadrature noise from a hard
#                          step. BFS values are still positive in this
#                          range (~0.1–0.4 pb) so the ramp is physical.
#   150 ≤ √s < 170 GeV   : pure BFS LO_EFT N^(3/2)LO Born
#                          (matches Whizard exact 4f Born to 1% over 155–170)
#   √s ≥ 170 GeV         : RACOONWW calibration spline (LEP2-era CC03 Born)
#                          retained for the 240 GeV reference point used
#                          when --lastecm is enabled.
#
# Note: the BFS and RACOONWW values disagree by ~13–18% at the boundary
# (singly-resonant content + 5% NLO-width-resummation correction + EFT-
# validity drift above 170 GeV). The discontinuity at 170 GeV is benign
# for the threshold-scan analysis (no scan points sit there).
_SQRTS_BFS_UPPER = 170.0
_S_BFS_UPPER = _SQRTS_BFS_UPPER ** 2

# Floor: BFS evaluation extends to 149 GeV (still positive there); a smooth
# weight ramps 0 → 1 across the 1-GeV window [149, 150]. Below 149 GeV,
# σ = 0. The smoothstep choice 3t² − 2t³ is C¹ — enough to eliminate the
# discontinuity-driven kink in the ISR-convolved variation-template ratios
# (was visible as a "kink near 159 GeV" before this softening).
_SQRTS_BFS_FLOOR = 149.0
_S_BFS_FLOOR = _SQRTS_BFS_FLOOR ** 2
_SQRTS_BFS_RAMP_TOP = 150.0


def _bfs_floor_weight(s_arr):
    """Quintic smoothstep ramp 0 → 1 across [149, 150] GeV applied to BFS σ̂.
    Returns a vector of weights ∈ [0, 1] for the input ``s_arr`` (GeV²).
    Below 149 GeV the weight is 0 (hard zero); above 150 GeV it is 1 (BFS
    unchanged); in between the weight is the C² quintic smoothstep
    6t⁵ − 15t⁴ + 10t³ (zero 1st AND 2nd derivative at both ends), so the
    ISR integrand is C² continuous across the boundary — the C¹ cubic
    smoothstep still leaked a small d² residual into the variation-ratio
    plots near 157 GeV.
    """
    sqrt_s = np.sqrt(np.asarray(s_arr, dtype=float))
    t = np.clip((sqrt_s - _SQRTS_BFS_FLOOR)
                / (_SQRTS_BFS_RAMP_TOP - _SQRTS_BFS_FLOOR), 0.0, 1.0)
    return t ** 3 * (10.0 + t * (-15.0 + 6.0 * t))


def _F_factor_grid():
    """
    F(s) = σ_ref × s × s_W^4 / (π α² × β_eff) at the calibration grid.
    """
    alpha = alpha_Gmu(M_W_DEFAULT)
    sW2 = sin2_thetaW_OS(M_W_DEFAULT)
    s_grid = _REF_SQRTS ** 2
    beta_grid = np.array([
        beta_complex(s, M_W_DEFAULT, GAMMA_W_DEFAULT).real
        for s in s_grid
    ])
    F_grid = _REF_SIGMA_BORN / GEV_M2_TO_PB * sW2 ** 2 * s_grid / (np.pi * alpha ** 2 * beta_grid)
    return s_grid, F_grid


_S_GRID, _F_GRID = _F_factor_grid()

# C² cubic spline through F(s) for smooth derivatives at grid knots.
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
    _F_SPLINE = _CubicSpline(_S_GRID, _F_GRID, bc_type="natural", extrapolate=True)

    def _interp_F_smooth(s_eff):
        return _F_SPLINE(s_eff)
except ImportError:
    def _interp_F_smooth(s_eff):
        return np.interp(s_eff, _S_GRID, _F_GRID)


def sigma_WW_partonic(s,
                  mW: float = M_W_DEFAULT,
                  gammaW: float = GAMMA_W_DEFAULT,
                  # Defaults below are the project's "best calculation":
                  # full BFS NLO+NNLO chain + δ_QCD + Whizard anchor (= what
                  # the production fit templates use). Toggle individual knobs
                  # off for diagnostic / Born-only comparisons.
                  include_NLO_hard_decay: bool = True,
                  include_BFS_NNLO: bool = True,
                  apply_delta_QCD: bool = True,
                  alpha_s: float = ALPHA_S_MW_DEFAULT,
                  apply_whizard_anchor: bool = True,
                  whizard_anchor_source: str = "grid",
                  coulomb_kc_safe: bool = False,
                  decay_uses_full_born: bool = True):
    """
    Off-shell-convolved σ(e+e- → W+W- → 4f), full off-shell, in pb.

    Three regions in absolute √s (m_W independent):

    * ``√s < 150 GeV``  →  σ = 0. Avoids the spurious M_Z pole in the
      BFS ξ(s)/χ(s) functions; σ is negligible anyway.
    * ``150 ≤ √s < 170 GeV``  →  BFS LO_EFT N^{3/2}LO Born from
      ``bfs_eft.sigma_BFS_LO_total_WW_pb`` with the (Γ_W^(0)/Γ_W)² BR
      correction; NLO loop chain (HSC + Coulomb_NLO + EW-decay), δ_QCD,
      and the Whizard anchor are applied by default (toggle via the
      flag arguments above). No matching to the RACOONWW grid — BFS
      gives the *full* Born (CC03 + singly-resonant) and matches the
      Whizard exact 4f Born to ~1 % over 155–170 GeV. Full analytic
      m_W, Γ_W dependence is preserved.
    * ``√s ≥ 170 GeV``  →  RACOONWW calibration spline (CC03 Born).
      Used only for the 240 GeV "last_ecm" reference point in the
      analysis; the threshold-scan grid (157–163 GeV) never enters
      this region. The BFS/spline values differ by ~13–18 % at 170
      GeV (singly-resonant content); this discontinuity is benign
      for the analysis fit but is a known feature in the diagnostic
      plots.

    Vectorised: accepts scalar or array ``s`` (m_W, Γ_W must be scalar).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)

    use_zero = s_arr < _S_BFS_FLOOR
    use_bfs = (s_arr >= _S_BFS_FLOOR) & (s_arr < _S_BFS_UPPER)
    use_cal = s_arr >= _S_BFS_UPPER

    sigma_pb = np.zeros_like(s_arr)

    if np.any(use_bfs):
        from framework.process.ww.xsec_calculator.bfs_eft import sigma_BFS_LO_total_WW_pb
        sigma_bfs_pb = sigma_BFS_LO_total_WW_pb(
            s_arr, mW, gammaW, order="N3/2LO",
            include_NLO_hard_decay=include_NLO_hard_decay,
            include_BFS_NNLO=include_BFS_NNLO,
            apply_delta_QCD=apply_delta_QCD,
            alpha_s=alpha_s,
            apply_whizard_anchor=apply_whizard_anchor,
            whizard_anchor_source=whizard_anchor_source,
            coulomb_kc_safe=coulomb_kc_safe,
            decay_uses_full_born=decay_uses_full_born,
        )
        # Smooth-floor weight is 1 above 150 GeV, ramps to 0 across [149, 150]
        # to keep σ̂(s) C¹ for the ISR convolution kernel.
        sigma_bfs_pb = sigma_bfs_pb * _bfs_floor_weight(s_arr)
        sigma_pb = np.where(use_bfs, sigma_bfs_pb, sigma_pb)

    if np.any(use_cal):
        bM_real = beta_complex(s_arr, mW, gammaW).real
        sqrt_s = np.sqrt(s_arr)
        sqrt_s_eff = sqrt_s * (M_W_DEFAULT / mW)
        s_eff = sqrt_s_eff ** 2
        F = _interp_F_smooth(s_eff)
        sigma_cal_pb = (
            (np.pi * alpha ** 2) / (sW2 ** 2 * s_arr)
            * np.maximum(bM_real, 0.0) * F * GEV_M2_TO_PB
        )
        sigma_pb = np.where(use_cal, sigma_cal_pb, sigma_pb)

    return np.where((sigma_pb > 0.0) & (~use_zero), sigma_pb, 0.0)


# ---------------------------------------------------------------------------
# Coulomb K-factor (Fadin-Khoze-Martin with finite Γ_W)
# ---------------------------------------------------------------------------

def coulomb_K_factor(s,
                     mW: float = M_W_DEFAULT,
                     gammaW: float = GAMMA_W_DEFAULT,
                     order: int = 1,
                     prescription: str = "on-shell"):
    """
    Coulomb-photon-exchange K-factor with finite Γ_W:

        K_C = 1 + (α√s)/(4p) × [π - 2·arctan((|κ|² - p²)/(2 p Re κ))]
              + (α² s ln 2)/(4|κ|²)              # O(α²) (Bardin-Riemann)

    p = (√s/2) × Re[β_M],   κ = √(-m_W (E + i Γ_W)),   E = √s - 2 m_W.

    Refs:
        Fadin, Khoze, Martin, Phys.Lett.B311 (1993) 311 (arctan form)
        Fadin, Khoze, Martin, Stirling, hep-ph/9507422 eq. (9-11), Z.Phys.C75 (1997) 53

    NOTE on conventions (2026-05-18). The default ``prescription="on-shell"``
    uses the BFS-EFT complex-p regularisation: p = (√s/2)·Re[β_M_complex]
    where β_M² = 1 − 4(m_W² − i·m_W·Γ_W)/s. This is the natural finite-Γ_W
    smoothing of the FKM 1995 eq. (3) on-shell kinematic momentum — analytic
    in (m_W, Γ_W) everywhere, so dσ/dm_W and dσ/dΓ_W have no derivative
    cusp at 2m_W. Above threshold by many widths it recovers FKM exactly.
    At threshold it gives K_C − 1 ≈ +7.3 % vs FKM's strict-Γ_W → 0 limit
    +6.6 %; the 0.7 pp difference is the W-width effect on the Coulomb.

    Set ``prescription="real-p-strict"`` to recover FKM's exact zero-width
    formula (p = sqrt(max(s/4 − m_W², 0))), at the cost of a derivative
    cusp in m_W at threshold — useful only for direct literature comparison.

    BFS arXiv:0707.0773 eq. (62) is yet another formulation (EFT Coulomb
    expansion in α, with on-shell limit), giving +5.2 % at threshold
    (first-order log) + 0.18 % (NLO two-photon). Applying both K_C and
    BFS eq. (62) overlaps at leading order. See ``BFSCorrections.delta_NLO``
    flag ``enabled_coulomb_NLO_subleading`` to pick up only the NLO
    two-photon piece K_C-safely.

    α = α(0) (Thomson limit) for the soft Coulomb photon. Vectorised in s.
    """
    s_arr = np.asarray(s, dtype=float)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW

    kappa = np.sqrt(np.asarray(-mW * (E + 1j * gammaW), dtype=complex))
    kappa = np.where(kappa.real < 0, -kappa, kappa)

    if prescription == "on-shell":
        # On-shell kinematic momentum with finite-Γ_W complex-p regularisation:
        #     p = (√s/2) · Re[β_M_complex],   β_M² = 1 − 4(m_W² − i·m_W·Γ_W)/s
        # — i.e. take the real part of the natural complex velocity in the
        # BFS unstable-particle EFT (m_W² → m_W² − i·m_W·Γ_W in the propagator).
        # Above threshold by many widths this recovers the FKM 1995 eq. (3)
        # on-shell momentum to O(Γ_W²/(s/4 − m_W²)). At threshold p tends to
        # √(m_W·Γ_W/2) ≈ 9 GeV instead of 0, which sets the natural Coulomb
        # scale of an unstable W. Critically, p is ANALYTIC in m_W and Γ_W
        # everywhere (no kink at 2m_W), so dσ/dm_W and dσ/dΓ_W are smooth.
        # K_C − 1 at threshold: ~+7.3 % vs FKM's strict-on-shell L'Hôpital
        # limit +6.6 % (Γ_W → 0); the 0.7 pp difference is the width effect.
        bM = beta_complex(s_arr, mW, gammaW)
        p = 0.5 * sqrt_s * bM.real
    elif prescription == "real-p-strict":
        # Strict-on-shell with p = sqrt(max(s/4 − m_W², 0)) — exactly the
        # FKM 1995 eq. (3) real momentum, identically zero below threshold.
        # Reproduces the FKM threshold value via a separate L'Hôpital limit
        # at p → 0, but introduces a derivative cusp in m_W at 2m_W (the
        # max() kink). Retained only for direct comparison with literature
        # that uses zero-width Coulomb at threshold.
        p2 = s_arr / 4.0 - mW ** 2
        p = np.sqrt(np.maximum(p2, 0.0))
    else:
        raise ValueError(
            f"prescription must be 'on-shell' (default, complex-p regularised) "
            f"or 'real-p-strict' (FKM zero-width, with derivative cusp); "
            f"got {prescription!r}"
        )

    abs_kappa2 = np.abs(kappa) ** 2
    re_kappa = kappa.real

    # For real-p-strict: p ≡ 0 at/below threshold → use the FKM L'Hôpital
    # limit K_1 → 1 + α√s · Re(κ)/|κ|². For on-shell (complex-regularised):
    # p ≥ √(m_W·Γ_W/2) > 0 everywhere → arctan formula is stable.
    K1_limit = 1.0 + ALPHA_EM_0 * sqrt_s * re_kappa / abs_kappa2

    denom = 2.0 * p * re_kappa
    safe = np.abs(denom) > 1e-12
    arctan_val = np.where(
        safe,
        np.arctan(np.where(safe, (abs_kappa2 - p * p) / np.where(safe, denom, 1.0), 0.0)),
        0.5 * np.pi * np.sign(abs_kappa2 - p * p),
    )

    p_safe = np.where(p > 1e-6, p, 1.0)
    K1_main = 1.0 + (ALPHA_EM_0 * sqrt_s / (4.0 * p_safe)) * (np.pi - 2.0 * arctan_val)
    K1 = np.where(p > 1e-6, K1_main, K1_limit)

    if order < 2:
        out = K1
    else:
        # Second-order |f|² expansion to O(α²) from FKM eq. (10):
        #   f(p,E) ≈ 1 + α√s/(2κ) + α²s ln 2/(4κ²)         (valid p ≪ |κ|)
        # |f|² to O(α²):
        #   |f|² ≈ 1 + α√s · Re(1/κ) + α²s · [1/(4|κ|²) + (ln 2 / 2)·Re(1/κ²)]
        #                              └── |f₁|² ──┘  └── 2 Re(f₂) ──┘
        # Previous form α²s ln 2/(4|κ|²) was wrong — it paired ln 2 (which
        # belongs to f₂ ∝ 1/κ²) with |κ|² from |f₁|², and missed the |f₁|²
        # piece entirely. At threshold Re(1/κ²) = 0, so only |f₁|² contributes.
        # Validated 2026-05-18: gives 0.22% at threshold (was 0.15%; FKM eq. 21
        # X²/6 = 0.18% — proper |f|² O(α²) differs from the near-threshold
        # X expansion at the percent level, by construction).
        kappa_complex = np.asarray(kappa, dtype=complex)
        inv_kappa2 = 1.0 / (kappa_complex ** 2)
        delta_alpha2 = ALPHA_EM_0 ** 2 * s_arr * (
            1.0 / (4.0 * abs_kappa2)
            + (np.log(2.0) / 2.0) * inv_kappa2.real
        )
        out = K1 + delta_alpha2

    if np.ndim(s) == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# BFS NLO + NNLO placeholders
# ---------------------------------------------------------------------------

@dataclass
class BFSCorrections:
    """
    NLO + dominant-NNLO matching corrections from
        Beneke, Falgari, Schwinn arXiv:0707.0773  (NLO)
        Actis, Beneke, Falgari, Schwinn arXiv:0807.0102  (dominant NNLO)

    Per-piece flags:
      * ``enabled_coulomb_NLO``: include FULL eq. (62) of arXiv:0707.0773
        (one-photon log term ~5 % + two-photon ~0.2 %). IR-finite.
        **OVERLAPS with the off-shell-resummed K_C** at leading order
        (~5 % double-counting at threshold). Use this if K_C is OFF
        (``WWGenerator(include_coulomb=False)``).
      * ``enabled_coulomb_NLO_subleading``: include ONLY the NLO two-
        photon term (second term of eq. 62), ~0.2 % at threshold. Safe
        to combine with K_C (no leading-order overlap). Mutually
        exclusive with ``enabled_coulomb_NLO``.
      * ``enabled_hard_NLO``  : NOT YET IMPLEMENTED. Eq. (56) — needs
        c_p,LR^(1,fin) from ref. [13].
      * ``enabled_soft_NLO``  : NOT YET IMPLEMENTED. Eq. (64)/(65) —
        IR poles need MS-bar ePDF for cancellation.
      * ``enabled_decay_NLO`` : already absorbed via fixed PDG BRs.
      * ``enabled_NNLO``      : eq. (3.1) of arXiv:0807.0102 — NOT YET.

    ``enabled`` (legacy bool) is a shortcut for
    ``enabled_coulomb_NLO_subleading=True`` — the K_C-safe combination
    that's the safe default for "add NLO Coulomb on top of K_C".
    """
    enabled: bool = False
    enabled_coulomb_NLO: bool = False
    enabled_coulomb_NLO_subleading: bool = False

    def __post_init__(self):
        # Legacy ``enabled=True`` shortcut: K_C-safe NLO additions only
        # (subleading Coulomb; hard/soft/NNLO when those get implemented).
        if self.enabled:
            self.enabled_coulomb_NLO_subleading = True
        if self.enabled_coulomb_NLO and self.enabled_coulomb_NLO_subleading:
            raise ValueError(
                "enabled_coulomb_NLO and enabled_coulomb_NLO_subleading are "
                "mutually exclusive: the former INCLUDES the latter."
            )

    def delta_NLO(self, s, mW: float, gammaW: float):
        """Return the relative NLO correction δ s.t. σ_partonic
        = σ_LO × (1 + δ_NLO + …). Vectorised in ``s``.

        Currently sums only the implemented pieces.
        """
        out = 0.0
        if self.enabled_coulomb_NLO or self.enabled_coulomb_NLO_subleading:
            from framework.process.ww.xsec_calculator.bfs_eft import (
                delta_sigma_Coulomb_NLO_specific_pb,
                sigma_LR0_specific_pb,
            )
            d_sigma_C = delta_sigma_Coulomb_NLO_specific_pb(
                s, mW, gammaW,
                apply_BR_correction=True,
                subleading_only=self.enabled_coulomb_NLO_subleading,
            )
            sigma_LR0 = sigma_LR0_specific_pb(s, mW, gammaW,
                                              apply_BR_correction=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(sigma_LR0 > 0, d_sigma_C / sigma_LR0, 0.0)
            out = out + rel
        return out

    def delta_NNLO(self, s, mW: float, gammaW: float):
        if not self.enabled:
            return 0.0
        raise NotImplementedError(
            "Fill in arXiv:0807.0102 eq. (3.1): NNLO Coulomb², "
            "single-Coulomb × soft interference, NLL-resummed hard function."
        )


# ---------------------------------------------------------------------------
# Partonic cross section for μν qq̄
# ---------------------------------------------------------------------------

# Number of specific channels contributing to each named channel. The
# per-channel σ is built directly from BFS specific-channel formulae
# (which carry the proper per-component BR correction: squared for the
# potential-region pieces σ^(0), σ^(1)_pot; linear for the hard-region
# pieces σ^(1/2), σ^(3/2),a, per §6.1 of arXiv:0707.0773), then scaled
# by this multiplicity.
#   "inclusive" μν qq̄: 4 = 2 W charges × 2 quark generations (ud̄, cs̄)
#   "munuud" μ⁻ν̄_μ ud̄: 1 = the BFS-specific channel itself
_CHANNEL_MULTIPLICITY = {
    "inclusive": 4,
    "munuud":    1,
}


def sigma_partonic_munuqq(s,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          channel: str = "inclusive",
                          include_coulomb: bool = True,
                          bfs: BFSCorrections | None = None,
                          br_convention: str = "pdg-constant",
                          # Defaults below are the project's "best calculation"
                          # — full BFS NLO+NNLO chain + δ_QCD + Whizard anchor.
                          # Validation: scripts/validate_bfs_nlo.py +
                          # scripts/investigations/bfs_nnlo/.
                          include_NLO_hard_decay: bool = True,
                          include_BFS_NNLO: bool = True,
                          apply_delta_QCD: bool = True,
                          alpha_s: float = ALPHA_S_MW_DEFAULT,
                          alpha_s_ref: float = ALPHA_S_MW_DEFAULT,
                          apply_whizard_anchor: bool = True,
                          whizard_anchor_source: str = "grid",
                          coulomb_kc_safe: bool = False,
                          decay_uses_full_born: bool = True):
    """
    Partonic σ(e+e- → μν qq̄) at LO + Coulomb (+ optional BFS NLO/NNLO).
    Returns σ in pb at partonic CM energy² = s (before ISR convolution).

    ``channel`` selects the named final state:
        "inclusive" (default) — μν qq̄, both W charges × (ud̄, cs̄)
        "munuud"              — μ⁻ν̄_μ ud̄ specific

    ``br_convention`` selects how the BR factor depends on (m_W, Γ_W) and
    where δ_QCD enters when ``apply_delta_QCD=True``:

      * ``"pdg-constant"`` (default) — fixed PDG-measured BR product:
            BR_inclusive = 2·BR(W→μν)·BR(W→had) = 0.1433  (≈ ``BR_INCLUSIVE_MUNUQQ``)
            BR_munuud    =  BR(W→μν)·BR(W→ud̄)   = 0.0357  (≈ ``BR_MUNUUD``)
        Independent of (m_W, Γ_W). Γ_W enters σ only via the propagator
        broadening in σ_WW. d BR/dΓ_W = 0 → d σ/dΓ_W reflects pure propagator
        broadening. Matches the YFSWW3/RACOONWW experimental convention
        (BR taken from data; Γ_W is the propagator parameter only).
        Differs from the LO theory BR 4/27 ≈ 0.148 by the 3.5 % radiative
        corrections folded into PDG.

        With ``apply_delta_QCD=True`` the QCD correction is applied to the BR
        (where it physically belongs: δ_QCD multiplies Γ_had inside
        BR(W→qq̄)) via the α_s-aware factor
            BR(α_s) = BR_PDG × δ_QCD(α_s) / δ_QCD(α_s_ref)
        with ``α_s_ref`` the card-declared nominal. At α_s = α_s_ref the
        ratio is 1 → BR_PDG is recovered exactly (no double-count). The
        α_s differential ∂σ/∂α_s is preserved. δ_QCD is NOT multiplied onto
        σ_WW in this convention.

      * ``"bfs-eft"`` — BFS section 6.1 / eq. 83 per-component:
            BR = (channel_mult/27) × (Γ_W^(0)(m_W)/Γ_W)²  for σ^(0), σ^(1)_pot
                                                          (potential region,
                                                          two cut propagators)
            BR = (channel_mult/27) × (Γ_W^(0)(m_W)/Γ_W)   for σ^(1/2), σ^(3/2),a
                                                          (hard region, one
                                                          W in NWA)
        Theory-fixed-partials: partial widths Γ_x^(0)(m_W) are SM-LO predictions
        of m_W only; total Γ_W is the fit parameter; BR shrinks as the partials
        are divided by a (potentially) larger total. d ln BR/dΓ_W = −2/Γ_W
        (resp. −1/Γ_W for σ^(1/2)). Reproduces BFS Tables 1, 2 round-trip
        when called via the BFS specific-channel helpers in bfs_eft.py.

    Vectorised: accepts scalar or array ``s``.
    """
    if bfs is None:
        bfs = BFSCorrections(enabled=False)
    if channel not in _CHANNEL_MULTIPLICITY:
        raise ValueError(f"channel={channel!r} not in {list(_CHANNEL_MULTIPLICITY)}")
    if br_convention not in ("bfs-eft", "pdg-constant"):
        raise ValueError(f"br_convention must be 'bfs-eft' or 'pdg-constant'; "
                         f"got {br_convention!r}")

    # Region-aware σ (handles ISR convolution sampling sub-threshold s_hat):
    #   √s < 150 GeV  → 0 (avoids spurious M_Z pole in BFS ξ,χ functions)
    #   150 ≤ √s < 170 → BFS computation
    #   √s ≥ 170 GeV  → RACOONWW calibration spline × BR factor
    s_arr = np.asarray(s, dtype=float)
    use_bfs = (s_arr >= _S_BFS_FLOOR) & (s_arr < _S_BFS_UPPER)
    use_cal = s_arr >= _S_BFS_UPPER

    sigma = np.zeros_like(s_arr)

    # Route δ_QCD by convention: for pdg-constant the QCD correction belongs
    # in the BR (Γ_had ∝ δ_QCD); for bfs-eft it multiplies σ per BFS §6.1.
    if br_convention == "pdg-constant":
        BR_pdg = {"inclusive": BR_INCLUSIVE_MUNUQQ,
                  "munuud":    BR_MUNUUD}[channel]
        if apply_delta_QCD:
            from framework.process.ww.xsec_calculator.bfs_eft import delta_QCD_factor
            BR_pdg = BR_pdg * (delta_QCD_factor(alpha_s)
                               / delta_QCD_factor(alpha_s_ref))
        apply_delta_QCD_on_sigma = False
    else:
        apply_delta_QCD_on_sigma = apply_delta_QCD

    if np.any(use_bfs):
        if br_convention == "bfs-eft":
            from framework.process.ww.xsec_calculator.bfs_eft import sigma_BFS_specific_munuud_pb
            sigma_specific = sigma_BFS_specific_munuud_pb(
                s_arr, mW, gammaW, order="N3/2LO",
                include_NLO_hard_decay=include_NLO_hard_decay,
                include_BFS_NNLO=include_BFS_NNLO,
                apply_delta_QCD=apply_delta_QCD_on_sigma,
                alpha_s=alpha_s,
                apply_whizard_anchor=apply_whizard_anchor,
                whizard_anchor_source=whizard_anchor_source,
                coulomb_kc_safe=coulomb_kc_safe,
                decay_uses_full_born=decay_uses_full_born,
            )
            sigma_bfs = sigma_specific * _CHANNEL_MULTIPLICITY[channel]
        else:   # pdg-constant: σ_WW × BR_PDG (BR carries δ_QCD)
            from framework.process.ww.xsec_calculator.bfs_eft import sigma_BFS_LO_total_WW_pb
            sigma_WW_total = sigma_BFS_LO_total_WW_pb(
                s_arr, mW, gammaW,
                order="N3/2LO",
                apply_BR_correction=False,
                include_NLO_hard_decay=include_NLO_hard_decay,
                include_BFS_NNLO=include_BFS_NNLO,
                apply_delta_QCD=apply_delta_QCD_on_sigma,
                alpha_s=alpha_s,
                apply_whizard_anchor=apply_whizard_anchor,
                whizard_anchor_source=whizard_anchor_source,
                coulomb_kc_safe=coulomb_kc_safe,
                decay_uses_full_born=decay_uses_full_born,
            )
            sigma_bfs = sigma_WW_total * BR_pdg
        # Smooth-floor weight kills the hard step at 150 GeV that was
        # seeding ISR quadrature noise (kink in σ-variation ratios).
        sigma_bfs = sigma_bfs * _bfs_floor_weight(s_arr)
        sigma = np.where(use_bfs, sigma_bfs, sigma)

    if np.any(use_cal):
        if br_convention == "bfs-eft":
            from framework.process.ww.xsec_calculator.bfs_eft import gamma_W_LO
            BR_x = (_CHANNEL_MULTIPLICITY[channel] / 27.0) * (gamma_W_LO(mW) / gammaW) ** 2
        else:
            BR_x = BR_pdg
        sigma_cal = sigma_WW_partonic(
            s_arr, mW, gammaW,
            include_NLO_hard_decay=include_NLO_hard_decay,
            include_BFS_NNLO=include_BFS_NNLO,
            apply_delta_QCD=apply_delta_QCD_on_sigma,
            alpha_s=alpha_s,
            apply_whizard_anchor=apply_whizard_anchor,
            whizard_anchor_source=whizard_anchor_source,
            coulomb_kc_safe=coulomb_kc_safe,
            decay_uses_full_born=decay_uses_full_born,
        ) * BR_x
        sigma = np.where(use_cal, sigma_cal, sigma)

    if include_coulomb:
        sigma = sigma * coulomb_K_factor(s, mW, gammaW)

    sigma = sigma * (1.0 + bfs.delta_NLO(s, mW, gammaW)
                          + bfs.delta_NNLO(s, mW, gammaW))

    if np.ndim(s) == 0:
        return float(sigma)
    return sigma


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("σ(e+e- → W+W-) Born CC03 — calibration grid round-trip")
    print("=" * 76)
    print(f"{'√s [GeV]':>10}  {'σ this code':>15}  {'σ RACOONWW':>15}  {'ratio':>8}")
    for sqrts, sigma_ref in zip(_REF_SQRTS, _REF_SIGMA_BORN):
        s = sqrts ** 2
        sigma_mine = sigma_WW_partonic(s)
        print(f"  {sqrts:8.2f}    {sigma_mine:12.4f} pb   {sigma_ref:12.4f} pb"
              f"   {sigma_mine / sigma_ref:6.4f}")

    print("\n" + "=" * 76)
    print("Inclusive μν qq̄ partonic σ at threshold scan points")
    print("=" * 76)
    print(f"{'√s [GeV]':>10}  {'σ_partonic [pb]':>17}  {'K_Coul':>8}  {'channel':>10}")
    for sqrts in [157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 162.5, 163.0, 165.0]:
        s = sqrts ** 2
        sigma_incl = sigma_partonic_munuqq(s, channel="inclusive")
        sigma_ud = sigma_partonic_munuqq(s, channel="munuud")
        kC = coulomb_K_factor(s)
        print(f"  {sqrts:8.2f}    {sigma_incl:15.5f}    {kC:6.4f}   inclusive")
        print(f"  {sqrts:8.2f}    {sigma_ud:15.5f}    {kC:6.4f}      munuud")
