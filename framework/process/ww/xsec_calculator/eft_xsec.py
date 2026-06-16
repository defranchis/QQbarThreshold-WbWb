"""σ(e+e- → μν qq̄) near the WW threshold for FCC-ee — partonic level.

Default partonic cross section (= what the production fit templates use):
BFS-EFT N^(3/2)LO Born + NLO loops + δ_QCD + Whizard 4f Born anchor,
with finite-Γ_W complex-velocity smoothing. Above √s = 170 GeV a
RACOONWW CC03 calibration spline is used (only the 240 GeV "last_ecm"
reference point if --lastecm is enabled).

Pipeline (``sigma_partonic_munuqq``), production defaults:

    σ̂_partonic(ŝ; m_W, Γ_W)
        = ( BFS N^(3/2)LO Born + Δσ_HSC + Δσ_Coul^NLO + Δσ_decay^EW + NNLO )
          × f_Whizard(δ, Γ_W)        ← BFS sec. 6.2 anchor
          × δ_QCD(α_s)               ← routed through BR in pdg-constant mode
          × BR_channel

The multiplicative FKM ``K_Coulomb`` (Fadin-Khoze-Martin) factor is OFF by
default (``include_coulomb=False``): the BFS Coulomb correction lives
additively in the NLO loops, so a multiplicative K_C would double-count it.
It is retained only for diagnostic comparisons against pre-BFS calculations.

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
ALPHA_MZ_PDG    = 1.0 / 128.943   # α(M_Z), PDG — QED coupling for the ISR ALPMZ
                                  # renorm scheme (eMELA). SINGLE SOURCE OF TRUTH
                                  # for the framework: imported by isr.py (bare-ISR
                                  # ALPMZ default) and theory_ladder.py (fallback).
                                  # The card PARAM_INPUTS["alpha_em_isr"] is the
                                  # user-editable INPUT that defaults to this value;
                                  # it keeps its own literal because cards/ww_default
                                  # cannot import eft_xsec (this module imports the
                                  # card's BRs below → would be a circular import).
                                  # The indep chain keeps its own copy by design
                                  # (isr_beta.ALPHA_MZ_EMELA — see that module).
GEV_M2_TO_PB    = 3.8937937217e8
PB_TO_FB        = 1000.0   # picobarn → femtobarn

# ---------------------------------------------------------------------------
# WW-chain defaults (single source of truth; imported by bfs_eft, isr,
# generator, compute_xsec_ww, cards). PRODUCTION defaults are PDG values
# (now plumbed end-to-end via the PARAM_INPUTS card knob; previously the
# card's m_t/M_H were silently ignored by the σ chain — see the
# 2026-05-27 calc /simplify pass and the 2026-05-27 parameter audit).
#
# The M_*_BFS_REF constants are the inputs at which arXiv:0707.0773 + 0807.0102
# evaluated the BFS Tables 1-4 and c^(1,fin) reference numbers; validation
# scripts (scripts/validate_bfs_nlo.py, scripts/investigations/bfs_nnlo/,
# scripts/investigations/c1fin_analytic/) pass these explicitly to reproduce
# paper closure. They differ from the PDG production defaults (M_H pre-discovery
# 115 vs 125.25; M_Z 91.188 vs 91.1876).
# ---------------------------------------------------------------------------
ALPHA_S_MW_DEFAULT = 0.1199     # α_s(M_W) in MS-bar, BFS reference
M_W_BFS_REF        = 80.377     # m_W at which BFS Tables / c_fin are tabulated
M_T_DEFAULT        = 172.5      # m_t (pole), FCC-ee FSR Table 2 (was 174.2 = BFS ref)
M_H_DEFAULT        = 125.25     # PDG (production); 115 = BFS Table 4 (pre-discovery)
M_H_BFS_REF        = 115.0      # M_H used by BFS arXiv:0707.0773 Tables/c_fin
M_T_BFS_REF        = 174.2      # m_t used by BFS arXiv:0707.0773 Tables/c_fin
M_Z_BFS_REF        = 91.188     # M_Z used by BFS arXiv:0707.0773 Tables/c_fin
                                # (3 sig figs in paper; 91.1876 PDG used as
                                # production default for M_Z)

# PDG branching ratios — primitives live in the steering card (single source
# of truth); the combinations below are derived once at module load.
from cards.ww_default import BR_W_MUNU, BR_W_HAD, BR_W_UD  # noqa: E402

# 2 × BR(W→μν) × BR(W→had): factor 2 because either W can be the muonic one.
BR_INCLUSIVE_MUNUQQ = 2.0 * BR_W_MUNU * BR_W_HAD
# BR(W→μν) × BR(W→ud̄/cs̄): legacy single-specific-channel convention from
# the BFS tables.
BR_MUNUUD          = BR_W_MUNU * BR_W_UD


def sin2_thetaW_OS(mW: float = M_W_DEFAULT, MZ: float = M_Z) -> float:
    return 1.0 - (mW / MZ) ** 2


def alpha_Gmu(mW: float = M_W_DEFAULT, MZ: float = M_Z,
              override: float | None = None) -> float:
    """α_em in the G_μ scheme: √2 G_F m_W² sin²θ_W / π.

    ``override`` short-circuits the derivation — pass a float to inject a
    user-supplied α_em (used by the PARAM_UNC.alpha_em nuisance via the
    PARAM_INPUTS.alpha_em card knob).  ``None`` (default) → derived value.
    """
    if override is not None:
        return float(override)
    sW2 = sin2_thetaW_OS(mW, MZ)
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

_SQRTS_BFS_FLOOR = 149.0
_S_BFS_FLOOR = _SQRTS_BFS_FLOOR ** 2
_SQRTS_BFS_RAMP_TOP = 150.0


def _bfs_floor_weight(s_arr):
    """C² quintic smoothstep 6t⁵ − 15t⁴ + 10t³ on [149, 150] GeV — keeps
    the ISR integrand C² continuous across the BFS lower floor."""
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
                  decay_uses_full_born: bool = True,
                  m_t: float = M_T_DEFAULT,
                  M_H: float = M_H_DEFAULT,
                  MZ: float = M_Z,
                  alpha_em: float | None = None):
    """
    Off-shell-convolved σ(e+e- → W+W- → 4f), full off-shell, in pb.

    Three regions in absolute √s (m_W independent):

    * ``√s < 149 GeV``  →  σ = 0. Avoids the spurious M_Z pole in the
      BFS ξ(s)/χ(s) functions; σ is negligible anyway. A C² quintic ramp
      (0→1 across [149, 150] GeV) softens the floor so the ISR convolution
      kernel stays smooth.
    * ``149 ≤ √s < 170 GeV``  →  BFS LO_EFT N^{3/2}LO Born from
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
    alpha = alpha_Gmu(mW, MZ, override=alpha_em)
    sW2 = sin2_thetaW_OS(mW, MZ)

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
            mt=m_t, MH=M_H, MZ=MZ,
            alpha_em=alpha_em,
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
                     gammaW: float = GAMMA_W_DEFAULT):
    """
    Coulomb-photon-exchange K-factor with finite Γ_W (FKM 1993 arctan form):

        K_C = 1 + (α√s)/(4p) × [π - 2·arctan((|κ|² - p²)/(2 p Re κ))]

    with p = (√s/2)·Re[β_M_complex] (complex-p regularisation analytic in
    m_W, Γ_W everywhere — no derivative cusp at 2 m_W) and
    κ = √(-m_W·(E + i Γ_W)), E = √s − 2 m_W.

    Refs: Fadin, Khoze, Martin, Phys.Lett.B311 (1993) 311; Fadin, Khoze,
    Martin, Stirling, hep-ph/9507422 (Z.Phys.C75 (1997) 53). α = α(0).

    Vestigial under the production default ``include_coulomb=False``
    ([[project-followup-kc-dropped-2026-05-26]]); kept for diagnostic
    comparisons against pre-BFS-EFT calculations.
    """
    s_arr = np.asarray(s, dtype=float)
    sqrt_s = np.sqrt(s_arr)
    E = sqrt_s - 2.0 * mW

    kappa = np.sqrt(np.asarray(-mW * (E + 1j * gammaW), dtype=complex))
    kappa = np.where(kappa.real < 0, -kappa, kappa)
    bM = beta_complex(s_arr, mW, gammaW)
    p = 0.5 * sqrt_s * bM.real

    abs_kappa2 = np.abs(kappa) ** 2
    re_kappa = kappa.real

    denom = 2.0 * p * re_kappa
    safe = np.abs(denom) > 1e-12
    arctan_val = np.where(
        safe,
        np.arctan(np.where(safe, (abs_kappa2 - p * p) / np.where(safe, denom, 1.0), 0.0)),
        0.5 * np.pi * np.sign(abs_kappa2 - p * p),
    )

    out = 1.0 + (ALPHA_EM_0 * sqrt_s / (4.0 * p)) * (np.pi - 2.0 * arctan_val)
    if np.ndim(s) == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# BFS NLO + NNLO placeholders
# ---------------------------------------------------------------------------

@dataclass
class BFSCorrections:
    """Diagnostic-only BFS NLO Coulomb adder (arXiv:0707.0773 eq. 62).

    ``enabled_coulomb_NLO=True`` adds the full eq. (62) (one-photon log
    ~5 % + two-photon ~0.2 %) as a relative correction on top of σ_LO.
    Overlaps with K_C at leading order — use only with K_C OFF.

    Production chains do NOT use this path; the BFS NLO Coulomb is added
    additively inside `_add_nlo_loops_to_LR` of bfs_eft.py. This dataclass
    survives for the `compute_xsec_ww.py --diagnostic-bfs-coulomb-nlo`
    CLI flag, which reproduces BFS paper plots of eq. (62) in isolation.
    """
    enabled_coulomb_NLO: bool = False

    def delta_NLO(self, s, mW: float, gammaW: float):
        """Relative NLO Coulomb correction δ s.t. σ = σ_LO × (1 + δ + …).
        Vectorised in ``s``. Returns 0 if the flag is off."""
        if not self.enabled_coulomb_NLO:
            return 0.0
        from framework.process.ww.xsec_calculator.bfs_eft import (
            delta_sigma_Coulomb_NLO_specific_pb,
            sigma_LR0_specific_pb,
        )
        d_sigma_C = delta_sigma_Coulomb_NLO_specific_pb(
            s, mW, gammaW, apply_BR_correction=True, subleading_only=False)
        sigma_LR0 = sigma_LR0_specific_pb(s, mW, gammaW, apply_BR_correction=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sigma_LR0 > 0, d_sigma_C / sigma_LR0, 0.0)

    def delta_NNLO(self, s, mW: float, gammaW: float):
        return 0.0


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
                          decay_uses_full_born: bool = True,
                          m_t: float = M_T_DEFAULT,
                          M_H: float = M_H_DEFAULT,
                          MZ: float = M_Z,
                          alpha_em: float | None = None):
    """
    Partonic σ(e+e- → μν qq̄): BFS N^(3/2)LO Born + NLO loops + NNLO +
    δ_QCD + Whizard anchor (production defaults; the FKM K_C factor is OFF
    by default — BFS Coulomb is in the NLO loops). Returns σ in pb at
    partonic CM energy² = s (before ISR convolution).

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
        bfs = BFSCorrections()
    if channel not in _CHANNEL_MULTIPLICITY:
        raise ValueError(f"channel={channel!r} not in {list(_CHANNEL_MULTIPLICITY)}")
    if br_convention not in ("bfs-eft", "pdg-constant"):
        raise ValueError(f"br_convention must be 'bfs-eft' or 'pdg-constant'; "
                         f"got {br_convention!r}")

    # Region-aware σ (handles ISR convolution sampling sub-threshold s_hat):
    #   √s < 149 GeV  → 0 (avoids spurious M_Z pole in BFS ξ,χ functions)
    #   149 ≤ √s < 170 → BFS computation (C² quintic ramp across [149,150])
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
                mt=m_t, MH=M_H, MZ=MZ,
                alpha_em=alpha_em,
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
                mt=m_t, MH=M_H, MZ=MZ,
                alpha_em=alpha_em,
            )
            sigma_bfs = sigma_WW_total * BR_pdg
        # Smooth-floor weight kills the hard step at 150 GeV that was
        # seeding ISR quadrature noise (kink in σ-variation ratios).
        sigma_bfs = sigma_bfs * _bfs_floor_weight(s_arr)
        sigma = np.where(use_bfs, sigma_bfs, sigma)

    if np.any(use_cal):
        if br_convention == "bfs-eft":
            from framework.process.ww.xsec_calculator.bfs_eft import gamma_W_LO
            BR_x = (_CHANNEL_MULTIPLICITY[channel] / 27.0) * (gamma_W_LO(mW, MZ=MZ, alpha_em=alpha_em) / gammaW) ** 2
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
            m_t=m_t, M_H=M_H, MZ=MZ,
            alpha_em=alpha_em,
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
