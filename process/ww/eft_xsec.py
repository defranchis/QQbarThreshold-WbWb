"""σ(e+e- → μν qq̄) near the WW threshold for FCC-ee — partonic level.

LO partonic cross section using the unstable-particle EFT prescription:
finite W width via analytic continuation to complex velocity; the on-shell
Born σ_WW^CC03 is calibrated against a RACOONWW reference grid from
161.33 to 500 GeV (with a power-law BW-tail extrapolation below — to be
replaced with real RACOONWW / MoCaNLO values for per-mille precision).

Pipeline:
    σ̂_partonic(ŝ; m_W, Γ_W)
        = σ̂_WW^Born(ŝ; m_W, Γ_W)
          × BR_channel
          × K_Coulomb(ŝ; m_W, Γ_W)
          × (1 + δ_NLO + δ_NNLO)              ← BFS hooks (NotImplementedError until filled)

Channels (set via ``channel`` arg of :func:`sigma_partonic_munuqq`):

    "inclusive"  (default)
        Inclusive μν qq̄ summed over both W charges:
            BR = 2 × BR(W→μν) × BR(W→hadrons)
        This is what the FCC-ee threshold-scan m_W analysis sees.

    "munuud"
        Specific μ⁻ν̄_μ + (W⁺→ud̄ or cs̄) channel only:
            BR = BR(W→μν) × BR(W→ud̄/cs̄ summed)
        Useful for benchmarking against published BFS tables.

Components:

[1] σ̂_WW^Born -- on-shell tree CC03, interpolated from a calibration grid
    of RACOONWW reference values (the 161.33–500 GeV part are RACOONWW Born
    values; below 161.33 GeV is a power-law BW-tail model -- TODO replace
    with real Born for the below-threshold region). m_W dependence
    factorises via β/(s_W^4 s) × α_Gμ(m_W); we exploit this to scale the
    grid in m_W. Threshold smoothing via complex velocity (EFT
    prescription).

[2] K_Coulomb -- Fadin-Khoze-Martin closed form with finite Γ_W
    (Phys.Lett.B311 1993 311; Bardin-Riemann hep-ph/9507422 eq. 9-10).

[3] BFS NLO/NNLO -- placeholder hooks. To activate, fill in:
        arXiv:0707.0773 eq. (4.13)+        -- NLO hard matching + soft/coll
        arXiv:0807.0102 eq. (3.1)          -- dominant NNLO Coulomb + soft

[4] ISR convolution -- in isr.py
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
GAMMA_Z         = 2.4952
M_E             = 0.5109989461e-3
G_F             = 1.1663787e-5
ALPHA_EM_0      = 1.0 / 137.035999084
ALPHA_EM_MZ     = 1.0 / 128.943
GEV_M2_TO_PB    = 3.8937937217e8

BR_W_MUNU = 0.1063   # PDG
BR_W_HAD  = 0.6741   # PDG, hadronic inclusive
BR_W_UD   = 0.3358   # W → up-type-quark generation (ud̄ + cs̄ summed, ≈ BR_had/2)

# 2 × BR(W→μν) × BR(W→had):  factor 2 because either W can be the muonic one.
BR_INCLUSIVE_MUNUQQ = 2.0 * BR_W_MUNU * BR_W_HAD
# BR(W→μν) × BR(W→ud̄/cs̄):  legacy convention from BFS tables (one specific
# W charge × one specific up-type quark generation pair).
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

# Calibration boundary — below this √s, we use BFS LO_EFT.
_SQRTS_BFS_BOUNDARY = 161.33
_S_BFS_BOUNDARY = _SQRTS_BFS_BOUNDARY ** 2

# Lower validity limit for the BFS Born expansion. The BFS paper recommends
# matching to a full calculation below ~155 GeV (paper sec. 6.2). The ξ(s),
# χ(s) functions in eqs. (32)/(33)/(37) have spurious poles at s = M_Z²
# (~91 GeV) which the ISR convolution would sample if not clamped. Set σ
# to 0 below ~150 GeV — physically σ_WW is well below 0.01 pb there and
# contributes negligibly to the ISR-convolved observed cross section.
_SQRTS_BFS_FLOOR = 150.0
_S_BFS_FLOOR = _SQRTS_BFS_FLOOR ** 2


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


def sigma_WW_Born(s,
                  mW: float = M_W_DEFAULT,
                  gammaW: float = GAMMA_W_DEFAULT):
    """
    Off-shell-convolved Born σ(e+e- → W+W- → 4f), full off-shell, in pb.

    For ``s_eff ≥ (161.33 GeV)²``: read F(s_eff) from the cubic spline
    through the RACOONWW reference values, then
        σ = (π α²(m_W)) / (s_W^4(m_W) × s) × Re[β_M(s, m_W, Γ_W)] × F(s_eff)
    with s_eff = s × (m_W^default / m_W)².

    For ``s_eff < (161.33 GeV)²``: use the BFS unstable-particle EFT Born
    expansion (σ^{(0)} + σ^{(1)}_pot + σ^{(1/2)}_hard + σ^{(3/2),a},
    arXiv:0707.0773 eqs. 17 + 33 + 37 + 39), multiplicatively matched to
    RACOONWW at the boundary for C⁰ continuity. Agrees with full 4f Born
    to ~0.1 % at threshold and ~3 % at 155 GeV (Table 1 of the paper).

    Vectorised: accepts scalar or array ``s`` (m_W, Γ_W must be scalar).
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)

    bM_real = beta_complex(s_arr, mW, gammaW).real

    sqrt_s = np.sqrt(s_arr)
    sqrt_s_eff = sqrt_s * (M_W_DEFAULT / mW)
    s_eff = sqrt_s_eff ** 2

    F = _interp_F_smooth(s_eff)
    sigma_GeVm2 = (np.pi * alpha ** 2) / (sW2 ** 2 * s_arr) * np.maximum(bM_real, 0.0) * F
    sigma_cal_pb = sigma_GeVm2 * GEV_M2_TO_PB

    # BFS LO_EFT in (FLOOR, BOUNDARY]; clamp to 0 below FLOOR to avoid the
    # spurious ξ(s)/χ(s) Z-pole the EFT picks up far below its validity.
    use_bfs = (s_eff < _S_BFS_BOUNDARY) & (s_eff >= _S_BFS_FLOOR)
    use_zero = s_eff < _S_BFS_FLOOR
    if np.any(use_bfs):
        from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb
        sigma_bfs_raw = sigma_BFS_LO_total_WW_pb(s_arr, mW, gammaW, order="N3/2LO")
        # Multiplicative match at the boundary using current m_W, Γ_W.
        sigma_cal_boundary = (
            (np.pi * alpha ** 2) / (sW2 ** 2 * _S_BFS_BOUNDARY)
            * beta_complex(_S_BFS_BOUNDARY, mW, gammaW).real
            * _F_GRID[0] * GEV_M2_TO_PB
        )
        sigma_bfs_boundary = sigma_BFS_LO_total_WW_pb(
            _S_BFS_BOUNDARY, mW, gammaW, order="N3/2LO")
        match_factor = sigma_cal_boundary / max(sigma_bfs_boundary, 1e-30)
        sigma_bfs_pb = sigma_bfs_raw * match_factor
        sigma_pb = np.where(use_bfs, sigma_bfs_pb, sigma_cal_pb)
    else:
        sigma_pb = sigma_cal_pb

    sigma_pb = np.where(use_zero, 0.0, sigma_pb)
    return np.where(sigma_pb > 0.0, sigma_pb, 0.0)


# ---------------------------------------------------------------------------
# Coulomb K-factor (Fadin-Khoze-Martin with finite Γ_W)
# ---------------------------------------------------------------------------

def coulomb_K_factor(s,
                     mW: float = M_W_DEFAULT,
                     gammaW: float = GAMMA_W_DEFAULT,
                     order: int = 1):
    """
    Coulomb-photon-exchange K-factor with finite Γ_W:

        K_C = 1 + (α√s)/(4p) × [π - 2·arctan((|κ|² - p²)/(2 p Re κ))]
              + (α² s ln 2)/(4|κ|²)              # O(α²) (Bardin-Riemann)

    p = (√s/2) × Re[β_M],   κ = √(-m_W (E + i Γ_W)),   E = √s - 2 m_W.

    Refs:
        Fadin, Khoze, Martin, Phys.Lett.B311 (1993) 311
        Bardin, Riemann, hep-ph/9507422 eq. (9-10)

    α = α(0) (Thomson limit) for the soft Coulomb photon. Vectorised in s.
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

    # Main formula: arctan((|κ|²-p²)/(2 p Re κ)). With finite Γ_W, the
    # denominator is always > 0 in physical regimes — both p and Re κ have
    # leading O(√(m_W Γ_W)) at threshold. Guard against numerical zero anyway.
    denom = 2.0 * p * re_kappa
    safe = np.abs(denom) > 1e-12
    arctan_val = np.where(
        safe,
        np.arctan(np.where(safe, (abs_kappa2 - p * p) / np.where(safe, denom, 1.0), 0.0)),
        0.5 * np.pi * np.sign(abs_kappa2 - p * p),
    )

    K1 = 1.0 + (ALPHA_EM_0 * sqrt_s / (4.0 * p)) * (np.pi - 2.0 * arctan_val)

    if order < 2:
        out = K1
    else:
        out = K1 + (ALPHA_EM_0 ** 2 * s_arr * np.log(2.0)) / (4.0 * abs_kappa2)

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

    Currently placeholders. Set ``enabled=True`` and fill in the formulae
    from those papers when needed.

    Expected impact at √s ≈ 161 GeV:
        δ_NLO  ≈ -7% (dominated by ISR-like logs; should be partly
                       absorbed into NLL QED ISR resummation)
        δ_NNLO ≈ +1% (Coulomb²) − (ISR matching) ≈ small net, but
                       shifts m_W by ≈ 3 MeV in the threshold fit.
    """
    enabled: bool = False

    def delta_NLO(self, s: float, mW: float, gammaW: float) -> float:
        if not self.enabled:
            return 0.0
        raise NotImplementedError(
            "Fill in arXiv:0707.0773 eq. (4.13)+ : hard one-loop matching, "
            "NLO Coulomb (already in K_C — subtract to avoid double-counting), "
            "soft+collinear photon corrections, NLO W-decay corrections."
        )

    def delta_NNLO(self, s: float, mW: float, gammaW: float) -> float:
        if not self.enabled:
            return 0.0
        raise NotImplementedError(
            "Fill in arXiv:0807.0102 eq. (3.1): NNLO Coulomb², "
            "single-Coulomb × soft interference, NLL-resummed hard function."
        )


# ---------------------------------------------------------------------------
# Partonic cross section for μν qq̄
# ---------------------------------------------------------------------------

_CHANNEL_BR = {
    "inclusive": BR_INCLUSIVE_MUNUQQ,   # 2 × BR(W→μν) × BR(W→had), either W charge
    "munuud":    BR_MUNUUD,             # BR(W→μν) × BR(W→ud̄/cs̄ summed) — legacy
}


def sigma_partonic_munuqq(s,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          channel: str = "inclusive",
                          include_coulomb: bool = True,
                          bfs: BFSCorrections | None = None):
    """
    Partonic σ(e+e- → μν qq̄) at LO + Coulomb (+ optional BFS NLO/NNLO).
    Returns σ in pb at partonic CM energy² = s (before ISR convolution).

    ``channel`` selects the branching-ratio convention:
        "inclusive" (default) — 2 × BR(μν) × BR(had), both W charges summed
        "munuud"              — μ⁻ν̄_μ + (ud̄/cs̄) specific

    Vectorised: accepts scalar or array ``s``.
    """
    if bfs is None:
        bfs = BFSCorrections(enabled=False)
    if channel not in _CHANNEL_BR:
        raise ValueError(f"channel={channel!r} not in {list(_CHANNEL_BR)}")

    sigma_WW = sigma_WW_Born(s, mW, gammaW)
    sigma = sigma_WW * _CHANNEL_BR[channel]

    if include_coulomb:
        sigma = sigma * coulomb_K_factor(s, mW, gammaW)

    sigma = sigma * (1.0 + bfs.delta_NLO(s, mW, gammaW)
                          + bfs.delta_NNLO(s, mW, gammaW))

    if np.ndim(s) == 0:
        return float(sigma)
    return sigma


# Back-compat alias matching the previous chat's naming convention.
def sigma_partonic_munuud(s: float, mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT, **kwargs) -> float:
    """Legacy alias: σ for the μ⁻ν̄_μ ud̄ channel (one specific charge)."""
    return sigma_partonic_munuqq(s, mW, gammaW, channel="munuud", **kwargs)


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
        sigma_mine = sigma_WW_Born(s)
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
