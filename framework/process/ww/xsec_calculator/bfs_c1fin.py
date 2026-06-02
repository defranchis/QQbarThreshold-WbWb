"""Analytic implementation of the BFS hard one-loop matching coefficients

c_p,LR^(1,fin), c_d,l^(1,fin), c_d,h^(1,fin) from the appendix of
arXiv:0707.0773 (Beneke-Falgari-Schwinn).

These were originally hard-coded as constants -10.076, -2.709, -2.034
(reference: m_W=80.377, M_Z=91.188, m_t=174.2, M_H=115 GeV). This module
exposes them as analytic functions of (m_W, m_t, M_H, M_Z) so the fit can
propagate their m-dependence.

The implementation is built from the BFS appendix's explicit ingredients:
  * L(k², m1², m2²) — finite combination of B_0(k², m1², m2²), eq. (BtoLL).
    Explicit formulae are given by BFS for L at (0, M_W², M_Z²),
    (M_Z², M_W², M_W²), (M_W², M_Z², M_W²). The L special case with
    arbitrary heavy mass M (=> M_H) reuses the (M_W², M_Z², M_W²) form.
    L(4M_W², M_Z², M_Z²) — equal-mass case — is computed from B_0.
  * ∂B_0(M_W², M_W², m²) for m = M_Z, M_H (BFS explicit, m → M_H by
    analytic substitution).
  * Six C_0 functions for the specific kinematic invariants needed by
    the bare matching coefficient (BFS lines 3508-3573), plus a one-mass
    triangle C_0(4M_W², 0, 0, 0, 0, m²) derived in closed form below
    (the 'two collinear externals + one massive internal' configuration).

The 'fin' subscript denotes the finite part after the BFS pole structure
is subtracted: for c_p,LR^(1) the external poles are
(-1/ε² - 3/(2ε))(-4M_W²/μ²)^(-ε); for c_d,l/h^(1) they are
(-1/(2ε²) - 5/(4ε))(M_W²/μ²)^(-ε) + Q_f Q̄_f (-1/ε² - 3/(2ε))(-M_W²/μ²)^(-ε).
The production-decay counterterm pole structure differs ((-4M_W²/μ²)^(-ε)
vs (M_W²/μ²)^(-ε)) and leaves a μ-independent constant
(A_ct/2)·(-ln 4 + iπ) in the decay finite parts.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spence as _spence


# ---------------------------------------------------------------------------
# Dilogarithm wrapper — scipy.special.spence(z) = Li_2(1 - z), complex-safe.
# ---------------------------------------------------------------------------

def Li2(z):
    """Standard dilogarithm Li_2(z), complex-safe via scipy.special.spence."""
    return _spence(1.0 - z)


# ---------------------------------------------------------------------------
# Auxiliary masses
# ---------------------------------------------------------------------------

def _M_WpmZ(M_W2: float, M_Z2: float):
    """M_{W±Z} ≡ M_W ± sqrt(M_W² - M_Z²); complex when M_W < M_Z."""
    M_W = np.sqrt(M_W2)
    root = np.sqrt(complex(M_W2 - M_Z2, 0.0))
    return M_W + root, M_W - root


# ---------------------------------------------------------------------------
# L(k², m1², m2²) — finite part of -B_0 + 2 + (1/ε)(m1²/μ²)^(-ε), μ-independent.
# ---------------------------------------------------------------------------

def L_0_m1_m2(m1_2: float, m2_2: float) -> complex:
    """L(0, m1², m2²) = 1 + m2²/(m1²-m2²) × ln(m1²/m2²).

    BFS provide this for (M_W², M_Z²). Identical structure in general.
    Symmetric under m1 ↔ m2 up to ln(m2²/m1²) per BFS line 3476.
    """
    if abs(m1_2 - m2_2) < 1e-12 * max(m1_2, m2_2):
        return 2.0 + 0.0j  # limit: L(0, m², m²) = 2
    return 1.0 + m2_2 / (m1_2 - m2_2) * np.log(m1_2 / m2_2) + 0.0j


def L_MW2_m_MW(M_W2: float, m_2: float) -> complex:
    """L(M_W², m², M_W²) — generalises BFS L(M_W², M_Z², M_W²) by m → m_Z, m_H.

        L(M_W², m², M_W²) =
            (2M_W² − m² + M_mW²)/(2 M_W²) × ln((m² − M_mW²)/(2 m²))
          + (2M_W² − m² − M_mW²)/(2 M_W²) × ln((m² + M_mW²)/(2 m²))

    with M_mW² = sqrt(m⁴ − 4 m² M_W²) (imaginary when m² < 4 M_W²).
    """
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    t1 = (2.0 * M_W2 - m_2 + M_mW2) / (2.0 * M_W2)
    t2 = (2.0 * M_W2 - m_2 - M_mW2) / (2.0 * M_W2)
    log1 = np.log((m_2 - M_mW2) / (2.0 * m_2))
    log2 = np.log((m_2 + M_mW2) / (2.0 * m_2))
    return t1 * log1 + t2 * log2


def L_pp2_m_m(p2: float, m_2: float) -> complex:
    """L(p², m², m²) for equal internal masses, general p² > 0.

    Below threshold (p² < 4 m²):
        β' = sqrt(4 m²/p² − 1) ∈ ℝ_+,
        L = 2 β' × arctan(1/β').
    Above threshold (p² > 4 m²), with +iε:
        β = sqrt(1 − 4 m²/p²),
        L = β × [ln((1+β)/(1−β)) − iπ].
    """
    r = 4.0 * m_2 / p2
    if r > 1.0:
        bp = np.sqrt(r - 1.0)
        return 2.0 * bp * np.arctan(1.0 / bp) + 0.0j
    b = np.sqrt(1.0 - r)
    return complex(b * np.log((1.0 + b) / (1.0 - b)), -np.pi * b)


# ---------------------------------------------------------------------------
# ∂B_0(M_W², M_W², m²) — BFS line 3499-3506, generalised to m = M_Z, M_H.
# ---------------------------------------------------------------------------

def dB0_MW2_MW_m(M_W2: float, m_2: float) -> complex:
    """∂B_0(M_W², M_W², m²) at fixed external momentum p² = M_W².

        ∂B_0 = −1/M_W² × { 1
                + (M_W² − m²)/(2 M_W²) × ln(m²/M_W²)
                + m² (3 M_W² − m²)/(M_W² M_mW²) × ln((m² − M_mW²)/(2 M_W √m²)) }

    Generalises BFS's (M_W², M_W², M_Z²) form by m → m_Z or m_H.
    """
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    M_W = np.sqrt(M_W2)
    m_sqrt = np.sqrt(complex(m_2, 0.0))
    a = 1.0
    b = (M_W2 - m_2) / (2.0 * M_W2) * np.log(m_2 / M_W2)
    c = m_2 * (3.0 * M_W2 - m_2) / (M_W2 * M_mW2) * np.log(
        (m_2 - M_mW2) / (2.0 * M_W * m_sqrt)
    )
    return -1.0 / M_W2 * (a + b + c)


# ---------------------------------------------------------------------------
# C_0 functions — BFS appendix (eq. c0expl1, line 3508-3573) + derivations
# ---------------------------------------------------------------------------

def C0_zero_MW2_mMW2_0_m_MW(M_W2: float, m_2: float) -> complex:
    """C_0(0, M_W², −M_W², 0, m², M_W²) — BFS line 3513-3520, parameterised by m.

    Original form has m² = M_Z²; this version accepts any m² ≥ 0.
    """
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    M_mW4 = M_mW2 * M_mW2
    inv = 1.0 / (4.0 * M_W2)
    if m_2 == 0.0:
        return 0.0 + 0.0j  # m^2->0 limit (defensive; not hit on current call paths)
    arg1 = 1.0 - 2.0 * M_W2 / m_2
    arg2 = (2.0 * M_W2 - m_2) / (4.0 * M_W2 - m_2)
    arg3 = (m_2 * m_2) / M_mW4   # m⁴/M_mW⁴
    arg4 = (2.0 * M_W2 - m_2) / M_mW2
    arg5 = (m_2 - 2.0 * M_W2) / M_mW2
    return inv * (
        2.0 * Li2(arg1)
        + 2.0 * Li2(arg2)
        - Li2(arg3)
        - 2.0 * Li2(arg4)
        - 2.0 * Li2(arg5)
        - np.pi ** 2 / 3.0
    )


def C0_zero_4MW2_0_0_m_m(M_W2: float, m_2: float) -> complex:
    """C_0(0, 4 M_W², 0, 0, m², m²) — BFS line 3523-3533, m = M_Z."""
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    M_mW4 = M_mW2 * M_mW2
    M_pZ, M_mZ = _M_WpmZ(M_W2, m_2)  # M_{W±} for the m mass (use m where Z is)
    M_WpZ2 = M_pZ * M_pZ
    M_WmZ2 = M_mZ * M_mZ
    inv = -1.0 / (8.0 * M_W2)
    return inv * (
        np.log(-M_mW4 / (m_2 * m_2)) ** 2
        + np.log(M_WpZ2 / m_2) ** 2
        + 2.0 * Li2((m_2 * m_2) / M_mW4)
        + 2.0 * Li2((4.0 * M_W2 - m_2) / M_WmZ2)
        + 2.0 * Li2((4.0 * M_W2 - m_2) / M_WpZ2)
        + np.pi ** 2
    )


def C0_mMW2_MW2_0_0_m_MW(M_W2: float, m_2: float) -> complex:
    """C_0(−M_W², M_W², 0, 0, m², M_W²) — BFS line 3536-3544 with m as parameter.

    The bare formula uses this with m² = 0, M_H², and M_Z². For m² → 0 the
    expression has a 0/0 limit; this implementation takes the explicit limit.
    """
    if m_2 == 0.0:
        # m²→0 limit: Li_2 reflection → result is −π²/(4 M_W²).
        return -np.pi ** 2 / (4.0 * M_W2) + 0.0j
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    inv = -1.0 / (2.0 * M_W2)
    a1 = -M_W2 / (M_W2 + 2.0 * m_2)
    b1 = M_W2 / (M_W2 + 2.0 * m_2)
    a2 = M_W2 / (M_W2 - m_2 - M_mW2)
    b2 = -M_W2 / (M_W2 - m_2 - M_mW2)
    a3 = M_W2 / (M_W2 - m_2 + M_mW2)
    b3 = -M_W2 / (M_W2 - m_2 + M_mW2)
    return inv * (
        Li2(a1) - Li2(b1)
        + Li2(a2) - Li2(b2)
        + Li2(a3) - Li2(b3)
        + np.pi ** 2 / 4.0
    )


def C0_MW2_0_0_MW_m_0(M_W2: float, m_2: float) -> complex:
    """C_0(M_W², 0, 0, M_W², m², 0) — BFS line 3547-3551, m = M_Z."""
    M_mW2 = np.sqrt(complex(m_2 * m_2 - 4.0 * m_2 * M_W2, 0.0))
    inv = 1.0 / M_W2
    a = 2.0 * M_W2 / (m_2 + M_mW2)
    b = (m_2 + M_mW2) / (2.0 * m_2)
    return inv * (Li2(a) + Li2(b) - np.pi ** 2 / 6.0)


def C0_MW2_mMW2_0_0_0_m(M_W2: float, m_2: float) -> complex:
    """C_0(M_W², −M_W², 0, 0, 0, m²) — BFS line 3554-3565, m = M_Z."""
    inv = 1.0 / (4.0 * M_W2)
    L = np.log(2.0 * M_W2 / m_2 + 1.0)
    arg1 = (m_2 * m_2) / ((2.0 * M_W2 + m_2) ** 2)
    arg2 = 1.0 - 2.0 * M_W2 / m_2
    arg3 = (2.0 * M_W2 - m_2) / (2.0 * M_W2 + m_2)
    arg4 = (m_2 - 2.0 * M_W2) / (2.0 * M_W2 + m_2)
    arg5 = m_2 / (2.0 * M_W2 + m_2)
    return inv * (
        L * (L - 2.0j * np.pi)
        - Li2(arg1)
        + 2.0 * Li2(arg2)
        + 2.0 * Li2(arg3)
        - 2.0 * Li2(arg4)
        + 6.0 * Li2(arg5)
        - 2.0 * np.pi ** 2 / 3.0
    )


def C0_MW2_0_0_0_0_m(M_W2: float, m_2: float) -> complex:
    """C_0(M_W², 0, 0, 0, 0, m²) — BFS line 3568-3573, m = M_Z."""
    inv = 1.0 / M_W2
    log_arg = (M_W2 + m_2) / m_2
    L = np.log(log_arg)
    return inv * (
        0.5 * L * L
        - 1.0j * np.pi * L
        + Li2(m_2 / (M_W2 + m_2))
        - np.pi ** 2 / 6.0
    )


def C0_4MW2_0_0_0_0_m(M_W2: float, m_2: float) -> complex:
    """C_0(4 M_W², 0, 0, 0, 0, m²) — 1-mass triangle, derived in this module.

    Topology: two adjacent massless internal lines, one massive (m²); two
    light-like externals and one with virtuality 4 M_W². Standard Feynman-
    parameter evaluation gives, for s ≡ 4 M_W² > 0 and m² > 0:

        C_0 = (1/s) × [ ln(s/m²) · ln(1 + s/m²) + Li_2(−s/m²) − iπ · ln(1 + s/m²) ]

    Equivalent to BFS's (1/M_W²) × {(1/2) ln²(1+s/m²) − iπ ln(1+s/m²)
    + Li_2(m²/(s+m²)) − π²/6} form (eq. line 3568-3573, with s in place of
    M_W²) by Li_2 reflection — checked numerically.
    """
    s = 4.0 * M_W2
    return (1.0 / s) * (
        np.log(s / m_2) * np.log(1.0 + s / m_2)
        + Li2(-s / m_2)
        - 1.0j * np.pi * np.log(1.0 + s / m_2)
    )


# ---------------------------------------------------------------------------
# Bare and counterterm finite lines
# ---------------------------------------------------------------------------

def _c_p_LR_bare_finite(M_W2: float, M_Z2: float, M_H2: float) -> complex:
    """Sum of the 'explicit finite' lines of c_p,LR^(1,bare) — BFS line 3196-3244.

    Excludes the 1/ε² and 1/ε poles (which cancel against the external pole
    structure -1/ε² -3/(2ε) times (-4M_W²/μ²)^(-ε)). The finite remainder
    is μ-independent.
    """
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    sw4 = sw2 * sw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2
    cw8 = cw4 * cw4
    cw10 = cw8 * cw2

    # Three-point functions used by the bare
    C0_a = C0_zero_MW2_mMW2_0_m_MW(M_W2, M_Z2)
    C0_b = C0_zero_4MW2_0_0_m_m(M_W2, M_Z2)
    C0_c1 = C0_mMW2_MW2_0_0_m_MW(M_W2, 0.0)       # m² = 0
    C0_c2 = C0_mMW2_MW2_0_0_m_MW(M_W2, M_H2)      # m² = M_H²
    C0_c3 = C0_mMW2_MW2_0_0_m_MW(M_W2, M_Z2)      # m² = M_Z²
    C0_d1 = C0_MW2_mMW2_0_0_0_m(M_W2, M_W2)       # m² = M_W² (substitute)
    C0_d2 = C0_MW2_mMW2_0_0_0_m(M_W2, M_Z2)       # m² = M_Z²
    C0_e1 = C0_4MW2_0_0_0_0_m(M_W2, M_Z2)         # m² = M_Z²
    C0_e2 = C0_4MW2_0_0_0_0_m(M_W2, M_W2)         # m² = M_W²

    # L(p², m1², m2²) terms
    # bare uses L(M_W², M_W², M_H²), L(M_W², M_W², M_Z²), L(4M_W², M_Z², M_Z²)
    # Apply symmetry L(k², m1², m2²) = L(k², m2², m1²) + ln(m2²/m1²) once:
    L_MW_MW_MH = L_MW2_m_MW(M_W2, M_H2) + np.log(M_H2 / M_W2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)
    L_4MW_MZ_MZ = L_pp2_m_m(4.0 * M_W2, M_Z2)

    fin = (
        (2.0 * cw2 - 1.0) * (24.0 * cw4 + 16.0 * cw2 - 1.0) * M_W2 * C0_a
        / (8.0 * cw4 * sw4)
        - (2.0 * cw2 - 1.0) * M_W2 * C0_b / (2.0 * cw4 * sw2)
        - ((cw4 + 17.0 * cw2 - 16.0) * M_H2 + M_W2) * M_W2 * C0_c1
        / (4.0 * M_H2 * sw2)
        + (M_H2 + M_W2) * M_W2 * C0_c2 / (4.0 * M_H2 * sw2)
        - (2.0 * cw8 + 32.0 * cw6 + 32.0 * cw4 - 11.0 * cw2 - 16.0) * M_W2 * C0_c3
        / (8.0 * cw2 * sw4)
        + 3.0 * (33.0 - 46.0 * cw2) * M_W2 * C0_d1 / (8.0 * sw4)
        + (4.0 * cw4 - 1.0) * (14.0 * cw6 + 15.0 * cw4 - 2.0 * cw2 - 1.0) * M_W2 * C0_d2
        / (16.0 * cw8 * sw4)
        - ((1.0 - 2.0 * cw2) ** 2) * (cw2 + 1.0) * ((4.0 * cw2 + 1.0) ** 2) * M_W2 * C0_e1
        / (16.0 * cw8 * sw2)
        - 25.0 * M_W2 * C0_e2 / (4.0 * sw2)
        + M_W2 * L_MW_MW_MH / (4.0 * M_H2 * sw2)
        + (-168.0 * cw8 - 214.0 * cw6 + 56.0 * cw4 + 32.0 * cw2 - 3.0) * L_MW_MW_MZ
        / (24.0 * cw2 * (1.0 - 4.0 * cw2) * sw2)
        + (1.0 - 2.0 * cw2) * (8.0 * cw4 + cw2 + 3.0) * L_4MW_MZ_MZ
        / (6.0 * cw2 * sw2)
        + 3.0 * (cw2 + 1.0) * np.log(M_W2 / M_Z2 + 1.0) / (16.0 * cw6)
        + (1.0 - 2.0 * cw2) * (64.0 * cw4 + 4.0 * cw2 + 1.0) * np.log(4.0 * M_W2 / M_Z2 - 1.0)
        / (24.0 * cw4)
        + (-512.0 * cw10 + 1536.0 * cw8 - 672.0 * cw6 + 44.0 * cw4 + 3.0 * cw2 - 3.0)
        * np.log(M_Z2 / M_W2) / (48.0 * cw4 * (1.0 - 4.0 * cw2) * sw2)
        + (-128.0 * cw10 + 304.0 * cw8 + 144.0 * cw6 - 38.0 * cw4 + 9.0 * cw2 + 3.0)
        * np.log(2.0) / (24.0 * cw6 * sw2)
        + (96.0 * cw6 - (10.0 - 2.0 * sw2 * np.pi ** 2) * cw4 - 9.0 * cw2 - 6.0)
        / (24.0 * cw4 * sw2)
        - (128.0 * cw8 - 64.0 * cw6 + 4.0 * cw4 + 23.0 * cw2 + 5.0) * 1.0j * np.pi
        / (48.0 * cw4 * sw2)
    )
    return fin


def _c_p_LR_ct_finite(M_W2: float, M_Z2: float, M_H2: float, mt2: float) -> complex:
    """Sum of the 'explicit finite' lines of c_p,LR^(1,ct) — BFS line 3256-3286.

    Contains the only direct m_t dependence in c_p,LR. M_H and M_Z dependences
    are also explicit here through the Higgs- and Z-channel logs/L's/∂B_0's.
    """
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    sw4 = sw2 * sw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2
    cw8 = cw4 * cw4

    L_MW_MH_MW = L_MW2_m_MW(M_W2, M_H2)
    L_0_MH_MW = L_0_m1_m2(M_H2, M_W2)
    L_0_MW_MZ = L_0_m1_m2(M_W2, M_Z2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)
    dB0_MW_MW_MH = dB0_MW2_MW_m(M_W2, M_H2)
    dB0_MW_MW_MZ = dB0_MW2_MW_m(M_W2, M_Z2)

    fin = (
        - (M_H2 * M_H2 - 3.0 * M_W2 * M_H2 + 6.0 * M_W2 * M_W2) * L_MW_MH_MW
        / (12.0 * M_W2 * M_W2 * sw2)
        - (M_H2 - 5.0 * M_W2) * L_0_MH_MW / (12.0 * M_W2 * sw2)
        - (8.0 * cw4 + 27.0 * cw2 - 5.0) * L_0_MW_MZ / (12.0 * cw2 * sw2)
        + (42.0 * cw4 - 11.0 * cw2 - 1.0) * L_MW_MW_MZ / (12.0 * cw4 * sw2)
        - (M_H2 * M_H2 - 4.0 * M_W2 * M_H2 + 12.0 * M_W2 * M_W2) * dB0_MW_MW_MH
        / (24.0 * M_W2 * sw2)
        + (48.0 * cw6 + 68.0 * cw4 - 16.0 * cw2 - 1.0) * M_W2 * dB0_MW_MW_MZ
        / (24.0 * cw4 * sw2)
        + (2.0 * M_H2 * M_H2 - 3.0 * M_H2 * M_W2 + 2.0 * M_W2 * M_W2)
        * np.log(M_H2 / M_W2) / (24.0 * M_W2 * (M_H2 - M_W2) * sw2)
        + M_H2 * M_H2 / (12.0 * M_W2 * M_W2 * sw2)
        - 3.0 * M_H2 / (16.0 * M_W2 * sw2)
        - 3.0 * mt2 * (mt2 * mt2 - M_W2 * M_W2) * np.log(1.0 - M_W2 / mt2)
        / (4.0 * M_W2 ** 3 * sw2)
        - 3.0 * mt2 / (8.0 * M_W2 * sw2)
        - 3.0 * mt2 * mt2 / (4.0 * M_W2 * M_W2 * sw2)
        - (12.0 * cw8 - 72.0 * cw6 + 26.0 * cw4 - 15.0 * cw2 - 2.0) * np.log(M_Z2 / M_W2)
        / (24.0 * cw4 * sw4)
        + (4.0 * cw4 - 22.0 * cw2 - 1.0) * np.log(2.0) / (4.0 * cw2 * sw2)
        + (2.0 * (35.0 - 6.0j * np.pi) * cw6
           + (-112.0 + 66.0j * np.pi) * cw4
           + (13.0 + 3.0j * np.pi) * cw2 + 2.0)
        / (24.0 * cw4 * sw2)
    )
    return fin


def _c_d_l_bare_finite(M_W2: float, M_Z2: float) -> complex:
    """c_d,l^(1,bare) finite lines — BFS line 3372-3387."""
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2

    C0_a = C0_MW2_0_0_0_0_m(M_W2, M_Z2)
    C0_b = C0_MW2_0_0_MW_m_0(M_W2, M_Z2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)

    fin = (
        (cw2 + 1.0) ** 2 * (2.0 * cw2 - 1.0) * M_W2 * C0_a / (4.0 * cw6 * sw2)
        + (cw2 + 2.0) * M_W2 * C0_b / sw2
        + (2.0 * cw2 + 1.0) * L_MW_MW_MZ / (2.0 * sw2)
        - (4.0 * cw6 - 2.0 * cw4 + 1.0) * np.log(M_Z2 / M_W2)
        / (4.0 * cw4 * sw2)
        - (-(24.0 + np.pi ** 2) * cw6
          + (np.pi ** 2 - 18.0j * np.pi) * cw4
          - 3.0j * np.pi * cw2
          + 6.0j * np.pi + 6.0)
        / (24.0 * cw4 * sw2)
    )
    return fin


def _c_d_l_ct_extra_finite(M_W2: float, M_Z2: float) -> complex:
    """Extra (non-c_p,LR^(1,ct)/2) finite lines of c_d,l^(1,ct) — BFS LVDCT."""
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    fin = (
        np.log(M_Z2 / M_W2) / (16.0 * cw2 * sw2)
        + (2.0 * cw2 + 1.0) / (32.0 * cw2 * sw2)
    )
    return fin + 0.0j


def _c_d_h_bare_finite(M_W2: float, M_Z2: float) -> complex:
    """c_d,h^(1,bare) finite lines — BFS line 3403-3423."""
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    cw4 = cw2 * cw2
    cw6 = cw4 * cw2
    cw8 = cw4 * cw4

    C0_a = C0_MW2_0_0_0_0_m(M_W2, M_Z2)
    C0_b = C0_MW2_0_0_MW_m_0(M_W2, M_Z2)
    L_MW_MW_MZ = L_MW2_m_MW(M_W2, M_Z2) + np.log(M_Z2 / M_W2)

    fin = (
        (8.0 * cw8 + 18.0 * cw6 + 11.0 * cw4 - 1.0) * M_W2 * C0_a / (36.0 * cw6 * sw2)
        + (cw2 + 2.0) * M_W2 * C0_b / sw2
        + (2.0 * cw2 + 1.0) * L_MW_MW_MZ / (2.0 * sw2)
        - (20.0 * cw6 + 6.0 * cw4 + 1.0) * np.log(M_Z2 / M_W2)
        / (36.0 * cw4 * sw2)
        + (120.0 * cw6 + (48.0 - 13.0 * sw2 * np.pi ** 2) * cw4 - 6.0)
        / (216.0 * cw4 * sw2)
        + (24.0 * cw6 + 22.0 * cw4 + cw2 - 2.0) * 1.0j * np.pi / (72.0 * cw4 * sw2)
    )
    return fin


def _c_d_h_ct_extra_finite(M_W2: float, M_Z2: float) -> complex:
    """Extra (non-c_p,LR^(1,ct)/2) finite lines of c_d,h^(1,ct) — BFS DHCT."""
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    cw4 = cw2 * cw2
    fin = (
        - (16.0 * cw4 - 32.0 * cw2 + 7.0) * np.log(M_Z2 / M_W2)
        / (144.0 * cw2 * sw2)
        - (16.0 * cw4 - 50.0 * cw2 + 7.0) / (288.0 * cw2 * sw2)
    )
    return fin + 0.0j


# ---------------------------------------------------------------------------
# Production-decay counterterm cross-talk residual: (A_ct/2)·(−ln 4 + iπ)
# ---------------------------------------------------------------------------

def _A_ct_over_2(M_W2: float, M_Z2: float) -> float:
    """A_ct/2 = (4 c_w^4 − 22 c_w² − 1) / (16 c_w² s_w²).

    A_ct is the coefficient of the (1/ε)(-4M_W²/μ²)^(-ε) term in c_p,LR^(1,ct);
    the c_d,l/h^(1) external pole structure carries (M_W²/μ²)^(-ε), and the
    mismatch leaves a μ-independent constant in the finite part.
    """
    cw2 = M_W2 / M_Z2
    sw2 = 1.0 - cw2
    cw4 = cw2 * cw2
    return (4.0 * cw4 - 22.0 * cw2 - 1.0) / (16.0 * cw2 * sw2)


def _residual_decay_pole_mismatch(M_W2: float, M_Z2: float) -> complex:
    """The (A_ct/2)·(−ln 4 + iπ) μ-independent residual to add to c_d,l/h^(1,fin)."""
    return _A_ct_over_2(M_W2, M_Z2) * (-np.log(4.0) + 1.0j * np.pi)


# ---------------------------------------------------------------------------
# Public API: c^(1,fin) as analytic functions of (m_W, m_t, M_H, M_Z)
# ---------------------------------------------------------------------------

def c_p_LR_1_fin(m_W: float, m_t: float, M_H: float, M_Z: float) -> complex:
    """c_p,LR^(1,fin)(m_W, m_t, M_H, M_Z), BFS arXiv:0707.0773 appendix.

    Reproduces −10.076 + 0.205 i at (m_W, m_t, M_H, M_Z) = (80.377, 174.2, 115, 91.188).
    """
    M_W2 = m_W * m_W
    M_Z2 = M_Z * M_Z
    M_H2 = M_H * M_H
    mt2 = m_t * m_t
    return _c_p_LR_bare_finite(M_W2, M_Z2, M_H2) + _c_p_LR_ct_finite(M_W2, M_Z2, M_H2, mt2)


def c_d_l_1_fin(m_W: float, m_t: float, M_H: float, M_Z: float) -> complex:
    """c_d,l^(1,fin) — leptonic decay matching coefficient.

    Reproduces −2.709 − 0.552 i at the reference inputs.
    """
    M_W2 = m_W * m_W
    M_Z2 = M_Z * M_Z
    M_H2 = M_H * M_H
    mt2 = m_t * m_t
    return (
        _c_d_l_bare_finite(M_W2, M_Z2)
        + 0.5 * _c_p_LR_ct_finite(M_W2, M_Z2, M_H2, mt2)
        + _c_d_l_ct_extra_finite(M_W2, M_Z2)
        + _residual_decay_pole_mismatch(M_W2, M_Z2)
    )


def c_d_h_1_fin(m_W: float, m_t: float, M_H: float, M_Z: float) -> complex:
    """c_d,h^(1,fin) — hadronic decay matching coefficient.

    Reproduces −2.034 − 0.597 i at the reference inputs.
    """
    M_W2 = m_W * m_W
    M_Z2 = M_Z * M_Z
    M_H2 = M_H * M_H
    mt2 = m_t * m_t
    return (
        _c_d_h_bare_finite(M_W2, M_Z2)
        + 0.5 * _c_p_LR_ct_finite(M_W2, M_Z2, M_H2, mt2)
        + _c_d_h_ct_extra_finite(M_W2, M_Z2)
        + _residual_decay_pole_mismatch(M_W2, M_Z2)
    )
