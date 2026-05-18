"""BFS unstable-particle EFT cross section for e+e- → W+W- near threshold.

Closed-form leading-order results from Beneke, Falgari, Schwinn, Signer,
Zanderighi, *"Four-fermion production near the W pair production
threshold"*, [arXiv:0707.0773](https://arxiv.org/abs/0707.0773).

Conventions match the paper. The natural BFS quantity is σ_LR^(0), the
LR-helicity cross section for the *specific* channel e+e- → μ⁻ν̄_μ ud̄;
helpers below convert to unpolarised totals.

Born-level EFT contributions implemented here (no loop corrections):

  • σ_LR^(0)             doubly-resonant LO,           eq. (17)
  • σ_LR^(1), σ_RL^(1)   NLO subleading potential,     eq. (33)
  • σ_LR^(1/2)           hard non-resonant (h1-h3),    eq. (37) + (38)
                          (h4-h7 single-resonant terms are < 0.5% per the
                           paper between 155 and 180 GeV — omitted)
  • σ_LR^(3/2),a         energy-dependent N^{3/2}LO,   eq. (39) + (40)

The N^{3/2}LO,b correction of eq. (41) is energy-independent and quoted
as ≲ 2 fb in the specific channel — neglected here. NLO LOOP corrections
(section 4 of the paper: hard, Coulomb, soft, collinear, decay) are NOT
implemented here — those go into ``BFSCorrections.delta_NLO`` separately.

The sum σ^(0) + σ^(1) + σ^(1/2) + σ^(3/2),a is what the paper labels
``EFT(N^{3/2}LO)`` in Table 1 — matches the Whizard exact 4f Born to
~0.1 % at 161 GeV, ~10 % at 155 GeV, ~0.1 % at 170 GeV.

Auxiliary kinematic functions, eq. (32):

  • ξ(s) = −3 M_W² (s − 2 M_Z² s_W²) / (s (s − M_Z²))
  • χ(s) = −6 M_W² M_Z² s_W²        / (s (s − M_Z²))

All cross sections returned in pb. Vectorised in ``s``.
"""

from __future__ import annotations

import numpy as np

from process.ww.eft_xsec import (
    ALPHA_EM_0,                         # not used for these LO formulae; kept for parity
    GEV_M2_TO_PB,
    M_W_DEFAULT, GAMMA_W_DEFAULT, M_Z,
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

    Returns ``(σ_LR^{1/2}, σ_RL^{1/2})`` in pb, for the specific channel.
    """
    s_arr = np.asarray(s, dtype=float)
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    xi, chi = _xi_chi(s_arr, mW)
    pref = (4.0 * alpha ** 3) / (27.0 * sW2 ** 3 * s_arr)
    sigma_LR_half = pref * (_K_H1 + _K_H2 * xi + _K_H3 * xi ** 2)
    sigma_RL_half = pref * (_K_H3 * chi ** 2)
    sigma_LR_half = sigma_LR_half * GEV_M2_TO_PB
    sigma_RL_half = sigma_RL_half * GEV_M2_TO_PB
    if apply_BR_correction:
        c = _BR_correction(mW, gammaW)
        sigma_LR_half = sigma_LR_half * c
        sigma_RL_half = sigma_RL_half * c
    return sigma_LR_half, sigma_RL_half


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


def sigma_BFS_LO_total_WW_pb(s, mW: float = M_W_DEFAULT,
                             gammaW: float = GAMMA_W_DEFAULT,
                             order: str = "N3/2LO",
                             apply_BR_correction: bool = True):
    """Total σ_WW = σ(e+e- → W+W-) at BFS LO_EFT, unpolarised initial state,
    summed over ALL 4-fermion final states.

    Each σ_LR^(...) carries the BFS 1/27 specific-channel BR factor. We
    strip it by × 27 and average over the four initial helicities by ÷4.
    The result is σ(e+e- → W+W-) including all 4f decays.

    Parameters
    ----------
    s : float or array
        Partonic CM energy² in GeV².
    mW, gammaW : float
        W mass and physical width in GeV. The width is used in the
        complex-velocity propagator AND propagates into the BR-correction
        factor (Γ_W^(0)(m_W) / Γ_W)² that re-weights the LO BR convention
        of 1/27 when Γ_W ≠ Γ_W^(0)(m_W). Section 6.1 of the paper.
    order : {"LO", "N1/2LO", "NLO", "N3/2LO"}
        Truncation of the BFS Born expansion.
    apply_BR_correction : bool
        If True (default), apply the (Γ_W^(0)/Γ_W)² BR factor correction.
        Set to False to reproduce Table 1 of the paper exactly (which uses
        Γ_W = Γ_W^(0) so the correction is trivially 1).

    Returns
    -------
    σ_total_WW in pb (same shape as ``s``).
    """
    sigma_LR_0 = sigma_LR0_specific_pb(s, mW, gammaW, apply_BR_correction=apply_BR_correction)
    sigma_LR = sigma_LR_0
    sigma_RL = np.zeros_like(np.asarray(sigma_LR_0, dtype=float))

    if order in ("N1/2LO", "NLO", "N3/2LO"):
        s_LR_12, s_RL_12 = sigma_LR_RL_half_specific_pb(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR_12
        sigma_RL = sigma_RL + s_RL_12
    if order in ("NLO", "N3/2LO"):
        # NLO Born potential expansion, eq. (33). Γ_W^(1) = 0 for Born.
        s_LR_NLO, s_RL_NLO = sigma_LR_RL_NLO_potential_specific_pb(
            s, mW, gammaW, gammaW_NLO=0.0,
            apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR_NLO
        sigma_RL = sigma_RL + s_RL_NLO
    if order == "N3/2LO":
        s_LR_32a, s_RL_32a = sigma_LR_RL_three_half_a_specific_pb(
            s, mW, gammaW, apply_BR_correction=apply_BR_correction)
        sigma_LR = sigma_LR + s_LR_32a
        sigma_RL = sigma_RL + s_RL_32a

    sigma_total_WW = (sigma_LR + sigma_RL) * 27.0 / 4.0
    if np.ndim(s) == 0:
        return float(sigma_total_WW)
    return sigma_total_WW


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
