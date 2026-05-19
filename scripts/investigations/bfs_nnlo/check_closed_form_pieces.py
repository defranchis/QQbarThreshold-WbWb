"""Closed-form NNLO pieces vs Table 1 of arXiv:0807.0102.

Three of the five N^{3/2}LO_EFT corrections in eq. (49) of the BFS NNLO
paper reduce to a small number of analytic terms reusing pieces of the
existing NLO chain:

  • Δσ^(C3)_LR       — triple-Coulomb, eq. (11). One line.
  • Δσ^(C×res)_LR   — residue × single-Coulomb, eq. (48). One line.
  • Δσ^(C×decay)_LR — decay correction × single-Coulomb, eq. (40).
                       = δ_decay_EW × Δσ^(C1)_LR (the α-piece of eq. 10).

All three are implemented in bfs_eft.py as σ_LR for the specific channel
μ⁻ν̄_μ ud̄. Table 1 of the paper tabulates Δσ^i = Δσ^i_LR / 4 (helicity-
averaged); we divide by 4 here for comparison. Table 1 uses the **raw
LO BR factor 1/27** with full Γ_W (no LO/NLO BR rescaling), so we call
the functions with ``apply_BR_correction=False`` for the round-trip.

Paper inputs (eqs. 50–51):
  M_W = 80.377, Γ_W = 2.09201, M_Z = 91.188 GeV
  m_e = 0.51099892 MeV, m_t = 174.2, M_H = 115 GeV
"""

import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    delta_sigma_NNLO_triple_Coulomb_specific_pb,
    delta_sigma_NNLO_C_residue_specific_pb,
    delta_sigma_NNLO_C_decay_specific_pb,
    delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb,
    delta_sigma_NNLO_C_soft_hard_specific_pb,
)


# Table 1 of arXiv:0807.0102 (Δσ^i = Δσ^i_LR / 4, in fb)
TABLE_1 = {
    "sqrts_GeV":  np.array([158.0, 161.0, 164.0, 167.0, 170.0]),
    "sumN32_fb":  np.array([-0.001, 0.147, 0.811, 1.287, 1.577]),
    "CxSH_fb":    np.array([-0.116, -0.321, -0.417, -0.389, -0.354]),
    "NLO_C_fb":   np.array([0.104,  0.226,  0.393,  0.473,  0.511]),
    "Cxdecay_fb": np.array([-0.037, -0.091, -0.134, -0.142, -0.142]),
    "Cxres_fb":   np.array([0.044,  0.324,  0.965,  1.345,  1.561]),
    "C3_fb":      np.array([0.004,  0.010,  0.003,  0.001,  0.000]),
}


def _print_compare(label, mine_fb, paper_fb, sqrts_GeV):
    print(f"\n{label}")
    print(f"  {'√s':>5}  {'mine':>9}  {'paper':>9}  {'mine-paper':>11}")
    for i, sq in enumerate(sqrts_GeV):
        diff = mine_fb[i] - paper_fb[i]
        print(f"  {sq:5.1f}  {mine_fb[i]:8.4f}  {paper_fb[i]:8.4f}  "
              f"{diff:+10.4f}")


if __name__ == "__main__":
    mW = 80.377
    gammaW = 2.09201
    sqrts = TABLE_1["sqrts_GeV"]
    s = sqrts ** 2

    print("=" * 64)
    print("BFS NNLO closed-form pieces — round-trip vs Table 1")
    print("  arXiv:0807.0102, M_W=80.377, Γ_W=2.09201, helicity-averaged")
    print("  apply_BR_correction=False (paper convention: raw 1/27)")
    print("=" * 64)

    cxSH_fb = delta_sigma_NNLO_C_soft_hard_specific_pb(
        s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0
    _print_compare("Δσ̂^(C×[S+H])  (eq. 34)  [fb, helicity-averaged]:",
                   cxSH_fb, TABLE_1["CxSH_fb"], sqrts)

    nlo_c_fb = delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb(
        s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0
    _print_compare("Δσ^(NLO-C)  (eq. 39)  [fb, helicity-averaged]:",
                   nlo_c_fb, TABLE_1["NLO_C_fb"], sqrts)

    cxdecay_fb = delta_sigma_NNLO_C_decay_specific_pb(
        s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0
    _print_compare("Δσ^(C×decay)  (eq. 40)  [fb, helicity-averaged]:",
                   cxdecay_fb, TABLE_1["Cxdecay_fb"], sqrts)

    cxres_fb = delta_sigma_NNLO_C_residue_specific_pb(
        s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0
    _print_compare("Δσ^(C×res)  (eq. 48)  [fb, helicity-averaged]:",
                   cxres_fb, TABLE_1["Cxres_fb"], sqrts)

    c3_fb = delta_sigma_NNLO_triple_Coulomb_specific_pb(
        s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0
    _print_compare("Δσ^(C3)  (eq. 11)  [fb, helicity-averaged]:",
                   c3_fb, TABLE_1["C3_fb"], sqrts)

    # Combined σ̂^(3/2) — sum of all five pieces, eq. (49)
    sum_fb = cxSH_fb + nlo_c_fb + cxdecay_fb + cxres_fb + c3_fb
    _print_compare("Σ = σ̂^(3/2)  (eq. 49 sum)  [fb, helicity-averaged]:",
                   sum_fb, TABLE_1["sumN32_fb"], sqrts)
