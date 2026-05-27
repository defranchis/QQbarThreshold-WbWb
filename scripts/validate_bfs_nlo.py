"""Comprehensive BFS-NLO validation against arXiv:0707.0773.

Runs the framework's σ chain in several piece-by-piece scenarios and
compares each to BFS paper Tables 1, 2, 4 (and the Coulomb / HSC magnitudes
quoted in the text). Each scenario isolates a different physics piece so a
discrepancy can be attributed to its source:

  A. Born N^(3/2)LO (no width-correction)        →  BFS Table 1
  B. Born N^(3/2)LO (with BR correction)          →  BFS Table 2
  C. Coulomb_NLO (eq. 62) / σ_LR^(0) at threshold →  "~+5 %" quoted in text
  D. HSC bracket / σ_LR^(0) at threshold          →  pure numerical evaluation
  E. δ_QCD(α_s)                                   →  eq. delta_qcd
  F. Born + HSC + EW_decay + Coul_NLO + ISR       →  BFS Table 4
  G. PDG BR vs bfs-eft BR (relative)              →  internal consistency

For BFS-paper comparison the ISR convolution is run with α_em = α_Gμ (BFS
prescription "G_μ scheme everywhere"), matching the paper's setup. For the
analysis default (PDG-constant + α_s-aware δ_QCD), see cards/ww_default.py.
"""

from __future__ import annotations

import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    delta_QCD_factor,
    delta_sigma_Coulomb_NLO_specific_pb,
    delta_sigma_NLO_hard_softcoll_specific_pb,
    delta_sigma_NLO_decay_specific_pb,
    sigma_BFS_LO_total_WW_pb,
    sigma_BFS_specific_munuud_pb,
    sigma_LR0_specific_pb,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    alpha_Gmu, M_T_BFS_REF, M_H_BFS_REF, M_Z_BFS_REF,
)

# BFS arXiv:0707.0773 reference EW inputs (m_t = 174.2, M_H = 115 pre-discovery,
# M_Z = 91.188). Pinned explicitly here so paper closure is preserved as the
# framework's production defaults move to PDG (m_W to 80.379, M_H to 125.25,
# M_Z to 91.1876).
_BFS_THEORY_KW = dict(mt=M_T_BFS_REF, MH=M_H_BFS_REF, MZ=M_Z_BFS_REF)
_BFS_THEORY_KW_PARTONIC = dict(m_t=M_T_BFS_REF, M_H=M_H_BFS_REF, MZ=M_Z_BFS_REF)
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

# Framework default ISR α is now α_Gμ at m_W = 80.377 (BFS prescription,
# line 2514 of arXiv:0707.0773). No monkeypatching needed.


# Paper Table 1 (LO width, no BR correction): m_W = 80.377, Γ_W = 2.04483
BFS_TABLE_1 = {     # √s [GeV] → σ paper N^(3/2)LO [fb]
    155: 31.30, 158: 62.50, 161: 160.89, 164: 318.80,
    167: 429.70, 170: 505.40,
}
# Paper Table 2 (NLO+QCD width, with BR correction): m_W = 80.377, Γ_W = 2.09201
# (Table 2 keeps Table 1's pole m_W; only Γ_W changes — BFS §6.1.)
BFS_TABLE_2 = {
    155: 30.54, 158: 60.83, 161: 154.44, 164: 303.70,
    167: 409.30, 170: 481.70,
}
# Paper Table 4 (specific channel μ⁻ν̄_μ ud̄): m_W = 80.377, Γ_W = 2.09201
# Columns: (Born, Born+ISR, NLO_with_ISR, NLO_ISR-tree)
BFS_TABLE_4 = {
    158: (61.67, 45.64,  49.19,  50.02),
    161: (154.19, 108.60, 117.81, 120.00),
    164: (303.00, 219.70, 234.90, 236.80),
    167: (408.80, 310.20, 328.20, 329.10),
    170: (481.70, 378.40, 398.00, 398.30),
}


def _fmt(x, w=8, dec=3):
    return f"{x:>{w}.{dec}f}"


def scenario_A():
    """Born N^(3/2)LO, no BR correction — reproduces BFS Table 1."""
    print("=" * 72)
    print("Scenario A: BFS Table 1 (LO width 2.04483, no BR correction)")
    print("=" * 72)
    print(f"  {'√s':>6} {'mine [fb]':>10} {'BFS [fb]':>10} {'ratio':>8}")
    mW, gW = 80.377, 2.04483
    for sq, paper in BFS_TABLE_1.items():
        mine = sigma_BFS_LO_total_WW_pb(np.array([sq**2]), mW, gW,
                                         order="N3/2LO",
                                         apply_BR_correction=False,
                                         **_BFS_THEORY_KW)[0]
        mine_specific = mine * (4.0/27.0) * 1e3   # → fb specific unpol (BR ÷ 27, × 4 for inclusive then div 4 = ÷27)
        # Actually mine total σ_WW × (1/27) gives specific unpol with LO BR.
        # But mine already has 27/4 stripped, then ÷ 27 gives total/27 = (LR+RL)/4 = specific unpol.
        mine_specific_unpol = mine / 27.0 * 1e3
        print(f"  {sq:>6} {_fmt(mine_specific_unpol)} {_fmt(paper)}    {mine_specific_unpol/paper:.4f}")


def scenario_B():
    """Born N^(3/2)LO, with BR correction — reproduces BFS Table 2."""
    print("\n" + "=" * 72)
    print("Scenario B: BFS Table 2 (NLO+QCD width 2.09201, with BR correction)")
    print("=" * 72)
    print(f"  {'√s':>6} {'mine [fb]':>10} {'BFS [fb]':>10} {'ratio':>8}")
    mW, gW = 80.377, 2.09201
    for sq, paper in BFS_TABLE_2.items():
        mine_specific = sigma_BFS_specific_munuud_pb(np.array([sq**2]), mW, gW,
                                                     order="N3/2LO",
                                                     **_BFS_THEORY_KW)[0] * 1e3
        print(f"  {sq:>6} {_fmt(mine_specific)} {_fmt(paper)}    {mine_specific/paper:.4f}")


def scenario_C():
    """Coulomb NLO (eq. 62) magnitude at threshold ≈ +5 % per BFS text."""
    print("\n" + "=" * 72)
    print("Scenario C: Coulomb NLO (eq. 62) magnitude relative to σ_LR^(0)")
    print("=" * 72)
    mW, gW = 80.377, 2.09201
    # at threshold E=0:
    s_th = (2*mW)**2
    sLR0 = sigma_LR0_specific_pb(np.array([s_th]), mW, gW,
                                  apply_BR_correction=True, MZ=M_Z_BFS_REF)[0]
    coul = delta_sigma_Coulomb_NLO_specific_pb(np.array([s_th]), mW, gW,
                                                apply_BR_correction=True,
                                                subleading_only=False,
                                                MZ=M_Z_BFS_REF)[0]
    print(f"  At √s = 2 m_W = {2*mW:.3f} GeV:")
    print(f"    σ_LR^(0)            = {sLR0:.5f} pb (specific LR helicity)")
    print(f"    Δσ_Coul^(1) (eq.62) = {coul:.5f} pb")
    print(f"    Coul / σ_LR^(0)     = {(coul/sLR0)*100:+.3f} %    (BFS text: ~+5 %)")
    print(f"  Two-photon piece only (subleading):")
    coul_sub = delta_sigma_Coulomb_NLO_specific_pb(np.array([s_th]), mW, gW,
                                                    apply_BR_correction=True,
                                                    subleading_only=True,
                                                    MZ=M_Z_BFS_REF)[0]
    print(f"    Δσ_Coul^(1,sub) / σ_LR^(0) = {(coul_sub/sLR0)*100:+.3f} %  (BFS: ~+0.2 %)")


def scenario_D():
    """HSC piece magnitude at threshold."""
    print("\n" + "=" * 72)
    print("Scenario D: Hard+Soft+Coll (HSC) magnitude relative to σ_LR^(0)")
    print("=" * 72)
    mW, gW = 80.377, 2.09201
    print(f"  c_p,LR^(1,fin) Re value used: {-10.076}  (BFS line 1797)")
    for sq in [158, 161, 164, 167, 170]:
        s = sq**2
        sLR0 = sigma_LR0_specific_pb(np.array([s]), mW, gW,
                                      apply_BR_correction=True,
                                      MZ=M_Z_BFS_REF)[0]
        hsc = delta_sigma_NLO_hard_softcoll_specific_pb(
            np.array([s]), mW, gW, apply_BR_correction=True,
            **_BFS_THEORY_KW)[0]
        print(f"  √s = {sq:3d}: σ_LR^(0) = {sLR0:.5f}, Δσ_HSC = {hsc:+.5f},"
              f" rel = {(hsc/sLR0)*100:+.3f} %")


def scenario_E():
    """δ_QCD multiplicative factor at α_s = 0.1199."""
    print("\n" + "=" * 72)
    print("Scenario E: δ_QCD(α_s) = 1 + α_s/π + 1.409(α_s/π)²")
    print("=" * 72)
    for a_s, label in [(0.110, "α_s lo"), (0.1199, "BFS ref"), (0.130, "α_s hi")]:
        d = delta_QCD_factor(a_s)
        print(f"  α_s = {a_s:.4f}  ({label}):  δ_QCD = {d:.5f}")


def scenario_F():
    """Full NLO chain (HSC + EW-decay + Coul_NLO + δ_QCD) × ISR → BFS Table 4.

    Apples-to-paper toggles:
      * include_coulomb=False — BFS Table 4 is a fixed-order N^(3/2)LO+NLO
        expansion: σ^(1)_pot carries the leading-α/v Coulomb piece, and
        eq.(62) carries the NLO α correction. The Fadin-Khoze-Martin K_C
        resummation overlaps these → must be off.
      * apply_whizard_anchor=True — BFS Table 4 caption: col 1 ("Born") is
        identical to Table 2's last column = Whizard 4f Born. So the
        apples reference is Whizard, not the EFT N^(3/2)LO sum. The anchor
        f(δ,Γ_W) brings σ_BFS·f onto Whizard by construction.
      * apply_delta_QCD = {False for Born(ISR), True for NLO} — BFS §6.1
        (lines 2622-2635 of wwpaper_v2.tex): δ_QCD is applied to the
        "entire NLO electroweak cross section" only. The Born(ISR) column
        is the Whizard reference convoluted with ISR; Whizard already has
        the QCD-corrected Γ_W in the propagator with LO partial widths at
        the vertex, so no extra δ_QCD multiplier.
      * isr_scheme="2leg" — paper uses LL+exp two-leg.
    """
    print("\n" + "=" * 72)
    print("Scenario F: Full NLO chain vs BFS Table 4 (specific μ⁻ν̄_μ ud̄)")
    print("=" * 72)
    mW, gW = 80.377, 2.09201
    print(f"  {'√s':>6} {'Born[isr]_mine':>15} {'Born[isr]_BFS':>15}"
          f" {'NLO_mine':>10} {'NLO_BFS':>10}"
          f" {'mine/BFS NLO':>14}")
    for sq, (born_paper, born_isr_paper, nlo_paper, _) in BFS_TABLE_4.items():
        born_isr_mine = sigma_observed_munuqq(
            float(sq), mW=mW, gammaW=gW, channel="munuud",
            br_convention="bfs-eft", include_coulomb=False,
            include_NLO_hard_decay=False, apply_delta_QCD=False,
            apply_whizard_anchor=True, isr_scheme="2leg",
            **_BFS_THEORY_KW_PARTONIC) * 1e3
        nlo_mine = sigma_observed_munuqq(
            float(sq), mW=mW, gammaW=gW, channel="munuud",
            br_convention="bfs-eft", include_coulomb=False,
            include_NLO_hard_decay=True, apply_delta_QCD=True,
            alpha_s=0.1199, apply_whizard_anchor=True,
            isr_scheme="2leg",
            **_BFS_THEORY_KW_PARTONIC) * 1e3
        print(f"  {sq:>6}  {_fmt(born_isr_mine, 12)}    {_fmt(born_isr_paper, 12)}"
              f"  {_fmt(nlo_mine, 8)}  {_fmt(nlo_paper, 8)}"
              f"  {nlo_mine/nlo_paper:>9.4f}")


def scenario_G():
    """Compare PDG-constant vs bfs-eft BR conventions on inclusive σ."""
    print("\n" + "=" * 72)
    print("Scenario G: PDG-constant vs bfs-eft BR conventions (inclusive μνqq̄)")
    print("=" * 72)
    from framework.process.ww.xsec_calculator.eft_xsec import BR_INCLUSIVE_MUNUQQ
    if True:
        mW, gW = 80.379, 2.085
        print(f"  PDG BR inclusive = {BR_INCLUSIVE_MUNUQQ:.5f}")
        print(f"  LO BR inclusive  = {4.0/27.0:.5f} × (Γ_W^(0)/Γ_W)² (≈ {4.0/27.0 * (2.0443/2.085)**2:.5f})")
        print(f"  {'√s':>6} {'σ_inclusive_pdg':>17} {'σ_inclusive_bfseft':>20} {'ratio':>10}")
        for sq in [157, 158, 161, 162.3, 163, 165, 170]:
            s_pdg = sigma_observed_munuqq(float(sq), mW=mW, gammaW=gW,
                                           channel="inclusive",
                                           br_convention="pdg-constant",
                                           include_NLO_hard_decay=True,
                                           apply_delta_QCD=True) * 1e3
            s_bfs = sigma_observed_munuqq(float(sq), mW=mW, gammaW=gW,
                                           channel="inclusive",
                                           br_convention="bfs-eft",
                                           include_coulomb=False,
                                           include_NLO_hard_decay=True,
                                           apply_delta_QCD=True) * 1e3
            print(f"  {sq:>6.1f}  {_fmt(s_pdg, 14)} fb   {_fmt(s_bfs, 14)} fb   {s_pdg/s_bfs:.4f}")


def scenario_I_anchor():
    """Whizard-anchor closure: with f applied, the BFS-N^(3/2)LO Born should
    reproduce BFS Tables 1 (LO width) and 2 (NLO+QCD width) Whizard 4f Born
    values to <0.1% by construction. Tests the spline + Γ_W-interp logic."""
    print("\n" + "=" * 72)
    print("Scenario I: Whizard-anchor closure (mine × f vs BFS Whizard reference)")
    print("=" * 72)
    print("Table 1 reference (m_W=80.377, Γ_W=2.04483, no BR corr):")
    mW1, gW1 = 80.377, 2.04483
    table1_whiz = {155: 34.43, 158: 63.39, 161: 160.62, 164: 318.30, 167: 428.60, 170: 505.10}
    for sq, target in table1_whiz.items():
        mine_anchor = sigma_BFS_LO_total_WW_pb(
            np.array([sq**2]), mW1, gW1, order="N3/2LO",
            apply_BR_correction=False, apply_whizard_anchor=True,
            **_BFS_THEORY_KW)[0]
        mine_unpol_specific = mine_anchor / 27.0 * 1e3
        print(f"  √s={sq}: mine_anchored = {mine_unpol_specific:.3f} fb, BFS_Whizard = {target:.3f}, ratio = {mine_unpol_specific/target:.5f}")
    print()
    print("Table 2 reference (m_W=80.377, Γ_W=2.09201, BR corr):")
    mW2, gW2 = 80.377, 2.09201
    table2_whiz = {155: 33.58, 158: 61.67, 161: 154.19, 164: 303.00, 167: 408.80, 170: 481.70}
    for sq, target in table2_whiz.items():
        mine_anchor = sigma_BFS_specific_munuud_pb(
            np.array([sq**2]), mW2, gW2, order="N3/2LO",
            apply_whizard_anchor=True, **_BFS_THEORY_KW)[0] * 1e3
        print(f"  √s={sq}: mine_anchored = {mine_anchor:.3f} fb, BFS_Whizard = {target:.3f}, ratio = {mine_anchor/target:.5f}")


def scenario_J_derivatives():
    """Check that the anchor preserves dσ/dm_W and dσ/dΓ_W derivatives
    within <1%. f(δ, Γ_W) varies smoothly, so the derivatives are
    dominated by σ_BFS_N32 with at most a small f' correction."""
    from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
    from framework.process.ww.xsec_calculator.eft_xsec import BR_INCLUSIVE_MUNUQQ
    print("\n" + "=" * 72)
    print("Scenario J: dσ/dm_W and dσ/dΓ_W derivatives — anchor effect")
    print("=" * 72)
    mW0, gW0 = 80.385, 2.085
    h = 0.001
    print("  √s [GeV]   dσ/dm_W [fb/MeV]                dσ/dΓ_W [fb/MeV]")
    print("            no-anchor      +anchor      Δ%   no-anchor    +anchor    Δ%")
    for sq in [157.0, 161.0, 162.3, 165.0]:
        def s_obs(mW, gW, anchor):
            return sigma_observed_munuqq(
                sq, mW=mW, gammaW=gW, channel="inclusive",
                br_convention="pdg-constant",
                include_NLO_hard_decay=True, apply_delta_QCD=True,
                apply_whizard_anchor=anchor,
                **_BFS_THEORY_KW_PARTONIC) / BR_INCLUSIVE_MUNUQQ * 1e3
        dm_no = (s_obs(mW0 + h, gW0, False) - s_obs(mW0 - h, gW0, False)) / (2*h)
        dm_a  = (s_obs(mW0 + h, gW0, True)  - s_obs(mW0 - h, gW0, True))  / (2*h)
        dg_no = (s_obs(mW0, gW0 + h, False) - s_obs(mW0, gW0 - h, False)) / (2*h)
        dg_a  = (s_obs(mW0, gW0 + h, True)  - s_obs(mW0, gW0 - h, True))  / (2*h)
        print(f"  {sq:>6.1f}    {dm_no:+8.3f}    {dm_a:+8.3f}   {(dm_a-dm_no)/dm_no*100:+.2f}    {dg_no:+8.3f}   {dg_a:+8.3f}  {(dg_a-dg_no)/dg_no*100:+.2f}")


def scenario_H():
    """c_p,LR^(1,fin) m_W / m_t / M_H slopes from the analytic BFS appendix.

    The framework now evaluates c_p,LR^(1,fin), c_d,l^(1,fin), c_d,h^(1,fin)
    analytically (bfs_c1fin.py, BFS appendix eq. PV0LR + CTLR + LVD0/LVDCT +
    HDV0/DHCT). Reference closure: -10.0758 + 0.2049 i (BFS quotes -10.076);
    decay reproduces -2.7093, -2.0341. This scenario reports the slopes that
    matter for the fit propagation of m_W, m_t, M_H uncertainties.
    """
    from framework.process.ww.xsec_calculator.bfs_eft import (
        _c_p_LR_1_fin_re, _c_d_l_1_fin_re, _c_d_h_1_fin_re,
    )
    print("\n" + "=" * 72)
    print("Scenario H: analytic c^(1,fin) slopes (m_W, m_t, M_H)")
    print("=" * 72)
    m_W0, m_t0, M_H0, M_Z0 = 80.377, M_T_BFS_REF, M_H_BFS_REF, M_Z_BFS_REF
    print(f"  reference: m_W={m_W0}, m_t={m_t0}, M_H={M_H0}, M_Z={M_Z0}")
    print(f"  Re(c_p,LR) = {_c_p_LR_1_fin_re(m_W0, m_t0, M_H0, M_Z0):.4f}   (BFS -10.076)")
    print(f"  Re(c_d,l)  = {_c_d_l_1_fin_re(m_W0, m_t0, M_H0, M_Z0):.4f}   (BFS  -2.709)")
    print(f"  Re(c_d,h)  = {_c_d_h_1_fin_re(m_W0, m_t0, M_H0, M_Z0):.4f}   (BFS  -2.034)")
    print()
    print("  Numerical slopes (central finite diff):")
    for label, fn in [("c_p,LR", _c_p_LR_1_fin_re),
                      ("c_d,l ", _c_d_l_1_fin_re),
                      ("c_d,h ", _c_d_h_1_fin_re)]:
        d_mw = (fn(m_W0 + 0.1, m_t0, M_H0, M_Z0) - fn(m_W0 - 0.1, m_t0, M_H0, M_Z0)) / 0.2
        d_mt = (fn(m_W0, m_t0 + 0.5, M_H0, M_Z0) - fn(m_W0, m_t0 - 0.5, M_H0, M_Z0)) / 1.0
        d_mh = (fn(m_W0, m_t0, M_H0 + 1.0, M_Z0) - fn(m_W0, m_t0, M_H0 - 1.0, M_Z0)) / 2.0
        print(f"    ∂Re({label})/∂m_W = {d_mw:+.4f} /GeV   "
              f"∂/∂m_t = {d_mt:+.4e} /GeV   ∂/∂M_H = {d_mh:+.4e} /GeV")
    print()
    print("  c_p,LR m_W slope of +1.30 per GeV translates to ~0.4 in Δc")
    print("  over ±0.3 GeV m_W — comparable to ~1 MeV in fit bias if held")
    print("  constant. m_t / M_H slopes are O(10⁻³)/GeV: negligible.")


def scenario_K_per_piece_table():
    """Per-piece partonic NLO contributions at BFS Table 4 inputs.

    Tabulates the three NLO-loop pieces (HSC, EW-decay, Coul_NLO eq.62) at
    the six BFS scan energies and gives each as both an absolute σ in fb
    and a fraction of σ_LR^(0). This is the table-form summary of
    Scenarios C and D — single reference for "the partonic NLO sum at
    each scan energy", reproducible from this script.

    All partonic, BR correction on, no ISR, no δ_QCD, no anchor.
    m_W=80.377, Γ_W=2.09201 (BFS Table 4 inputs).
    """
    import numpy as np
    from framework.process.ww.xsec_calculator.bfs_eft import (
        sigma_LR0_specific_pb,
        sigma_BFS_specific_munuud_pb,
        delta_sigma_NLO_hard_softcoll_specific_pb,
        delta_sigma_NLO_decay_specific_pb,
        delta_sigma_Coulomb_NLO_specific_pb,
    )
    print("\n" + "=" * 72)
    print("Scenario K: per-piece partonic NLO contributions (BFS Table 4 inputs)")
    print("=" * 72)
    mW, gW = 80.377, 2.09201
    sqrts = np.array([158., 161., 164., 167., 170.])
    s = sqrts ** 2
    sLR0  = sigma_LR0_specific_pb(s, mW, gW, apply_BR_correction=True,
                                    MZ=M_Z_BFS_REF)
    sBorn = sigma_BFS_specific_munuud_pb(s, mW, gW, order="N3/2LO",
                                          include_NLO_hard_decay=False,
                                          include_BFS_NNLO=False,
                                          apply_delta_QCD=False,
                                          apply_whizard_anchor=False,
                                          **_BFS_THEORY_KW)
    hsc  = delta_sigma_NLO_hard_softcoll_specific_pb(s, mW, gW, apply_BR_correction=True,
                                                      **_BFS_THEORY_KW)
    dec  = delta_sigma_NLO_decay_specific_pb(s, mW, gW, apply_BR_correction=True,
                                              **_BFS_THEORY_KW)
    coul = delta_sigma_Coulomb_NLO_specific_pb(s, mW, gW, apply_BR_correction=True,
                                                subleading_only=False,
                                                MZ=M_Z_BFS_REF)
    sum_loops = hsc + dec + coul

    print(f"  m_W={mW} GeV, Γ_W={gW} GeV, channel μ⁻ν̄_μ ud̄, BR correction on.")
    print()
    print("  Absolute σ [fb] (partonic, no ISR, no δ_QCD):")
    print(f"  {'√s':>6} {'σ_LR^(0)':>10} {'σ_Born':>10} {'ΔHSC':>10} {'Δdecay':>10} {'ΔCoul^(1)':>10} {'Σ NLO':>10}")
    for i, sq in enumerate(sqrts):
        print(f"  {sq:>6.0f} {sLR0[i]*1e3:>10.3f} {sBorn[i]*1e3:>10.3f}"
              f" {hsc[i]*1e3:>+10.3f} {dec[i]*1e3:>+10.3f}"
              f" {coul[i]*1e3:>+10.3f} {sum_loops[i]*1e3:>+10.3f}")
    print()
    print("  Each NLO piece as % of σ_LR^(0):")
    print(f"  {'√s':>6} {'HSC%':>10} {'decay%':>10} {'Coul^(1)%':>10} {'Σ NLO%':>10}")
    for i, sq in enumerate(sqrts):
        print(f"  {sq:>6.0f} {hsc[i]/sLR0[i]*100:>+10.3f} {dec[i]/sLR0[i]*100:>+10.3f}"
              f" {coul[i]/sLR0[i]*100:>+10.3f} {sum_loops[i]/sLR0[i]*100:>+10.3f}")
    print()
    print("  Sub-leading Coulomb piece (two-photon, eq.62 second term):")
    coul_sub = delta_sigma_Coulomb_NLO_specific_pb(s, mW, gW, apply_BR_correction=True,
                                                    subleading_only=True,
                                                    MZ=M_Z_BFS_REF)
    print(f"  {'√s':>6} {'ΔCoul^(1,sub) [fb]':>22} {'sub/σ_LR^(0) %':>18}")
    for i, sq in enumerate(sqrts):
        print(f"  {sq:>6.0f} {coul_sub[i]*1e3:>22.4f} {coul_sub[i]/sLR0[i]*100:>+17.3f}%")


def main():
    scenario_A()
    scenario_B()
    scenario_C()
    scenario_D()
    scenario_E()
    scenario_F()
    scenario_G()
    scenario_H()
    scenario_I_anchor()
    scenario_J_derivatives()
    scenario_K_per_piece_table()


if __name__ == "__main__":
    main()
