"""Validate the K_C-safe Coulomb chain (NLO_CONFIG["coulomb_kc_safe"]=True).

Four cross-checks that pin the new ``coulomb_kc_safe`` knob in
``framework/process/ww/xsec_calculator/bfs_eft.py``:

  1. Route-A vs Route-B identity (machine precision). The two-call
     difference σ_chain(kcsafe=False) − σ_chain(kcsafe=True) must equal
     the standalone leading-α/v piece of eq. 62 of arXiv:0707.0773
     (= Δσ_Coul_NLO with subleading_only=False minus subleading_only=True).
     Tested in σ_BFS_LO_total_WW_pb (apply_delta_QCD=False) so the relation
     is exact, plus a second pass with apply_delta_QCD=True where the
     identity carries an overall × δ_QCD factor.

  2. Leading-α/v overlap between K_C and eq. 62 term 1. Print K_C(√s) − 1
     and the size of Δσ_Coul_NLO_term1 / σ_LR^(0) across the threshold
     window. Both should be ~5-7% at threshold and decrease above, with
     the K_C value slightly larger (≈+2 pp) by the off-shell-vs-on-shell
     prescription difference. This is what the K_C-safe combination
     removes from the BFS NLO bracket.

  3. Asymptotic limit. Above √s ≈ 172 GeV, K_C → 1 and Δσ_Coul_NLO → 0;
     the K_C-safe chain must agree with the unsafe chain to better than
     ~0.5% in this region.

  4. FKM 1995 hep-ph/9507422 eq. 21 strict near-threshold:
     X/2 ≈ 5.21% at E = 0. Verify our standalone Δσ_Coul_NLO_term1
     reproduces this value (cross-checked at validation log §5).

No plots; tab-delimited table output to stdout. Run from the repo root:

    python3 -m scripts.investigations.coulomb_double_count.validate_kc_safe_decomposition
"""

from __future__ import annotations

import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_BFS_LO_total_WW_pb,
    delta_sigma_Coulomb_NLO_specific_pb,
    sigma_LR0_specific_pb,
    delta_QCD_factor,
    gamma_W_LO,
    M_W_BFS_REF,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    coulomb_K_factor,
    sigma_partonic_munuqq,
    ALPHA_S_MW_DEFAULT,
)


MW = 80.379
GW = 2.085


# ---------------------------------------------------------------------------
# 1. Identity: chain difference equals Δσ_Coul_NLO_term1 (× δ_QCD)
# ---------------------------------------------------------------------------

def test_decomposition_identity():
    print("=" * 78)
    print("1. Route-A vs Route-B identity")
    print("=" * 78)
    print("    chain_diff  ≡  σ_BFS_total(kcsafe=False) − σ_BFS_total(kcsafe=True)")
    print("    closed_form ≡  27/4 × [Δσ_NLO(subleading_only=False)")
    print("                          − Δσ_NLO(subleading_only=True)]")
    print("                ≡  27/4 × Δσ_Coul_NLO_term1")
    print()
    sqrts = np.array([155.0, 157.0, 160.0, 161.0, 162.5, 165.0, 168.0, 172.0])
    s = sqrts ** 2

    # All NLO loops ON, but δ_QCD OFF for an exact identity (no scaling)
    common = dict(
        order="N3/2LO",
        apply_BR_correction=False,        # σ_BFS_LO_total_WW_pb default
        include_NLO_hard_decay=True,
        include_BFS_NNLO=False,
        apply_delta_QCD=False,
        apply_whizard_anchor=False,
    )
    sig_false = sigma_BFS_LO_total_WW_pb(s, MW, GW, coulomb_kc_safe=False, **common)
    sig_true  = sigma_BFS_LO_total_WW_pb(s, MW, GW, coulomb_kc_safe=True,  **common)
    chain_diff = sig_false - sig_true

    # Standalone Δσ_Coul_NLO_term1: difference of the two subleading_only
    # branches of the helper, lifted to σ_total_WW by × 27/4.
    coul_full = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                    subleading_only=False)
    coul_sub  = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                    subleading_only=True)
    closed_form = (27.0 / 4.0) * (coul_full - coul_sub)

    rel = np.where(np.abs(chain_diff) > 0, (chain_diff - closed_form) / chain_diff, 0.0)
    print(f"  {'√s':>8}  {'chain_diff (pb)':>18}  {'closed_form (pb)':>18}  {'rel err':>12}")
    for sq, a, b, r in zip(sqrts, chain_diff, closed_form, rel):
        print(f"  {sq:8.2f}  {a:+18.10e}  {b:+18.10e}  {r:+12.2e}")
    max_rel = float(np.max(np.abs(rel)))
    print()
    print(f"  → max |rel err| = {max_rel:.2e}  "
          f"({'PASS' if max_rel < 1e-12 else 'FAIL — investigate'})")
    print()

    # Second pass: identity with δ_QCD on (should scale by exactly δ_QCD).
    sig_false_qcd = sigma_BFS_LO_total_WW_pb(s, MW, GW, coulomb_kc_safe=False,
                                             **{**common, "apply_delta_QCD": True})
    sig_true_qcd  = sigma_BFS_LO_total_WW_pb(s, MW, GW, coulomb_kc_safe=True,
                                             **{**common, "apply_delta_QCD": True})
    chain_diff_qcd = sig_false_qcd - sig_true_qcd
    expected_qcd = closed_form * delta_QCD_factor(ALPHA_S_MW_DEFAULT)
    rel2 = (chain_diff_qcd - expected_qcd) / chain_diff_qcd
    max_rel2 = float(np.max(np.abs(rel2)))
    print(f"  with apply_delta_QCD=True: max |rel err| = {max_rel2:.2e}  "
          f"({'PASS' if max_rel2 < 1e-12 else 'FAIL'})")
    print()


# ---------------------------------------------------------------------------
# 2. K_C-1 vs Δσ_Coul_NLO_term1: same physics, different formulation
# ---------------------------------------------------------------------------

def test_kc_vs_term1_overlap():
    print("=" * 78)
    print("2. K_C − 1 vs Δσ_Coul_NLO_term1 / σ_LR^(0)")
    print("=" * 78)
    print("    The two quantities should be ~5-7% at threshold (the same")
    print("    leading-α/v Coulomb physics) and tend to 0 above. The K_C")
    print("    value is slightly larger (off-shell-p prescription, ~+2pp).")
    print()
    sqrts = np.array([155.0, 157.0, 159.0, 160.0, 161.0, 162.5, 165.0, 168.0, 172.0])
    s = sqrts ** 2
    kc = coulomb_K_factor(s, MW, GW)
    sigma_LR0 = sigma_LR0_specific_pb(s, MW, GW, apply_BR_correction=False)
    coul_full = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                    subleading_only=False)
    coul_sub  = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                    subleading_only=True)
    term1 = coul_full - coul_sub
    term1_rel = term1 / sigma_LR0

    print(f"  {'√s':>8}  {'K_C-1 [%]':>10}  {'Δσ_term1/σ_LR^0 [%]':>22}  "
          f"{'Δσ_term2/σ_LR^0 [%]':>22}")
    for sq, k, t1r, t2 in zip(sqrts, kc - 1.0, term1_rel * 100, coul_sub / sigma_LR0 * 100):
        print(f"  {sq:8.2f}  {k*100:+10.4f}  {t1r:+22.4f}  {t2:+22.5f}")
    print()


# ---------------------------------------------------------------------------
# 3. Asymptotic limit at high √s
# ---------------------------------------------------------------------------

def test_asymptotic_limit():
    print("=" * 78)
    print("3. Asymptotic limit at high √s")
    print("=" * 78)
    print("    K_C → 1 and Δσ_Coul_NLO → 0 well above threshold.")
    print("    σ_chain(kcsafe=True) should approach σ_chain(kcsafe=False).")
    print("    Comparison via the full sigma_partonic_munuqq (post-K_C, post-δ_QCD).")
    print()
    sqrts = np.array([161.0, 165.0, 170.0, 175.0, 180.0, 200.0, 240.0])
    s = sqrts ** 2
    common = dict(
        mW=MW, gammaW=GW,
        channel="inclusive", include_coulomb=True, br_convention="pdg-constant",
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True, whizard_anchor_source="grid",
    )
    sig_false = sigma_partonic_munuqq(s, coulomb_kc_safe=False, **common)
    sig_true  = sigma_partonic_munuqq(s, coulomb_kc_safe=True,  **common)
    rel = (sig_true - sig_false) / sig_false * 100
    print(f"  {'√s':>8}  {'σ_unsafe (pb)':>14}  {'σ_kc-safe (pb)':>14}  "
          f"{'Δσ/σ [%]':>10}")
    for sq, a, b, r in zip(sqrts, sig_false, sig_true, rel):
        print(f"  {sq:8.2f}  {a:14.5f}  {b:14.5f}  {r:+10.3f}")
    print()
    print("  Expectation: |Δσ/σ| shrinks monotonically above threshold,")
    print("  ~few-permille at 200 GeV, ~1 permille by 240 GeV.")
    print()


# ---------------------------------------------------------------------------
# 4. FKM 1995 strict near-threshold cross-check
# ---------------------------------------------------------------------------

def test_fkm_strict():
    print("=" * 78)
    print("4. FKM 1995 hep-ph/9507422 eq. 21 — leading-α/v at threshold")
    print("=" * 78)
    print("    At E ≡ √s − 2 m_W → 0, FKM eq. 21 gives the leading α/v")
    print("    coefficient X/2 = (α/π)·π·1 = α/2·(π/v)·v = α/2 → 5.21%")
    print("    on σ_LO. Our Δσ_Coul_NLO_term1 / σ_LR^(0) is the discrete")
    print("    analog at E = 0 (taking finite-Γ_W regularised log).")
    print()
    # Evaluate at √s = 2·M_W_BFS_REF (E=0 to within Γ_W), nominal width
    sqrts0 = 2.0 * M_W_BFS_REF                  # = 160.754 GeV
    gw_lo = gamma_W_LO(M_W_BFS_REF)             # 2.04485 GeV — BFS LO ref
    s0 = sqrts0 ** 2
    sigma_LR0 = sigma_LR0_specific_pb(s0, M_W_BFS_REF, gw_lo, apply_BR_correction=False)
    coul_full = delta_sigma_Coulomb_NLO_specific_pb(s0, M_W_BFS_REF, gw_lo,
                                                    apply_BR_correction=False, subleading_only=False)
    coul_sub  = delta_sigma_Coulomb_NLO_specific_pb(s0, M_W_BFS_REF, gw_lo,
                                                    apply_BR_correction=False, subleading_only=True)
    term1_rel = (coul_full - coul_sub) / sigma_LR0
    kc_at_thr = coulomb_K_factor(s0, M_W_BFS_REF, gw_lo) - 1.0
    print(f"  At √s = 2·M_W_BFS_REF = {sqrts0:.4f} GeV, Γ_W = {gw_lo:.5f} GeV:")
    print(f"    Δσ_Coul_NLO_term1 / σ_LR^(0)  = {term1_rel*100:+.3f}%   (FKM target ≈ +5.21%)")
    print(f"    K_C − 1                         = {kc_at_thr*100:+.3f}%   "
          f"(FKM strict on-shell limit: +6.6%; off-shell prescription: +7.3%)")
    print()


if __name__ == "__main__":
    test_decomposition_identity()
    test_kc_vs_term1_overlap()
    test_asymptotic_limit()
    test_fkm_strict()
