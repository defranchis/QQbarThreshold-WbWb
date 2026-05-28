"""Quantify the Born-side m_W bias on the post-NNLO chain — v3.

What's different from v2:

  - m_W reference fixed to M_W_BFS_REF = 80.377 GeV (BFS Table 2 reference;
    the 2026-05-26 Table-2 closure bug fix).  v2 used 80.379, which baked
    a known ~2 MeV mismatch against the BFS reference column.
  - BFS Table 2 reference values pulled from the canonical constant
    `_BFS_TABLE_2_WHIZ` in bfs_eft.py rather than hard-coded in the script.
  - Chain is whatever the current `sigma_BFS_specific_munuud_pb` default
    produces at order N3/2LO + apply_whizard_anchor=True (no NLO loops,
    no NNLO, no δ_QCD — same closure target as v2).  Since v2 ran,
    the chain has picked up: analytic c^(1,fin) (commit 7d87038),
    σ^(3/2),a BR-power fix, m_W=80.377 in Table-2 closure code paths.
  - Reports a side-by-side row with the v2 (MW=80.379) numbers so the
    impact of the m_W-reference fix is visible.

NOTE on NNLO: the Scenario-I closure compares to the BFS Table 2 WHIZARD
column, which is itself Born-only.  Apples-to-apples means we keep our
side Born-only here (anchor on, NLO/NNLO off).  NNLO does change the
PRODUCTION-chain σ but does not shift this residual — it's an additive
term on top of the anchored Born.  The reason this rerun is still
informative is the bug fix and analytic c^(1,fin), not the NNLO landing.
"""
import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_BFS_specific_munuud_pb,
    _BFS_TABLE_2_SQRTS,
    _BFS_TABLE_2_WHIZ,
    _BFS_TABLE_2_GW,
)
from framework.process.ww.xsec_calculator.eft_xsec import M_W_BFS_REF


# Scan window in BFS Table 2 (drop 155 and 170 — outside FCC-ee primary
# scan; the bias estimator uses the four points that bracket [161, 164]).
SQRTS_FULL = np.array(_BFS_TABLE_2_SQRTS, dtype=float)         # 155..170
WHIZ_FULL  = np.array(_BFS_TABLE_2_WHIZ, dtype=float)          # paper, fb
keep = (SQRTS_FULL >= 158.0) & (SQRTS_FULL <= 167.0)
SQRTS = SQRTS_FULL[keep]
WHIZ  = WHIZ_FULL[keep]

MW_REF = M_W_BFS_REF        # 80.377 GeV — BFS Table 2 reference
GW_REF = _BFS_TABLE_2_GW    # 2.09201 GeV — BFS Table 2 reference


def _mine(mW_val):
    return np.array([
        1e3 * sigma_BFS_specific_munuud_pb(
            s_i ** 2, mW_val, GW_REF, order="N3/2LO",
            apply_whizard_anchor=True)
        for s_i in SQRTS
    ])


def _bias_estimators(mine, label):
    r = mine / WHIZ - 1.0
    print(f"\n--- {label} ---")
    print("BFS reference points (Table 2, scan window):")
    print("  √s [GeV]   mine [fb]   BFS_Whiz [fb]   residual")
    for x, m, w, rr in zip(SQRTS, mine, WHIZ, r):
        print(f"  {x:7.1f}   {m:9.3f}   {w:13.3f}    {rr*100:+.4f} %")

    sref = 162.5
    x = SQRTS - sref
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, r, rcond=None)
    print(f"  Linear fit r(√s) ≈ {a*100:+.4f} %  +  {b*100:+.4f} %/GeV × (√s - {sref})")

    DMW = 1e-3  # GeV
    sig_pl = np.array([
        1e3 * sigma_BFS_specific_munuud_pb(
            s_i ** 2, MW_REF + DMW, GW_REF, order="N3/2LO",
            apply_whizard_anchor=True)
        for s_i in SQRTS
    ])
    sig_mi = np.array([
        1e3 * sigma_BFS_specific_munuud_pb(
            s_i ** 2, MW_REF - DMW, GW_REF, order="N3/2LO",
            apply_whizard_anchor=True)
        for s_i in SQRTS
    ])
    dsigma_dmW = (sig_pl - sig_mi) / (2 * DMW)
    rel_dsdm = dsigma_dmW / mine

    mean_d = np.mean(rel_dsdm)
    centered = rel_dsdm - mean_d
    bias_unif = -np.mean(r * centered) / np.mean(centered ** 2)

    # FCC-ee-like weights: more lumi at 161 and 164 than at 158, 167.
    w_scan = np.array([1.5, 1.5, 1.0, 0.5])
    w_scan = w_scan / w_scan.sum()
    mean_d_w = np.sum(w_scan * rel_dsdm)
    centered_w = rel_dsdm - mean_d_w
    bias_fcc = (-np.sum(w_scan * r * centered_w)
                / np.sum(w_scan * centered_w ** 2))

    # 2-point inside primary FCC-ee scan window
    i161, i164 = list(SQRTS).index(161.0), list(SQRTS).index(164.0)
    dr = r[i164] - r[i161]
    dd = rel_dsdm[i164] - rel_dsdm[i161]
    bias_2pt = -dr / dd

    print("  m_W bias estimators (Δm_W, MeV):")
    print(f"    Uniform 4-pt        : {bias_unif*1e3:+.3f}")
    print(f"    FCC-ee weighted     : {bias_fcc*1e3:+.3f}")
    print(f"    Conservative 2-pt   : {bias_2pt*1e3:+.3f}")

    return {"uniform": bias_unif*1e3, "fcc": bias_fcc*1e3, "two_pt": bias_2pt*1e3}


print(f"MW_REF = {MW_REF} GeV (BFS Table 2)")
print(f"GW_REF = {GW_REF} GeV (BFS Table 2)")
print(f"SQRTS  = {list(SQRTS)} GeV")

r_correct = _bias_estimators(_mine(MW_REF),    "Current chain, MW = 80.377 (correct BFS ref)")
r_v2      = _bias_estimators(_mine(80.379),    "Current chain, MW = 80.379 (v2's setting; for comparison)")

print("\n--- Summary table ---")
print(f"  {'estimator':22s} {'80.377 [MeV]':>14s} {'80.379 [MeV]':>14s} {'shift':>10s}")
for k in ("uniform", "fcc", "two_pt"):
    print(f"  {k:22s} {r_correct[k]:>14.3f} {r_v2[k]:>14.3f} {r_correct[k]-r_v2[k]:>+10.3f}")
