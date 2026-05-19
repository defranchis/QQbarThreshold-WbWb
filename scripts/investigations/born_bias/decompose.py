"""Decompose the Scenario B residual into per-component contributions.

For each √s and each Born component {σ^(0), σ^(1/2), σ^(1)_pot, σ^(3/2),a}:
  compute (mine_component_BRcorr - BFS_implied_component) / σ_total
i.e., the fractional residual contributed by each piece.

The BFS implied per-component values are not published, but we can deduce
them from the BFS Table 2 totals minus my own Table 1 (BR=False) values,
since Scenario A matches Table 1 to 4-5 digits. So:

  mine_X(BR=True)  vs  BFS_X(BR=True implied from Table-2 sum)

Strategy: write the Table 2 BFS total at each √s. Sum of my BR=True
components should equal this. The shortfall/surplus locates which
component's BR propagation is wrong.

Even simpler: vary the BR convention on each component one at a time
and see which choice makes Scenario B close to 4-5 digits.
"""
import numpy as np
from process.ww.xsec_calculator.bfs_eft import (
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
    gamma_W_LO,
    _BR_correction,
)

# Table 2 inputs and reference
SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
BFS_T2_TOTAL = np.array([30.54, 60.83, 154.44, 303.70, 409.30, 481.70])  # fb
MW = 80.379
GW = 2.09201

c_sq = _BR_correction(MW, GW)         # (Γ_W^(0)/Γ_W)²
c_lin = np.sqrt(c_sq)                  # Γ_W^(0)/Γ_W

print(f"c_sq = {c_sq:.6f},  c_lin = sqrt(c_sq) = {c_lin:.6f}")
print(f"At Table 2:  BR_corr_squared = {c_sq:.4f}  →  σ^(0), σ^(1)_pot, σ^(3/2),a multiplied by this")
print(f"             BR_corr_linear  = {c_lin:.4f}  →  σ^(1/2) currently multiplied by this in my code")

# Per-piece values WITHOUT any BR correction (these match BFS Table 1 to 4-5 digits)
def pieces_no_BR(s):
    s0 = sigma_LR0_specific_pb(s, MW, GW, apply_BR_correction=False)
    s12_LR, s12_RL = sigma_LR_RL_half_specific_pb(s, MW, GW, apply_BR_correction=False)
    sNLO_LR, sNLO_RL = sigma_LR_RL_NLO_potential_specific_pb(s, MW, GW, 0.0, apply_BR_correction=False)
    s32a_LR, s32a_RL = sigma_LR_RL_three_half_a_specific_pb(s, MW, GW, apply_BR_correction=False)
    return {
        "sigma_0":    s0,
        "sigma_12":   (s12_LR + s12_RL),
        "sigma_NLO":  (sNLO_LR + sNLO_RL),
        "sigma_32a":  (s32a_LR + s32a_RL),
    }

# Compute "specific" (= unpolarised / 4 since RL=0 at LO) = component / 4
# Then total in fb = ×1e3
def total(pieces, br_factor_per_component):
    """br_factor_per_component: dict mapping component -> BR factor to apply."""
    total = sum(pieces[k] * br_factor_per_component[k] for k in pieces) / 4.0
    return total * 1e3   # pb → fb

print("\n=== Current code (σ^(1/2) BR=linear, rest BR=squared) ===")
factors_cur = {"sigma_0": c_sq, "sigma_12": c_lin, "sigma_NLO": c_sq, "sigma_32a": c_sq}
print(f"{'√s':>5}  {'mine [fb]':>9}  {'BFS [fb]':>9}  {'residual':>10}")
for sq, bfs in zip(SQRTS, BFS_T2_TOTAL):
    p = pieces_no_BR(sq**2)
    m = total(p, factors_cur)
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:9.3f}  {bfs:9.3f}  {r*100:+8.4f}%")

# Hypothesis 1: σ^(1/2) BR should be squared too
print("\n=== Hypothesis 1: σ^(1/2) BR also squared ===")
factors_h1 = {"sigma_0": c_sq, "sigma_12": c_sq, "sigma_NLO": c_sq, "sigma_32a": c_sq}
print(f"{'√s':>5}  {'mine [fb]':>9}  {'BFS [fb]':>9}  {'residual':>10}")
for sq, bfs in zip(SQRTS, BFS_T2_TOTAL):
    p = pieces_no_BR(sq**2)
    m = total(p, factors_h1)
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:9.3f}  {bfs:9.3f}  {r*100:+8.4f}%")

# Hypothesis 2: σ^(1/2) BR = 1 (no BR factor)
print("\n=== Hypothesis 2: σ^(1/2) no BR correction ===")
factors_h2 = {"sigma_0": c_sq, "sigma_12": 1.0, "sigma_NLO": c_sq, "sigma_32a": c_sq}
print(f"{'√s':>5}  {'mine [fb]':>9}  {'BFS [fb]':>9}  {'residual':>10}")
for sq, bfs in zip(SQRTS, BFS_T2_TOTAL):
    p = pieces_no_BR(sq**2)
    m = total(p, factors_h2)
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:9.3f}  {bfs:9.3f}  {r*100:+8.4f}%")
