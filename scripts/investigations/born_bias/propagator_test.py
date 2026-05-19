"""Test hypothesis: BFS Table 2 uses Γ_W^(0) in the propagator,
not the full Γ_W^NLO.

Strategy: compute σ_total at Table 2 inputs but pass gammaW = Γ_W^(0)(80.379)
to the partonic functions (which puts Γ_W^(0) in propagators), while applying
the BR correction with the full Γ_W = 2.09201 to convert from LO BR (1/27)
to the NLO BR.
"""
import numpy as np
from process.ww.bfs_eft import (
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
    gamma_W_LO,
)

SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
BFS_T2 = np.array([30.54, 60.83, 154.44, 303.70, 409.30, 481.70])  # fb

MW = 80.379
GW_FULL = 2.09201
GW_LO = gamma_W_LO(MW)  # = 2.04500

c_sq_full = (GW_LO / GW_FULL) ** 2
c_lin_full = GW_LO / GW_FULL

print(f"Γ_W^(0)(80.379) = {GW_LO:.5f} GeV")
print(f"Γ_W^NLO         = {GW_FULL:.5f} GeV")
print(f"c_sq (full)     = {c_sq_full:.5f}")
print(f"c_lin (full)    = {c_lin_full:.5f}")

# Hypothesis A (current code): propagator uses GW_FULL
# Hypothesis B (BFS prescription if "fixed-width with Γ_W^(0)"): propagator uses GW_LO

def piece_eval(gw_in_propagator, sqrts):
    rows = []
    for sq in sqrts:
        s = sq ** 2
        s0 = sigma_LR0_specific_pb(s, MW, gw_in_propagator, apply_BR_correction=False)
        s12_LR, s12_RL = sigma_LR_RL_half_specific_pb(s, MW, gw_in_propagator, apply_BR_correction=False)
        sNLO_LR, sNLO_RL = sigma_LR_RL_NLO_potential_specific_pb(s, MW, gw_in_propagator, 0.0, apply_BR_correction=False)
        s32a_LR, s32a_RL = sigma_LR_RL_three_half_a_specific_pb(s, MW, gw_in_propagator, apply_BR_correction=False)
        sig0 = s0/4.0 * 1e3
        sig12 = (s12_LR + s12_RL)/4.0 * 1e3
        sigNLO = (sNLO_LR + sNLO_RL)/4.0 * 1e3
        sig32a = (s32a_LR + s32a_RL)/4.0 * 1e3
        rows.append((sig0, sig12, sigNLO, sig32a))
    return rows

print("\n=== Hyp A: propagator uses Γ_W^NLO=2.09201, BR = c_sq for doubly-res, c_lin for σ^(1/2) ===")
print(f"{'√s':>5}  {'mine':>8}  {'BFS':>8}  {'resid':>8}")
rows_A = piece_eval(GW_FULL, SQRTS)
for (sq, bfs), (s0, s12, sNLO, s32a) in zip(zip(SQRTS, BFS_T2), rows_A):
    m = (s0 + sNLO + s32a) * c_sq_full + s12 * c_lin_full
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:8.3f}  {bfs:8.3f}  {r*100:+7.4f}%")

print("\n=== Hyp B: propagator uses Γ_W^(0)=2.04500, BR = c_sq for doubly-res, c_lin for σ^(1/2) ===")
rows_B = piece_eval(GW_LO, SQRTS)
print(f"{'√s':>5}  {'mine':>8}  {'BFS':>8}  {'resid':>8}")
for (sq, bfs), (s0, s12, sNLO, s32a) in zip(zip(SQRTS, BFS_T2), rows_B):
    m = (s0 + sNLO + s32a) * c_sq_full + s12 * c_lin_full
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:8.3f}  {bfs:8.3f}  {r*100:+7.4f}%")

print("\n=== Hyp C: propagator uses Γ_W^(0)=2.04500, BR = c_sq for ALL (including σ^(1/2)) ===")
print(f"{'√s':>5}  {'mine':>8}  {'BFS':>8}  {'resid':>8}")
for (sq, bfs), (s0, s12, sNLO, s32a) in zip(zip(SQRTS, BFS_T2), rows_B):
    m = (s0 + s12 + sNLO + s32a) * c_sq_full
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:8.3f}  {bfs:8.3f}  {r*100:+7.4f}%")

print("\n=== Hyp D: propagator uses Γ_W^(0), no BR factor applied at all ===")
print(f"{'√s':>5}  {'mine':>8}  {'BFS':>8}  {'resid':>8}")
for (sq, bfs), (s0, s12, sNLO, s32a) in zip(zip(SQRTS, BFS_T2), rows_B):
    m = s0 + s12 + sNLO + s32a
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {m:8.3f}  {bfs:8.3f}  {r*100:+7.4f}%")
