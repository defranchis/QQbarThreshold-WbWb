"""Back out the implied BR factor BFS uses on σ^(1/2), at each √s.

Solve:  total_BFS = (σ^(0) + σ^(1)_pot + σ^(3/2),a) × c_sq + σ^(1/2) × c_X
for c_X. If c_X is the same at every √s, that's BFS's convention.
"""
import numpy as np
from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
    _BR_correction,
)

SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
BFS_T2 = np.array([30.54, 60.83, 154.44, 303.70, 409.30, 481.70])  # fb
MW, GW = 80.379, 2.09201
c_sq = _BR_correction(MW, GW)
c_lin = np.sqrt(c_sq)

print(f"c_sq = {c_sq:.5f}  c_lin = {c_lin:.5f}")
print(f"{'√s':>5}  {'σ_0[fb]':>9}  {'σ_1/2[fb]':>10}  {'σ_1_pot':>9}  {'σ_3/2,a':>9}  "
      f"  {'σ_no_half[fb]':>14}  c_X(implied)  c_X/c_sq  c_X/c_lin")
for sq, bfs in zip(SQRTS, BFS_T2):
    s = sq**2
    s0 = sigma_LR0_specific_pb(s, MW, GW, apply_BR_correction=False)
    s12_LR, s12_RL = sigma_LR_RL_half_specific_pb(s, MW, GW, apply_BR_correction=False)
    sNLO_LR, sNLO_RL = sigma_LR_RL_NLO_potential_specific_pb(s, MW, GW, 0.0, apply_BR_correction=False)
    s32a_LR, s32a_RL = sigma_LR_RL_three_half_a_specific_pb(s, MW, GW, apply_BR_correction=False)
    # As specific channel (÷ 4 helicity avg), in fb
    sig0 = s0/4.0 * 1e3
    sig12 = (s12_LR + s12_RL)/4.0 * 1e3
    sigNLO = (sNLO_LR + sNLO_RL)/4.0 * 1e3
    sig32a = (s32a_LR + s32a_RL)/4.0 * 1e3
    sum_no_half = sig0 + sigNLO + sig32a
    # Solve: BFS = sum_no_half × c_sq + sig12 × c_X
    c_X = (bfs - sum_no_half * c_sq) / sig12
    print(f"  {sq:4.0f}  {sig0:9.3f}  {sig12:10.3f}  {sigNLO:9.3f}  {sig32a:9.3f}  "
          f"  {sum_no_half:14.3f}  {c_X:+.5f}    {c_X/c_sq:.4f}  {c_X/c_lin:.4f}")

print("\nIf c_X is √s-independent → BFS uses that convention for σ^(1/2).")
print("If c_X drifts → σ^(1/2) BR alone can't explain the residual; something else is involved.")
