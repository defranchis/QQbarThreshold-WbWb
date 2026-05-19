"""Test if the Whizard anchor is the source of the residual ~0.8 % gap.

If I disable the anchor, my Born partonic is BFS-EFT N^(3/2)LO directly.
BFS Table 3 Born column compares to Whizard 4f Born (which BFS Table 2
column 5 differs from BFS-EFT N^(3/2)LO by 0.5-3 %).

So mine-no-anchor vs BFS Table 3 should equal (mine BFS-EFT × ISR) vs
(BFS-Whizard × ISR). With the EFT vs Whizard partonic ratios from Table 2.
"""
import numpy as np
from process.ww.xsec_calculator.isr import sigma_observed_munuqq

BFS_Born_ISR = {158: 45.64, 161: 108.60, 164: 219.7, 167: 310.2, 170: 378.4}
sqrts = np.array(list(BFS_Born_ISR.keys()), dtype=float)

print("=== Born×ISR vs BFS Table 3, anchor on/off ===")
print(f"{'√s':>5}  {'BFS T3':>9}  {'anchor=on':>11}  {'anchor=off':>12}  {'on/BFS':>8}  {'off/BFS':>9}")

for sq in sqrts:
    bfs_val = BFS_Born_ISR[int(sq)]
    sig_on = sigma_observed_munuqq(np.array([sq]),
        mW=80.377, gammaW=2.09201,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True,
        isr_scheme="2leg")[0] * 1e3
    sig_off = sigma_observed_munuqq(np.array([sq]),
        mW=80.377, gammaW=2.09201,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=False,
        isr_scheme="2leg")[0] * 1e3
    print(f"  {sq:4.0f}  {bfs_val:8.3f}   {sig_on:10.3f}    {sig_off:11.3f}   {sig_on/bfs_val:.4f}   {sig_off/bfs_val:.4f}")

# Also compare to BFS Table 2's σ_Born ratios:
# Table 2 EFT N^(3/2)LO at Table 2 inputs (mW=80.379, Γ_W=2.09201):
# 155: 30.54, 158: 60.83, 161: 154.44, 164: 303.7, 167: 409.3, 170: 481.7
# Table 2 Whizard at same: 33.58, 61.67, 154.19, 303.0, 408.8, 481.7
# So at Born level (no ISR), EFT/Whizard ratio is:
BFS_Tab2_EFT_partonic = {158: 60.83, 161: 154.44, 164: 303.7, 167: 409.3, 170: 481.7}
BFS_Tab2_WHIZ_partonic = {158: 61.67, 161: 154.19, 164: 303.0, 167: 408.8, 170: 481.7}
print("\nPartonic EFT/Whizard ratio at Table 2 inputs (for reference):")
for sq in sqrts:
    e = BFS_Tab2_EFT_partonic[int(sq)]
    w = BFS_Tab2_WHIZ_partonic[int(sq)]
    print(f"  {sq:4.0f}  EFT={e:8.2f}  Whiz={w:8.2f}  EFT/Whiz={e/w:.5f}")
