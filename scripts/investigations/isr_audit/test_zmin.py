"""Test the BFS-style 155-GeV cutoff: z_min(s) = (155/√s)²."""
import numpy as np
from process.ww.isr import sigma_observed_munuqq, sigma_ISR_2leg_convolution
from process.ww.eft_xsec import sigma_partonic_munuqq

BFS_Born_ISR = {158: 45.64, 161: 108.60, 164: 219.7, 167: 310.2, 170: 378.4}
sqrts = np.array(list(BFS_Born_ISR.keys()), dtype=float)

print("=== Test BFS-style cutoff: √(x₁x₂s) ≥ 155 GeV ===")
print("Default z_min=0.10 (my code) vs z_min_BFS = (155/√s)² (BFS convention)")
print(f"{'√s':>5}  {'BFS Tab 3':>10}  {'z_min=0.10':>12}  {'z_min_BFS':>11}  {'ratio_BFS_cut':>14}")

for sq in sqrts:
    bfs_val = BFS_Born_ISR[int(sq)]
    # Default
    sig_default = sigma_observed_munuqq(np.array([sq]),
        mW=80.377, gammaW=2.09201,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True,
        isr_scheme="2leg",
        z_min=0.10)[0] * 1e3
    # BFS cutoff: x_min = 155/√s per leg (so √(x₁x₂s) ≥ 155 GeV means x₁x₂ ≥ (155/√s)²)
    # For symmetric per-leg cutoff: x_i ≥ 155/√s
    x_min_BFS = 155.0/sq
    z_min_BFS = x_min_BFS ** 2
    sig_bfs_cut = sigma_observed_munuqq(np.array([sq]),
        mW=80.377, gammaW=2.09201,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True,
        isr_scheme="2leg",
        z_min=z_min_BFS)[0] * 1e3
    print(f"  {sq:4.0f}  {bfs_val:9.3f}   {sig_default:11.3f}    {sig_bfs_cut:10.3f}    {sig_bfs_cut/bfs_val:.5f}")
