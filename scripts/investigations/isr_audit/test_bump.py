"""Check if the 168-GeV bump is a Whizard-anchor spline artifact."""
import numpy as np
from framework.process.ww.xsec_calculator.bfs_eft import whizard_anchor_factor, sigma_BFS_specific_munuud_pb
from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq

# Fine grid through the suspect region
sqrts = np.linspace(155.0, 170.0, 151)   # 0.1 GeV pitch

# 1. Look at the anchor factor f(δ, Γ_W) directly
mW, gW = 80.379, 2.085
f = np.array([whizard_anchor_factor(sq**2, mW, gW) for sq in sqrts])

# 2. Look at σ_partonic with vs without anchor
sig_on = sigma_partonic_munuqq(sqrts**2, mW=mW, gammaW=gW,
                                channel="inclusive", apply_whizard_anchor=True)
sig_off = sigma_partonic_munuqq(sqrts**2, mW=mW, gammaW=gW,
                                channel="inclusive", apply_whizard_anchor=False)

# Print f and σ ratios at suspect √s values
print(f"{'√s':>6}  {'f(δ,Γ_W)':>10}  {'σ_on/σ_off':>11}  {'1st diff f':>12}")
for i, sq in enumerate(sqrts):
    if sq < 165.5 or sq > 170.0:
        continue
    df = (f[i] - f[i-1]) / 0.1 if i > 0 else 0.0
    ratio = sig_on[i] / sig_off[i] if sig_off[i] > 0 else 0.0
    print(f"  {sq:5.1f}  {f[i]:10.6f}   {ratio:10.6f}  {df:+11.6f} /GeV")

print("\nBFS reference √s for the anchor spline: 155, 158, 161, 164, 167, 170 GeV")
print("Look for non-monotonic d f/d√s between 167 and 170 → cubic-spline overshoot.")
