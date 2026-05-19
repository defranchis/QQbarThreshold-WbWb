"""Extract dσ_WW/dΓ_W from BFS Whizard 4f Born reference values
(Tables 1+2 of arXiv:0707.0773). These are the GOLD STANDARD — full 4f
matrix element from Whizard, no truncation. They give us an honest
ΔΓ_W = 47 MeV finite-difference reference."""
import numpy as np

# BFS Table 1: m_W = 80.377, Γ_W = 2.04483 GeV  (Whizard exact 4f Born, fb)
SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
WHIZ_T1 = np.array([34.43, 63.39, 160.62, 318.30, 428.60, 505.10])
mW_T1, gW_T1 = 80.377, 2.04483

# BFS Table 2: m_W = 80.379, Γ_W = 2.09201 GeV (Whizard exact 4f Born, fb)
# Note: Table 2 specific σ has BR_correction = (Γ_W^(0)/Γ_W)² ≈ 0.955 baked in.
WHIZ_T2 = np.array([33.58, 61.67, 154.19, 303.00, 408.80, 481.70])
mW_T2, gW_T2 = 80.379, 2.09201

# Strip BR factor to get σ_WW total (no BR factor, all 4f channels summed):
# Table 1 uses Γ_W^(0)(80.377)/2.04483 = 1.0 ⇒ BR_corr = 1.
# Table 2 uses Γ_W^(0)(80.379)/2.09201 ≈ 0.97753 ⇒ BR_corr = 0.95556.
sigma_T1_total = WHIZ_T1 * 27.0   # fb (sigma_WW total = sigma_specific × 27)
sigma_T2_total = WHIZ_T2 * 27.0 / 0.95556

print("=== σ_WW total (Whizard 4f Born from BFS Tables) ===")
print(f"{'√s':>5}  {'T1 (Γ_W=2.045)':>16}  {'T2 (Γ_W=2.092)':>16}  {'Δσ [fb]':>9}  {'dσ/dΓ_W [pb/GeV]':>18}")
delta_gw = gW_T2 - gW_T1
for sq, s1, s2 in zip(SQRTS, sigma_T1_total, sigma_T2_total):
    dsg = (s2 - s1) / delta_gw / 1e3   # fb/GeV → pb/GeV
    print(f"  {sq:4.0f}  {s1:14.1f}    {s2:14.1f}    {s2-s1:+8.1f}    {dsg:+15.4f}")

print(f"\nm_W shift between tables: {mW_T2-mW_T1:+.3f} GeV (2 MeV) — small effect on dσ/dΓ_W.")
print("CROSSING in Whizard data: between 161 and 164 GeV (sign change).")
print("This is the GOLD STANDARD reference — full 4f matrix element, no truncation.")
print("Azzurri's claim of crossing at 162.3 GeV is CONSISTENT with this.")

# Now our chain at the SAME inputs
print(f"\n=== Our chain at the same inputs (BFS Tables 1+2 conditions) ===")
from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb

# Two columns: σ_WW_total computed by our chain at Table 1 / Table 2 inputs.
# Use FULL chain (NLO+anchor+δ_QCD).
sigma_T1_mine = sigma_BFS_LO_total_WW_pb(
    SQRTS**2, mW_T1, gW_T1,
    order="N3/2LO", apply_BR_correction=False,
    include_NLO_hard_decay=True, apply_delta_QCD=True,
    apply_whizard_anchor=True)
sigma_T2_mine = sigma_BFS_LO_total_WW_pb(
    SQRTS**2, mW_T2, gW_T2,
    order="N3/2LO", apply_BR_correction=False,
    include_NLO_hard_decay=True, apply_delta_QCD=True,
    apply_whizard_anchor=True)

print(f"{'√s':>5}  {'mine T1':>10}  {'mine T2':>10}  {'mine dσ/dΓ_W':>14}  {'whiz dσ/dΓ_W':>14}")
for sq, s1m, s2m, s1w, s2w in zip(SQRTS, sigma_T1_mine, sigma_T2_mine,
                                    sigma_T1_total/1e3, sigma_T2_total/1e3):
    dsg_mine = (s2m - s1m) / delta_gw
    dsg_whiz = (s2w - s1w) / delta_gw
    print(f"  {sq:4.0f}  {s1m:9.4f}   {s2m:9.4f}   {dsg_mine:+13.4f}   {dsg_whiz:+13.4f}")
