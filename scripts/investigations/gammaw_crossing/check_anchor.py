"""Verify our BORN+anchor reproduces Whizard at the BFS reference points."""
import numpy as np
from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb

SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
# Whizard 4f Born total σ_WW (no BR) from BFS Tables × 27 (stripping 1/27)
WHIZ_T1_total = np.array([34.43, 63.39, 160.62, 318.30, 428.60, 505.10]) * 27.0   # fb
WHIZ_T2_total = np.array([33.58, 61.67, 154.19, 303.00, 408.80, 481.70]) * 27.0   # fb (with BR_corr baked in!)

# Table 1 inputs: BR_corr trivially 1 (since Γ_W = Γ_W^(0))
mine_T1 = sigma_BFS_LO_total_WW_pb(
    SQRTS**2, 80.377, 2.04483, order="N3/2LO",
    apply_BR_correction=False,
    include_NLO_hard_decay=False, apply_delta_QCD=False,
    apply_whizard_anchor=True)
# Table 2 inputs
mine_T2_no_BR = sigma_BFS_LO_total_WW_pb(
    SQRTS**2, 80.379, 2.09201, order="N3/2LO",
    apply_BR_correction=False,   # ← apply_BR_correction=False for total σ_WW
    include_NLO_hard_decay=False, apply_delta_QCD=False,
    apply_whizard_anchor=True)
mine_T2_BR  = sigma_BFS_LO_total_WW_pb(
    SQRTS**2, 80.379, 2.09201, order="N3/2LO",
    apply_BR_correction=True,    # ← with BR correction
    include_NLO_hard_decay=False, apply_delta_QCD=False,
    apply_whizard_anchor=True)

print("All values in pb. Reference = Whizard 4f Born values from BFS Tables × 27.\n")
print(f"{'√s':>5}  {'mine T1':>10}  {'WhizT1':>10}  {'mine T2 (no BR)':>16}  {'mine T2 (BR=T)':>16}  {'WhizT2 raw':>11}")
for i, sq in enumerate(SQRTS):
    print(f"  {sq:4.0f}  {mine_T1[i]:10.4f}  {WHIZ_T1_total[i]/1000:10.4f}  "
          f"{mine_T2_no_BR[i]:14.4f}    {mine_T2_BR[i]:14.4f}    {WHIZ_T2_total[i]/1000:11.4f}")
print("\nNote: 'WhizT2 raw' includes BR_corr=(Γ_W^(0)/Γ_W)² baked in by BFS table convention.")
print("To compare to OUR σ_WW_total (no BR factor), should we be looking at  T2 raw / 0.955  or  T2 raw  ?")
print("Recall the Table 2 column header says 'with the NLO+QCD width' applied to the propagator AND")
print("BR replacement (Γ_W^(0))²/Γ_W² in the specific-channel 1/27 factor. For TOTAL σ_WW (sum over")
print("all 4f channels, BR=1), the BR factor disappears, so T2_total = T1_table_at_Γ=2.092 (different propagator).")
