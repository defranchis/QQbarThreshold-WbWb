"""Apples-to-apples: σ_WW BORN ONLY (no NLO loops, no δ_QCD) — like Whizard.
If our anchored chain at BORN level matches Whizard dσ/dΓ_W, the issue is the
NLO loops adding spurious Γ_W dependence. If it still differs, the BFS Born
expansion's complex-velocity Γ_W treatment is the issue."""
import numpy as np
from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb

SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
gW_T1, gW_T2 = 2.04483, 2.09201
mW_T1, mW_T2 = 80.377, 80.379

# BFS Table Whizard 4f reference: σ_WW total (stripping BR factor)
WHIZ_T1 = np.array([34.43, 63.39, 160.62, 318.30, 428.60, 505.10]) * 27.0
WHIZ_T2 = np.array([33.58, 61.67, 154.19, 303.00, 408.80, 481.70]) * 27.0 / 0.95556
dsg_whiz = (WHIZ_T2 - WHIZ_T1) / (gW_T2 - gW_T1) / 1e3   # pb/GeV

configs = [
    ("FULL chain  (Born + NLO loops + δ_QCD + anchor)",
     dict(include_NLO_hard_decay=True, apply_delta_QCD=True,
          apply_whizard_anchor=True)),
    ("BORN + anchor (no NLO loops, no δ_QCD)",
     dict(include_NLO_hard_decay=False, apply_delta_QCD=False,
          apply_whizard_anchor=True)),
    ("BORN, no anchor (pure BFS Born N^(3/2)LO)",
     dict(include_NLO_hard_decay=False, apply_delta_QCD=False,
          apply_whizard_anchor=False)),
    ("BORN + NLO loops + δ_QCD (no anchor)",
     dict(include_NLO_hard_decay=True, apply_delta_QCD=True,
          apply_whizard_anchor=False)),
]

print(f"{'√s':>5}  {'Whiz 4f':>9}", end="")
for label, _ in configs:
    print(f"  {label[:38]:>38}", end="")
print()

for i, sq in enumerate(SQRTS):
    print(f"  {sq:4.0f}  {dsg_whiz[i]:+9.3f}", end="")
    for label, kw in configs:
        s1 = sigma_BFS_LO_total_WW_pb(np.array([sq**2]), mW_T1, gW_T1,
            order="N3/2LO", apply_BR_correction=False, **kw)[0]
        s2 = sigma_BFS_LO_total_WW_pb(np.array([sq**2]), mW_T2, gW_T2,
            order="N3/2LO", apply_BR_correction=False, **kw)[0]
        dsg = (s2 - s1) / (gW_T2 - gW_T1)
        sign_emoji = "✓" if (dsg * dsg_whiz[i] > 0) else "✗"
        print(f"  {dsg:+12.3f} pb/GeV {sign_emoji:>2}    ", end="")
    print()
