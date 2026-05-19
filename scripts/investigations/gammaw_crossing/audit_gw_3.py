"""Check dσ/dΓ_W at successive BFS Born orders to isolate where the
crossing comes from. CC03 ≈ BFS LO σ^(0) alone (pure doubly-resonant)."""
import numpy as np
from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb

mW, gW = 80.385, 2.085
DGW = 1e-3
sqrts = np.array([157.0, 160.0, 161.0, 162.3, 163.0, 164.0, 165.0, 167.0, 170.0, 173.0])
s = sqrts ** 2

orders = ["LO", "N1/2LO", "NLO", "N3/2LO"]
print(f"BFS Born orders, σ_WW partonic [pb], no ISR, no NLO loops, no anchor:\n")
print(f"{'order':<10}", end="")
for sq in sqrts:
    print(f"{sq:>7.1f}", end="")
print()
print("-" * (10 + 7*len(sqrts)) + "\n=== σ_WW [pb] ===")
for o in orders:
    sw = sigma_BFS_LO_total_WW_pb(
        s, mW, gW, order=o, apply_BR_correction=False,
        include_NLO_hard_decay=False, apply_delta_QCD=False,
        apply_whizard_anchor=False)
    print(f"{o:<10}", end="")
    for v in sw:
        print(f"{v:>7.3f}", end="")
    print()
print("\n=== dσ_WW/dΓ_W [pb/GeV] ===")
for o in orders:
    sp = sigma_BFS_LO_total_WW_pb(
        s, mW, gW+DGW, order=o, apply_BR_correction=False,
        include_NLO_hard_decay=False, apply_delta_QCD=False,
        apply_whizard_anchor=False)
    sm = sigma_BFS_LO_total_WW_pb(
        s, mW, gW-DGW, order=o, apply_BR_correction=False,
        include_NLO_hard_decay=False, apply_delta_QCD=False,
        apply_whizard_anchor=False)
    ds = (sp - sm) / (2*DGW)
    print(f"{o:<10}", end="")
    for v in ds:
        print(f"{v:>+7.3f}", end="")
    print()

print("\nCROSSING (interpolated location where dσ/dΓ_W = 0):")
for o in orders:
    sp = sigma_BFS_LO_total_WW_pb(np.linspace(160, 180, 201)**2, mW, gW+DGW,
        order=o, apply_BR_correction=False,
        include_NLO_hard_decay=False, apply_delta_QCD=False,
        apply_whizard_anchor=False)
    sm = sigma_BFS_LO_total_WW_pb(np.linspace(160, 180, 201)**2, mW, gW-DGW,
        order=o, apply_BR_correction=False,
        include_NLO_hard_decay=False, apply_delta_QCD=False,
        apply_whizard_anchor=False)
    ds = (sp - sm) / (2*DGW)
    sqrts_grid = np.linspace(160, 180, 201)
    sign_change = np.where(np.diff(np.sign(ds)))[0]
    if len(sign_change) > 0:
        idx = sign_change[0]
        x0 = sqrts_grid[idx] - ds[idx] * (sqrts_grid[idx+1] - sqrts_grid[idx]) / (ds[idx+1] - ds[idx])
        print(f"  {o:<10}  crossing at √s ≈ {x0:.2f} GeV  (Azzurri quotes 162.3 GeV)")
    else:
        print(f"  {o:<10}  no crossing in [160, 180] GeV")
