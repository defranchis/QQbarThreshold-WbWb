"""Does dropping σ^(1/2) single-resonant pieces shift our dσ/dΓ_W crossing?
YFSWW3 v1.18 has only CC03 doubly-resonant; modern 4f codes have single-res too.
σ^(1/2) in BFS contains the h4-h7 single-resonant diagrams."""
import numpy as np
from process.ww.bfs_eft import (
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
)

mW, gW = 80.385, 2.085
DGW = 1e-3
sqrts = np.array([157.0, 160.0, 161.0, 162.3, 163.0, 165.0, 167.0, 170.0])
s = sqrts ** 2


def sigma_WW(s, mW_in, gW_in, *, include_half=True, include_NLO_pot=True,
              include_32a=True):
    """Build σ_WW total from selected BFS Born components."""
    sLR = sigma_LR0_specific_pb(s, mW_in, gW_in, apply_BR_correction=False)
    sRL = np.zeros_like(np.asarray(sLR, dtype=float))
    if include_half:
        a, b = sigma_LR_RL_half_specific_pb(s, mW_in, gW_in, apply_BR_correction=False)
        sLR = sLR + a; sRL = sRL + b
    if include_NLO_pot:
        a, b = sigma_LR_RL_NLO_potential_specific_pb(s, mW_in, gW_in, 0.0,
                                                      apply_BR_correction=False)
        sLR = sLR + a; sRL = sRL + b
    if include_32a:
        a, b = sigma_LR_RL_three_half_a_specific_pb(s, mW_in, gW_in,
                                                     apply_BR_correction=False)
        sLR = sLR + a; sRL = sRL + b
    return (sLR + sRL) * 27.0 / 4.0   # specific → total WW


configs = [
    ("σ^(0) ONLY (pure doubly-resonant CC03-like)",
     dict(include_half=False, include_NLO_pot=False, include_32a=False)),
    ("σ^(0) + σ^(1)_pot (doubly-resonant + NLO potential)",
     dict(include_half=False, include_NLO_pot=True, include_32a=False)),
    ("σ^(0) + σ^(1/2) (CC03 + h1-h7 single-resonant)",
     dict(include_half=True, include_NLO_pot=False, include_32a=False)),
    ("FULL N^(3/2)LO Born (all pieces)",
     dict(include_half=True, include_NLO_pot=True, include_32a=True)),
]

# Extended grid for crossing detection
sqrts_fine = np.linspace(157, 180, 92)
s_fine = sqrts_fine ** 2

print(f"=== dσ_WW/dΓ_W [pb/GeV] vs BFS Born components ===\n")
print(f"{'config':<58}", end="")
for sq in sqrts:
    print(f"{sq:>7.1f}", end="")
print("  crossing√s")
print("-" * (58 + 7*len(sqrts)))

for label, kw in configs:
    sp = sigma_WW(s, mW, gW+DGW, **kw)
    sm = sigma_WW(s, mW, gW-DGW, **kw)
    ds = (sp - sm) / (2*DGW)
    # Find crossing on fine grid
    sp_fine = sigma_WW(s_fine, mW, gW+DGW, **kw)
    sm_fine = sigma_WW(s_fine, mW, gW-DGW, **kw)
    ds_fine = (sp_fine - sm_fine) / (2*DGW)
    idx = np.where(np.diff(np.sign(ds_fine)))[0]
    if len(idx) > 0:
        i = idx[0]
        x0 = sqrts_fine[i] - ds_fine[i] * (sqrts_fine[i+1]-sqrts_fine[i]) / (ds_fine[i+1]-ds_fine[i])
        cross = f"{x0:>6.2f}"
    else:
        cross = "  >180"
    print(f"{label:<58}", end="")
    for v in ds:
        print(f"{v:>+7.2f}", end="")
    print(f"   {cross}")
