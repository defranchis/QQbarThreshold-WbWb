"""Audit dσ/dΓ_W vs √s under every toggle to find any Γ_W dependence
we may have missed vs Azzurri's expected zero-crossing at 162.3 GeV."""
import numpy as np
from process.ww.eft_xsec import sigma_partonic_munuqq

mW, gW = 80.385, 2.085   # Azzurri central values
DGW = 1e-3   # 1 MeV

sqrts = np.array([157.0, 159.0, 161.0, 162.3, 163.0, 165.0, 167.0, 170.0, 175.0, 180.0])
s = sqrts ** 2

def dsig_dgw(**kw):
    sp = sigma_partonic_munuqq(s, mW, gW+DGW, channel="inclusive", **kw)
    sm = sigma_partonic_munuqq(s, mW, gW-DGW, channel="inclusive", **kw)
    return (sp - sm) / (2*DGW)   # pb/GeV = fb/MeV

# All chain pieces enabled
labels = [
    ("FULL chain (anchor + NLO loops + δ_QCD + K_C + pdg-BR)", dict()),
    ("ANCHOR OFF (NLO loops + δ_QCD + K_C + pdg-BR)", dict(apply_whizard_anchor=False)),
    ("NLO loops OFF (anchor + δ_QCD + K_C + pdg-BR)", dict(include_NLO_hard_decay=False)),
    ("δ_QCD OFF (anchor + NLO loops + K_C + pdg-BR)", dict(apply_delta_QCD=False)),
    ("K_C OFF (anchor + NLO loops + δ_QCD + pdg-BR)", dict(include_coulomb=False)),
    ("BORN-only (no NLO loops, no δ_QCD, no anchor, K_C, pdg-BR)",
     dict(apply_whizard_anchor=False, include_NLO_hard_decay=False,
          apply_delta_QCD=False)),
    ("BORN-only, BR=bfs-eft (Γ_W²)", 
     dict(apply_whizard_anchor=False, include_NLO_hard_decay=False,
          apply_delta_QCD=False, br_convention="bfs-eft")),
    ("BORN-only, K_C OFF",
     dict(apply_whizard_anchor=False, include_NLO_hard_decay=False,
          apply_delta_QCD=False, include_coulomb=False)),
]

print(f"{'config':<58}", end="")
for sq in sqrts:
    print(f"{sq:>7.1f}", end="")
print()
print("-" * (58 + 7*len(sqrts)))
for label, kw in labels:
    vals = dsig_dgw(**kw) * 1e3   # pb/GeV → fb/MeV
    print(f"{label:<58}", end="")
    for v in vals:
        print(f"{v:>+7.1f}", end="")
    print()
print()
print("CROSSING = √s where dσ/dΓ_W changes sign.")
print("Azzurri Fig. 1: crossing at √s ≈ 162.3 GeV. Where is ours?")
