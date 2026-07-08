"""Decompose the anchor-suppressed alpha_em response:
 - confirm morph WHIZARD sigma is alpha-blind
 - confirm anchor factor f = sigma_whiz/sigma_EFT scales as 1/alpha^2 (so f*sigma_LR
   is alpha-independent at Born, leaving only the loop K-factor residual)
 - show the residual ~0.12 response is the NLO/NNLO+decay loops, NOT Born.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np
from framework.process.ww.xsec_calculator.eft_xsec import alpha_Gmu, M_Z
from framework.process.ww.xsec_calculator import bfs_eft as B
from framework.process.ww.xsec_calculator.grid_morph import whizard_sigma_morph

MW, GW = 80.379, 2.085
a0 = alpha_Gmu(MW, M_Z)
a1 = 1.01 * a0
SQRTS = [157.0, 160.0, 163.0]

print("1) morph WHIZARD sigma is alpha-blind (no alpha arg) — values fb:")
for rs in SQRTS:
    s = rs*rs
    print(f"   sqrt(s)={rs}: sigma_whiz = {whizard_sigma_morph(s, MW, GW):.6f} fb")
print()

print("2) anchor factor f = sigma_whiz/sigma_EFT — does it scale ~1/alpha^2?")
print(f"   {'sqrt(s)':>8} {'f(None)':>12} {'f(+1%)':>12} {'f ratio':>10} {'(a0/a1)^2':>10}")
for rs in SQRTS:
    s = rs*rs
    f0 = B.whizard_anchor_factor(s, MW, GW, apply_BR_correction=False,
                                 source="morph", alpha_em=None)
    f1 = B.whizard_anchor_factor(s, MW, GW, apply_BR_correction=False,
                                 source="morph", alpha_em=a1)
    print(f"   {rs:8.1f} {f0:12.5f} {f1:12.5f} {f1/f0:10.5f} {(a0/a1)**2:10.5f}")
print("   -> f ratio ~ (a0/a1)^2 = 0.98030 confirms f carries 1/alpha^2;")
print("      f*sigma_LR_Born is therefore alpha-INDEPENDENT (Born cancels).")
print()

print("3) where the residual response comes from: total WW sigma, anchor ON,")
print("   Born-only vs +loops (response = dsig/sig / 0.01):")
def tot(s, a, nlo, nnlo, dqcd):
    return B.sigma_BFS_LO_total_WW_pb(
        s, MW, GW, order="N3/2LO", apply_BR_correction=False,
        include_NLO_hard_decay=nlo, include_BFS_NNLO=nnlo,
        apply_delta_QCD=dqcd, apply_whizard_anchor=True,
        whizard_anchor_source="morph", alpha_em=a)
for label, nlo, nnlo, dqcd in [
        ("Born only        ", False, False, False),
        ("+NLO loops        ", True,  False, False),
        ("+NLO+NNLO         ", True,  True,  False),
        ("+NLO+NNLO+dQCD    ", True,  True,  True)]:
    row = []
    for rs in SQRTS:
        s = rs*rs
        s0 = tot(s, None, nlo, nnlo, dqcd)
        s1 = tot(s, a1,   nlo, nnlo, dqcd)
        row.append((s1 - s0)/s0/0.01)
    print(f"   {label} response@{SQRTS}: " + "  ".join(f"{r:+.4f}" for r in row))
