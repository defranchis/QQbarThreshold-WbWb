"""Back out simultaneously: (r_squared, r_linear) — the implied BR factors
for doubly-resonant pieces and σ^(1/2), assuming both differ from mine.

At each √s:
   BFS_T2 = σ_doubly × r_sq²  +  σ^(1/2) × r_lin

With two unknowns (r_sq, r_lin) need at least 2 √s to solve. Use 161 and 164
(scan-window center), then test at all 6 √s.
"""
import numpy as np
from process.ww.xsec_calculator.bfs_eft import (
    sigma_LR0_specific_pb, sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb, gamma_W_LO,
)
SQRTS = np.array([155.0, 158.0, 161.0, 164.0, 167.0, 170.0])
BFS_T2 = np.array([30.54, 60.83, 154.44, 303.70, 409.30, 481.70])
MW, GW = 80.379, 2.09201

# Per-piece values (specific channel, fb), no BR correction
def get_pieces():
    rows = []
    for sq in SQRTS:
        s = sq ** 2
        s0 = sigma_LR0_specific_pb(s, MW, GW, apply_BR_correction=False)
        s12 = sum(sigma_LR_RL_half_specific_pb(s, MW, GW, apply_BR_correction=False))
        sNLO = sum(sigma_LR_RL_NLO_potential_specific_pb(s, MW, GW, 0.0, apply_BR_correction=False))
        s32a = sum(sigma_LR_RL_three_half_a_specific_pb(s, MW, GW, apply_BR_correction=False))
        sig0 = s0/4 * 1e3; sig12 = s12/4 * 1e3
        sigNLO = sNLO/4 * 1e3; sig32a = s32a/4 * 1e3
        rows.append((sig0 + sigNLO + sig32a, sig12))
    return rows

pieces = get_pieces()

# Solve from 161 and 164 only
i1, i2 = 2, 3   # 161, 164
A_dr_1, A_12_1 = pieces[i1]
A_dr_2, A_12_2 = pieces[i2]
# r_sq² * A_dr + r_lin * A_12 = BFS
# 2 equations, 2 unknowns: x = r_sq², y = r_lin
import numpy.linalg as la
M = np.array([[A_dr_1, A_12_1], [A_dr_2, A_12_2]])
b = np.array([BFS_T2[i1], BFS_T2[i2]])
x, y = la.solve(M, b)
r_sq = np.sqrt(x)
r_lin = y
GW_implied_doubly = gamma_W_LO(MW)/r_sq
GW_implied_half = gamma_W_LO(MW)/r_lin
print(f"From 161 & 164 system:")
print(f"  r²(doubly-res) = {x:.6f}    r(doubly-res) = {r_sq:.6f}")
print(f"  r(σ^(1/2))     = {y:.6f}")
print(f"  Implied Γ_W in BR(doubly) = Γ_W^(0)/r_sq = {GW_implied_doubly:.5f} GeV  (mine 2.09201)")
print(f"  Implied Γ_W in BR(half)   = Γ_W^(0)/r_lin = {GW_implied_half:.5f} GeV  (mine 2.09201)")
print(f"  My values: r_sq² = {(gamma_W_LO(MW)/GW)**2:.6f}, r_lin = {gamma_W_LO(MW)/GW:.6f}")
print(f"  Difference vs mine: r_sq² off by {(x - (gamma_W_LO(MW)/GW)**2)*100:+.5f} %, "
      f"r_lin off by {(y - gamma_W_LO(MW)/GW)*100:+.5f} %")

# Apply derived BR factors at ALL √s
print(f"\nApplying (r_sq²={x:.5f}, r_lin={y:.5f}) to all 6 √s:")
print(f"{'√s':>5}  {'σ_doubly':>9}  {'σ_half':>8}  {'mine':>8}  {'BFS':>8}  {'resid':>8}")
for sq, bfs, (sd, sh) in zip(SQRTS, BFS_T2, pieces):
    m = sd*x + sh*y
    r = m/bfs - 1.0
    print(f"  {sq:4.0f}  {sd:9.3f}  {sh:8.3f}  {m:8.3f}  {bfs:8.3f}  {r*100:+8.5f}%")
print("\n→ If residual is uniformly small across all √s with this (r_sq, r_lin),")
print("   then the issue IS a uniform BR-factor difference. If residual still")
print("   shows √s structure, then σ^(3/2),b or higher-order EFT effects remain.")
