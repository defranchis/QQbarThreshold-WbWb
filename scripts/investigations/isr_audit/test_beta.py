"""Look at my β value vs BFS's β value at √s=161."""
import numpy as np
from process.ww.isr import beta_ISR
from process.ww.eft_xsec import alpha_Gmu, M_E

# At √s = 161 GeV
s = 161.0**2
L = np.log(s / M_E**2)
# BFS uses m_e = 0.51099892 MeV per page 34. My code: 0.5109989461 MeV.
# Negligible difference.
alpha_my = alpha_Gmu(80.377)
beta_my = (2*alpha_my/np.pi) * (L - 1)
print(f"M_E (mine) = {M_E*1e6:.4f} keV  (BFS: 0.51099892 MeV = 511.0 keV — essentially identical)")
print(f"α_Gμ(m_W=80.377) (mine) = {alpha_my:.8f}")
print(f"L = ln(s/m_e²) at √s=161: {L:.5f}")
print(f"β = (2α/π)(L-1) = {beta_my:.6f}")
print()

# What if BFS uses Γ_W in the L instead of √s? Try L = ln(Γ_W² / m_e²)
L_GW = np.log(2.04483**2 / M_E**2)
beta_GW = (2*alpha_my/np.pi) * (L_GW - 1)
print(f"If L = ln(Γ_W² / m_e²) = {L_GW:.5f}, β = {beta_GW:.6f} (smaller — would damp less)")

# What if BFS uses Q² = 4 M_W² instead of s?
L_4MW = np.log((2*80.377)**2 / M_E**2)
beta_4MW = (2*alpha_my/np.pi) * (L_4MW - 1)
print(f"If L = ln(4 M_W² / m_e²) = {L_4MW:.5f}, β = {beta_4MW:.6f} (essentially same as s, since s ≈ 4 M_W²)")

# Effect of NLL "+(α/(6π)) L" correction (LEP2 YR mentions virtual e+e- pair contributions
# can be partially accounted by replacing α → α[1 + αL/(6π)] in φ)
beta_NLL = (2 * alpha_my * (1 + alpha_my * L / (6*np.pi)) / np.pi) * (L - 1)
print(f"With NLL pair-conversion fix α → α(1+αL/(6π)): β = {beta_NLL:.6f}")
print(f"  (this is {(beta_NLL/beta_my - 1)*100:.3f} % larger than nominal β — would damp MORE)")
