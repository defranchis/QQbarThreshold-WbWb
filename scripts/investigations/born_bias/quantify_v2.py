"""Quantify the m_W bias from the anchored-Born residual — v2.

v1 was bogus: I cubic-splined BFS's 6 sparse Whizard values and the
spline oscillated between points, generating fictitious residuals up to
±1.5 % that swamped the real (~0.1 %) residual at the reference points.

v2: use only the actual BFS reference points where the residual is
well-defined. We have BFS Whizard at √s ∈ {155, 158, 161, 164, 167, 170}.
For an FCC-ee scan in [158, 167 GeV], the residual is bounded by what we
see AT these reference points (the anchor was constructed to interpolate
smoothly; between two adjacent BFS points the residual cannot exceed the
endpoint values by much unless the EFT-truncation error itself has
structure that the anchor's spline misses).

m_W bias estimator:
   Δm_W = − ⟨ r · (∂lnσ/∂m_W − ⟨∂lnσ/∂m_W⟩) ⟩
          / ⟨ (∂lnσ/∂m_W − ⟨∂lnσ/∂m_W⟩)² ⟩
where ⟨·⟩ is the sample mean over the relevant √s grid (here, BFS reference
points in the scan window).
"""
import numpy as np
from process.ww.xsec_calculator.bfs_eft import sigma_BFS_specific_munuud_pb

# Table 2 BFS reference points in the scan window
SQRTS = np.array([158.0, 161.0, 164.0, 167.0])
WHIZ  = np.array([61.67, 154.19, 303.00, 408.80])   # fb (BFS Table 2 Whizard col)

MW = 80.379
GW = 2.09201

# mine_anchored at the SAME √s
mine = np.array([1e3 * sigma_BFS_specific_munuud_pb(s_i**2, MW, GW,
                                                    order="N3/2LO",
                                                    apply_whizard_anchor=True)
                 for s_i in SQRTS])

r = mine / WHIZ - 1.0

print("BFS reference points (Table 2, scan window):")
print("  √s [GeV]   mine [fb]   BFS_Whiz [fb]   residual")
for x, m, w, rr in zip(SQRTS, mine, WHIZ, r):
    print(f"  {x:7.1f}   {m:9.3f}   {w:13.3f}    {rr*100:+.4f} %")

# Linear fit of r vs √s for visualisation
sref = 162.5
x = SQRTS - sref
A = np.vstack([np.ones_like(x), x]).T
(a, b), *_ = np.linalg.lstsq(A, r, rcond=None)
print(f"\nLinear fit r(√s) ≈ {a*100:+.4f} %  +  {b*100:+.4f} %/GeV × (√s - {sref})")
print(f"  Flat piece (lumi-absorbed): {a*100:+.4f} %")
print(f"  Slope piece (m_W bias):     {b*100:+.4f} %/GeV")

# m_W sensitivities at the same √s (relative, fb/GeV → fb/MeV)
DMW = 1e-3
sig_pl = np.array([1e3 * sigma_BFS_specific_munuud_pb(s_i**2, MW+DMW, GW,
                                                      order="N3/2LO",
                                                      apply_whizard_anchor=True)
                   for s_i in SQRTS])
sig_mi = np.array([1e3 * sigma_BFS_specific_munuud_pb(s_i**2, MW-DMW, GW,
                                                      order="N3/2LO",
                                                      apply_whizard_anchor=True)
                   for s_i in SQRTS])
dsigma_dmW = (sig_pl - sig_mi) / (2*DMW)  # fb/GeV
rel_dsdm = dsigma_dmW / mine               # 1/GeV
print("\n  √s [GeV]   dσ/dm_W [fb/GeV]   (1/σ) dσ/dm_W [1/GeV]")
for x, d, rd in zip(SQRTS, dsigma_dmW, rel_dsdm):
    print(f"  {x:7.1f}   {d:13.2f}    {rd:+.4f}")

mean_d = np.mean(rel_dsdm)
centered_d = rel_dsdm - mean_d
num = -np.mean(r * centered_d)
den = np.mean(centered_d ** 2)
bias_mW = num / den
print(f"\nm_W bias estimator (uniform weights, lumi profiled out):")
print(f"  Δm_W = {bias_mW*1e3:+.3f} MeV  (uniform weights across 4 ref points)")

# Now with FCC-ee-like weights: more lumi at 161 and 163 than at 158, 167.
# Approximate scan plan (5 ab^-1 spread): 1.5/1.5/1.0/0.5 weight at (158,161,164,167).
w_scan = np.array([1.5, 1.5, 1.0, 0.5])
w_scan = w_scan / w_scan.sum()

# Use weighted means
mean_d_w = np.sum(w_scan * rel_dsdm)
centered_d_w = rel_dsdm - mean_d_w
num_w = -np.sum(w_scan * r * centered_d_w)
den_w = np.sum(w_scan * centered_d_w**2)
bias_mW_w = num_w / den_w
print(f"  Δm_W = {bias_mW_w*1e3:+.3f} MeV  (FCC-ee-like weights)")

# Restrict to the heart of the scan: 161 and 164 only (the two BFS ref points
# that are inside the FCC-ee primary scan window 161-163)
SQRTS_h = np.array([161.0, 164.0])
r_h = r[1:3]
rel_dsdm_h = rel_dsdm[1:3]
# Two-point analysis: just take the slope of r vs the slope of dσ/dm_W
delta_r = r_h[1] - r_h[0]            # change in residual across [161, 164]
delta_lnsig_dmW = rel_dsdm_h[1] - rel_dsdm_h[0]
bias_2pt = -delta_r / delta_lnsig_dmW
print(f"\nTwo-point analysis (161 & 164 GeV only):")
print(f"  Δr = {delta_r*100:+.4f} %    Δ(1/σ dσ/dm_W) = {delta_lnsig_dmW:+.4f} GeV⁻¹")
print(f"  Δm_W ≈ {bias_2pt*1e3:+.3f} MeV")
