"""Compare absolute σ_WW(total) and dσ/dΓ_W to Azzurri's Fig. 1 directly."""
import numpy as np
from process.ww.xsec_calculator.eft_xsec import sigma_WW_partonic, BR_INCLUSIVE_MUNUQQ, sigma_partonic_munuqq
from process.ww.xsec_calculator.isr import sigma_observed_munuqq

mW, gW = 80.385, 2.085   # Azzurri central
DGW = 1e-3

sqrts = np.array([157.0, 160.0, 161.0, 162.3, 163.0, 165.0, 167.0, 170.0, 173.0])
s = sqrts ** 2

# σ_WW total — Azzurri's y-axis (Fig. 1)
sigma_WW_pb = sigma_WW_partonic(s, mW, gW)
print(f"=== Compare to Azzurri Fig. 1 (σ_WW in pb, full off-shell) ===")
print(f"{'√s':>5}  {'our σ_WW partonic [pb]':>20}  {'Azzurri Fig.1 eyeball [pb]':>26}")
azzurri_eyeball = {157: 1.0, 160: 1.5, 161: 2.0, 162.3: 3.5, 163: 4.5, 165: 7.0, 167: 9.0, 170: 10.5, 173: 11.0}
for sq, sw in zip(sqrts, sigma_WW_pb):
    az = azzurri_eyeball.get(float(sq), float('nan'))
    print(f"  {sq:4.1f}  {sw:18.4f}     {az:24.2f}")

# dσ_WW/dΓ_W (no BR, partonic — Azzurri's quantity)
sp = sigma_WW_partonic(s, mW, gW+DGW)
sm = sigma_WW_partonic(s, mW, gW-DGW)
dsig_dgw_pb_per_GeV = (sp - sm) / (2*DGW)
print(f"\n=== dσ_WW/dΓ_W partonic, no ISR (Azzurri equivalent at Fig. 1 zoomed) ===")
print(f"{'√s':>5}  {'dσ_WW/dΓ_W [pb/GeV]':>22}")
for sq, ds in zip(sqrts, dsig_dgw_pb_per_GeV):
    print(f"  {sq:4.1f}  {ds:+22.4f}")

# Now WITH ISR (σ_observed)
sigma_obs_pb = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW, channel="inclusive") / BR_INCLUSIVE_MUNUQQ
sigma_obs_pb_p = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW+DGW, channel="inclusive") / BR_INCLUSIVE_MUNUQQ
sigma_obs_pb_m = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW-DGW, channel="inclusive") / BR_INCLUSIVE_MUNUQQ
dsig_obs = (sigma_obs_pb_p - sigma_obs_pb_m) / (2*DGW)
print(f"\n=== σ_WW observed (with LL+exp ISR) — what we plot in our diagnostics ===")
print(f"{'√s':>5}  {'σ_obs [pb]':>12}  {'dσ_obs/dΓ_W [pb/GeV]':>22}")
for sq, sw, ds in zip(sqrts, sigma_obs_pb, dsig_obs):
    print(f"  {sq:4.1f}  {sw:10.4f}    {ds:+22.4f}")
