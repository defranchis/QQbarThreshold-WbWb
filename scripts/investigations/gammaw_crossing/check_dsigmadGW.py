"""Check the dσ/dΓ_W shape — looking for a bump near 168 GeV."""
import numpy as np
from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

mW, gW = 80.379, 2.085
DGW = 1e-3   # 1 MeV

sqrts = np.linspace(155.0, 175.0, 201)   # 0.1 GeV pitch
sig_pl = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW+DGW, channel="inclusive")
sig_mi = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW-DGW, channel="inclusive")
dsig_dgw = (sig_pl - sig_mi) / (2*DGW) * 1e3   # fb/MeV

# Look for non-smooth structure
print("dσ/dΓ_W vs √s (with current best chain, anchor ON):")
print(f"{'√s':>6}  {'dσ/dΓ_W [fb/MeV]':>17}  {'2nd diff':>10}")
for i in range(2, len(sqrts)-2):
    sq = sqrts[i]
    if not (164.0 <= sq <= 171.0):
        continue
    d2 = dsig_dgw[i+1] - 2*dsig_dgw[i] + dsig_dgw[i-1]   # 2nd diff
    print(f"  {sq:5.1f}  {dsig_dgw[i]:15.3f}    {d2:+10.4f}")

# Same w/o anchor for comparison
sig_pl_off = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW+DGW, channel="inclusive",
                                    apply_whizard_anchor=False)
sig_mi_off = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW-DGW, channel="inclusive",
                                    apply_whizard_anchor=False)
dsig_dgw_off = (sig_pl_off - sig_mi_off) / (2*DGW) * 1e3

print("\nFor comparison: dσ/dΓ_W with anchor OFF:")
print(f"{'√s':>6}  {'on [fb/MeV]':>13}  {'off [fb/MeV]':>13}  {'on-off':>10}")
for i in range(len(sqrts)):
    sq = sqrts[i]
    if not (164.0 <= sq <= 171.0) or sq*10 % 5 != 0:
        continue
    diff = dsig_dgw[i] - dsig_dgw_off[i]
    print(f"  {sq:5.1f}  {dsig_dgw[i]:12.3f}   {dsig_dgw_off[i]:12.3f}   {diff:+9.3f}")
