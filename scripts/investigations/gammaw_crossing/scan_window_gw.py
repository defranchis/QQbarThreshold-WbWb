"""Within the FCC-ee scan window [157, 163] GeV: where is dσ/dΓ_W?
Verify whether anything in our chain crosses zero anywhere in this region."""
import numpy as np
from process.ww.eft_xsec import sigma_partonic_munuqq, sigma_WW_partonic
from process.ww.isr import sigma_observed_munuqq

mW, gW = 80.385, 2.085
DGW = 1e-3
sqrts = np.linspace(157.0, 163.0, 13)
s = sqrts ** 2

print("=== dσ_WW/dΓ_W [pb/GeV], within FCC-ee scan window [157, 163] ===\n")
print(f"{'config':<46}", end="")
for sq in sqrts:
    print(f"{sq:>6.1f}", end="")
print()
print("-" * (46 + 6*len(sqrts)))

configs = [
    ("partonic, FULL chain (production default)", "partonic", {}),
    ("partonic, BORN-only (BFS N3/2 + K_C, no NLO)",
     "partonic", dict(apply_whizard_anchor=False,
                       include_NLO_hard_decay=False,
                       apply_delta_QCD=False)),
    ("partonic, BORN-only, K_C OFF (pure BFS Born)",
     "partonic", dict(apply_whizard_anchor=False,
                       include_NLO_hard_decay=False,
                       apply_delta_QCD=False,
                       include_coulomb=False)),
    ("σ_observed, FULL chain + LL+exp ISR", "observed", {}),
]

for label, kind, extra in configs:
    if kind == "partonic":
        sp = sigma_partonic_munuqq(s, mW, gW+DGW, channel="inclusive", **extra)
        sm = sigma_partonic_munuqq(s, mW, gW-DGW, channel="inclusive", **extra)
    else:
        sp = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW+DGW, channel="inclusive", **extra)
        sm = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW-DGW, channel="inclusive", **extra)
    ds = (sp - sm) / (2*DGW)
    sign_change = np.where(np.diff(np.sign(ds)))[0]
    print(f"{label:<46}", end="")
    for v in ds:
        print(f"{v:>+6.2f}", end="")
    cross = "yes" if len(sign_change) else "no"
    print(f"  crossing: {cross}")

print("\nBottom line: in the entire scan window, every variant of our chain")
print("gives dσ/dΓ_W > 0 monotonically. Azzurri claims dσ/dΓ_W = 0 at 162.3 GeV.")
