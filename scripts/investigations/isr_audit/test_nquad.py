"""Convergence test of 2-leg with increasing n_quad."""
import numpy as np
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

for n in [16, 32, 64, 128, 256]:
    sig = sigma_observed_munuqq(np.array([161.0]),
        mW=80.377, gammaW=2.09201,
        channel="munuud", br_convention="bfs-eft",
        include_coulomb=False, include_NLO_hard_decay=False,
        apply_delta_QCD=False, apply_whizard_anchor=True,
        isr_scheme="2leg",
        n_quad=n)[0] * 1e3
    print(f"  n_quad = {n:3d}: σ = {sig:.4f} fb  (BFS 108.600)")
