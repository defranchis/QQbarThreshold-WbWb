"""Probe: response of the WW sigma chain to a +1% shift in the sigma-chain
electroweak coupling alpha_em (= alpha_Gmu), with/without the WHIZARD morph
anchor. Measures d(sigma)/sigma / (d(alpha)/alpha) at sqrt(s) = 157,160,163.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq, alpha_Gmu, M_Z,
)
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

MW = 80.379
GW = 2.085
DERIVED_ALPHA = alpha_Gmu(MW, M_Z)          # sigma-chain Born coupling, override=None path
SHIFT = 1.01
ALPHA_HI = SHIFT * DERIVED_ALPHA
SQRTS = [157.0, 160.0, 163.0]

print(f"derived alpha_Gmu(mW={MW}, MZ={M_Z}) = {DERIVED_ALPHA:.8e}")
print(f"shifted (+1%)                        = {ALPHA_HI:.8e}")
print()

def response_table(title, fn):
    print("=" * 78)
    print(title)
    print(f"{'sqrt(s)':>9} {'sigma(None)':>16} {'sigma(+1%)':>16} "
          f"{'dsig/sig':>11} {'response':>9}")
    for rs in SQRTS:
        s = rs * rs
        s0 = fn(s, None)
        s1 = fn(s, ALPHA_HI)
        dss = (s1 - s0) / s0
        resp = dss / (SHIFT - 1.0)
        print(f"{rs:9.1f} {s0:16.8e} {s1:16.8e} {dss:11.5f} {resp:9.4f}")
    print()

# (A) Partonic Born only, anchor OFF: expect response ~ +2 (sigma ~ alpha^2).
response_table(
    "A) PARTONIC Born only, anchor OFF (expect ~+2.0)",
    lambda s, a: sigma_partonic_munuqq(
        s, MW, GW, channel="inclusive", include_coulomb=False,
        include_NLO_hard_decay=False, include_BFS_NNLO=False,
        apply_delta_QCD=False, apply_whizard_anchor=False,
        whizard_anchor_source="morph", alpha_em=a),
)

# (B) Partonic full chain, anchor OFF (no WHIZARD anchor): expect ~+2 minus small K drift.
response_table(
    "B) PARTONIC full BFS chain (NLO+NNLO+dQCD), anchor OFF",
    lambda s, a: sigma_partonic_munuqq(
        s, MW, GW, channel="inclusive", include_coulomb=False,
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=False,
        whizard_anchor_source="morph", alpha_em=a),
)

# (C) Partonic full chain, morph anchor ON (PRODUCTION sigma chain, pre-ISR).
response_table(
    "C) PARTONIC full BFS chain, morph anchor ON (production, pre-ISR)",
    lambda s, a: sigma_partonic_munuqq(
        s, MW, GW, channel="inclusive", include_coulomb=False,
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True,
        whizard_anchor_source="morph", alpha_em=a),
)

# (D) FULL OBSERVED chain (ISR convolution) with morph anchor ON = production.
#     alpha_em (sigma chain) is overridden; alpha_em_isr held fixed (ISR coupling
#     is the separate nuisance) to isolate the production-coupling response.
def obs(s, a):
    return sigma_observed_munuqq(
        np.sqrt(s), MW, GW, channel="inclusive", include_coulomb=False,
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True,
        whizard_anchor_source="morph",
        isr_scheme="single_conv", isr_nll=True,
        isr_emela_ren_scheme="ALPMZ", isr_emela_fac_scheme="DELTA",
        alpha_em_isr=1.0 / 128.943,
        alpha_em=a)
response_table(
    "D) FULL OBSERVED chain (ISR + morph anchor ON) = PRODUCTION",
    obs,
)
