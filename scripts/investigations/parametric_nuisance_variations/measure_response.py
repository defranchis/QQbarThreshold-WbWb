"""Measure the relative σ-response to m_t, M_H, α_em_isr so we can size the
template variations for the new profiled nuisances.

The variation only needs to lift the σ-response comfortably above the morph
noise floor (~0.02 %) while staying linear. We measure the relative response
per natural unit, then report what variation gives a ~0.5 % response (≈25×
noise) and check linearity by comparing a small and a large step.

m_t / M_H enter the hard NLO matching c^(1,fin) → measured on the partonic σ
(no ISR, fast). α_em_isr enters the ISR β_e exponent → measured on the
observed σ with the fast analytic LL+exp ISR.
"""
import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq, M_T_DEFAULT, M_H_DEFAULT,
)
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

MW, GW = 80.379, 2.085
ECMS = np.array([160.0, 161.0, 162.0, 163.0])
S = ECMS ** 2

NOISE = 2e-4  # 0.02 % morph noise floor


def rel_response(sig_nom, sig_var):
    return (sig_var - sig_nom) / sig_nom


def partonic(m_t=M_T_DEFAULT, M_H=M_H_DEFAULT):
    return sigma_partonic_munuqq(S, mW=MW, gammaW=GW, m_t=m_t, M_H=M_H)


def observed_ll(alpha_em_isr):
    # analytic LL+exp ISR (isr_nll=False, isr_emela_ll=False) — fast
    return sigma_observed_munuqq(ECMS, mW=MW, gammaW=GW,
                                 alpha_em_isr=alpha_em_isr,
                                 isr_nll=False, isr_emela_ll=False)


def report(name, unit, base, dvals, fn, prior_in_unit):
    print(f"\n=== {name}  (unit: {unit}) ===")
    sig0 = fn(base)
    print(f"  nominal {name} = {base}")
    print(f"  {'Δ':>10s} {'rel.resp @each ecm (%)':>40s} {'per-unit slope (%/unit) @162':>30s}")
    slopes = []
    for d in dvals:
        sigd = fn(base + d)
        r = rel_response(sig0, sigd) * 100.0
        # per-unit slope at the 162 GeV point (index 2)
        slope = r[2] / d
        slopes.append(slope)
        rstr = " ".join(f"{x:+7.3f}" for x in r)
        print(f"  {d:>+10.4g} [{rstr}]   {slope:>+12.4f}")
    # linearity: slope should be ~constant across the steps
    s = np.array(slopes)
    print(f"  slope spread (max-min)/mean = {(s.max()-s.min())/abs(s.mean())*100:.2f}%  → linearity")
    slope162 = slopes[0]  # smallest-step slope, most linear
    # variation for a 0.5% response (25x noise) at 162 GeV:
    if abs(slope162) > 0:
        v_target = 0.5 / abs(slope162)
        resp_at_prior = abs(slope162) * prior_in_unit
        print(f"  → variation for 0.5% response: {v_target:.4g} {unit}")
        print(f"  → σ-response at the PRIOR ({prior_in_unit:g} {unit}) = "
              f"{resp_at_prior:.4f}%  (noise floor 0.02%)")
    return slopes


# m_t: enters c^(1,fin); probe ±0.25, ±1, ±2 GeV for linearity
report("m_t", "GeV", M_T_DEFAULT, [0.25, 1.0, 2.0], lambda v: partonic(m_t=v),
       prior_in_unit=0.007)

# M_H: log-weak; probe 0.5, 2, 5 GeV
report("M_H", "GeV", M_H_DEFAULT, [0.5, 2.0, 5.0], lambda v: partonic(M_H=v),
       prior_in_unit=0.005)

# α_em_isr: enters β_e linearly; probe 2e-5, 1e-4, 2e-4 abs
ALPHA_MZ = 1.0 / 128.943
report("alpha_em_isr", "abs-α", ALPHA_MZ, [2e-5, 1e-4, 2e-4],
       lambda v: observed_ll(v), prior_in_unit=2.4e-7)
