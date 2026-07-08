"""Quantify the (m_W, Γ_W) slope shift induced by the BFS decay
substitution knob.

If the decay substitution is mostly a normalisation shift on σ_obs,
the m_W bias from flipping the knob is sub-MeV: the fit absorbs the
~+0.2-0.9 % normalisation via BR/luminosity nuisances, and only the
*differential* shift in dσ/dm_W matters.

This script prints, at scan-relevant √s:
  σ_obs(m_W ± δ),  σ_obs(Γ_W ± δ)   — numerical centred differences
  Δ_swap(slope)                    — fraction of slope changed by the knob
For comparison with the per-Γ_W and per-m_W variation templates that
drive the morph.

USE: PYTHONPATH=. python3 scripts/investigations/c1fin_analytic/chain_decay_swap_slopes.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

MW0 = 80.3692
GW0 = 2.085
DM  = 0.010
DG  = 0.010
SCAN = (158.0, 161.0, 162.3, 164.0, 165.0, 167.0, 170.0)


def _sig(sq, mW, gW, swap_on):
    return float(sigma_observed_munuqq(
        sq,
        mW=mW, gammaW=gW, channel="inclusive",
        br_convention="pdg-constant",
        include_coulomb=False,
        include_NLO_hard_decay=True,
        include_BFS_NNLO=True,
        apply_delta_QCD=True,
        apply_whizard_anchor=True, whizard_anchor_source="grid",
        isr_scheme="single_conv",
        decay_uses_full_born=swap_on,
    ))


def main():
    print(f"Reference: m_W={MW0} GeV, Γ_W={GW0} GeV, ±δm={DM*1e3:.0f} MeV, "
          f"±δΓ={DG*1e3:.0f} MeV")
    print()
    print("σ_obs at reference (knob OFF vs ON):")
    print(f"  {'√s':>5}  {'σ OFF (pb)':>12}  {'σ ON (pb)':>12}  "
          f"{'Δσ (fb)':>10}  {'Δσ/σ %':>9}")
    for sq in SCAN:
        s_off = _sig(sq, MW0, GW0, False)
        s_on  = _sig(sq, MW0, GW0, True)
        d = s_on - s_off
        print(f"  {sq:>5.1f}  {s_off:>12.6f}  {s_on:>12.6f}  "
              f"{d*1e3:>+10.4f}  {100*d/s_off:>+8.3f}%")

    print()
    print("Centred dσ/dm_W (pb/GeV) and dσ/dΓ_W (pb/GeV):")
    print(f"  {'√s':>5}  {'dσ/dmW OFF':>12}  {'dσ/dmW ON':>12}  {'Δslope %':>10}"
          f"  {'dσ/dΓW OFF':>12}  {'dσ/dΓW ON':>12}  {'Δslope %':>10}")
    for sq in SCAN:
        # m_W slope, knob OFF/ON
        dm_off = (_sig(sq, MW0+DM, GW0, False) - _sig(sq, MW0-DM, GW0, False)) / (2*DM)
        dm_on  = (_sig(sq, MW0+DM, GW0, True)  - _sig(sq, MW0-DM, GW0, True))  / (2*DM)
        # Γ_W slope, knob OFF/ON
        dg_off = (_sig(sq, MW0, GW0+DG, False) - _sig(sq, MW0, GW0-DG, False)) / (2*DG)
        dg_on  = (_sig(sq, MW0, GW0+DG, True)  - _sig(sq, MW0, GW0-DG, True))  / (2*DG)
        d_dm_pct = 100.0 * (dm_on - dm_off) / dm_off if dm_off else float('nan')
        d_dg_pct = 100.0 * (dg_on - dg_off) / dg_off if dg_off else float('nan')
        print(f"  {sq:>5.1f}  {dm_off:>12.5f}  {dm_on:>12.5f}  {d_dm_pct:>+9.3f}%"
              f"  {dg_off:>12.5f}  {dg_on:>12.5f}  {d_dg_pct:>+9.3f}%")

    print()
    print("Approx Δm_W bias from flipping the knob (∂σ/∂m_W and Δσ both centered):")
    print(f"  At each √s:  Δm_W ≈ Δσ / (dσ/dm_W).  Pure-normalisation effect")
    print(f"  (m_W absorbed by BR nuisance) would land at ~0 MeV.")
    print(f"  {'√s':>5}  {'Δσ (pb)':>10}  {'dσ/dmW':>10}  {'Δm_W (MeV)':>12}")
    for sq in SCAN:
        s_off = _sig(sq, MW0, GW0, False)
        s_on  = _sig(sq, MW0, GW0, True)
        d = s_on - s_off
        dm_off = (_sig(sq, MW0+DM, GW0, False) - _sig(sq, MW0-DM, GW0, False)) / (2*DM)
        dmw_mev = 1000.0 * d / dm_off if dm_off else float('nan')
        print(f"  {sq:>5.1f}  {d:>+10.4f}  {dm_off:>+10.4f}  {dmw_mev:>+12.2f}")


if __name__ == "__main__":
    main()
