"""Re-run of the 2107.04444 (Azzurri) Asimov cross-check against the
current production chain (item 5 of the open-issues list).

Uses the ``prod/`` templates built by ``build_templates.py`` so the fit
sees the up-to-date card (NLO loops + NNLO + δ_QCD-in-BR + morph anchor
+ LL+exp ISR + ALPMZ-scheme α_em_isr) without depending on whatever the
production ``output_xsec/ww/`` directory currently holds.

Target (after channel-scope √0.143 correction): Δm_W ≲ 0.6 MeV, ΔΓ_W ≲ 1.3 MeV,
plus a check that the dσ/dΓ_W zero now lives near 162.3 GeV → ρ(m_W,Γ_W) → 0.

USE: PYTHONPATH=. python3 scripts/investigations/old_new_AB/run_2107_followup.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from cards import ww_default as card
from framework.common.fit_core import ecm_to_str
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit


BASE = "output_xsec/ww_AB_2026-05-29/prod"

SQRTS_E1 = 157.1
SQRTS_E2 = 162.3
LUMI_FRACTION_E2 = 0.40
TOTAL_LUMI_INVPB = 12.0e6


def main():
    # Point card I/O at the fresh prod templates we just built.
    orig_dirs = dict(card.INPUT_DIRS)
    card.INPUT_DIRS["nominal"] = os.path.join(BASE, "nominal")
    card.INPUT_DIRS["BEC"]     = os.path.join(BASE, "BEC")
    card.INPUT_DIRS["pseudo"]  = os.path.join(BASE, "nominal")
    try:
        generator = WWGenerator.from_card(card)
        fit = WWFit(card, generator, asimov=True,
                    mass_scheme=getattr(card, "MASS_SCHEME", "OS"), debug=False)

        L2 = TOTAL_LUMI_INVPB * LUMI_FRACTION_E2
        L1 = TOTAL_LUMI_INVPB - L2
        scan_list = [ecm_to_str(SQRTS_E1), ecm_to_str(SQRTS_E2)]
        lumi_dict = {ecm_to_str(SQRTS_E1): L1, ecm_to_str(SQRTS_E2): L2}
        fit.init_scenario(
            scan_list=scan_list,
            total_lumi=TOTAL_LUMI_INVPB,
            last_lumi=card.SCENARIO["last_lumi"],
            lumi_dict=lumi_dict,
        )

        print("=" * 78)
        print("  2107.04444 §2.4 optimal scenario — current production chain")
        print("=" * 78)
        print(f"   Scan: E_1 = {SQRTS_E1} GeV   E_2 = {SQRTS_E2} GeV")
        print(f"   Lumi: L_1 = {L1/1e6:.2f} ab⁻¹   L_2 = {L2/1e6:.2f} ab⁻¹")
        print(f"   Chain: {generator.describe()}")
        print()

        fit.fit_parameters()
        res = fit.fit_results(printout=False)
        by_name = dict(zip(fit.param_names, res))
        mW = by_name["mass"]
        gW = by_name["width"]
        try:
            cov = fit.minuit.covariance
            i = list(fit.param_names).index("mass")
            j = list(fit.param_names).index("width")
            rho = float(cov[i, j] / (cov[i, i] ** 0.5 * cov[j, j] ** 0.5))
        except Exception:
            rho = float("nan")

        # μνqq̄ inclusive only; paper sums all WW decay channels.
        # Channel-scope correction: σ stat scales as 1/√N_events; paper has
        # 1/0.1433 ~ 7× more events → our σ shrinks by √0.1433 ≈ 0.378.
        ch_factor = np.sqrt(0.1433)

        print(f"   m_W stat (μνqq̄ only):       {mW.s*1e3:.3f} MeV")
        print(f"   ΔΓ_W stat (μνqq̄ only):      {gW.s*1e3:.3f} MeV")
        print(f"   ρ(m_W, Γ_W):                 {rho:+.3f}")
        print()
        print(f"   m_W stat × √BR_incl (all chans): {mW.s*ch_factor*1e3:.3f} MeV")
        print(f"   ΔΓ_W stat × √BR_incl :          {gW.s*ch_factor*1e3:.3f} MeV")
        print()
        print("   Paper projection (YFSWW3, all channels): Δm_W=0.5  ΔΓ_W=1.2 MeV")
        print(f"   Target (this followup): Δm_W ≲ 0.6 MeV, ΔΓ_W ≲ 1.3 MeV")
        print(f"   ρ → 0 check: paper sits near the dσ/dΓ_W=0 crossing at 162.3 GeV.")
        print()
        print(f"   ⇒ Δm_W (channel-corrected) =  {mW.s*ch_factor*1e3:.3f} MeV")
        print(f"      vs paper target          =  ≲ 0.6 MeV   "
              f"({'PASS' if mW.s*ch_factor*1e3 <= 0.6 else 'OVER'})")
        print(f"   ⇒ ΔΓ_W (channel-corrected) =  {gW.s*ch_factor*1e3:.3f} MeV")
        print(f"      vs paper target          =  ≲ 1.3 MeV   "
              f"({'PASS' if gW.s*ch_factor*1e3 <= 1.3 else 'OVER'})")
        print(f"   ⇒ ρ = {rho:+.3f}   (paper near 0 — value depends on dσ/dΓ_W slope @ 162.3)")
    finally:
        card.INPUT_DIRS.update(orig_dirs)


if __name__ == "__main__":
    main()
