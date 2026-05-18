"""Cross-check fit using the 2107.04444 (Azzurri) optimal scenario.

Paper §2.4 / eq. (15)+(16): optimal 2-point scan for ``min(Δm_W+ΔΓ_W)``:
    E_1 = 157.1 GeV,  E_2 = 162.3 GeV,  f = 0.40 (lumi fraction at E_2)
    L_total = 12 ab⁻¹  →  L_1 = 7.2 ab⁻¹,  L_2 = 4.8 ab⁻¹
Quoted projection: Δm_W = 0.5 MeV, ΔΓ_W = 1.2 MeV (statistical only).

The paper uses YFSWW3 (full NLO EW + ISR + Coulomb + …) and counts
events in *all* WW decay channels (BR multiplier ≈ 1).

Our framework only does the **μνqq̄ inclusive** channel (BR ≈ 0.143)
with **BFS LO_EFT Born + Coulomb + LL+YFS ISR**. Apples-to-oranges,
but a useful shape and sensitivity cross-check.

Run from WW_threshold/:
    python3 -m scripts.fit_2107_04444_scenario
"""

from __future__ import annotations

import numpy as np

from cards import ww_default as card
from common.fit_core import ecm_to_str
from process.ww.fit import WWFit
from process.ww.generator import WWGenerator


# 2107.04444 §2.4 optimal scenario
SQRTS_E1 = 157.1   # GeV
SQRTS_E2 = 162.3   # GeV
LUMI_FRACTION_E2 = 0.40
TOTAL_LUMI_INVPB = 12.0e6   # = 12 ab⁻¹


def main():
    generator = WWGenerator(order=card.ORDER)
    fit = WWFit(card, generator, asimov=True, mass_scheme=card.MASS_SCHEME, debug=False)

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
    print(" 2107.04444 §2.4 optimal scenario applied to BFS-LO_EFT framework")
    print("=" * 78)
    print(f"  Scan points:  E_1 = {SQRTS_E1} GeV   E_2 = {SQRTS_E2} GeV")
    print(f"  Luminosities: L_1 = {L1/1e6:.2f} ab⁻¹   L_2 = {L2/1e6:.2f} ab⁻¹")
    print(f"  Channel:      inclusive μν qq̄ (BR = {card.PARAMETERS['mass']})")
    print()

    fit.fit_parameters()
    res = fit.fit_results()
    print()

    print("-" * 78)
    print("Paper projection (YFSWW3, all WW decay channels combined):")
    print("    Δm_W = 0.5 MeV,  ΔΓ_W = 1.2 MeV   (statistical only)")
    print("-" * 78)
    print("Differences vs the paper to keep in mind:")
    print("  • paper counts ALL WW decay channels (factor ~7 more events than μνqq̄)")
    print("  • paper uses YFSWW3 (NLO EW + ISR + ...) — mine is BFS LO_EFT Born")
    print("  • paper's dσ/dΓ_W has a crossing at 162.3 GeV (YFSWW3 radiative effect);")
    print("    BFS LO_EFT Born doesn't reproduce that crossing → ρ(m_W,Γ_W) > 0")
    print("    here whereas the paper has the two POIs decorrelated by sampling")
    print("    near the crossing point.")


if __name__ == "__main__":
    main()
