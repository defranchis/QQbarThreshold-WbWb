"""Smoke test for the new α(M_Z)/ALPMZ ISR central + isr_scale_factor knob.

End-to-end verification:
  1. WWGenerator.from_card builds without error against ww_default.
  2. sigma_observed_munuqq at √s = 162 GeV with isr_scale_factor ∈ {0.5, 1.0, 2.0}.
  3. Cross-checks against both ISR paths:
       - analytic LL+exp (isr_nll=False, isr_emela_ll=False): scale enters via
         β_ISR's log(ξ²·s/m_e²).
       - eMELA NLL (isr_nll=True): scale enters via Q = ξ·√s in DGLAP evolution,
         AND via β_ISR for the integration grid.
       - eMELA LL (isr_emela_ll=True): same as eMELA NLL but LL truncation.

Expected: LL scale variation > NLL scale variation (NLL absorbs the leading
scale log). Not a precise validation — just confirms the plumbing works and
the numbers move in the right direction at threshold.
"""
import os
import sys

# Ensure framework imports work when run from project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "..")))

import numpy as np
from cards import ww_default
from framework.process.ww.generator import WWGenerator
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq


def main():
    sqrt_s = np.array([162.0])  # one bin, near σ_WW maximum
    mW = 80.379
    gammaW = 2.085

    gen = WWGenerator.from_card(ww_default)
    print("Generator built. Describe:")
    print(" ", gen.describe())
    print()

    common = {
        "mW": mW, "gammaW": gammaW,
        "channel": gen.channel,
        "include_coulomb": gen.include_coulomb,
        "bfs": gen.bfs,
        "include_NLO_hard_decay": gen.include_NLO_hard_decay,
        "include_BFS_NNLO": gen.include_BFS_NNLO,
        "apply_delta_QCD": gen.apply_delta_QCD,
        "alpha_s": gen.alpha_s,
        "alpha_s_ref": gen.alpha_s,
        "br_convention": gen.br_convention,
        "apply_whizard_anchor": gen.apply_whizard_anchor,
        "whizard_anchor_source": gen.whizard_anchor_source,
        "isr_emela_pert_order": gen.isr_emela_pert_order,
        "isr_emela_fac_scheme": gen.isr_emela_fac_scheme,
        "isr_emela_ren_scheme": gen.isr_emela_ren_scheme,
        "alpha_em": gen.alpha_em,
        "alpha_em_isr": gen.alpha_em_isr,
        "coulomb_kc_safe": gen.coulomb_kc_safe,
        "decay_uses_full_born": gen.decay_uses_full_born,
        "m_t": gen.m_t, "M_H": gen.M_H, "MZ": gen.MZ,
    }

    print(f"Card α_em_isr = {gen.alpha_em_isr:.6e}  (1/α = {1.0/gen.alpha_em_isr:.3f})")
    print(f"Card eMELA ren scheme = {gen.isr_emela_ren_scheme}")
    print(f"Card isr_scale_factor = {gen.isr_scale_factor}")
    print()

    scales = [0.5, 1.0, 2.0]
    print(f"=== ISR scale variation at √s = {sqrt_s[0]} GeV ===")

    for path_name, path_kwargs in [
        ("analytic LL+exp",      {"isr_scheme": "single_conv",
                                  "isr_nll": False, "isr_emela_ll": False}),
        ("eMELA LL (BETA-DGLAP)", {"isr_scheme": "2leg",
                                  "isr_nll": False, "isr_emela_ll": True}),
        ("eMELA NLL (CodePdf)",   {"isr_scheme": "2leg",
                                  "isr_nll": True, "isr_emela_ll": False}),
    ]:
        print(f"\n--- {path_name} ---")
        sigs = []
        for xi in scales:
            sig = sigma_observed_munuqq(
                sqrt_s, **common, **path_kwargs,
                isr_scale_factor=xi,
            )
            sig = float(np.atleast_1d(sig)[0])
            sigs.append(sig)
            print(f"  ξ = {xi:.1f}:  σ_obs = {sig:.6f} pb")
        sig_lo, sig_c, sig_hi = sigs
        env_pct = max(abs(sig_hi - sig_c), abs(sig_c - sig_lo)) / sig_c * 100.0
        print(f"  envelope: max |Δσ/σ| = {env_pct:.3f}%")


if __name__ == "__main__":
    main()
