"""Option-5 hybrid + BFS's σ_LR^(0) → σ_Born substitution in decay correction.

BFS line 2660: "replacing the leading-order cross section σ^(0) by the full
Born cross section σ_Born in the decay correction (eq:delta-decay)".

This test takes our framework's Δσ_decay = δ_decay × σ_LR^(0) and adds the
ADJUSTMENT δ_decay × (σ_Born_full − σ_LR^(0)) to mimic BFS's prescription
inside the LL+exp convolution, then assembles the option-5 hybrid σ_test
and compares to BFS Table 4 NLO column.

USE: PYTHONPATH=. python3 scripts/investigations/c1fin_analytic/option5_with_decay_swap.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq, sigma_ISR_2leg_convolution
from framework.process.ww.xsec_calculator.bfs_eft import (
    delta_QCD_factor,
    sigma_LR0_specific_pb, sigma_BFS_specific_munuud_pb,
    delta_decay_EW_relative,
)

BFS_TABLE_4 = {
    158: (61.67,  45.64,  49.19,  50.02),
    161: (154.19, 108.60, 117.81, 120.00),
    164: (303.00, 219.70, 234.90, 236.80),
    167: (408.80, 310.20, 328.20, 329.10),
    170: (481.70, 378.40, 398.00, 398.30),
}


def _decay_adjustment_partonic_pb(s, mW, gW):
    """Δσ_decay_BFS - Δσ_decay_ours, per-helicity, BR-corr units.

    Our: δ_decay × σ_LR^(0)_specific. BFS: δ_decay × σ_Born_full_specific
    (where σ_Born_full = our N(3/2)LO sum, helicity-averaged ×4 to match
    per-helicity convention). Adjustment per helicity = δ_decay × (4σ_Born_unpol - σ_LR^(0)).
    """
    s_arr = np.asarray(s, dtype=float)
    sLR0 = sigma_LR0_specific_pb(s_arr, mW, gW, apply_BR_correction=True)
    sBorn_unpol = sigma_BFS_specific_munuud_pb(
        s_arr, mW, gW, order="N3/2LO",
        include_NLO_hard_decay=False, include_BFS_NNLO=False,
        apply_delta_QCD=False, apply_whizard_anchor=False,
    )
    # σ_Born_full per-helicity (un-/4): multiply by 4
    sBorn_full_perhel = 4.0 * sBorn_unpol
    return delta_decay_EW_relative(mW) * (sBorn_full_perhel - sLR0)


def _llexp_of_partonic_per_hel_pb(s_obs, mW, gW, partonic_fn_pb):
    """LL+exp 2-leg convolution of a partonic σ_LR-per-helicity function,
    returning the σ_observed_specific in pb (helicity-averaged × BR per-component).

    sigma_ISR_2leg_convolution wants σ_partonic in the per-helicity-LR convention
    of sigma_LR0_specific_pb; helicity averaging happens in the wrapping caller.
    For the decay adjustment we only need (× σ_partonic_change / 4 unpol) at
    the LL+exp-convolved level, since LL+exp is linear in σ̂_partonic.
    """
    # ad-hoc minimal wrapper around the 2-leg engine
    return sigma_ISR_2leg_convolution(
        np.sqrt(s_obs),
        lambda s_p, mw, gw: partonic_fn_pb(s_p, mw, gw) / 4.0,
        mW=mW, gammaW=gW,
    )


def main():
    mW, gW = 80.377, 2.09201
    alpha_s = 0.1199
    dQCD = delta_QCD_factor(alpha_s)
    base_kwargs = dict(
        mW=mW, gammaW=gW, channel="munuud",
        br_convention="bfs-eft", include_coulomb=False,
        apply_delta_QCD=False, apply_whizard_anchor=True, isr_scheme="2leg",
    )

    print(f"δ_QCD(α_s={alpha_s}) = {dQCD:.5f}")
    print()
    print("Option-5+ hybrid: δ_QCD · (BFS Born(ISR) + ΔNLO_LL+exp_ours + Δ_decay_swap)")
    print("=" * 96)
    hdr = ("√s", "BornISR_BFS", "ΔNLO_ours", "Δ_dec_swap", "test×δ_QCD", "NLO_BFS", "ratio", "Δ%")
    print(f"  {hdr[0]:>4} {hdr[1]:>12} {hdr[2]:>10} {hdr[3]:>11} {hdr[4]:>11} {hdr[5]:>10} {hdr[6]:>9} {hdr[7]:>9}")
    print(f"  {'':->88}")

    for sq, (_, born_isr_paper, nlo_paper, _) in BFS_TABLE_4.items():
        # ΔNLO_ours (no δ_QCD)
        sigma_w = sigma_observed_munuqq(float(sq), include_NLO_hard_decay=True,
                                        include_BFS_NNLO=False, **base_kwargs) * 1e3
        sigma_wo = sigma_observed_munuqq(float(sq), include_NLO_hard_decay=False,
                                         include_BFS_NNLO=False, **base_kwargs) * 1e3
        dnlo_pure = float(sigma_w) - float(sigma_wo)

        # Δ_decay_swap at the OBSERVED level: LL+exp of the partonic adjustment
        adj_pb = _llexp_of_partonic_per_hel_pb(
            sq * sq, mW, gW,
            partonic_fn_pb=lambda s_p, mw, gw: _decay_adjustment_partonic_pb(s_p, mw, gw))
        adj_fb = float(adj_pb) * 1e3

        test = born_isr_paper + dnlo_pure + adj_fb
        test_qcd = dQCD * test
        ratio = test_qcd / nlo_paper
        print(f"  {sq:>4} {born_isr_paper:>12.2f} {dnlo_pure:>+10.3f} {adj_fb:>+11.3f}"
              f" {test_qcd:>11.3f} {nlo_paper:>10.2f} {ratio:>9.4f} {100*(ratio-1):>+8.2f}%")


if __name__ == "__main__":
    main()
