"""Option-5 closure test: hybrid post-ISR anchor (BFS prescription).

BFS Table 3/4 σ_NLO is constructed as

   σ_NLO_BFS(s) = δ_QCD · ( WHIZARD_Born(ISR)(s)  +  LL+exp{Δσ̂_NLO}(s) )

Our standard chain computes σ_obs = δ_QCD · LL+exp{ anchored σ̂_Born_partonic + Δσ̂_NLO }.
The two differ in the Born×ISR piece: BFS use WHIZARD-internal-ISR on the
4f Born, we use our analytic LL+exp BETA on the EFT-anchored partonic Born.

This test isolates the Born×ISR recipe-mismatch piece by replacing our
LL+exp(Born_partonic) with BFS's quoted Born(ISR) and keeping our analytic
LL+exp on the NLO addition. If the hybrid closes to BFS NLO column, the
residual in the standard chain is purely the LL+exp-vs-WHIZARD-ISR-kernel
mismatch on the Born side.

USE: PYTHONPATH=. python3 scripts/investigations/c1fin_analytic/option5_postisr_anchor_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
from framework.process.ww.xsec_calculator.bfs_eft import delta_QCD_factor


# BFS Table 4 (specific channel μ⁻ν̄_μ ud̄): m_W = 80.377, Γ_W = 2.09201
# Columns: (Born, Born+ISR, NLO_with_ISR, NLO_ISR-tree). NLO column has δ_QCD
# applied per BFS §6.1 ("entire NLO electroweak cross section"); Born(ISR) does not.
BFS_TABLE_4 = {
    158: (61.67,  45.64,  49.19,  50.02),
    161: (154.19, 108.60, 117.81, 120.00),
    164: (303.00, 219.70, 234.90, 236.80),
    167: (408.80, 310.20, 328.20, 329.10),
    170: (481.70, 378.40, 398.00, 398.30),
}


def main():
    mW, gW = 80.377, 2.09201
    alpha_s = 0.1199
    dQCD = delta_QCD_factor(alpha_s)
    # Base: K_C off, BFS-EFT BR, 2-leg LL+exp, anchor on. δ_QCD applied AFTER
    # so we can split the "BFS-Born(ISR) + ΔNLO" components cleanly.
    base_kwargs = dict(
        mW=mW, gammaW=gW, channel="munuud",
        br_convention="bfs-eft", include_coulomb=False,
        apply_delta_QCD=False,            # apply by hand AFTER
        apply_whizard_anchor=True,
        isr_scheme="2leg",
    )

    print(f"δ_QCD(α_s={alpha_s}) = {dQCD:.5f}")
    print()
    print("Option-5 hybrid: σ_test = δ_QCD · (BFS_Born_ISR + LL+exp{Δσ̂_NLO}_ours)")
    print("=" * 100)
    hdr = ("√s", "BornISR_BFS", "ΔNLO_pure", "test_pure",
           "test×δ_QCD", "NLO_BFS", "ratio", "Δ%")
    print(f"  {hdr[0]:>4} {hdr[1]:>12} {hdr[2]:>10} {hdr[3]:>10} "
          f"{hdr[4]:>11} {hdr[5]:>10} {hdr[6]:>9} {hdr[7]:>9}")
    print(f"  {'':->84}")

    for sq, (born_paper, born_isr_paper, nlo_paper, _) in BFS_TABLE_4.items():
        sigma_w_nlo = sigma_observed_munuqq(
            float(sq), include_NLO_hard_decay=True,
            include_BFS_NNLO=False, **base_kwargs) * 1e3
        sigma_wo_nlo = sigma_observed_munuqq(
            float(sq), include_NLO_hard_decay=False,
            include_BFS_NNLO=False, **base_kwargs) * 1e3
        dnlo_pure = float(sigma_w_nlo) - float(sigma_wo_nlo)
        test_pure = born_isr_paper + dnlo_pure
        test_with_qcd = dQCD * test_pure
        ratio = test_with_qcd / nlo_paper
        print(f"  {sq:>4} {born_isr_paper:>12.2f} {dnlo_pure:>+10.3f}"
              f" {test_pure:>10.3f} {test_with_qcd:>11.3f}"
              f" {nlo_paper:>10.2f} {ratio:>9.4f} {100*(ratio-1):>+8.2f}%")

    print()
    print("Compare standard chain (single LL+exp on Born+NLO sum, δ_QCD on inside):")
    print(f"  {'√s':>4} {'σ_obs_std':>11} {'NLO_BFS':>10} {'ratio':>9} {'Δ%':>9}")
    print(f"  {'':->44}")
    for sq, (_, _, nlo_paper, _) in BFS_TABLE_4.items():
        sigma_std = sigma_observed_munuqq(
            float(sq), mW=mW, gammaW=gW, channel="munuud",
            br_convention="bfs-eft", include_coulomb=False,
            include_NLO_hard_decay=True, apply_delta_QCD=True,
            alpha_s=alpha_s, apply_whizard_anchor=True,
            isr_scheme="2leg") * 1e3
        ratio = float(sigma_std) / nlo_paper
        print(f"  {sq:>4} {float(sigma_std):>11.3f} {nlo_paper:>10.2f}"
              f" {ratio:>9.4f} {100*(ratio-1):>+8.2f}%")

    print()
    print("Also compute the Born(ISR) column comparison:")
    print(f"  {'√s':>4} {'BornISR_ours':>14} {'BornISR_BFS':>13} {'ratio':>9} {'Δ%':>9}")
    print(f"  {'':->52}")
    for sq, (_, born_isr_paper, _, _) in BFS_TABLE_4.items():
        # σ_obs_ours_Born_ISR = our anchored Born × LL+exp, no NLO loops
        sigma_born_isr = sigma_observed_munuqq(
            float(sq), include_NLO_hard_decay=False,
            include_BFS_NNLO=False, **base_kwargs) * 1e3
        ratio = float(sigma_born_isr) / born_isr_paper
        print(f"  {sq:>4} {float(sigma_born_isr):>14.3f} {born_isr_paper:>13.2f}"
              f" {ratio:>9.4f} {100*(ratio-1):>+8.2f}%")


if __name__ == "__main__":
    main()
