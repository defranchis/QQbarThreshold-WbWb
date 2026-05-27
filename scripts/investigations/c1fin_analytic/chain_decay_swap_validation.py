"""Chain-level validation of the BFS σ̂_LR^(0) → σ_Born decay substitution.

Confirms that the new `decay_uses_full_born` knob in the production chain
reproduces:
  (a) the closure of the option-5 hybrid + decay swap against BFS Table 4
      NLO column (the closure-budget reference, see report §5.5);
  (b) numerically, the option-5+swap result inside ~MC-stat — both should
      land at +0.0-0.4 % vs BFS Table 4 NLO.

Reads BFS Table 4 NLO column from the paper (PDF column 3 of Table 3 of
arXiv:0707.0773, reproduced here for convenience).

USE: PYTHONPATH=. python3 scripts/investigations/c1fin_analytic/chain_decay_swap_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

# BFS arXiv:0707.0773 Table 3: σ(e- e+ → μ- ν̄_μ ud̄ X) in fb at the scan
# energies, m_W=80.377, Γ_W=2.09201, α_s for δ_QCD applied. Columns:
#   Born — exact 4f Born (WHIZARD).
#   Born(ISR) — convolved with ISR (BFS Born(ISR)).
#   NLO — full NLO including the σ̂_LR^(0)→σ_Born decay substitution and δ_QCD.
#   NLO(ISR-tree) — tree-level ISR-improved hybrid.
BFS_TABLE_3 = {
    158: (61.67,  45.64,  49.19,  50.02),
    161: (154.19, 108.60, 117.81, 120.00),
    164: (303.00, 219.70, 234.90, 236.80),
    167: (408.80, 310.20, 328.20, 329.10),
    170: (481.70, 378.40, 398.00, 398.30),
}


def main():
    mW, gW = 80.377, 2.09201
    base = dict(
        mW=mW, gammaW=gW, channel="munuud",
        br_convention="bfs-eft",
        include_coulomb=False,
        include_NLO_hard_decay=True,
        include_BFS_NNLO=False,
        apply_delta_QCD=True,
        apply_whizard_anchor=True, whizard_anchor_source="spline",
        isr_scheme="2leg",
    )
    print("σ_obs(μν ud̄) at scan-window √s, two prescriptions:")
    print("  knob OFF  →  Δσ_decay = δ_decay × σ̂_LR^(0)   (historical chain)")
    print("  knob ON   →  Δσ_decay = δ_decay × σ_LR_Born   (BFS recipe, default)")
    print()
    print(f"  {'√s':>5}  {'BFS NLO':>10}  {'chain OFF':>11}  {'chain ON':>11}  "
          f"{'Δ_swap':>9}  {'Δ% OFF':>9}  {'Δ% ON':>9}")
    print(f"  {'':->88}")

    for sq, (_, _, nlo_paper, _) in BFS_TABLE_3.items():
        s_off = float(sigma_observed_munuqq(float(sq), decay_uses_full_born=False, **base)) * 1e3
        s_on  = float(sigma_observed_munuqq(float(sq), decay_uses_full_born=True,  **base)) * 1e3
        dswap = s_on - s_off
        d_off = 100.0 * (s_off - nlo_paper) / nlo_paper
        d_on  = 100.0 * (s_on  - nlo_paper) / nlo_paper
        print(f"  {sq:>5d}  {nlo_paper:>10.2f}  {s_off:>11.3f}  {s_on:>11.3f}  "
              f"{dswap:>+9.3f}  {d_off:>+8.2f}%  {d_on:>+8.2f}%")


if __name__ == "__main__":
    main()
