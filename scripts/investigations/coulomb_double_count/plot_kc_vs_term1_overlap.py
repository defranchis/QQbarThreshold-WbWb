"""K_C−1 vs Δσ_Coul_NLO term1/σ_LR^(0) — visual of the same-physics overlap.

The Fadin-Khoze-Martin resummation K_C and the first term of BFS eq.(62)
of arXiv:0707.0773 describe the SAME leading-α/v Coulomb physics in two
different formulations:

    K_C - 1    : resummed, off-shell prescription, ~+7% at threshold
    Δσ_term1 / σ_LR^(0) : unresummed first-order, ~+5% at threshold

The ~2pp offset is the framework's off-shell-momentum (finite-Γ_W)
regularisation. Both vanish well above threshold (~2pp / 3pp by 172 GeV).
The opt-in coulomb_kc_safe=True removes Δσ_term1 from the NLO chain so
that K_C and eq.(62) don't double-count the leading α/v.

Also shown: Δσ_term2 / σ_LR^(0), the genuine NLO α² two-photon piece
(~+0.2% at threshold) that survives in both branches.

Run from the repo root:

    python3 -m scripts.investigations.coulomb_double_count.plot_kc_vs_term1_overlap
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    delta_sigma_Coulomb_NLO_specific_pb,
    sigma_LR0_specific_pb,
)
from framework.process.ww.xsec_calculator.eft_xsec import coulomb_K_factor


OUT_DIR = "plots/coulomb_double_count"
os.makedirs(OUT_DIR, exist_ok=True)

MW = 80.379
GW = 2.085


def main():
    sqrts = np.linspace(155.0, 172.0, 171)
    s = sqrts ** 2

    kc_minus_1 = coulomb_K_factor(s, MW, GW) - 1.0
    sigma_LR0 = sigma_LR0_specific_pb(s, MW, GW, apply_BR_correction=False)
    coul_full = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                    subleading_only=False)
    coul_sub = delta_sigma_Coulomb_NLO_specific_pb(s, MW, GW, apply_BR_correction=False,
                                                   subleading_only=True)
    term1_rel = (coul_full - coul_sub) / sigma_LR0    # one-photon, leading α/v
    term2_rel = coul_sub / sigma_LR0                   # two-photon, NLO α²

    # Two panels: (top) overlap; (bottom) prescription offset and term-2 size.
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))

    ax[0].plot(sqrts, kc_minus_1 * 100, lw=1.8, color="C0",
               label=r"$K_C - 1$  (resummed; off-shell prescription)")
    ax[0].plot(sqrts, term1_rel * 100, lw=1.8, color="C3", ls="--",
               label=r"$\Delta\sigma^{(1)}_{\rm Coul,\,term\,1}\,/\,\sigma_{LR}^{(0)}$"
                     r"  (unresummed, FKM $X/2$)")
    ax[0].plot(sqrts, term2_rel * 100, lw=1.3, color="grey",
               label=r"$\Delta\sigma^{(1)}_{\rm Coul,\,term\,2}\,/\,\sigma_{LR}^{(0)}$"
                     r"  (NLO $\alpha^2$, FKM $X^2/6$)")
    ax[0].axvline(2 * MW, color="k", lw=0.5, ls=":", alpha=0.6)
    ax[0].text(2 * MW + 0.05, ax[0].get_ylim()[1] * 0.95 if ax[0].get_ylim()[1] > 0 else 6,
               r"$\sqrt{s}=2m_W$", fontsize=9, va="top")
    ax[0].set_ylabel(r"size relative to $\sigma_{LR}^{(0)}$  [%]")
    ax[0].set_title(r"Leading-$\alpha/v$ Coulomb: two formulations of the same physics")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(True, alpha=0.3)

    # Bottom: prescription offset = K_C - 1 - term1_rel (in pp), plus the
    # term-2 size to show it stays sub-percent throughout.
    offset_pp = (kc_minus_1 - term1_rel) * 100      # percent points
    ax[1].plot(sqrts, offset_pp, lw=1.5, color="C2",
               label=r"$(K_C - 1) - \Delta\sigma_{\rm term\,1}/\sigma_{LR}^{(0)}$  "
                     r"(off-shell prescription offset)")
    ax[1].axhline(0, color="k", lw=0.5, alpha=0.5)
    ax[1].set_ylabel(r"$K_C{-}1$ minus term-1  [pp]")
    ax[1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "kc_vs_term1_overlap.pdf")
    png = os.path.join(OUT_DIR, "kc_vs_term1_overlap.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # Console table for the same data (used as the in-report table).
    print()
    print("=" * 70)
    print("Numerical readout at canonical √s points:")
    print("=" * 70)
    print(f"  {'√s':>8}  {'K_C-1 [%]':>10}  {'term1 [%]':>10}  "
          f"{'offset [pp]':>12}  {'term2 [%]':>10}")
    for sq in (155.0, 157.0, 159.0, 161.0, 162.5, 165.0, 168.0, 172.0):
        s0 = sq ** 2
        k = float(coulomb_K_factor(s0, MW, GW)) - 1.0
        sLR0 = float(sigma_LR0_specific_pb(s0, MW, GW, apply_BR_correction=False))
        cf = float(delta_sigma_Coulomb_NLO_specific_pb(s0, MW, GW,
                                                       apply_BR_correction=False,
                                                       subleading_only=False))
        cs = float(delta_sigma_Coulomb_NLO_specific_pb(s0, MW, GW,
                                                       apply_BR_correction=False,
                                                       subleading_only=True))
        t1 = (cf - cs) / sLR0
        t2 = cs / sLR0
        print(f"  {sq:8.2f}  {k*100:+10.4f}  {t1*100:+10.4f}  "
              f"{(k-t1)*100:+12.4f}  {t2*100:+10.4f}")


if __name__ == "__main__":
    main()
