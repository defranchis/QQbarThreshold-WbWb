"""Diagnostic D: σ_chain / σ_WHIZARD_4f_Born for three Coulomb configurations.

For each config, compute the multiplicative ``K-factor'' that the chain
applies on top of the WHIZARD 4f Born:

    R(√s) ≡ σ_chain(√s) / σ_WHIZARD_4f_Born(√s)
          = K_C × δ_QCD × BR_PDG × [1 + Σ δ_NLO/σ_LR^(0) + δ_NNLO/σ_LR^(0)]

(The Whizard anchor f(δ, Γ_W) cancels with σ_BFS_Born in the chain's
numerator, so σ_WHIZARD_4f_Born ≡ σ_BFS_Born × f is exactly what the
chain's Born sum becomes after the anchor.)

Three configs:

  off    : include_coulomb=False, coulomb_kc_safe=False
           (BFS-consistent path — K_C off, full eq.(62) on)
  unsafe : include_coulomb=True,  coulomb_kc_safe=False
           (current production chain — K_C on AND full eq.(62) on,
            double-counts leading-α/v Coulomb)
  safe   : include_coulomb=True,  coulomb_kc_safe=True
           (proposed K_C-safe — K_C on, eq.(62) subleading α² only)

Physics checks the plot should pass:

  1. R_unsafe - R_off ≈ (K_C - 1) × R_off   (current chain's extra K_C - 1
     is just the K_C resummation, applied on top of an already-NLO-Coulomb-on σ)

  2. R_safe - R_off ≈ (K_C - 1 - δ_term1/σ_LR^(0)) × R_off
     ≈ +2 pp × R_off near threshold (the off-shell prescription offset),
     vanishing well above threshold.

  3. Above-threshold asymptotic limit (computed by calling
     sigma_BFS_LO_total_WW_pb DIRECTLY, bypassing the 170 GeV region
     cutoff in sigma_partonic_munuqq): R_unsafe → R_safe as K_C → 1.
     At √s = 250 GeV K_C-1 is at the per-mille level, so R_unsafe / R_safe
     should equal 1 to within a few per-mille; at √s = 500 GeV, sub-permil.

Run from the repo root:

    python3 -m scripts.investigations.coulomb_double_count.check_chain_over_whizard_ratio
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_BFS_LO_total_WW_pb,
    delta_QCD_factor,
    whizard_anchor_factor,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq,
    coulomb_K_factor,
    BR_INCLUSIVE_MUNUQQ,
    ALPHA_S_MW_DEFAULT,
)


OUT_DIR = "plots/coulomb_double_count"
os.makedirs(OUT_DIR, exist_ok=True)

MW = 80.379
GW = 2.085


def chain_sigma(s, *, include_coulomb, coulomb_kc_safe, source="spline"):
    return sigma_partonic_munuqq(
        s, mW=MW, gammaW=GW,
        channel="inclusive", include_coulomb=include_coulomb,
        br_convention="pdg-constant",
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True,
        whizard_anchor_source=source,
        coulomb_kc_safe=coulomb_kc_safe,
    )


def whizard_born_sigma(s, *, source="spline"):
    """σ_WHIZARD_4f_Born for the inclusive μν qq̄ channel, in pb."""
    sigma_bfs_born_total = sigma_BFS_LO_total_WW_pb(
        s, MW, GW, order="N3/2LO",
        include_NLO_hard_decay=False, include_BFS_NNLO=False,
        apply_delta_QCD=False,
        apply_whizard_anchor=True, whizard_anchor_source=source,
    )
    return sigma_bfs_born_total * BR_INCLUSIVE_MUNUQQ


def main():
    # ----- in-region scan -----
    sqrts = np.linspace(154.5, 169.5, 151)
    s = sqrts ** 2

    sig_off    = chain_sigma(s, include_coulomb=False, coulomb_kc_safe=False)
    sig_unsafe = chain_sigma(s, include_coulomb=True,  coulomb_kc_safe=False)
    sig_safe   = chain_sigma(s, include_coulomb=True,  coulomb_kc_safe=True)
    sig_whz    = whizard_born_sigma(s)

    R_off    = sig_off    / sig_whz
    R_unsafe = sig_unsafe / sig_whz
    R_safe   = sig_safe   / sig_whz

    KC = coulomb_K_factor(s, MW, GW)

    # ----- above-region asymptotic check (bypasses 170 GeV cutoff) -----
    # Call sigma_BFS_LO_total_WW_pb directly; multiply K_C and BR by hand.
    sqrts_high = np.array([170.0, 180.0, 200.0, 250.0, 350.0, 500.0])
    s_high = sqrts_high ** 2
    # The spline anchor extrapolates above 170 GeV; the per-√s f(δ) factor
    # affects σ_chain and σ_whz equally so it cancels in the ratio.
    sig_bfs_unsafe_high = sigma_BFS_LO_total_WW_pb(
        s_high, MW, GW, order="N3/2LO",
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True,
        whizard_anchor_source="spline", coulomb_kc_safe=False,
    )
    sig_bfs_safe_high = sigma_BFS_LO_total_WW_pb(
        s_high, MW, GW, order="N3/2LO",
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, apply_whizard_anchor=True,
        whizard_anchor_source="spline", coulomb_kc_safe=True,
    )
    sig_whz_high = whizard_born_sigma(s_high)
    # Apply K_C and BR_PDG by hand (sigma_BFS_LO_total_WW_pb returns the
    # full-WW total without BR, K_C is multiplied externally by the chain).
    KC_high = coulomb_K_factor(s_high, MW, GW)
    R_unsafe_high = sig_bfs_unsafe_high * KC_high * BR_INCLUSIVE_MUNUQQ / sig_whz_high
    R_safe_high   = sig_bfs_safe_high   * KC_high * BR_INCLUSIVE_MUNUQQ / sig_whz_high

    # ----- plot -----
    fig, ax = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1]))

    ax[0].plot(sqrts, R_off,    lw=1.7, color="C2",
               label=r"$R_{\rm off}$  ($K_C$ off, full eq.(62))  — BFS-validated")
    ax[0].plot(sqrts, R_unsafe, lw=1.7, color="C0",
               label=r"$R_{\rm unsafe}$  ($K_C$ on, full eq.(62))  — current chain")
    ax[0].plot(sqrts, R_safe,   lw=1.7, color="C1", ls="--",
               label=r"$R_{\rm safe}$  ($K_C$ on, eq.(62) subleading only)")
    ax[0].axvline(2 * MW, color="k", lw=0.5, ls=":", alpha=0.6)
    ax[0].set_ylabel(r"$R(\sqrt{s}) \equiv \sigma_{\rm chain}/\sigma_{\rm WHIZARD}$")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(r"Chain K-factor over WHIZARD 4f Born, three Coulomb configs")

    # bottom: differences in percentage points of R
    ax[1].plot(sqrts, (R_unsafe - R_off) / R_off * 100, lw=1.5, color="C0",
               label=r"($R_{\rm unsafe} - R_{\rm off}$) / $R_{\rm off}$  "
                     r"≈ $(K_C - 1)$  (resummation contribution)")
    ax[1].plot(sqrts, (R_safe - R_off) / R_off * 100, lw=1.5, color="C1", ls="--",
               label=r"($R_{\rm safe} - R_{\rm off}$) / $R_{\rm off}$  "
                     r"≈ off-shell prescription residual")
    ax[1].plot(sqrts, (KC - 1) * 100, lw=1.0, color="grey", alpha=0.7,
               label=r"$K_C - 1$  (reference)")
    ax[1].axhline(0, color="k", lw=0.5, alpha=0.5)
    ax[1].set_ylabel(r"relative shift  [%]")
    ax[1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "chain_over_whizard_ratio.pdf")
    png = os.path.join(OUT_DIR, "chain_over_whizard_ratio.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # ----- console tables -----
    print()
    print("=" * 78)
    print("In-region R(√s) at canonical scan points")
    print("=" * 78)
    print(f"  {'√s':>8}  {'R_off':>8}  {'R_unsafe':>9}  {'R_safe':>8}  "
          f"{'(R_uns/R_off-1) [%]':>20}  {'(R_safe/R_off-1) [%]':>20}")
    for sq in (155.0, 157.0, 159.0, 161.0, 162.5, 165.0, 168.0):
        i = int(np.argmin(np.abs(sqrts - sq)))
        ro, ru, rs = R_off[i], R_unsafe[i], R_safe[i]
        print(f"  {sqrts[i]:8.2f}  {ro:8.5f}  {ru:9.5f}  {rs:8.5f}  "
              f"{(ru/ro - 1)*100:+20.4f}  {(rs/ro - 1)*100:+20.4f}")

    print()
    print("=" * 78)
    print("Above-region asymptotic limit (bypasses 170 GeV region cutoff)")
    print("R_unsafe and R_safe should converge as K_C → 1")
    print("=" * 78)
    print(f"  {'√s':>8}  {'K_C-1 [%]':>10}  {'R_unsafe':>9}  {'R_safe':>9}  "
          f"{'R_unsafe/R_safe - 1 [%]':>24}")
    for sq, kc1, ru, rs in zip(sqrts_high, KC_high - 1, R_unsafe_high, R_safe_high):
        print(f"  {sq:8.2f}  {kc1*100:+10.4f}  {ru:9.5f}  {rs:9.5f}  "
              f"{(ru/rs - 1)*100:+24.5f}")

    print()
    print("Interpretation:")
    print("  - Bottom-panel grey curve (K_C - 1) should match the orange")
    print("    (R_unsafe - R_off)/R_off curve almost exactly (≈ same physics).")
    print("  - Bottom-panel dashed (R_safe - R_off)/R_off should sit at")
    print("    ≈ +2 pp at threshold, dropping to ~0.4 pp by 168 GeV —")
    print("    exactly the off-shell vs FKM prescription difference plot.")
    print("  - In the asymptotic table, R_unsafe/R_safe → 1 monotonically")
    print("    as K_C - 1 → 0; no bug if the two converge to within < 0.05% at 500 GeV.")


if __name__ == "__main__":
    main()
