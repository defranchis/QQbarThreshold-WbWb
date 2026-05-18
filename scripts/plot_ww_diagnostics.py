"""Diagnostic plots for the WW threshold-scan calculation.

Two PDF figures written to ``plots/ww_diagnostics/``:

* ``xsec_vs_sqrts.pdf`` — σ(√s) at successive BFS Born orders (eq. 17 →
  17+37 → 17+33+37 → 17+33+37+39) and then with the Coulomb K-factor
  and LL+YFS ISR convolution layered on. Two panels: linear scale
  155–170 GeV, and ratios relative to the full N^(3/2)LO Born.

* ``sensitivity_vs_sqrts.pdf`` — dσ/dm_W and dσ/dΓ_W vs √s computed by
  central finite difference, both for the partonic cross section (no
  ISR) and the observed cross section (with LL+YFS ISR). All in
  fb/MeV; both POIs on a single figure with two panels.

Run from the WW_threshold/ directory:

    python3 -m scripts.plot_ww_diagnostics
"""

from __future__ import annotations

import os

import numpy as np

from process.ww.bfs_eft import (
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
)
from process.ww.eft_xsec import (
    GAMMA_W_DEFAULT, M_W_DEFAULT,
    BR_INCLUSIVE_MUNUQQ,
    coulomb_K_factor, sigma_partonic_munuqq, sigma_WW_Born,
)
from process.ww.isr import sigma_observed_munuqq


PLOT_DIR = "plots/ww_diagnostics"


def _bfs_total_WW_order(s, mW, gammaW, order: str):
    """σ_WW (total, unpolarised, summed over 4f decays) at the requested
    BFS Born truncation. Returns pb, vectorised in ``s``.

    ``order`` ∈ {LO, N1/2LO, NLO, N3/2LO}.
    """
    sLR = sigma_LR0_specific_pb(s, mW, gammaW)
    sRL = np.zeros_like(np.asarray(sLR, dtype=float))
    if order in ("N1/2LO", "NLO", "N3/2LO"):
        s12_LR, s12_RL = sigma_LR_RL_half_specific_pb(s, mW)
        sLR = sLR + s12_LR
        sRL = sRL + s12_RL
    if order in ("NLO", "N3/2LO"):
        s_NLO_LR, s_NLO_RL = sigma_LR_RL_NLO_potential_specific_pb(s, mW, gammaW, gammaW_NLO=0.0)
        sLR = sLR + s_NLO_LR
        sRL = sRL + s_NLO_RL
    if order == "N3/2LO":
        s32_LR, s32_RL = sigma_LR_RL_three_half_a_specific_pb(s, mW)
        sLR = sLR + s32_LR
        sRL = sRL + s32_RL
    return (sLR + sRL) * 27.0 / 4.0   # specific → total WW, unpolarised


# ---------------------------------------------------------------------------
# Plot 1: σ vs √s
# ---------------------------------------------------------------------------

def plot_xsec_vs_sqrts():
    import matplotlib.pyplot as plt

    sqrts = np.linspace(155.0, 170.0, 151)
    s = sqrts ** 2
    mW, gW = M_W_DEFAULT, GAMMA_W_DEFAULT

    # BFS orders for σ_WW (then × BR for inclusive μν qq̄)
    orders = ["LO", "N1/2LO", "NLO", "N3/2LO"]
    labels = {
        "LO":     r"BFS LO (eq. 17)",
        "N1/2LO": r"+ N$^{1/2}$LO non-res. (eq. 37)",
        "NLO":    r"+ NLO Born pot. (eq. 33)",
        "N3/2LO": r"+ N$^{3/2}$LO E-dep (eq. 39)",
    }
    colors = {"LO": "#4575b4", "N1/2LO": "#74add1", "NLO": "#fdae61", "N3/2LO": "#a50026"}

    # Physics-layer curves on top of the best BFS Born ("framework" σ_WW
    # = RACOONWW above 161.33, BFS-matched below)
    sigma_WW = sigma_WW_Born(s, mW, gW)
    K_C = coulomb_K_factor(s, mW, gW)
    sigma_partonic = sigma_partonic_munuqq(s, mW, gW, channel="inclusive")
    sigma_observed = sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW, channel="inclusive")

    fig, (ax_abs, ax_rat) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1.5]})

    # Top: absolute σ (inclusive μν qq̄, in fb)
    bfs_curves = {}
    for o in orders:
        sigma_WW_bfs = _bfs_total_WW_order(s, mW, gW, o)
        sigma_incl = sigma_WW_bfs * BR_INCLUSIVE_MUNUQQ
        bfs_curves[o] = sigma_incl * 1e3   # pb → fb
        ax_abs.plot(sqrts, bfs_curves[o], label=labels[o], color=colors[o],
                    linewidth=1.3, linestyle="-")

    # Framework curves
    ax_abs.plot(sqrts, sigma_WW * BR_INCLUSIVE_MUNUQQ * 1e3,
                label=r"σ$_{WW}$ × BR (framework, RACOONWW above 161.33)",
                color="black", linestyle=":", linewidth=1.6)
    ax_abs.plot(sqrts, sigma_partonic * 1e3,
                label=r"$+\,K_{\rm Coulomb}$  (partonic)",
                color="green", linestyle="--", linewidth=1.6)
    ax_abs.plot(sqrts, sigma_observed * 1e3,
                label=r"$+\,$LL+YFS ISR  (observed)",
                color="red", linestyle="-", linewidth=2.0)

    ax_abs.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_abs.text(2 * mW + 0.05, ax_abs.get_ylim()[1] * 0.95, r"$2\,m_W$",
                color="grey", alpha=0.6, fontsize=9, ha="left", va="top")
    ax_abs.set_ylabel(r"$\sigma(e^+e^- \to \mu\nu q\bar q)$ [fb]")
    ax_abs.set_title(r"WW threshold cross section vs $\sqrt{s}$  "
                      f"($m_W = {mW:.4f}$ GeV, $\\Gamma_W = {gW:.3f}$ GeV)")
    ax_abs.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_abs.grid(alpha=0.25)

    # Bottom: ratio to N^(3/2)LO Born
    denom = bfs_curves["N3/2LO"]
    eps = 1e-9
    for o in orders:
        ax_rat.plot(sqrts, bfs_curves[o] / np.maximum(denom, eps),
                    color=colors[o], linewidth=1.3)
    ax_rat.plot(sqrts,
                (sigma_WW * BR_INCLUSIVE_MUNUQQ * 1e3) / np.maximum(denom, eps),
                color="black", linestyle=":", linewidth=1.4)
    ax_rat.plot(sqrts, (sigma_partonic * 1e3) / np.maximum(denom, eps),
                color="green", linestyle="--", linewidth=1.4)
    ax_rat.plot(sqrts, (sigma_observed * 1e3) / np.maximum(denom, eps),
                color="red", linewidth=1.6)

    ax_rat.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_rat.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_rat.set_ylabel(r"ratio to N$^{3/2}$LO Born")
    ax_rat.set_ylim(0.0, 1.6)
    ax_rat.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "xsec_vs_sqrts.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2: dσ/dm_W and dσ/dΓ_W vs √s
# ---------------------------------------------------------------------------

def _ddiff(sigma_fn, x_default: float, *, h: float, **kwargs):
    """Central finite difference of ``sigma_fn(x, **kwargs)`` w.r.t. ``x``."""
    plus  = sigma_fn(x_default + h, **kwargs)
    minus = sigma_fn(x_default - h, **kwargs)
    return (plus - minus) / (2.0 * h)


def plot_sensitivity_vs_sqrts():
    import matplotlib.pyplot as plt

    sqrts = np.linspace(155.0, 170.0, 151)
    s = sqrts ** 2
    mW, gW = M_W_DEFAULT, GAMMA_W_DEFAULT

    h_mW = 0.001   # 1 MeV
    h_gW = 0.001   # 1 MeV

    # Partonic sensitivity (no ISR)
    dsig_dmW_part = (sigma_partonic_munuqq(s, mW + h_mW, gW, channel="inclusive")
                     - sigma_partonic_munuqq(s, mW - h_mW, gW, channel="inclusive")) / (2 * h_mW)
    dsig_dgW_part = (sigma_partonic_munuqq(s, mW, gW + h_gW, channel="inclusive")
                     - sigma_partonic_munuqq(s, mW, gW - h_gW, channel="inclusive")) / (2 * h_gW)

    # Observed sensitivity (with ISR)
    dsig_dmW_obs = (sigma_observed_munuqq(sqrts, mW=mW + h_mW, gammaW=gW, channel="inclusive")
                    - sigma_observed_munuqq(sqrts, mW=mW - h_mW, gammaW=gW, channel="inclusive")) / (2 * h_mW)
    dsig_dgW_obs = (sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW + h_gW, channel="inclusive")
                    - sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW - h_gW, channel="inclusive")) / (2 * h_gW)

    # pb/GeV  →  fb/MeV  is a factor of 1 (1 pb/GeV = 1 fb/MeV).
    fig, (ax_mW, ax_gW) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    ax_mW.plot(sqrts, dsig_dmW_part, color="C0", linewidth=1.6,
               linestyle="--", label="partonic (no ISR)")
    ax_mW.plot(sqrts, dsig_dmW_obs, color="C0", linewidth=2.0,
               label="observed (with LL+YFS ISR)")
    ax_mW.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_mW.axhline(0.0, color="grey", alpha=0.4, linewidth=0.6)
    ax_mW.set_ylabel(r"$d\sigma/dm_W$ [fb/MeV]")
    ax_mW.set_title(r"Sensitivity of $\sigma(\mu\nu q\bar q)$ to $m_W$ and $\Gamma_W$  "
                     f"($m_W = {mW:.4f}$ GeV, $\\Gamma_W = {gW:.3f}$ GeV)")
    ax_mW.legend(loc="best", fontsize=9)
    ax_mW.grid(alpha=0.25)

    ax_gW.plot(sqrts, dsig_dgW_part, color="C3", linewidth=1.6,
               linestyle="--", label="partonic (no ISR)")
    ax_gW.plot(sqrts, dsig_dgW_obs, color="C3", linewidth=2.0,
               label="observed (with LL+YFS ISR)")
    ax_gW.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_gW.axhline(0.0, color="grey", alpha=0.4, linewidth=0.6)
    ax_gW.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gW.set_ylabel(r"$d\sigma/d\Gamma_W$ [fb/MeV]")
    ax_gW.legend(loc="best", fontsize=9)
    ax_gW.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "sensitivity_vs_sqrts.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    print(f"Writing diagnostic plots to {PLOT_DIR}/ …")
    plot_xsec_vs_sqrts()
    plot_sensitivity_vs_sqrts()
    print("Done.")


if __name__ == "__main__":
    main()
