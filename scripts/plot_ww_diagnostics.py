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
    gamma_W_LO,
    sigma_LR0_specific_pb,
    sigma_LR_RL_half_specific_pb,
    sigma_LR_RL_NLO_potential_specific_pb,
    sigma_LR_RL_three_half_a_specific_pb,
)
from process.ww.eft_xsec import (
    GAMMA_W_DEFAULT, M_W_DEFAULT,
    coulomb_K_factor, sigma_partonic_munuqq as _sigma_partonic_munuqq_raw,
    sigma_WW_Born,
)
from process.ww.isr import sigma_observed_munuqq as _sigma_observed_munuqq_raw
from cards import ww_default as _card

# Diagnostic plots always reflect the *card-configured* NLO behaviour so the
# plots match what the fit actually sees. Curried wrappers below pick up the
# full chain configuration from cards/ww_default.py.
_NLO_KW = {
    "include_NLO_hard_decay": _card.NLO_CONFIG.get("include_NLO_hard_decay", True),
    "apply_delta_QCD":        _card.NLO_CONFIG.get("apply_delta_QCD", True),
    "br_convention":          _card.NLO_CONFIG.get("br_convention", "pdg-constant"),
    "alpha_s":                _card.THEORY_INPUTS.get("alpha_s_MW", 0.1199),
    "apply_whizard_anchor":   _card.NLO_CONFIG.get("apply_whizard_anchor", True),
}
# Extra ISR knobs only meaningful for sigma_observed (post-convolution σ).
_ISR_KW = {
    "isr_scheme":   _card.NLO_CONFIG.get("isr_scheme", "single_conv"),
    "alpha_em_isr": _card.NLO_CONFIG.get("alpha_em_isr", None),
}


def sigma_partonic_munuqq(*args, **kwargs):
    for k, v in _NLO_KW.items():
        kwargs.setdefault(k, v)
    return _sigma_partonic_munuqq_raw(*args, **kwargs)


def sigma_observed_munuqq(*args, **kwargs):
    for k, v in _NLO_KW.items():
        kwargs.setdefault(k, v)
    for k, v in _ISR_KW.items():
        kwargs.setdefault(k, v)
    return _sigma_observed_munuqq_raw(*args, **kwargs)


PLOT_DIR = "plots/ww_diagnostics"


# LEP-EWWG combined σ_WW (CC03 Born) measurements.
# Source: ALEPH/DELPHI/L3/OPAL Phys.Rept.532 (2013) 119 ("Electroweak
# Measurements in Electron-Positron Collisions at W-Boson-Pair Energies
# at LEP"), Table 5.1. Each row is (√s [GeV], σ_WW [pb], stat+syst
# combined uncertainty [pb]). Below 175 GeV: results from individual
# experiments. These are CC03 cross sections — i.e. doubly-resonant only,
# with singly-resonant W background subtracted. The numerical values
# below are accurate to the precision quoted in the paper; refine if a
# more recent combination is preferred.
LEP_DATA = {
    "sqrts_GeV":   np.array([161.33, 172.12, 182.66, 188.63, 191.58,
                              195.52, 199.51, 201.62, 204.86, 206.55]),
    "sigma_pb":    np.array([ 3.69,  11.93,  16.13,  16.21,  16.86,
                              16.93,  17.06,  16.66,  17.30,  16.94]),
    "sigma_err":   np.array([ 0.45,   0.74,   0.37,   0.21,   0.49,
                              0.32,   0.31,   0.42,   0.31,   0.27]),
}


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

    # LO inclusive BR factor consistent with sigma_partonic_munuqq:
    # 4/27 × (Γ_W^(0)(m_W)/Γ_W)². At default (m_W, Γ_W) this is
    # ≈ 0.142, vs PDG BR_INCLUSIVE_MUNUQQ ≈ 0.143 (0.6 % smaller).
    BR_LO_incl = (4.0 / 27.0) * (gamma_W_LO(mW) / gW) ** 2

    fig, (ax_abs, ax_rat) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1.5]})

    # Top: absolute σ (inclusive μν qq̄, in fb)
    bfs_curves = {}
    for o in orders:
        sigma_WW_bfs = _bfs_total_WW_order(s, mW, gW, o)
        sigma_incl = sigma_WW_bfs * BR_LO_incl
        bfs_curves[o] = sigma_incl * 1e3   # pb → fb
        ax_abs.plot(sqrts, bfs_curves[o], label=labels[o], color=colors[o],
                    linewidth=1.3, linestyle="-")

    # Framework curves
    ax_abs.plot(sqrts, sigma_WW * BR_LO_incl * 1e3,
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
                (sigma_WW * BR_LO_incl * 1e3) / np.maximum(denom, eps),
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


def plot_vs_LEP():
    """σ_WW (no BR multiplication) vs √s, comparing BFS LO_EFT Born and
    RACOONWW CC03 spline to the LEP-EWWG combined measurements
    (Phys.Rep.532 (2013) 119, Table 5.1). LEP quotes CC03 Born σ_WW
    after ISR-unfolding and singly-resonant subtraction, so the right
    benchmark is the RACOONWW spline; the BFS curve is full Born and is
    expected to lie ~13 % above LEP."""
    import matplotlib.pyplot as plt

    sqrts_pred = np.linspace(155.0, 210.0, 221)
    s = sqrts_pred ** 2
    mW, gW = M_W_DEFAULT, GAMMA_W_DEFAULT

    # σ_WW total — use the framework function which is BFS in 150-170 GeV
    # and RACOONWW spline above 170 GeV.
    from process.ww.bfs_eft import sigma_BFS_LO_total_WW_pb
    sigma_BFS_full = sigma_BFS_LO_total_WW_pb(s, mW, gW, order="N3/2LO")
    # Pure RACOONWW spline (CC03 Born) for reference.
    from process.ww.eft_xsec import _F_SPLINE, _F_GRID, sin2_thetaW_OS, alpha_Gmu, beta_complex
    alpha = alpha_Gmu(mW)
    sW2 = sin2_thetaW_OS(mW)
    bMr = beta_complex(s, mW, gW).real
    F = _F_SPLINE(s)
    sigma_RACOONWW = ((np.pi * alpha ** 2) / (sW2 ** 2 * s)
                      * np.maximum(bMr, 0.0) * F) * 3.8937937217e8  # GEV_M2_TO_PB

    # Framework's actual prediction (BFS<170, spline≥170)
    sigma_framework = sigma_WW_Born(s, mW, gW)

    fig, (ax_abs, ax_rat) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1.5]})

    ax_abs.plot(sqrts_pred, sigma_BFS_full, color="C3", linewidth=1.5,
                label=r"BFS LO_EFT N$^{3/2}$LO Born (full off-shell)")
    ax_abs.plot(sqrts_pred, sigma_RACOONWW, color="C0", linewidth=1.5,
                linestyle="--", label="RACOONWW spline (CC03 Born)")
    ax_abs.plot(sqrts_pred, sigma_framework, color="black", linewidth=2.0,
                linestyle=":",
                label=r"Framework (BFS for √s<170, RACOONWW above)")

    ax_abs.errorbar(LEP_DATA["sqrts_GeV"], LEP_DATA["sigma_pb"],
                    yerr=LEP_DATA["sigma_err"], fmt="o", color="black",
                    markersize=5, capsize=3, label="LEP-EWWG combined (CC03)")

    ax_abs.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_abs.axvline(170.0, color="grey", alpha=0.4, linestyle=":", linewidth=0.8)
    ax_abs.text(170.0, ax_abs.get_ylim()[1] * 0.05, "BFS→spline", color="grey",
                fontsize=8, ha="center", va="bottom", rotation=90)
    ax_abs.set_ylabel(r"$\sigma(e^+e^- \to W^+W^-)$ [pb]")
    ax_abs.set_title(r"WW total cross section: BFS Born vs RACOONWW vs LEP data  "
                      f"($m_W = {mW:.4f}$, $\\Gamma_W = {gW:.3f}$)")
    ax_abs.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_abs.grid(alpha=0.25)

    # Ratio panel: (prediction or data) / RACOONWW spline (CC03 reference)
    eps = 1e-9
    ax_rat.plot(sqrts_pred, sigma_BFS_full / np.maximum(sigma_RACOONWW, eps),
                color="C3", linewidth=1.5)
    ax_rat.plot(sqrts_pred, sigma_RACOONWW / np.maximum(sigma_RACOONWW, eps),
                color="C0", linewidth=1.5, linestyle="--")
    ax_rat.plot(sqrts_pred, sigma_framework / np.maximum(sigma_RACOONWW, eps),
                color="black", linewidth=2.0, linestyle=":")

    # LEP point ratios — recompute spline at LEP energies
    F_lep = _F_SPLINE(LEP_DATA["sqrts_GeV"] ** 2)
    bMr_lep = np.array([beta_complex(sq ** 2, mW, gW).real for sq in LEP_DATA["sqrts_GeV"]])
    sig_ref_lep = ((np.pi * alpha ** 2) / (sW2 ** 2 * LEP_DATA["sqrts_GeV"] ** 2)
                   * np.maximum(bMr_lep, 0.0) * F_lep) * 3.8937937217e8
    ax_rat.errorbar(LEP_DATA["sqrts_GeV"],
                    LEP_DATA["sigma_pb"] / sig_ref_lep,
                    yerr=LEP_DATA["sigma_err"] / sig_ref_lep,
                    fmt="o", color="black", markersize=4, capsize=2)

    ax_rat.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_rat.axvline(2 * mW, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_rat.axvline(170.0, color="grey", alpha=0.4, linestyle=":", linewidth=0.8)
    ax_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_rat.set_ylabel("ratio to RACOONWW (CC03)")
    ax_rat.set_ylim(0.8, 1.4)
    ax_rat.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "xsec_vs_LEP.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def plot_ratios_vs_mW_GammaW():
    """Two-panel ratio plot: σ_obs(s; m_W ± δ) / σ_obs(s; m_W) for a set of
    m_W variations, and same for Γ_W. Shows the lineshape distortion
    induced by physical-parameter shifts at the order of the FCC-ee
    target precision (a few MeV). Counterpart of the WbWb
    ``plot_parameter_variations`` figure."""
    import matplotlib.pyplot as plt

    sqrts = np.linspace(155.0, 170.0, 151)
    mW0, gW0 = M_W_DEFAULT, GAMMA_W_DEFAULT

    sigma_nom = sigma_observed_munuqq(sqrts, mW=mW0, gammaW=gW0, channel="inclusive")

    # ±10, ±30 MeV variations — same convention as WbWb's parameter-variation
    # plot (card mass/width "variation" defaults are 30 MeV).
    variations_MeV = [-30, -10, +10, +30]
    palette = {-30: "#08519c", -10: "#6baed6", +10: "#fb6a4a", +30: "#a50f15"}

    fig, (ax_mW, ax_gW) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)

    for d in variations_MeV:
        sigma_v = sigma_observed_munuqq(sqrts, mW=mW0 + 1e-3 * d, gammaW=gW0,
                                        channel="inclusive")
        ax_mW.plot(sqrts, sigma_v / sigma_nom,
                   color=palette[d], linewidth=1.6,
                   label=fr"$\delta m_W = {d:+d}$ MeV")
    ax_mW.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_mW.axvline(2 * mW0, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_mW.set_ylabel(r"$\sigma_{\rm obs}(m_W + \delta m_W) / \sigma_{\rm obs}(m_W)$")
    ax_mW.set_title(r"Lineshape sensitivity to $m_W$ and $\Gamma_W$  "
                     fr"($m_W = {mW0:.4f}$ GeV, $\Gamma_W = {gW0:.3f}$ GeV, with LL+YFS ISR)")
    ax_mW.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_mW.grid(alpha=0.25)

    for d in variations_MeV:
        sigma_v = sigma_observed_munuqq(sqrts, mW=mW0, gammaW=gW0 + 1e-3 * d,
                                        channel="inclusive")
        ax_gW.plot(sqrts, sigma_v / sigma_nom,
                   color=palette[d], linewidth=1.6,
                   label=fr"$\delta \Gamma_W = {d:+d}$ MeV")
    ax_gW.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_gW.axvline(2 * mW0, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_gW.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gW.set_ylabel(r"$\sigma_{\rm obs}(\Gamma_W + \delta\Gamma_W) / \sigma_{\rm obs}(\Gamma_W)$")
    ax_gW.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_gW.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "ratios_mW_GammaW.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def plot_normalised_xsec_hypotheses():
    """Normalised σ vs √s for different (m_W, Γ_W) hypotheses, shown as
    residuals from the nominal-hypothesis shape so the threshold-shape
    information is visible.

    For each hypothesis we define R_h(√s) = σ(√s; h) / σ(s_ref; h), with
    s_ref = 170 GeV (upper edge, well into the rise). Each curve R_h is
    its own self-normalised line-shape — independent of total
    luminosity / BR / overall scale. The plot shows R_h(√s) − R_nom(√s)
    so the nominal hypothesis is the zero line and the other hypotheses
    show the residual shape distortion induced by ±10, ±30 MeV
    variations of m_W / Γ_W. The ratios at s_ref are identically 1, so
    all curves cross 0 there by construction.

    Two panels: m_W variations (top) and Γ_W variations (bottom)."""
    import matplotlib.pyplot as plt

    sqrts = np.linspace(155.0, 170.0, 151)
    mW0, gW0 = M_W_DEFAULT, GAMMA_W_DEFAULT
    s_ref = 170.0
    variations_MeV = [-30, -10, +10, +30]
    palette = {-30: "#08519c", -10: "#6baed6",
               +10: "#fb6a4a", +30: "#a50f15"}

    # Nominal self-normalised shape (= R_nom)
    sigma_nom = sigma_observed_munuqq(sqrts, mW=mW0, gammaW=gW0,
                                       channel="inclusive")
    sigma_nom_ref = sigma_observed_munuqq(s_ref, mW=mW0, gammaW=gW0,
                                           channel="inclusive")
    R_nom = sigma_nom / sigma_nom_ref

    fig, (ax_mW, ax_gW) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)

    for d in variations_MeV:
        sigma_v = sigma_observed_munuqq(sqrts, mW=mW0 + 1e-3 * d, gammaW=gW0,
                                        channel="inclusive")
        sigma_v_ref = sigma_observed_munuqq(s_ref, mW=mW0 + 1e-3 * d, gammaW=gW0,
                                            channel="inclusive")
        R_h = sigma_v / sigma_v_ref
        ax_mW.plot(sqrts, R_h - R_nom,
                   color=palette[d], linewidth=1.6,
                   label=fr"$\delta m_W = {d:+d}$ MeV")
    ax_mW.axhline(0.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_mW.axvline(2 * mW0, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_mW.axvline(s_ref, color="grey", alpha=0.4, linestyle=":", linewidth=0.8)
    ax_mW.set_ylabel(fr"$R_h(\sqrt{{s}}) - R_{{\rm nom}}(\sqrt{{s}})$"
                      "\n"
                      fr"$R_h \equiv \sigma_h(\sqrt{{s}})/\sigma_h({s_ref:.0f}\,\mathrm{{GeV}})$,  $h = m_W + \delta m_W$")
    ax_mW.set_title(r"Self-normalised line-shape residuals: $m_W$ and $\Gamma_W$ hypotheses  "
                     fr"(nominal $m_W = {mW0:.4f}$, $\Gamma_W = {gW0:.3f}$ GeV; with LL+YFS ISR)")
    ax_mW.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_mW.grid(alpha=0.25)

    for d in variations_MeV:
        sigma_v = sigma_observed_munuqq(sqrts, mW=mW0, gammaW=gW0 + 1e-3 * d,
                                        channel="inclusive")
        sigma_v_ref = sigma_observed_munuqq(s_ref, mW=mW0, gammaW=gW0 + 1e-3 * d,
                                            channel="inclusive")
        R_h = sigma_v / sigma_v_ref
        ax_gW.plot(sqrts, R_h - R_nom,
                   color=palette[d], linewidth=1.6,
                   label=fr"$\delta \Gamma_W = {d:+d}$ MeV")
    ax_gW.axhline(0.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_gW.axvline(2 * mW0, color="grey", alpha=0.4, linestyle="--", linewidth=0.8)
    ax_gW.axvline(s_ref, color="grey", alpha=0.4, linestyle=":", linewidth=0.8)
    ax_gW.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gW.set_ylabel(fr"$R_h(\sqrt{{s}}) - R_{{\rm nom}}(\sqrt{{s}})$"
                      "\n"
                      fr"$R_h \equiv \sigma_h(\sqrt{{s}})/\sigma_h({s_ref:.0f}\,\mathrm{{GeV}})$,  $h = \Gamma_W + \delta\Gamma_W$")
    ax_gW.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_gW.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "normalised_xsec_hypotheses.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def plot_azzurri_style_pm1GeV():
    """Reproduction of Fig. 1 of Azzurri 2107.04444: σ_WW vs √s with
    ±1 GeV variations on m_W and Γ_W. Two panels:

      * m_W variations (purple band in the paper): central, m_W ± 1 GeV
        at fixed Γ_W = 2.085. The ±1 GeV variation shifts the lineshape
        along the energy axis (threshold position moves).

      * Γ_W variations (green band in the paper): central, Γ_W ± 1 GeV
        at fixed m_W = 80.385. The paper claims a "crossing point" at
        E_CM ≈ 2 m_W + 1.5 GeV ≈ 162.3 GeV where σ_WW is insensitive
        to Γ_W.

    Uses the paper's central values m_W = 80.385, Γ_W = 2.085 (slightly
    different from the framework's M_W_DEFAULT = 80.3692, GAMMA_W_DEFAULT
    = 2.085). σ_WW here is σ(e+e- → W+W-) (total, with LL+YFS ISR) —
    derived from σ_observed_munuqq / BR_INCLUSIVE_MUNUQQ since the
    default BR is PDG-constant and ISR commutes with the BR scaling."""
    import matplotlib.pyplot as plt

    sqrts = np.linspace(155.0, 170.0, 121)
    mW_paper = 80.385
    gW_paper = 2.085

    def _sigma_WW_obs(sqrts_arr, mW, gW):
        # σ_observed_munuqq is σ_WW_obs × BR_INCLUSIVE_MUNUQQ in PDG default.
        # Divide back to recover σ_WW_obs (total W-pair, with ISR).
        return sigma_observed_munuqq(sqrts_arr, mW=mW, gammaW=gW,
                                      channel="inclusive") / BR_INCLUSIVE_MUNUQQ

    sigma_nom = _sigma_WW_obs(sqrts, mW_paper, gW_paper)
    sigma_mW_p = _sigma_WW_obs(sqrts, mW_paper + 1.0, gW_paper)
    sigma_mW_m = _sigma_WW_obs(sqrts, mW_paper - 1.0, gW_paper)
    sigma_gW_p = _sigma_WW_obs(sqrts, mW_paper, gW_paper + 1.0)
    sigma_gW_m = _sigma_WW_obs(sqrts, mW_paper, gW_paper - 1.0)

    fig, (ax_mW, ax_gW) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    # --- m_W panel ---
    ax_mW.fill_between(sqrts, sigma_mW_m, sigma_mW_p,
                        color="#9e6bbf", alpha=0.35,
                        label=r"$m_W \pm 1$ GeV band")
    ax_mW.plot(sqrts, sigma_nom, color="black", linewidth=1.8,
                label=rf"central: $m_W={mW_paper:.3f}$, $\Gamma_W={gW_paper:.3f}$ GeV")
    ax_mW.plot(sqrts, sigma_mW_p, color="#54278f", linewidth=1.0,
                linestyle="--", label=rf"$m_W={mW_paper+1:.3f}$ GeV")
    ax_mW.plot(sqrts, sigma_mW_m, color="#54278f", linewidth=1.0,
                linestyle=":",  label=rf"$m_W={mW_paper-1:.3f}$ GeV")
    ax_mW.axvline(2 * mW_paper, color="grey", alpha=0.4,
                   linestyle="--", linewidth=0.8)
    ax_mW.text(2 * mW_paper + 0.05, 0.1, r"$2\,m_W$", color="grey",
                alpha=0.7, fontsize=9, ha="left", va="bottom")
    ax_mW.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_mW.set_ylabel(r"$\sigma_{\rm WW}$ [pb]  (with LL+YFS ISR)")
    ax_mW.set_title(r"$m_W$ variation ($\pm 1$ GeV)")
    ax_mW.set_xlim(155, 170)
    ax_mW.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_mW.grid(alpha=0.25)

    # --- Γ_W panel ---
    ax_gW.fill_between(sqrts, sigma_gW_m, sigma_gW_p,
                        color="#4daf4a", alpha=0.30,
                        label=r"$\Gamma_W \pm 1$ GeV band")
    ax_gW.plot(sqrts, sigma_nom, color="black", linewidth=1.8,
                label=rf"central: $m_W={mW_paper:.3f}$, $\Gamma_W={gW_paper:.3f}$ GeV")
    ax_gW.plot(sqrts, sigma_gW_p, color="#1b7837", linewidth=1.0,
                linestyle="--", label=rf"$\Gamma_W={gW_paper+1:.3f}$ GeV")
    ax_gW.plot(sqrts, sigma_gW_m, color="#1b7837", linewidth=1.0,
                linestyle=":",  label=rf"$\Gamma_W={gW_paper-1:.3f}$ GeV")
    ax_gW.axvline(2 * mW_paper, color="grey", alpha=0.4,
                   linestyle="--", linewidth=0.8)
    ax_gW.axvline(162.3, color="red", alpha=0.6,
                   linestyle="-.", linewidth=0.9,
                   label=r"Azzurri 'crossing' $\approx 162.3$ GeV")
    ax_gW.text(2 * mW_paper + 0.05, 0.1, r"$2\,m_W$", color="grey",
                alpha=0.7, fontsize=9, ha="left", va="bottom")
    ax_gW.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gW.set_title(r"$\Gamma_W$ variation ($\pm 1$ GeV)")
    ax_gW.set_xlim(155, 170)
    ax_gW.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_gW.grid(alpha=0.25)

    plt.suptitle("Azzurri 2107.04444 Fig. 1 style — BFS LO_EFT N$^{3/2}$LO Born + Coulomb + LL+YFS ISR",
                  fontsize=11)
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "azzurri_style_pm1GeV.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# Local import for the Azzurri plot helper above
from process.ww.eft_xsec import BR_INCLUSIVE_MUNUQQ


def plot_bes_effect_on_variations():
    """Effect of FCC-ee BES on σ_WW for m_W ± 1 GeV and Γ_W ± 1 GeV variations.

    Shows directly whether the beam-energy spread smooths out the parameter-
    dependence signal. Two rows × two columns:

      * Top row: m_W ± 1 GeV variations, no-BES (left) vs +BES (right)
      * Bottom row: Γ_W ± 1 GeV variations, no-BES vs +BES

    Same y-axis scale across panels so the user can visually verify that
    the band shape is preserved under BES smearing."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from common.smearing import convolute_gauss

    BES_pct = 0.105   # FCC FSR Vol 1 Table 14 (W+W- BS)
    mW0, gW0 = 80.385, 2.085

    sqrts = np.linspace(150.0, 175.0, 2501)   # 10 MeV pitch
    sqrts_show = (sqrts >= 155.0) & (sqrts <= 170.0)

    def _sigma_WW(mW, gW):
        return sigma_observed_munuqq(sqrts, mW=mW, gammaW=gW,
                                      channel="inclusive") / BR_INCLUSIVE_MUNUQQ

    def _smear(arr):
        df = pd.DataFrame({"ecm": sqrts, "xsec": arr})
        return convolute_gauss(df, BES_pct, peak_ecm=162.5).to_numpy()[:, 1]

    sigma_nom         = _sigma_WW(mW0, gW0)
    sigma_nom_bes     = _smear(sigma_nom)
    sigma_mW_p        = _sigma_WW(mW0 + 1.0, gW0)
    sigma_mW_p_bes    = _smear(sigma_mW_p)
    sigma_mW_m        = _sigma_WW(mW0 - 1.0, gW0)
    sigma_mW_m_bes    = _smear(sigma_mW_m)
    sigma_gW_p        = _sigma_WW(mW0, gW0 + 1.0)
    sigma_gW_p_bes    = _smear(sigma_gW_p)
    sigma_gW_m        = _sigma_WW(mW0, gW0 - 1.0)
    sigma_gW_m_bes    = _smear(sigma_gW_m)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    (ax_mNo, ax_mYes), (ax_gNo, ax_gYes) = axs

    sq = sqrts[sqrts_show]
    def _plot_band(ax, lo, hi, central, color_lo, color_hi, color_band,
                    lbl_lo, lbl_hi, lbl_cen):
        ax.fill_between(sq, lo[sqrts_show], hi[sqrts_show],
                         color=color_band, alpha=0.30,
                         label=r"$\pm 1$ GeV band")
        ax.plot(sq, central[sqrts_show], color="black", linewidth=1.7,
                 label=lbl_cen)
        ax.plot(sq, hi[sqrts_show], color=color_hi, linewidth=1.0,
                 linestyle="--", label=lbl_hi)
        ax.plot(sq, lo[sqrts_show], color=color_lo, linewidth=1.0,
                 linestyle=":", label=lbl_lo)
        ax.axvline(2 * mW0, color="grey", alpha=0.3,
                    linestyle="--", linewidth=0.7)
        ax.set_xlim(155, 170)
        ax.set_ylim(0, 12)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.grid(alpha=0.25)

    # m_W variations
    _plot_band(ax_mNo, sigma_mW_m, sigma_mW_p, sigma_nom,
               "#54278f", "#54278f", "#9e6bbf",
               rf"$m_W={mW0-1:.3f}$ GeV", rf"$m_W={mW0+1:.3f}$ GeV",
               rf"central")
    _plot_band(ax_mYes, sigma_mW_m_bes, sigma_mW_p_bes, sigma_nom_bes,
               "#54278f", "#54278f", "#9e6bbf",
               rf"$m_W={mW0-1:.3f}$ GeV", rf"$m_W={mW0+1:.3f}$ GeV",
               rf"central")
    ax_mNo.set_ylabel(r"$\sigma_{\rm WW}$ [pb]")
    ax_mNo.set_title(r"$m_W \pm 1$ GeV: no BES (LL+YFS ISR only)")
    ax_mYes.set_title(rf"$m_W \pm 1$ GeV: + FCC-ee BES (0.105 % / beam)")

    # Γ_W variations
    _plot_band(ax_gNo, sigma_gW_m, sigma_gW_p, sigma_nom,
               "#1b7837", "#1b7837", "#4daf4a",
               rf"$\Gamma_W={gW0-1:.3f}$ GeV", rf"$\Gamma_W={gW0+1:.3f}$ GeV",
               rf"central")
    _plot_band(ax_gYes, sigma_gW_m_bes, sigma_gW_p_bes, sigma_nom_bes,
               "#1b7837", "#1b7837", "#4daf4a",
               rf"$\Gamma_W={gW0-1:.3f}$ GeV", rf"$\Gamma_W={gW0+1:.3f}$ GeV",
               rf"central")
    ax_gNo.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gNo.set_ylabel(r"$\sigma_{\rm WW}$ [pb]")
    ax_gYes.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gNo.set_title(r"$\Gamma_W \pm 1$ GeV: no BES")
    ax_gYes.set_title(rf"$\Gamma_W \pm 1$ GeV: + FCC-ee BES (0.105 % / beam)")

    plt.suptitle(r"BES smearing effect on $m_W$ and $\Gamma_W$ variations  "
                  rf"(central $m_W = {mW0:.3f}$, $\Gamma_W = {gW0:.3f}$ GeV)",
                  fontsize=11)
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "bes_effect_on_variations.pdf")
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
    plot_vs_LEP()
    plot_ratios_vs_mW_GammaW()
    plot_normalised_xsec_hypotheses()
    plot_azzurri_style_pm1GeV()
    plot_bes_effect_on_variations()
    print("Done.")


if __name__ == "__main__":
    main()
