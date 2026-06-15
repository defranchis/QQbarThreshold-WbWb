"""m_W / Γ_W response of the independent WW line shape — ±10 MeV, per channel,
combined, and vs the BFS chain.

Plots the ratio  σ(POI ± 10 MeV) / σ(POI nominal)  vs √s:
  • the three pure-WW channels (lnuqq, qqqq, μτ), +10 MeV, overlaid;
  • the combined pure-WW total (12·lnuqq + 4·qqqq + 9·μτ), ±10 MeV;
  • the BFS-EFT chain (sigma_observed_munuqq, pdg-constant BR), ±10 MeV.

Two figures (m_W, Γ_W).  σ̂(√ŝ) is denoised with a low-order polynomial fit (the
±10 MeV response, ~0.3 %, is below the ~0.5 % per-point MC error).

Physics point (Γ_W): the full off-shell MoCaNLO σ(4f) ∝ BR₁·BR₂ =
(Γ_partial/Γ_total)², so +δΓ_total cuts the rate by ≈ −2 δΓ/Γ_total even before
any line-shape change.  The BFS chain runs in *pdg-constant* mode (BR fixed), so
its Γ_W response is line-shape-only.  Multiplying BFS by (Γ_tot/(Γ_tot±δΓ))²
restores the BR effect and reconciles the two — demonstrating the difference is
the BR treatment, not the line shape.

Output: plots/indep_mocanlo/channel_response_{mass,width}.{pdf,png}
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.partonic_grid import load_grids
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS

CHANNELS = ("lnuqq", "qqqq", "mutau")
CH_LABEL = {"lnuqq": r"$\ell\nu q\bar q$", "qqqq": r"$q\bar q q\bar q$",
            "mutau": r"$\ell\nu\ell\nu$ ($\mu\tau$)"}
CH_COLOR = {"lnuqq": "tab:blue", "qqqq": "tab:red", "mutau": "tab:green"}

# Low edge pinned at 157.0: the partonic σ̂ grid floors at 156 GeV, so for
# √s ≲ 156.5 the ISR convolution truncates the radiative-return tail (σ̂=0 below
# the floor) — at √s=156.0 only the x=1 endpoint contributes, producing a
# spurious turnover.  157.0 is the lowest √s where the grid-based convolution
# matches the continuous BFS chain (<1e-5).  Extend the σ̂ grid below 156 to go
# lower.
SQRT_S = np.arange(157.0, 164.001, 0.1)
POLY_DEG = 3
MW0, GW0, STEP = 80.379, 2.085, 0.010   # GeV


def _poly_fn(ecm, sigma, err, lo, hi):
    w = 1.0 / np.maximum(err, 1e-9 * np.maximum(np.abs(sigma), 1.0))
    p = np.poly1d(np.polyfit(ecm, sigma, POLY_DEG, w=w))

    def fn(x):
        x = np.asarray(x, dtype=float)
        return np.clip(np.where((x >= lo) & (x <= hi), p(np.clip(x, lo, hi)), 0.0),
                       0.0, None)
    return fn


#: (m_W, Γ_W) [GeV] for each ±10 MeV varpoint, for the pdg-constant BR factor.
VP_VALS = {"nominal": (MW0, GW0), "mp10": (MW0 + STEP, GW0),
           "mm10": (MW0 - STEP, GW0), "wp10": (MW0, GW0 + STEP),
           "wm10": (MW0, GW0 - STEP)}


def mocanlo_channel(grids, channel, varpoint, cfg, br_convention="off-shell"):
    g = grids[(channel, varpoint)]
    lo, hi = g.ecm[0], g.ecm[-1]
    obs = isr_beta.sigma_observed(
        SQRT_S, _poly_fn(g.ecm, g.sigma_nlo, g.err_nlo, lo, hi), cfg)
    if br_convention == "pdg-constant":
        # divide out the off-shell BR² = (Γ_partial(m_W)/Γ_W)² ∝ m_W⁶/Γ_W²
        # (same universal factor as WWGeneratorMoCaNLO._br_factor)
        mW, gW = VP_VALS[varpoint]
        obs = obs * (gW / GW0) ** 2 * (MW0 / mW) ** 6
    return obs


def bfs_lineshapes():
    """BFS-EFT σ_observed (μνqq inclusive, pdg-constant BR, LL ISR for speed)
    at the 5 varpoints, on SQRT_S."""
    from framework.process.ww.generator import WWGenerator
    from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
    gen = WWGenerator(isr_nll=False, isr_emela_ll=False)

    def sig(mW, gW):
        return sigma_observed_munuqq(
            SQRT_S, mW=mW, gammaW=gW, channel=gen.channel,
            include_coulomb=gen.include_coulomb, bfs=gen.bfs,
            include_NLO_hard_decay=gen.include_NLO_hard_decay,
            include_BFS_NNLO=gen.include_BFS_NNLO, apply_delta_QCD=gen.apply_delta_QCD,
            alpha_s=gen.alpha_s, alpha_s_ref=gen.alpha_s,
            br_convention=gen.br_convention, apply_whizard_anchor=gen.apply_whizard_anchor,
            whizard_anchor_source=gen.whizard_anchor_source, isr_scheme=gen.isr_scheme,
            isr_nll=False, alpha_em=gen.alpha_em, alpha_em_isr=gen.alpha_em_isr,
            coulomb_kc_safe=gen.coulomb_kc_safe,
            decay_uses_full_born=gen.decay_uses_full_born,
            m_t=gen.m_t, M_H=gen.M_H, MZ=gen.MZ)

    return {"nominal": sig(MW0, GW0), "mp10": sig(MW0 + STEP, GW0),
            "mm10": sig(MW0 - STEP, GW0), "wp10": sig(MW0, GW0 + STEP),
            "wm10": sig(MW0, GW0 - STEP)}


def _plot_convention(grids, cfg, bfs, br_convention, outdir):
    """Produce the mass + width response figures for one BR convention.

    off-shell    : native σ(4f)∝BR²; BFS is line-shape-only so the Γ_W curves
                   differ — BFS×(Γ₀/Γ)² (dotted) is overlaid to show the BR effect.
    pdg-constant : MoCaNLO BR divided out → both calcs are BR-fixed, so the
                   MoCaNLO and BFS curves should now overlay for BOTH POIs.
    """
    vps = ("nominal", "mp10", "mm10", "wp10", "wm10")
    mc = {c: {v: mocanlo_channel(grids, c, v, cfg, br_convention) for v in vps}
          for c in CHANNELS}
    comb = {v: sum(PURE_WW_WEIGHTS[c] * mc[c][v] for c in CHANNELS) for v in vps}
    br_up = (GW0 / (GW0 + STEP)) ** 2
    br_dn = (GW0 / (GW0 - STEP)) ** 2
    pdg = br_convention == "pdg-constant"
    sfx = "_pdgconst" if pdg else ""
    conv_lbl = "pdg-constant BR" if pdg else "off-shell"

    for poi, up, dn, fname, is_width in (
        ("m_W", "mp10", "mm10", "channel_response_mass", False),
        (r"\Gamma_W", "wp10", "wm10", "channel_response_width", True),
    ):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for c in CHANNELS:
            ax.plot(SQRT_S, mc[c][up] / mc[c]["nominal"], color=CH_COLOR[c],
                    lw=1.2, alpha=0.85, label=f"{CH_LABEL[c]} +10")
        ax.plot(SQRT_S, comb[up] / comb["nominal"], color="black", lw=2.6,
                label="combined +10")
        ax.plot(SQRT_S, comb[dn] / comb["nominal"], color="black", lw=2.6,
                ls="--", label="combined −10")
        ax.plot(SQRT_S, bfs[up] / bfs["nominal"], color="magenta", lw=2.2,
                label="BFS +10")
        ax.plot(SQRT_S, bfs[dn] / bfs["nominal"], color="magenta", lw=2.2,
                ls="--", label="BFS −10")
        # In off-shell mode the BFS Γ_W curve lacks the BR rate effect; overlay
        # BFS×(Γ₀/Γ)² to show it. In pdg-constant mode MoCaNLO is itself BR-fixed,
        # so MoCaNLO and raw BFS should already coincide — no ×BR curve needed.
        if is_width and not pdg:
            ax.plot(SQRT_S, bfs[up] / bfs["nominal"] * br_up, color="darkcyan",
                    lw=1.8, ls=":", label=r"BFS +10 $\times\,(\Gamma_0/\Gamma)^2$")
            ax.plot(SQRT_S, bfs[dn] / bfs["nominal"] * br_dn, color="darkcyan",
                    lw=1.8, ls=":")
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
        ax.set_ylabel(rf"$\sigma({poi}\pm10\,$MeV$)\,/\,\sigma$(nominal)")
        ax.set_title(rf"Independent WW vs BFS ({conv_lbl}): "
                     rf"${poi}$ response of the line shape")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"{fname}{sfx}.{ext}"), dpi=140)
        plt.close(fig)

        print(f"=== [{conv_lbl}] {poi} response +10 MeV (ratio) ===")
        for s0 in (157, 159, 161, 162.5, 164):
            i = int(np.argmin(abs(SQRT_S - s0)))
            print(f"  √s={s0:6.1f}: MoCaNLO(comb)={comb[up][i]/comb['nominal'][i]:.5f}"
                  f"  BFS={bfs[up][i]/bfs['nominal'][i]:.5f}")


def main():
    grids = load_grids(scheme_alpha="gf", lepton_cut=None)
    cfg = isr_beta.ISRConfig()
    bfs = bfs_lineshapes()
    outdir = os.path.join(_REPO, "plots", "indep_mocanlo")
    os.makedirs(outdir, exist_ok=True)
    for br_convention in ("off-shell", "pdg-constant"):
        _plot_convention(grids, cfg, bfs, br_convention, outdir)
    print(f"\nwrote {outdir}/channel_response_{{mass,width}}{{,_pdgconst}}.{{pdf,png}}")


if __name__ == "__main__":
    main()
