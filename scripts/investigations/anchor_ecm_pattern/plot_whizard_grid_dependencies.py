"""Plot pure WHIZARD m_W and Γ_W dependence read directly from the
1295-pt grid (`whizard/work/grid/grid.csv`). No EFT chain, no anchor,
no ISR — just σ_Whiz(s, m_W, Γ_W) for the BFS specific channel
e+ e− → μ⁻ ν̄_μ u d̄.

Four panels:
  (a) σ vs √s at several m_W values (Γ_W fixed at 2.085).
  (b) σ(m_W) / σ(m_W = 80.379) vs √s, same Γ_W slice.
  (c) σ vs √s at several Γ_W values (m_W fixed at 80.379).
  (d) σ(Γ_W) / σ(Γ_W = 2.085) vs √s, same m_W slice.

This is the apples-to-apples "ground truth" for the anchor refactor
discussion: what WHIZARD actually predicts at the grid nodes, no
interpretation.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
GRID_CSV = Path(__file__).resolve().parents[3].parent / "whizard" / "work" / "grid" / "grid.csv"
OUT_DIR = HERE

MW_NOMINAL = 80.379
GW_NOMINAL = 2.085


def load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None,
                     names=["sqrts", "sigma_fb", "err_fb", "mW", "gammaW"])
    return df


def _slice(df, *, mW=None, gammaW=None, tol=1e-4):
    sub = df
    if mW is not None:
        sub = sub[np.isclose(sub.mW, mW, atol=tol)]
    if gammaW is not None:
        sub = sub[np.isclose(sub.gammaW, gammaW, atol=tol)]
    return sub.sort_values("sqrts")


def plot_dependencies():
    df = load_grid(GRID_CSV)
    mW_values = sorted(df.mW.unique())
    gW_values = sorted(df.gammaW.unique())
    print(f"grid m_W values: {mW_values}")
    print(f"grid Γ_W values: {gW_values}")

    fig, axes = plt.subplots(3, 2, figsize=(15, 16), sharex=True)
    (ax_m_abs, ax_m_rat), (ax_g_abs, ax_g_rat), (ax_gWW_abs, ax_gWW_rat) = axes

    # --- m_W slice at Γ_W = 2.085 -----------------------------------------
    palette_m = plt.cm.viridis(np.linspace(0.05, 0.9, len(mW_values)))
    nom_m = _slice(df, mW=MW_NOMINAL, gammaW=GW_NOMINAL)
    sqrts_m = nom_m.sqrts.values
    sig_nom_m = nom_m.sigma_fb.values

    for color, mw in zip(palette_m, mW_values):
        sub = _slice(df, mW=mw, gammaW=GW_NOMINAL)
        if len(sub) == 0:
            continue
        label = rf"$m_W = {mw:.3f}$ GeV"
        if np.isclose(mw, MW_NOMINAL):
            label += "  (nominal)"
        ax_m_abs.plot(sub.sqrts, sub.sigma_fb, "o-", color=color,
                      linewidth=1.4, markersize=3.5, label=label)
        if len(sub) == len(sig_nom_m):
            ax_m_rat.plot(sub.sqrts, sub.sigma_fb.values / sig_nom_m,
                          "o-", color=color, linewidth=1.4, markersize=3.5,
                          label=label)

    for ax in (ax_m_abs, ax_m_rat):
        ax.axvline(2 * MW_NOMINAL, color="grey", linestyle="--",
                   alpha=0.4, linewidth=0.8)
    ax_m_abs.set_ylabel(r"$\sigma_{\rm Whiz}(\mu\nu u\bar d)$ [fb]")
    ax_m_abs.set_title(rf"$m_W$ scan, $\Gamma_W$ fixed at {GW_NOMINAL:.3f} GeV")
    ax_m_abs.set_yscale("log")
    ax_m_abs.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_m_abs.grid(alpha=0.25, which="both")
    ax_m_rat.set_ylabel(r"$\sigma(m_W) / \sigma(m_W = 80.379)$")
    ax_m_rat.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_m_rat.legend(loc="best", fontsize=9, framealpha=0.9)
    ax_m_rat.grid(alpha=0.25)

    # --- Γ_W slice at m_W = 80.379 ----------------------------------------
    palette_g = plt.cm.plasma(np.linspace(0.05, 0.9, len(gW_values)))
    nom_g = _slice(df, mW=MW_NOMINAL, gammaW=GW_NOMINAL)
    sig_nom_g = nom_g.sigma_fb.values

    for color, gw in zip(palette_g, gW_values):
        sub = _slice(df, mW=MW_NOMINAL, gammaW=gw)
        if len(sub) == 0:
            continue
        label = rf"$\Gamma_W = {gw:.4f}$ GeV"
        if np.isclose(gw, GW_NOMINAL):
            label += "  (nominal)"
        ax_g_abs.plot(sub.sqrts, sub.sigma_fb, "o-", color=color,
                      linewidth=1.4, markersize=3.5, label=label)
        if len(sub) == len(sig_nom_g):
            ax_g_rat.plot(sub.sqrts, sub.sigma_fb.values / sig_nom_g,
                          "o-", color=color, linewidth=1.4, markersize=3.5,
                          label=label)

    for ax in (ax_g_abs, ax_g_rat):
        ax.axvline(2 * MW_NOMINAL, color="grey", linestyle="--",
                   alpha=0.4, linewidth=0.8)
    ax_g_abs.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_g_abs.set_ylabel(r"$\sigma_{\rm Whiz}(\mu\nu u\bar d)$ [fb]")
    ax_g_abs.set_title(rf"$\Gamma_W$ scan, $m_W$ fixed at {MW_NOMINAL:.3f} GeV")
    ax_g_abs.set_yscale("log")
    ax_g_abs.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_g_abs.grid(alpha=0.25, which="both")
    ax_g_rat.set_ylabel(r"$\sigma(\Gamma_W) / \sigma(\Gamma_W = 2.085)$")
    ax_g_rat.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_g_rat.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_g_rat.grid(alpha=0.25)

    # --- σ_WW (BR-stripped): σ_grid × 27 × (Γ_W/Γ_W^(0))² -----------------
    # WHIZARD's specific σ has an implicit (Γ_W^(0)/Γ_W)² BR factor because
    # partial widths are held at SM-LO while we vary total Γ_W. Multiplying
    # by (Γ_W/Γ_W^(0))² recovers σ(e+e- → W+W-) — the pure W-pair production
    # cross section that Azzurri 2107.04444 plots.
    GW_LO = 2.04483   # Γ_W^(0)(m_W=80.377) per BFS Table 1
    for color, gw in zip(palette_g, gW_values):
        sub = _slice(df, mW=MW_NOMINAL, gammaW=gw)
        if len(sub) == 0:
            continue
        br_strip = (gw / GW_LO) ** 2
        sigma_WW = sub.sigma_fb.values * 27.0 * br_strip
        label = rf"$\Gamma_W = {gw:.4f}$ GeV"
        if np.isclose(gw, GW_NOMINAL):
            label += "  (nominal)"
        ax_gWW_abs.plot(sub.sqrts, sigma_WW, "o-", color=color,
                        linewidth=1.4, markersize=3.5, label=label)
        sub_nom = _slice(df, mW=MW_NOMINAL, gammaW=GW_NOMINAL)
        sigma_WW_nom = sub_nom.sigma_fb.values * 27.0 * (GW_NOMINAL/GW_LO)**2
        if len(sigma_WW) == len(sigma_WW_nom):
            ax_gWW_rat.plot(sub.sqrts, sigma_WW / sigma_WW_nom,
                            "o-", color=color, linewidth=1.4, markersize=3.5,
                            label=label)

    for ax in (ax_gWW_abs, ax_gWW_rat):
        ax.axvline(2 * MW_NOMINAL, color="grey", linestyle="--",
                   alpha=0.4, linewidth=0.8)
        ax.axvline(162.3, color="red", linestyle="-.", alpha=0.5,
                   linewidth=0.9)
    ax_gWW_abs.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gWW_abs.set_ylabel(r"$\sigma_{WW}$ [fb]  (BR$^2$ stripped)")
    ax_gWW_abs.set_title(r"$\sigma(e^+e^- \to W^+W^-)$ from $\sigma_{\rm grid}\times 27\times(\Gamma_W/\Gamma_W^{(0)})^2$")
    ax_gWW_abs.set_yscale("log")
    ax_gWW_abs.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_gWW_abs.grid(alpha=0.25, which="both")
    ax_gWW_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_gWW_rat.set_ylabel(r"$\sigma_{WW}(\Gamma_W) / \sigma_{WW}(\Gamma_W = 2.085)$")
    ax_gWW_rat.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
    ax_gWW_rat.text(162.3, ax_gWW_rat.get_ylim()[1] * 0.99 if False else 1.012,
                    "  Azzurri 162.3 GeV", color="red", alpha=0.6,
                    fontsize=8, ha="left", va="top")
    ax_gWW_rat.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_gWW_rat.grid(alpha=0.25)

    fig.suptitle("WHIZARD 3.1.5 grid — direct $m_W$ and $\\Gamma_W$ dependence "
                 "(no EFT, no anchor, no ISR)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "whizard_grid_mW_gammaW_dependence"
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=140)
    print(f"wrote {base}.pdf + .png")


if __name__ == "__main__":
    plot_dependencies()
