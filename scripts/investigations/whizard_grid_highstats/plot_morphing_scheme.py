"""Morphing-scheme construction at fixed √s: naive independent quadratic
morphing, then the bilinear cross-term extension that absorbs the joint
(m_W, Γ_W) coupling at the threshold rise.

Two output plots, both showing doubly-off-axis residuals:

  morphing_naive.png     σ ≈ σ_nom × R_m(m_W) × R_Γ(Γ_W)
                          → up to 0.14% residual at √s ≈ 161 GeV (peak rise)

  morphing_bilinear.png  σ ≈ σ_nom × R_m × R_Γ × [1 + β·Δm·ΔΓ]
                          → residual drops to MC-bar level (~0.04%)

β is fitted at each √s from the doubly-off-axis residual of the naive
product; it is largest at the peak rise (β ≈ +0.25 /GeV² at 161 GeV),
near-zero above the peak. Operationally this adds exactly one extra
scalar per √s to the morph parameter set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from morph import filter_uniform_gw, load_operational_grid, MW0, GW0

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)

SQRTS_PANELS = [156.0, 161.0, 162.5, 167.0]


def fit_and_predict(slice_df: pd.DataFrame, *, use_cross: bool):
    """At one √s slice, fit per-axis quadratic morphs and optionally a
    bilinear cross term. Return (off-axis dataframe with prediction columns,
    β value)."""
    sl_m = slice_df[np.isclose(slice_df.gammaW, GW0)].sort_values("mW")
    sl_g = slice_df[np.isclose(slice_df.mW,    MW0)].sort_values("gammaW")
    nom  = slice_df[(np.isclose(slice_df.mW, MW0))
                    & (np.isclose(slice_df.gammaW, GW0))]
    if len(sl_m) < 3 or len(sl_g) < 3 or len(nom) == 0:
        return None, 0.0
    coef_m = np.polyfit(sl_m.mW.values,    sl_m.sigma_fb.values, 2)
    coef_g = np.polyfit(sl_g.gammaW.values, sl_g.sigma_fb.values, 2)
    sigma_nom_m = np.polyval(coef_m, MW0)
    sigma_nom_g = np.polyval(coef_g, GW0)
    sigma_nom   = float(nom.sigma_fb.iloc[0])

    off = slice_df[(~np.isclose(slice_df.mW, MW0))
                    & (~np.isclose(slice_df.gammaW, GW0))].copy()
    if len(off) == 0:
        return None, 0.0
    Rm = np.polyval(coef_m, off.mW.values)    / sigma_nom_m
    Rg = np.polyval(coef_g, off.gammaW.values) / sigma_nom_g
    sigma_pred_no_cross = sigma_nom * Rm * Rg

    beta = 0.0
    if use_cross:
        dm = off.mW.values    - MW0
        dG = off.gammaW.values - GW0
        A = sigma_pred_no_cross * dm * dG
        b = off.sigma_fb.values - sigma_pred_no_cross
        w = 1.0 / off.err_fb.values**2
        beta = (np.sum(w * A * b) / np.sum(w * A * A)) if np.sum(w * A * A) > 0 else 0.0
        off["sigma_pred"] = sigma_pred_no_cross * (
            1.0 + beta * (off.mW.values - MW0) * (off.gammaW.values - GW0))
    else:
        off["sigma_pred"] = sigma_pred_no_cross

    off["resid_pct"] = (off.sigma_pred - off.sigma_fb) / off.sigma_fb * 100
    off["mc_pct"]    = (off.err_fb / off.sigma_fb) * 100
    return off, beta


def plot_scheme(df: pd.DataFrame, *, use_cross: bool, out_png: Path, out_pdf: Path,
                title: str):
    fig, axes = plt.subplots(1, len(SQRTS_PANELS),
                              figsize=(7 * len(SQRTS_PANELS), 6), sharey=True)
    summary = []
    for ax, s in zip(axes, SQRTS_PANELS):
        off, beta = fit_and_predict(df[np.isclose(df.sqrts, s, atol=1e-3)],
                                     use_cross=use_cross)
        if off is None or len(off) == 0:
            ax.set_title(f"√s={s} (insufficient data)"); continue
        gw_vals = sorted(off.gammaW.unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(gw_vals)))
        for c, gw in zip(cmap, gw_vals):
            sel = np.isclose(off.gammaW.values, gw, atol=1e-3)
            ax.errorbar(off.mW.values[sel], off.resid_pct.values[sel],
                        yerr=off.mc_pct.values[sel], fmt="o", color=c,
                        markersize=6, capsize=2,
                        label=rf"$\Gamma_W={gw:.3f}$")
        mc_max = off.mc_pct.max()
        ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.10,
                   label=r"$\pm$MC stat")
        ax.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
        title_pieces = [rf"$\sqrt{{s}}={s:.1f}$ GeV"]
        if use_cross:
            title_pieces.append(rf"$\beta={beta:+.3f}$ /GeV²")
        ax.set_title(",  ".join(title_pieces))
        ax.set_xlabel(r"$m_W$ [GeV]")
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r"$(\sigma_{\rm pred} - \sigma_{\rm truth})/\sigma$ [%]")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
        summary.append((s, float(np.abs(off.resid_pct).max()),
                        float(np.median(np.abs(off.resid_pct))),
                        float(mc_max), beta))

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    print(f"\n{('bilinear' if use_cross else 'naive'):<10} morphing residuals:")
    print(f"{'√s':>6}  {'max|resid|':>11} {'med|resid|':>11} {'MC':>8}"
          + ('  β [/GeV²]' if use_cross else ''))
    for s, mx, md, mc, beta in summary:
        extra = f"  {beta:+.3f}" if use_cross else ""
        print(f"{s:>6.1f}    {mx:>9.4f}%  {md:>9.4f}% {mc:>6.4f}%{extra}")
    print(f"wrote {out_png}")


def main():
    df = filter_uniform_gw(load_operational_grid())

    plot_scheme(df, use_cross=False,
                out_png=PLOTS / "morphing_naive.png",
                out_pdf=PLOTS / "morphing_naive.pdf",
                title=r"Naive independent quadratic morphing  "
                      r"$\sigma \approx \sigma_0 \cdot R_m(m_W) \cdot R_\Gamma(\Gamma_W)$")

    plot_scheme(df, use_cross=True,
                out_png=PLOTS / "morphing_bilinear.png",
                out_pdf=PLOTS / "morphing_bilinear.pdf",
                title=r"Independent quadratic morphing + bilinear cross term  "
                      r"$\sigma_{\rm pred} = \sigma_0 \cdot R_m \cdot R_\Gamma \cdot"
                      r" [1 + \beta(\Delta m_W)(\Delta \Gamma_W)]$")


if __name__ == "__main__":
    main()
