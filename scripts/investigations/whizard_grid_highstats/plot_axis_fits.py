"""Dissection of the per-√s 1D morph fits — R_m(m_W) and R_Γ(Γ_W).

At every grid √s the morph models the m_W and Γ_W dependence (at the
nominal value of the other parameter) with an independent quadratic
LSQ fit. This script takes those fits apart:

  col 0  the fitted morph-factor family R = σ/σ_nom vs the parameter,
         one curve per √s (colour = √s) — shows how the curvature of
         the m_W / Γ_W response evolves across the threshold.
  col 1  the fractional residual (σ − quad fit)/σ vs the parameter,
         per √s — exposes any leftover structure the quadratic misses,
         against the per-point MC band.
  col 2  RMS fractional residual of a linear / quadratic / cubic fit
         vs √s — quantifies why quadratic is the right model order.

Row 0: m_W axis (9 grid points). Row 1: Γ_W axis (5 PDG-uniform points).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import load_grid, filter_uniform_gw, MW0, GW0

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"

CMAP = mpl.cm.viridis


def axis_slices(df, fixed_col, fixed_val, var_col):
    """For each √s, the (x, σ, err) of the 1D slice along var_col at the
    nominal value of fixed_col. Returns dict √s -> (x, sigma, err)."""
    out = {}
    sub = df[np.isclose(df[fixed_col], fixed_val)]
    for s in sorted(sub.sqrts.unique()):
        sl = sub[np.isclose(sub.sqrts, s, atol=1e-3)].sort_values(var_col)
        if len(sl) >= 4:
            out[s] = (sl[var_col].values, sl.sigma_fb.values, sl.err_fb.values)
    return out


def fit_row(axes, slices, x0, xlabel, rlabel, orders):
    """Fill one (3-panel) row for one morph axis."""
    ax_fam, ax_res, ax_mod = axes
    svals = np.array(sorted(slices))
    norm = mpl.colors.Normalize(vmin=svals.min(), vmax=svals.max())

    mc_bars, mod_rms = [], {d: [] for d in orders}
    for s in svals:
        x, y, e = slices[s]
        color = CMAP(norm(s))
        cq = np.polyfit(x, y, 2)
        ynom = np.polyval(cq, x0)
        xfine = np.linspace(x.min(), x.max(), 100)
        # family: fitted quadratic morph factor
        ax_fam.plot(xfine, np.polyval(cq, xfine) / ynom, "-", lw=0.8,
                    color=color, alpha=0.7)
        ax_fam.plot(x, y / ynom, "o", ms=2.5, color=color, alpha=0.6)
        # residual of the quadratic
        resid = (y - np.polyval(cq, x)) / y * 100
        ax_res.plot(x, resid, "-o", lw=0.7, ms=2.5, color=color, alpha=0.6)
        mc_bars.append(np.median(e / y) * 100)
        # model-order RMS residual
        for d in orders:
            cd = np.polyfit(x, y, d)
            r = (y - np.polyval(cd, x)) / y
            mod_rms[d].append(np.sqrt(np.mean(r ** 2)) * 100)

    mc = float(np.median(mc_bars))
    ax_res.axhspan(-mc, mc, color="grey", alpha=0.18, label=rf"$\pm$MC ({mc:.3f}%)")
    ax_res.axhline(0, color="grey", lw=0.6)
    ax_fam.axvline(x0, color="grey", ls=":", alpha=0.4)
    ax_res.axvline(x0, color="grey", ls=":", alpha=0.4)
    ax_fam.set_xlabel(xlabel); ax_fam.set_ylabel(rlabel)
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel(r"$(\sigma-\mathrm{quad})/\sigma$ [%]")
    ax_res.legend(loc="upper right", fontsize=8)
    for ax in (ax_fam, ax_res):
        ax.grid(alpha=0.25)

    style = {1: ("C0", "s", "linear"), 2: ("C3", "o", "quadratic"),
             3: ("C2", "^", "cubic")}
    for d in orders:
        col, mk, lab = style[d]
        ax_mod.plot(svals, mod_rms[d], mk + "-", ms=4, color=col, lw=1.0,
                    label=lab)
    ax_mod.axhline(mc, color="grey", ls="--", lw=0.9, label="MC bar")
    ax_mod.set_yscale("log")
    ax_mod.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_mod.set_ylabel("RMS fractional residual [%]")
    ax_mod.grid(alpha=0.25, which="both")
    ax_mod.legend(loc="best", fontsize=8)
    return norm


def main():
    df = load_grid(HIGHSTATS_CSV)
    if DENSIFY_CSV.exists():
        import pandas as pd
        df = pd.concat([df, load_grid(DENSIFY_CSV)], ignore_index=True)
    df_g = filter_uniform_gw(df)

    m_slices = axis_slices(df,   "gammaW", GW0, "mW")
    g_slices = axis_slices(df_g, "mW",     MW0, "gammaW")
    print(f"m_W axis: {len(m_slices)} √s slices (9 m_W each)")
    print(f"Γ_W axis: {len(g_slices)} √s slices (5 PDG-uniform Γ_W each)")

    fig, axes = plt.subplots(2, 3, figsize=(21, 11), constrained_layout=True)
    norm = fit_row(axes[0], m_slices, MW0, r"$m_W$ [GeV]", r"$R_m=\sigma/\sigma_0$",
                   orders=[1, 2, 3])
    fit_row(axes[1], g_slices, GW0, r"$\Gamma_W$ [GeV]",
            r"$R_\Gamma=\sigma/\sigma_0$", orders=[1, 2, 3])
    axes[0][0].set_title(r"$m_W$ morph factor — fitted family", fontsize=12)
    axes[0][1].set_title(r"$m_W$ — quadratic-fit residual", fontsize=12)
    axes[0][2].set_title(r"$m_W$ — fit-model order vs $\sqrt{s}$", fontsize=12)
    axes[1][0].set_title(r"$\Gamma_W$ morph factor — fitted family", fontsize=12)
    axes[1][1].set_title(r"$\Gamma_W$ — quadratic-fit residual", fontsize=12)
    axes[1][2].set_title(r"$\Gamma_W$ — fit-model order vs $\sqrt{s}$", fontsize=12)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right",
                      fraction=0.015, pad=0.01)
    cb.set_label(r"$\sqrt{s}$ [GeV]")

    fig.suptitle("Per-√s 1D morph fits dissected — quadratic model for the "
                 r"$m_W$ and $\Gamma_W$ dependence")
    out_pdf = PLOTS / "axis_fits.pdf"
    out_png = PLOTS / "axis_fits.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    # printed summary: worst quadratic RMS residual on each axis
    for name, sl, x0 in [("m_W", m_slices, MW0), ("Γ_W", g_slices, GW0)]:
        worst = 0.0
        for s, (x, y, e) in sl.items():
            cq = np.polyfit(x, y, 2)
            r = np.sqrt(np.mean(((y - np.polyval(cq, x)) / y) ** 2)) * 100
            worst = max(worst, r)
        print(f"  {name}: worst quadratic RMS residual = {worst:.4f}%")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
