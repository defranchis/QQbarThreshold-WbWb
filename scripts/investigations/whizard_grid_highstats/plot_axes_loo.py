"""1D interpolation residual on the m_W and Γ_W axes of the highstats grid.

For each of m_W and Γ_W, at fixed nominal value of the other and 4 √s
slices spanning the threshold:

    for each held-out point k:
        fit linear and quadratic LSQ through the remaining points
        evaluate at the held-out value
        residual = (fit − truth) / truth × 100

Quantifies how well a polynomial in either axis reproduces the data at
the off-grid midpoints — the relevant test for choosing the operational
m_W / Γ_W interpolation order (quadratic is sufficient on highstats; see
plot_morphing.py for the next layer that absorbs the joint coupling).

The Γ_W LOO uses the 5 uniformly-spaced PDG values; the BFS-reference
duplicates (2.04483, 2.09201) are excluded because the irregular
sub-MeV spacing destabilises the polynomial fits.

Two output PNGs:
    axes_loo_mw.png    — m_W axis LOO, 9 highstats m_W points
    axes_loo_gw.png    — Γ_W axis LOO, 5 PDG uniform Γ_W points
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from morph import filter_uniform_gw, load_grid, MW0, GW0

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                      # WW_threshold/
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"

SQRTS_SLICES = [156.0, 161.0, 162.5, 167.0]


def loo_polynomials(x, y):
    """Leave-one-out predictions from linear and quadratic LSQ fits."""
    n = len(x)
    lin, quad = np.empty(n), np.empty(n)
    for k in range(n):
        m = np.ones(n, dtype=bool); m[k] = False
        lin[k]  = np.polyval(np.polyfit(x[m], y[m], 1), x[k])
        quad[k] = np.polyval(np.polyfit(x[m], y[m], 2), x[k])
    return lin, quad


def plot_loo_axis(df: pd.DataFrame, *, varied: str, fixed_val: float,
                  varied_label: str, x_col: str, fixed_col: str,
                  out_png: Path, out_pdf: Path, title_suffix: str):
    """Render the 4-panel LOO plot for one axis (varied) with the other axis
    held at `fixed_val`."""
    fig, axes = plt.subplots(1, len(SQRTS_SLICES),
                              figsize=(7 * len(SQRTS_SLICES), 5.5), sharey=True)
    summary = []
    for ax, s in zip(axes, SQRTS_SLICES):
        sl = df[(np.isclose(df.sqrts, s, atol=1e-3))
                 & (np.isclose(df[fixed_col], fixed_val, atol=1e-3))].sort_values(x_col)
        if len(sl) < 4:
            ax.set_title(f"√s={s} (no data)"); continue
        x, y, err = sl[x_col].values, sl.sigma_fb.values, sl.err_fb.values
        lin_pred, quad_pred = loo_polynomials(x, y)
        lin_res  = (lin_pred  - y) / y * 100
        quad_res = (quad_pred - y) / y * 100
        mc_pct   = (err       / y) * 100
        interior = np.ones(len(x), dtype=bool); interior[0] = interior[-1] = False

        ax.errorbar(x, lin_res,  yerr=mc_pct, fmt="o", color="C0",
                    markersize=6, capsize=2, label="linear LOO")
        ax.errorbar(x, quad_res, yerr=mc_pct, fmt="s", color="C2",
                    markersize=6, capsize=2, label="quadratic LOO")
        ax.axhspan(-mc_pct.max(), mc_pct.max(), color="grey", alpha=0.10,
                   label=r"$\pm$MC stat")
        ax.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
        ax.axvspan(x[0]  - (x[1]  - x[0])  * 0.4, x[0],  color="orange", alpha=0.06)
        ax.axvspan(x[-1], x[-1] + (x[-1] - x[-2]) * 0.4, color="orange", alpha=0.06)
        ax.set_title(rf"$\sqrt{{s}}={s:.1f}$ GeV,  {title_suffix}")
        ax.set_xlabel(rf"held-out ${varied_label}$ [GeV]")
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel(r"$(\hat\sigma_{\rm LOO} - \sigma_{\rm truth})/\sigma$ [%]")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        summary.append((s,
                        float(np.abs(lin_res[interior]).max()),
                        float(np.abs(quad_res[interior]).max()),
                        float(mc_pct.max())))

    fig.suptitle(f"LOO residuals on highstats grid — {varied} axis", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n{varied} axis LOO summary (interior held-outs only):")
    print(f"{'√s':>6}  {'linear max':>11} {'quad max':>10} {'MC max':>8}")
    for s, lin_max, quad_max, mc in summary:
        print(f"{s:>6.1f}    {lin_max:>9.4f}%   {quad_max:>8.4f}%   {mc:>6.4f}%")
    print(f"wrote {out_png}")


def main():
    df = filter_uniform_gw(load_grid(HIGHSTATS_CSV))
    print(f"highstats: {len(df)} rows, "
          f"{df.mW.nunique()} m_W × {df.gammaW.nunique()} Γ_W × {df.sqrts.nunique()} √s")

    plot_loo_axis(df, varied="m_W", fixed_val=GW0, varied_label="m_W",
                  x_col="mW", fixed_col="gammaW",
                  out_png=PLOTS / "axes_loo_mw.png",
                  out_pdf=PLOTS / "axes_loo_mw.pdf",
                  title_suffix=rf"$\Gamma_W={GW0:.3f}$ GeV")

    plot_loo_axis(df, varied="Γ_W", fixed_val=MW0, varied_label="\\Gamma_W",
                  x_col="gammaW", fixed_col="mW",
                  out_png=PLOTS / "axes_loo_gw.png",
                  out_pdf=PLOTS / "axes_loo_gw.pdf",
                  title_suffix=rf"$m_W={MW0:.3f}$ GeV")


if __name__ == "__main__":
    main()
