"""Sub-step validation of the morphing scheme against held-out 1-MeV-step
(m_W, Γ_W) points at √s ∈ {161, 162, 163} GeV (the grid_validate/
campaign). The validation points are NOT used to build the morph — the
residual here directly bounds the m_W bias the morph would introduce at
sub-step resolutions.

  morph input:  grid_fine + outer 0.5-GeV wings (load_operational_grid)
  test:         grid_validate/grid.csv         (must exist or script
                                                 prints a no-op message)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (build_operational_morph, load_grid, sigma_morph,
                    GRID_VALIDATE_CSV, MW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)


def main():
    if not GRID_VALIDATE_CSV.exists():
        print(f"validate grid not found at {GRID_VALIDATE_CSV}.")
        print("Run `python3 whizard/aggregate.py --mode all --allow-gaps` "
              "after the grid_validate cluster jobs land.")
        return

    sqrts_axis, splines, _ = build_operational_morph()
    print(f"morph fitted at {len(sqrts_axis)} √s values "
          f"(grid_fine + outer 0.5-GeV wings)")

    df_val = load_grid(GRID_VALIDATE_CSV)
    print(f"validation set: {len(df_val)} points")
    df_val["sigma_pred"] = [float(sigma_morph(r.sqrts, r.mW, r.gammaW, splines=splines))
                            for r in df_val.itertuples()]
    df_val["resid_pct"]  = (df_val.sigma_pred - df_val.sigma_fb) / df_val.sigma_fb * 100
    df_val["mc_pct"]     = (df_val.err_fb / df_val.sigma_fb) * 100

    sqrts_panels = sorted(df_val.sqrts.unique())
    n = len(sqrts_panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    summary = []
    for ax, s in zip(axes, sqrts_panels):
        sub = df_val[np.isclose(df_val.sqrts, s, atol=1e-3)].sort_values(["mW", "gammaW"])
        gw_vals = sorted(sub.gammaW.unique())
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(gw_vals)))
        for c, gw in zip(cmap, gw_vals):
            sel = np.isclose(sub.gammaW.values, gw, atol=1e-4)
            ax.errorbar(sub.mW.values[sel], sub.resid_pct.values[sel],
                        yerr=sub.mc_pct.values[sel], fmt="o", color=c,
                        markersize=6, capsize=2, label=rf"$\Gamma_W={gw:.4f}$")
        mc_max = sub.mc_pct.max()
        ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.10, label=r"$\pm$MC stat")
        ax.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
        ax.axvline(MW0, color="grey", linestyle=":", alpha=0.3)
        ax.set_title(rf"$\sqrt{{s}}={s:.0f}$ GeV")
        ax.set_xlabel(r"$m_W$ [GeV] (1-MeV step)")
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            # loc="center": mplhep's top-anchored ylabel hangs below the axes
            # for a label this long and gets clipped by bbox_inches="tight".
            ax.set_ylabel(r"$(\sigma_{\rm morph} - \sigma_{\rm WHIZARD})/\sigma$ [%]",
                          loc="center")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        summary.append((s, float(np.max(np.abs(sub.resid_pct))),
                        float(np.median(np.abs(sub.resid_pct))), float(mc_max), len(sub)))

    fig.suptitle("Sub-step validation: morph vs WHIZARD at 1-MeV (m_W, Γ_W) offsets",
                 y=1.02)
    # w_pad: the edge x-tick labels (80.377/80.381) of adjacent panels merge
    # at the default panel separation.
    fig.tight_layout(w_pad=3.0)
    out_pdf = PLOTS / "validate_substep.pdf"
    out_png = PLOTS / "validate_substep.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    print(f"\n{'√s':>6}  {'max|resid|':>12} {'median|resid|':>15} {'MC bar':>9} {'n':>4}")
    for s, mx, md, mc, n in summary:
        print(f"{s:>6.0f}    {mx:>10.4f}%    {md:>11.4f}%     {mc:>6.4f}%   {n:>3}")
    print(f"\nwrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
