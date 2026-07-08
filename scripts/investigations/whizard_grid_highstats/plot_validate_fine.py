"""Sub-MeV held-out closure of the morphing scheme against grid_validate_fine.

The grid_validate_fine campaign generated dedicated WHIZARD points at
sub-MeV (m_W, Γ_W) offsets from the morph nominal — 0.1-MeV steps to
±1 MeV and 0.2-MeV steps to ±3 MeV, as 1D scans of m_W (at Γ_W^0) and
Γ_W (at m_W^0), at √s ∈ {159, 161, 163} GeV. MC precision ~0.005 %
makes a 0.1-MeV step (~0.019 % in σ) cleanly resolvable. None of these
points enters the morph fit.

This is the strongest available bound on the morph's sub-MeV
(m_W, Γ_W) interpolation bias — the regime the fit operates in.

Two output figures (one per scan axis):
  validate_fine_mW.{pdf,png}   — m_W scan (Γ_W = Γ_W^0 = 2.085 GeV)
  validate_fine_gW.{pdf,png}   — Γ_W scan (m_W = m_W^0 = 80.379 GeV)
Each shows the per-point residual vs the scanned parameter, with the
MC band overlaid, at each √s.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (build_operational_morph, load_grid, sigma_morph,
                    GRID_VALIDATE_FINE_CSV, MW0, GW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one_axis(df, *, scan_col, fixed_col, fixed_val, x_label, ref_val,
                  out_stem, title):
    """Plot the residual along one 1D scan axis. `scan_col` is the varied
    parameter ('mW' or 'gammaW'); `fixed_col` is held at `fixed_val`."""
    sub = df[np.isclose(df[fixed_col], fixed_val, atol=1e-6)].copy()
    if len(sub) == 0:
        print(f"  no points for {fixed_col}={fixed_val}")
        return
    sqrts_panels = sorted(sub.sqrts.unique())
    fig, axes = plt.subplots(1, len(sqrts_panels),
                              figsize=(6 * len(sqrts_panels), 5.5),
                              sharey=True)
    if len(sqrts_panels) == 1:
        axes = [axes]
    summary = []
    for ax, s in zip(axes, sqrts_panels):
        sl = sub[np.isclose(sub.sqrts, s, atol=1e-3)].sort_values(scan_col)
        # MeV offset from the nominal
        dx_MeV = (sl[scan_col].values - ref_val) * 1000.0
        ax.errorbar(dx_MeV, sl["resid"].values, yerr=sl["mc"].values,
                    fmt="o", ms=4, color="C0", capsize=2, alpha=0.85)
        mc_max = float(sl["mc"].max())
        ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.12,
                   label=rf"$\pm$MC (max {mc_max:.4f}%)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", ls=":", alpha=0.5)
        ax.set_title(rf"$\sqrt{{s}}={s:.0f}$ GeV")
        ax.set_xlabel(rf"{x_label} − {x_label}$^0$  [MeV]")
        ax.grid(alpha=0.25)
        if ax is axes[0]:
            # loc="center": mplhep's top-anchored ylabel hangs below the axes
            # for a label this long and gets clipped by bbox_inches="tight".
            ax.set_ylabel(r"$(\sigma_\mathrm{morph}-\sigma_\mathrm{WHIZARD})"
                          r"/\sigma$ [%]", loc="center")
            ax.legend(loc="best", fontsize=8)
        summary.append((s, len(sl),
                        float(np.abs(sl["resid"]).max()),
                        float(np.median(np.abs(sl["resid"]))),
                        mc_max))

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_pdf = PLOTS / f"{out_stem}.pdf"
    out_png = PLOTS / f"{out_stem}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  {out_stem} summary")
    print(f"  {'√s':>6} {'n':>5} {'max|r|':>11} {'med|r|':>11} {'MC max':>9}")
    for s, n, mx, md, mc in summary:
        print(f"  {s:>6.0f} {n:>5} {mx:>9.4f}% {md:>9.4f}% {mc:>7.4f}%")
    print(f"  wrote {out_pdf}")


def main():
    if not GRID_VALIDATE_FINE_CSV.exists():
        print(f"validate_fine grid not found at {GRID_VALIDATE_FINE_CSV}.")
        return

    sqrts_axis, splines, _ = build_operational_morph()
    print(f"morph fitted at {len(sqrts_axis)} √s values "
          f"(grid_fine + outer 0.5-GeV wings)")

    df = load_grid(GRID_VALIDATE_FINE_CSV)
    print(f"validate_fine: {len(df)} points, "
          f"√s ∈ {sorted(df.sqrts.unique())}, "
          f"m_W unique={df.mW.nunique()}, Γ_W unique={df.gammaW.nunique()}")
    df["pred"]  = [float(sigma_morph(r.sqrts, r.mW, r.gammaW, splines=splines))
                   for r in df.itertuples()]
    df["resid"] = (df.pred - df.sigma_fb) / df.sigma_fb * 100
    df["mc"]    = df.err_fb / df.sigma_fb * 100

    plot_one_axis(df, scan_col="mW", fixed_col="gammaW", fixed_val=GW0,
                  x_label=r"$m_W$", ref_val=MW0,
                  out_stem="validate_fine_mW",
                  title=r"Sub-MeV held-out closure — $m_W$ scan "
                        rf"(at $\Gamma_W=\Gamma_W^0={GW0:.3f}$ GeV)")
    plot_one_axis(df, scan_col="gammaW", fixed_col="mW", fixed_val=MW0,
                  x_label=r"$\Gamma_W$", ref_val=GW0,
                  out_stem="validate_fine_gW",
                  title=r"Sub-MeV held-out closure — $\Gamma_W$ scan "
                        rf"(at $m_W=m_W^0={MW0:.3f}$ GeV)")

    # Overall summary across both scans
    mx = float(np.abs(df["resid"]).max())
    md = float(np.median(np.abs(df["resid"])))
    print(f"\noverall: max|resid|={mx:.4f}%  median|resid|={md:.4f}% "
          f"  (n={len(df)})")


if __name__ == "__main__":
    main()
