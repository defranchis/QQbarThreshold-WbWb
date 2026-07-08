"""Overview of the WHIZARD grid the operational morph is built on.

Two panels:
  left   the (m_W, Γ_W) sampling plane — 9 × 7 rectangular grid, with
         the nominal morph point and the two BFS-reference Γ_W columns
         (excluded from the morph fit) marked.
  right  per-point Monte-Carlo relative precision err/σ vs √s at the
         nominal (m_W, Γ_W), separating the dense grid_fine campaign
         (0.1 GeV step in [155, 165], ~0.008 % MC) from the 0.5-GeV
         outer wings (taken from grid_highstats as spline support).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (load_grid, GRID_FINE_CSV, GRID_HIGHSTATS_CSV,
                    MW0, GW0, GW_UNIFORM)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)


def main():
    fine = load_grid(GRID_FINE_CSV)
    hs   = load_grid(GRID_HIGHSTATS_CSV)
    fine_sqrts = set(np.round(fine.sqrts.unique(), 4))
    wings = hs[~np.round(hs.sqrts, 4).isin(fine_sqrts)].copy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- (m_W, Γ_W) sampling plane --------------------------------------
    mw = np.array(sorted(fine.mW.unique()))
    gw = np.array(sorted(fine.gammaW.unique()))
    MW, GW = np.meshgrid(mw, gw)
    axL.plot(MW.ravel(), GW.ravel(), "o", ms=7, color="#1f5fa8",
             label=f"grid nodes ({len(mw)}×{len(gw)})")
    bfs_ref = [g for g in gw if round(g, 5) not in GW_UNIFORM]
    for g in bfs_ref:
        axL.axhline(g, color="#d62728", ls="--", lw=1.0, alpha=0.7)
    axL.plot([], [], "--", color="#d62728", label="BFS-ref $\\Gamma_W$ (fit-excluded)")
    axL.plot(MW0, GW0, "*", ms=22, color="gold", mec="black", mew=1.0,
             label="morph nominal", zorder=5)
    axL.set_xlabel(r"$m_W$ [GeV]")
    axL.set_ylabel(r"$\Gamma_W$ [GeV]")
    axL.set_title(r"$(m_W,\Gamma_W)$ plane (every $\sqrt{s}$)", fontsize=13)
    axL.grid(alpha=0.25)
    axL.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # --- MC relative precision vs √s ------------------------------------
    def nom_slice(df):
        sl = df[np.isclose(df.mW, MW0) & np.isclose(df.gammaW, GW0)]
        return sl.sort_values("sqrts")
    fine_n = nom_slice(fine)
    wing_n = nom_slice(wings)
    axR.plot(fine_n.sqrts, fine_n.err_fb / fine_n.sigma_fb * 100, "o", ms=4,
             color="#1f5fa8",
             label=f"grid_fine ({fine.sqrts.nunique()} √s @ 0.1 GeV)")
    axR.plot(wing_n.sqrts, wing_n.err_fb / wing_n.sigma_fb * 100, "s", ms=6,
             color="#ff7f0e",
             label=f"outer wings ({wings.sqrts.nunique()} √s @ 0.5 GeV)")
    axR.axhline(0.008, color="grey", ls=":", lw=1.2,
                label=r"$\sim$0.008% (grid_fine floor)")
    axR.axhline(0.016, color="grey", ls="-.", lw=0.8,
                label=r"$\sim$0.016% (wings floor)")
    axR.set_xlabel(r"$\sqrt{s}$ [GeV]")
    axR.set_ylabel(r"MC relative precision  $\delta\sigma/\sigma$ [%]")
    axR.set_title(r"MC precision at $(m_W^0,\Gamma_W^0)$", fontsize=13)
    axR.set_ylim(0, None)
    axR.grid(alpha=0.25)
    axR.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out_pdf = PLOTS / "grid_overview.pdf"
    out_png = PLOTS / "grid_overview.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"grid_fine: {len(fine)} pts, {fine.sqrts.nunique()} √s × "
          f"{len(mw)} m_W × {len(gw)} Γ_W")
    print(f"wings:     {len(wings)} pts, {wings.sqrts.nunique()} √s")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
