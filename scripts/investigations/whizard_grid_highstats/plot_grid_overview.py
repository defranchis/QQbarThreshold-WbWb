"""Overview of the WHIZARD highstats grid the morph is built on.

Two panels:
  left   the (m_W, Γ_W) sampling plane — 9 × 7 rectangular grid, with
         the nominal morph point and the two BFS-reference Γ_W columns
         (excluded from the morph fit) marked.
  right  per-point Monte-Carlo relative precision err/σ vs √s at the
         nominal (m_W, Γ_W), for the highstats and densify campaigns —
         shows the ~0.016% statistical floor the morph residuals are
         compared against.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import load_grid, MW0, GW0, GW_UNIFORM

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"


def main():
    hs = load_grid(HIGHSTATS_CSV)
    dn = load_grid(DENSIFY_CSV) if DENSIFY_CSV.exists() else None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- (m_W, Γ_W) sampling plane --------------------------------------
    mw = np.array(sorted(hs.mW.unique()))
    gw = np.array(sorted(hs.gammaW.unique()))
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
    hs_n = nom_slice(hs)
    axR.plot(hs_n.sqrts, hs_n.err_fb / hs_n.sigma_fb * 100, "o", ms=6,
             color="#1f5fa8", label=f"highstats ({hs.sqrts.nunique()} √s @ 0.5 GeV)")
    if dn is not None:
        dn_n = nom_slice(dn)
        axR.plot(dn_n.sqrts, dn_n.err_fb / dn_n.sigma_fb * 100, "s", ms=6,
                 color="#ff7f0e",
                 label=f"densify ({dn.sqrts.nunique()} √s @ 0.25 GeV)")
    axR.axhline(0.016, color="grey", ls=":", lw=1.2, label=r"$\sim$0.016% floor")
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
    print(f"highstats: {len(hs)} pts, {hs.sqrts.nunique()} √s × "
          f"{len(mw)} m_W × {len(gw)} Γ_W")
    if dn is not None:
        print(f"densify:   {len(dn)} pts, {dn.sqrts.nunique()} √s")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
