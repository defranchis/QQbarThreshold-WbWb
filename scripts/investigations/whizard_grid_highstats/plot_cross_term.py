"""Dissection of the bilinear (m_W·Γ_W) cross term β.

The product of the two 1D morphs, σ_nom·R_m·R_Γ, misses the joint
(m_W, Γ_W) coupling: near the threshold rise a shift in m_W (threshold
position) and a change in Γ_W (smearing) do not act independently.
The morph absorbs this with one scalar per √s,

    σ = σ_nom · R_m · R_Γ · [1 + β(√s)·(m_W−m_W⁰)·(Γ_W−Γ_W⁰)].

This script takes β apart:

  rows 0-1  residual of the naive product morph (no β) and of the full
            bilinear morph, as 2D maps over the (m_W, Γ_W) plane at
            three √s — the naive map shows a clear m_W·Γ_W saddle that
            the cross term flattens.
  row 2     (a) linearity check: fractional residual of the naive morph
            vs (Δm_W·ΔΓ_W) at the peak-rise √s, against the fitted
            slope β; (b) β(√s); (c) max |residual| over the doubly-
            off-axis points, naive vs bilinear, vs √s.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (load_grid, filter_uniform_gw, fit_morph_at_sqrts,
                    MW0, GW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"

HEATMAP_SQRTS = [157.0, 159.0, 161.0, 163.0, 165.0]   # full analysis range
LINFIT_SQRTS  = 161.0


def naive(m, mw, gw):
    """σ_nom·R_m·R_Γ — product of the two 1D morphs, no cross term."""
    Rm = np.polyval(m["coef_m"], mw) / np.polyval(m["coef_m"], MW0)
    Rg = np.polyval(m["coef_g"], gw) / np.polyval(m["coef_g"], GW0)
    return m["sigma_nom"] * Rm * Rg


def main():
    df = filter_uniform_gw(load_grid(HIGHSTATS_CSV))
    if DENSIFY_CSV.exists():
        import pandas as pd
        df = pd.concat([df, filter_uniform_gw(load_grid(DENSIFY_CSV))],
                       ignore_index=True)
    mw_ax = np.array(sorted(df.mW.unique()))
    gw_ax = np.array(sorted(df.gammaW.unique()))
    sqrts = np.array(sorted(df.sqrts.unique()))

    morphs = {s: fit_morph_at_sqrts(df[np.isclose(df.sqrts, s, atol=1e-3)])
              for s in sqrts}
    morphs = {s: m for s, m in morphs.items() if m is not None}

    def resid_maps(s):
        """(naive%, bilinear%) residual arrays shaped (n_gw, n_mw)."""
        m = morphs[s]
        sub = df[np.isclose(df.sqrts, s, atol=1e-3)]
        Zn = np.full((len(gw_ax), len(mw_ax)), np.nan)
        Zb = np.full((len(gw_ax), len(mw_ax)), np.nan)
        for r in sub.itertuples():
            i = np.argmin(np.abs(gw_ax - r.gammaW))
            j = np.argmin(np.abs(mw_ax - r.mW))
            sn = naive(m, r.mW, r.gammaW)
            sb = sn * (1 + m["beta"] * (r.mW - MW0) * (r.gammaW - GW0))
            Zn[i, j] = (sn - r.sigma_fb) / r.sigma_fb * 100
            Zb[i, j] = (sb - r.sigma_fb) / r.sigma_fb * 100
        return Zn, Zb

    maps = {s: resid_maps(s) for s in HEATMAP_SQRTS}
    vmax = max(np.nanmax(np.abs(maps[s][0])) for s in HEATMAP_SQRTS)

    keys_n = [f"n{i}" for i in range(len(HEATMAP_SQRTS))]
    keys_b = [f"b{i}" for i in range(len(HEATMAP_SQRTS))]
    mosaic = [keys_n, keys_b, ["lin", "lin", "beta", "mx", "mx"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(24, 14),
                                  constrained_layout=True)
    extent = [mw_ax[0] - 0.0125, mw_ax[-1] + 0.0125,
              gw_ax[0] - 0.010, gw_ax[-1] + 0.010]
    im = None
    for col, s in enumerate(HEATMAP_SQRTS):
        Zn, Zb = maps[s]
        for row, (Z, tag) in enumerate([(Zn, r"naive $\sigma_0 R_m R_\Gamma$"),
                                        (Zb, r"$+\,\beta$ term")]):
            ax = axd[("n" if row == 0 else "b") + str(col)]
            im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.plot(MW0, GW0, "*", ms=14, color="gold", mec="black", mew=0.8)
            ax.set_title(rf"$\sqrt{{s}}={s:.0f}$ — {tag}", fontsize=10)
            ax.set_xlabel(r"$m_W$ [GeV]")
            if col == 0:
                ax.set_ylabel(r"$\Gamma_W$ [GeV]")
    cb = fig.colorbar(im, ax=[axd[k] for k in keys_n + keys_b],
                      location="right", fraction=0.02, pad=0.01)
    cb.set_label(r"$(\sigma_\mathrm{morph}-\sigma_\mathrm{WHIZARD})/\sigma$ [%]")

    # --- linearity of the cross term ------------------------------------
    ax = axd["lin"]
    m = morphs[LINFIT_SQRTS]
    sub = df[np.isclose(df.sqrts, LINFIT_SQRTS, atol=1e-3)]
    off = sub[(~np.isclose(sub.mW, MW0)) & (~np.isclose(sub.gammaW, GW0))]
    x = (off.mW.values - MW0) * (off.gammaW.values - GW0)
    y = np.array([(off.sigma_fb.values[k] - naive(m, off.mW.values[k],
                   off.gammaW.values[k])) / naive(m, off.mW.values[k],
                   off.gammaW.values[k]) for k in range(len(off))])
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, m["beta"] * xx, "-", color="C3", lw=2.0,
            label=rf"slope $\beta={m['beta']:+.3f}$ GeV$^{{-2}}$")
    ax.plot(x, y, "o", ms=7, color="C0", alpha=0.8, label="doubly-off-axis pts")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(r"$(m_W-m_W^0)\,(\Gamma_W-\Gamma_W^0)$ [GeV$^2$]")
    ax.set_ylabel(r"$(\sigma_\mathrm{WHIZARD}-\sigma_\mathrm{naive})/\sigma$")
    ax.set_title(rf"cross-term linearity at $\sqrt{{s}}={LINFIT_SQRTS:.0f}$ GeV",
                 fontsize=11)
    ax.grid(alpha=0.25); ax.legend(loc="best", fontsize=9)

    # --- (2,1) β(√s) ----------------------------------------------------
    ax = axd["beta"]
    sv = np.array(sorted(morphs))
    bv = np.array([morphs[s]["beta"] for s in sv])
    ax.plot(sv, bv, "o-", ms=4, color="C3", lw=1.0)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(2 * MW0, color="grey", ls=":", alpha=0.5,
               label=rf"$2m_W^0={2*MW0:.2f}$")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\beta$ [GeV$^{-2}$]")
    ax.set_title(r"bilinear coefficient $\beta(\sqrt{s})$", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(loc="best", fontsize=9)

    # --- (2,2) naive vs bilinear max residual vs √s ---------------------
    ax = axd["mx"]
    naive_max, bil_max, mc = [], [], []
    for s in sv:
        m = morphs[s]
        sub = df[np.isclose(df.sqrts, s, atol=1e-3)]
        off = sub[(~np.isclose(sub.mW, MW0)) & (~np.isclose(sub.gammaW, GW0))]
        rn, rb = [], []
        for r in off.itertuples():
            sn = naive(m, r.mW, r.gammaW)
            sb = sn * (1 + m["beta"] * (r.mW - MW0) * (r.gammaW - GW0))
            rn.append(abs(sn - r.sigma_fb) / r.sigma_fb * 100)
            rb.append(abs(sb - r.sigma_fb) / r.sigma_fb * 100)
        naive_max.append(max(rn)); bil_max.append(max(rb))
        mc.append(np.median(off.err_fb / off.sigma_fb) * 100)
    ax.plot(sv, naive_max, "s-", ms=4, color="C0", lw=1.0,
            label="naive (no β)")
    ax.plot(sv, bil_max, "o-", ms=4, color="C3", lw=1.0,
            label="+ bilinear β")
    ax.plot(sv, mc, ":", color="grey", lw=1.2, label="MC bar")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"max $|$residual$|$ over doubly-off-axis [%]")
    ax.set_title("cross-term gain vs √s", fontsize=11)
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="best", fontsize=9)

    fig.suptitle("Bilinear cross term dissected — the joint "
                 r"$(m_W,\Gamma_W)$ coupling and its $\beta$ absorption")
    out_pdf = PLOTS / "cross_term.pdf"
    out_png = PLOTS / "cross_term.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"β peaks at {bv.max():+.3f} GeV^-2 at √s={sv[np.argmax(bv)]:.2f}")
    print(f"naive max residual  {max(naive_max):.3f}%  →  "
          f"bilinear {max(bil_max):.3f}%")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
