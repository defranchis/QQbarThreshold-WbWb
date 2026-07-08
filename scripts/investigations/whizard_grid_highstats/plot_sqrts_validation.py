"""Blind validation of the √s interpolation — thinned-grid LOO on grid_fine.

The operational morph is built from grid_fine (0.1 GeV √s in [155, 165])
plus the outer 0.5-GeV wings. To stress-test the cubic spline at genuine
un-fitted √s values, the morph is rebuilt from the half-density subset
(0.2-GeV step: 155.0, 155.2, …, 165.0) and used to predict the
intermediate 0.1-GeV midpoints (155.1, 155.3, …, 164.9) — points the
training spline never sees.

This is stronger than leave-one-out (which removes a node the spline
would otherwise anchor on) because the held-out points sit between two
training nodes at the full grid_fine step distance. With the 0.5-GeV
wings retained as outer support the test is purely about interpolation,
not extrapolation.

  left-top     per-point residual (morph − WHIZARD)/σ vs √s, coloured
               by axis class (nom / m-only / Γ-only / doubly off).
  left-bottom  median and max |residual| vs √s, against the MC bar.
  right        pull = (morph − WHIZARD)/δσ_MC histogram; unit-width
               and zero-centred if the interpolation is MC-limited.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (load_operational_grid, filter_uniform_gw,
                    build_morph_from_df, sigma_morph,
                    GRID_FINE_CSV, MW0, GW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)

CLASSES = [("nom", "C0", "o", "nominal"),
           ("m_only", "C1", "s", r"only $m_W$ off"),
           ("g_only", "C2", "^", r"only $\Gamma_W$ off"),
           ("both", "C3", "x", "doubly off-axis")]

FINE_LO, FINE_HI = 155.0, 165.0


def classify(mw, gw):
    on_m, on_g = np.isclose(mw, MW0), np.isclose(gw, GW0)
    return ("nom" if on_m and on_g else "m_only" if on_g
            else "g_only" if on_m else "both")


def main():
    df = filter_uniform_gw(load_operational_grid())
    # Split grid_fine √s into training (even index) and held-out (odd index).
    fine_mask = (df.sqrts >= FINE_LO - 1e-6) & (df.sqrts <= FINE_HI + 1e-6)
    fine_sqrts = np.array(sorted(df.loc[fine_mask, "sqrts"].unique()))
    # 0.1-GeV step → even indices ≈ 0.2 GeV, odd indices are the midpoints.
    train_sqrts = set(np.round(fine_sqrts[::2], 4))
    held_sqrts  = set(np.round(fine_sqrts[1::2], 4))
    wing_sqrts  = set(np.round(df.loc[~fine_mask, "sqrts"].unique(), 4))

    train = df[np.round(df.sqrts, 4).isin(train_sqrts | wing_sqrts)].copy()
    held  = df[np.round(df.sqrts, 4).isin(held_sqrts)].copy()
    print(f"training set: {train.sqrts.nunique()} √s "
          f"({len(train_sqrts)} grid_fine @ 0.2 GeV + {len(wing_sqrts)} wings)")
    print(f"blind set:    {held.sqrts.nunique()} √s "
          f"({len(held)} held-out 0.1-GeV midpoints)")

    sqrts_axis, splines, _ = build_morph_from_df(train, denoise=True)
    held["pred"]  = [float(sigma_morph(r.sqrts, r.mW, r.gammaW, splines=splines))
                     for r in held.itertuples()]
    held["resid"] = (held.pred - held.sigma_fb) / held.sigma_fb * 100
    held["pull"]  = (held.pred - held.sigma_fb) / held.err_fb
    held["mc"]    = held.err_fb / held.sigma_fb * 100
    held["cls"]   = [classify(r.mW, r.gammaW) for r in held.itertuples()]

    fig, axd = plt.subplot_mosaic([["res", "pull"], ["sum", "pull"]],
                                  figsize=(17, 9), constrained_layout=True,
                                  gridspec_kw={"height_ratios": [3, 2],
                                               "width_ratios": [2, 1]})

    # --- residual vs √s -------------------------------------------------
    ax = axd["res"]
    mc_max = held.mc.max()
    ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.15,
               label=rf"$\pm$MC (max {mc_max:.3f}%)")
    ax.axhline(0, color="grey", lw=0.6)
    for cls, col, mk, lab in CLASSES:
        sub = held[held.cls == cls]
        ax.plot(sub.sqrts, sub.resid, mk, ms=4, color=col, alpha=0.7, label=lab)
    ax.set_ylabel(r"$(\sigma_\mathrm{morph}-\sigma_\mathrm{WHIZARD})/\sigma$ [%]")
    ax.set_title(r"√s blind test: morph trained on 0.2-GeV thinned grid_fine, "
                 r"predicting the 0.1-GeV midpoints", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # --- median / max |resid| vs √s ------------------------------------
    ax = axd["sum"]
    sv = np.array(sorted(held.sqrts.unique()))
    med = [held[np.isclose(held.sqrts, s)].resid.abs().median() for s in sv]
    mx  = [held[np.isclose(held.sqrts, s)].resid.abs().max()    for s in sv]
    ax.plot(sv, med, "o-", color="C0", ms=4, label="median |resid|")
    ax.plot(sv, mx,  "s-", color="C3", ms=4, label="max |resid|")
    ax.axhline(mc_max, color="grey", ls="--", lw=0.9, label="MC bar (max)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$|$residual$|$ [%]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=8)

    # --- pull histogram -------------------------------------------------
    ax = axd["pull"]
    pull = held.pull.values
    ax.hist(pull, bins=40, color="C0", alpha=0.75, density=True,
            edgecolor="white")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.5,
            label="unit Gaussian")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(r"pull $(\sigma_\mathrm{morph}-\sigma_\mathrm{WHIZARD})/"
                  r"\delta\sigma_\mathrm{MC}$")
    ax.set_ylabel("density")
    ax.set_title(rf"pull: mean ${pull.mean():+.2f}$, std ${pull.std():.2f}$",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("√s interpolation — thinned-grid LOO blind test on grid_fine")
    out_pdf = PLOTS / "sqrts_validation.pdf"
    out_png = PLOTS / "sqrts_validation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  max |resid| = {held.resid.abs().max():.4f}%, "
          f"median = {held.resid.abs().median():.4f}%")
    print(f"  pull: mean {pull.mean():+.3f}, std {pull.std():.3f}")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
