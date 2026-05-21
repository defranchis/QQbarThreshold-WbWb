"""Blind validation of the √s interpolation.

The densify campaign placed WHIZARD points at √s = 157.25, 157.75, …,
162.75 GeV — exactly the 0.25-GeV midpoints between the highstats
0.5-GeV nodes. Building the morph from the highstats grid *only* and
predicting those densify points is therefore a genuine held-out test
of the cubic-spline √s interpolation at real, un-fitted √s values
(stronger than leave-one-out, which removes a node the spline would
otherwise anchor on).

  left-top     per-point residual (morph − WHIZARD)/σ vs √s, coloured
               by axis class, against the per-point MC band.
  left-bottom  median and max |residual| vs √s.
  right        pull = (morph − WHIZARD)/δσ_MC histogram — unit-width
               and zero-centred if the interpolation is MC-limited
               with no systematic bias.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (build_morph_from_grid, load_grid, filter_uniform_gw,
                    sigma_morph, MW0, GW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"

CLASSES = [("nom", "C0", "o", "nominal"),
           ("m_only", "C1", "s", r"only $m_W$ off"),
           ("g_only", "C2", "^", r"only $\Gamma_W$ off"),
           ("both", "C3", "x", "doubly off-axis")]


def classify(mw, gw):
    on_m, on_g = np.isclose(mw, MW0), np.isclose(gw, GW0)
    return ("nom" if on_m and on_g else "m_only" if on_g
            else "g_only" if on_m else "both")


def main():
    # morph from highstats ONLY — the densify √s are the blind test set
    sqrts_axis, splines, _ = build_morph_from_grid(HIGHSTATS_CSV)
    print(f"morph built from highstats only: {len(sqrts_axis)} √s nodes "
          f"@ 0.5 GeV")

    dn = filter_uniform_gw(load_grid(DENSIFY_CSV))
    dn["pred"]  = [float(sigma_morph(r.sqrts, r.mW, r.gammaW, splines=splines))
                   for r in dn.itertuples()]
    dn["resid"] = (dn.pred - dn.sigma_fb) / dn.sigma_fb * 100
    dn["pull"]  = (dn.pred - dn.sigma_fb) / dn.err_fb
    dn["mc"]    = dn.err_fb / dn.sigma_fb * 100
    dn["cls"]   = [classify(r.mW, r.gammaW) for r in dn.itertuples()]
    print(f"blind test set: {len(dn)} densify points at "
          f"{dn.sqrts.nunique()} off-node √s (0.25-GeV offset)")

    fig, axd = plt.subplot_mosaic([["res", "pull"], ["sum", "pull"]],
                                  figsize=(17, 9), constrained_layout=True,
                                  gridspec_kw={"height_ratios": [3, 2],
                                               "width_ratios": [2, 1]})

    # --- residual vs √s -------------------------------------------------
    ax = axd["res"]
    mc_max = dn.mc.max()
    ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.15,
               label=rf"$\pm$MC (max {mc_max:.3f}%)")
    ax.axhline(0, color="grey", lw=0.6)
    for cls, col, mk, lab in CLASSES:
        sub = dn[dn.cls == cls]
        ax.plot(sub.sqrts, sub.resid, mk, ms=6, color=col, alpha=0.7, label=lab)
    ax.set_ylabel(r"$(\sigma_\mathrm{morph}-\sigma_\mathrm{WHIZARD})/\sigma$ [%]")
    ax.set_title("√s-interpolation residual at the 0.25-GeV-offset densify "
                 "points (highstats-only morph)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # --- median / max |resid| vs √s ------------------------------------
    ax = axd["sum"]
    sv = np.array(sorted(dn.sqrts.unique()))
    med = [dn[np.isclose(dn.sqrts, s)].resid.abs().median() for s in sv]
    mx  = [dn[np.isclose(dn.sqrts, s)].resid.abs().max()    for s in sv]
    ax.plot(sv, med, "o-", color="C0", ms=5, label="median |resid|")
    ax.plot(sv, mx,  "s-", color="C3", ms=5, label="max |resid|")
    ax.axhline(mc_max, color="grey", ls="--", lw=0.9, label="MC bar (max)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$|$residual$|$ [%]")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=8)

    # --- pull histogram -------------------------------------------------
    ax = axd["pull"]
    pull = dn.pull.values
    ax.hist(pull, bins=22, color="C0", alpha=0.75, density=True,
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

    fig.suptitle("√s interpolation — blind test on the densify points")
    out_pdf = PLOTS / "sqrts_validation.pdf"
    out_png = PLOTS / "sqrts_validation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  max |resid| = {dn.resid.abs().max():.4f}%, "
          f"median = {dn.resid.abs().median():.4f}%")
    print(f"  pull: mean {pull.mean():+.3f}, std {pull.std():.3f}")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
