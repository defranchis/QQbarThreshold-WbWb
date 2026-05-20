"""Predictions from the full morph + cubic-spline-on-√s scheme.

Three deliverable plots produced from a single morph build:

  morph_smooth_sigma_sqrts.png
      σ(√s) at several (m_W, Γ_W) variations, with grid points overlaid
      as open markers. Smoke test that the morph reproduces sensible
      threshold curves between grid √s knots.

  morph_azzurri.png
      σ_observed(√s) with the production BR² re-introduction factor
      (Γ_W / Γ_W^(0)_REF)². Reproduces the classic Azzurri Γ_W-induced
      crossing at √s ≈ 162 GeV (the morph crossing lands at 161.9 GeV;
      consistent with the production-anchor calibration).

  4d_morphing_loo.png
      Leave-one-out residual on the √s axis: hold each grid √s out,
      build the 8 cubic splines from the remaining points, predict the
      full 9 × 5 (m_W, Γ_W) plane at the held-out √s. Bottom panel
      (log-y) tracks median and max |residual| vs √s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from morph import (build_morph_from_grid, build_splines, fit_morph_at_sqrts,
                   filter_uniform_gw, load_grid, sigma_morph,
                   GW_UNIFORM, GW0, MW0)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from framework.process.ww.xsec_calculator.bfs_eft import gamma_W_LO
from framework.process.ww.xsec_calculator.eft_xsec import M_W_BFS_REF

PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"

GAMMA_W_LO_REF = gamma_W_LO(M_W_BFS_REF)


def plot_smooth(splines, sqrts_axis, df):
    """σ(√s) at various (m_W, Γ_W) variations + grid points."""
    s_fine = np.linspace(sqrts_axis[0], sqrts_axis[-1], 400)
    curves = [
        ("m_W − 100 MeV", 80.279, GW0,   "C0",    "-"),
        ("m_W −  50 MeV", 80.329, GW0,   "C0",    "--"),
        ("nominal",       MW0,    GW0,   "black", "-"),
        ("m_W +  50 MeV", 80.429, GW0,   "C3",    "--"),
        ("m_W + 100 MeV", 80.479, GW0,   "C3",    "-"),
        ("Γ_W − 40 MeV",  MW0,    2.045, "C2",    "-."),
        ("Γ_W + 40 MeV",  MW0,    2.125, "C5",    "-."),
        ("(m+50, Γ+40)",  80.429, 2.125, "C4",    ":"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                              gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    ax_abs, ax_rat = axes
    sig_nom = sigma_morph(s_fine, MW0, GW0, splines=splines)
    for label, mw, gw, color, style in curves:
        sig = sigma_morph(s_fine, mw, gw, splines=splines)
        ax_abs.plot(s_fine, sig, style, color=color, linewidth=1.6, label=label)
        ax_rat.plot(s_fine, sig / sig_nom - 1, style, color=color, linewidth=1.4)
        on_grid = df[(np.isclose(df.mW, mw, atol=1e-3))
                      & (np.isclose(df.gammaW, gw, atol=1e-3))].sort_values("sqrts")
        if len(on_grid):
            ax_abs.plot(on_grid.sqrts, on_grid.sigma_fb, "o", color=color,
                        mfc="white", markersize=4, alpha=0.7)
    ax_abs.axvline(2 * MW0, color="grey", linestyle=":", alpha=0.4)
    ax_abs.set_ylabel(r"$\sigma_{\rm morph}(\mu\nu u\bar d)$ [fb]")
    ax_abs.set_title("Smooth σ(√s) from the morphing scheme — m_W and Γ_W variations  "
                     "(open markers = underlying grid points)")
    ax_abs.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax_abs.grid(alpha=0.25)
    ax_rat.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
    ax_rat.set_ylabel(r"$\sigma/\sigma_{\rm nom} - 1$")
    ax_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_rat.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "morph_smooth_sigma_sqrts.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "morph_smooth_sigma_sqrts.png", dpi=150, bbox_inches="tight")
    print(f"wrote {PLOTS / 'morph_smooth_sigma_sqrts.png'}")


def plot_azzurri(splines, sqrts_axis):
    """σ_observed(√s) with BR² re-introduced; shows the Γ_W crossing at 162 GeV."""
    s_fine = np.linspace(sqrts_axis[0], sqrts_axis[-1], 400)
    def sigma_obs(s, mw, gw):
        return sigma_morph(s, mw, gw, splines=splines) * (gw / GAMMA_W_LO_REF) ** 2
    sigma_nom = sigma_obs(s_fine, MW0, GW0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                              gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    ax_abs, ax_rat = axes
    gw_curves = [(2.045, "C0", "-"), (2.065, "C0", "--"), (GW0, "black", "-"),
                  (2.105, "C3", "--"), (2.125, "C3", "-")]
    for gw, color, style in gw_curves:
        sig = sigma_obs(s_fine, MW0, gw)
        ax_abs.plot(s_fine, sig, style, color=color, linewidth=1.6,
                    label=rf"$\Gamma_W={gw:.3f}$ GeV  ($m_W={MW0:.3f}$)")
        ax_rat.plot(s_fine, sig / sigma_nom - 1, style, color=color, linewidth=1.4,
                    label=rf"$\Gamma_W={gw:.3f}$")
    for mw, color, style in [(80.279, "C1", "-."), (80.479, "C4", "-.")]:
        sig = sigma_obs(s_fine, mw, GW0)
        ax_abs.plot(s_fine, sig, style, color=color, linewidth=1.6,
                    label=rf"$m_W={mw:.3f}$ GeV")
        ax_rat.plot(s_fine, sig / sigma_nom - 1, style, color=color, linewidth=1.4,
                    label=rf"$m_W={mw:.3f}$")
    ax_abs.axvline(2 * MW0, color="grey", linestyle=":", alpha=0.4)
    ax_abs.set_ylabel(r"$\sigma_{\rm obs} = \sigma_{\rm morph} \times "
                     r"(\Gamma_W/\Gamma_W^{(0),{\rm REF}})^2$ [fb]")
    ax_abs.set_title(r"Azzurri-style $\sigma_{\rm obs}(\sqrt{s})$ from the morphing scheme"
                     rf"  ($\Gamma_W^{{(0),{{\rm REF}}}}={GAMMA_W_LO_REF:.5f}$ GeV)")
    ax_abs.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax_abs.grid(alpha=0.25)
    ax_rat.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
    ax_rat.axvline(2 * MW0, color="grey", linestyle=":", alpha=0.4,
                   label=rf"$2 m_W = {2*MW0:.2f}$")
    ax_rat.set_ylabel(r"$\sigma_{\rm obs}/\sigma_{\rm obs,nom} - 1$")
    ax_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_rat.grid(alpha=0.25)
    ax_rat.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "morph_azzurri.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "morph_azzurri.png", dpi=150, bbox_inches="tight")

    diff = sigma_obs(s_fine, MW0, 2.125) - sigma_nom
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    for idx in sign_changes:
        s_cross = s_fine[idx] + (s_fine[idx+1] - s_fine[idx]) * (
            -diff[idx] / (diff[idx+1] - diff[idx]))
        print(f"  Γ_W=2.125 vs nominal Azzurri crossing at √s ≈ {s_cross:.3f} GeV")
    print(f"wrote {PLOTS / 'morph_azzurri.png'}")


def plot_4d_loo(df):
    """LOO on the √s axis: hold each √s out, refit splines from the rest,
    predict the full (m_W, Γ_W) plane at the held-out √s."""
    sqrts_axis = np.array(sorted(df.sqrts.unique()))
    morphs, kept = [], []
    for s in sqrts_axis:
        m = fit_morph_at_sqrts(df[np.isclose(df.sqrts, s, atol=1e-3)])
        if m is not None:
            morphs.append(m); kept.append(s)
    sqrts_axis = np.array(kept)

    rows = []
    for i, s_held in enumerate(sqrts_axis):
        sqrts_train = np.delete(sqrts_axis, i)
        splines = build_splines(sqrts_train, morphs[:i] + morphs[i+1:])
        for r in df[np.isclose(df.sqrts, s_held, atol=1e-3)].itertuples():
            sigma_pred = float(sigma_morph(s_held, r.mW, r.gammaW, splines=splines))
            rows.append({"sqrts": s_held, "mW": r.mW, "gammaW": r.gammaW,
                         "resid_pct": (sigma_pred - r.sigma_fb) / r.sigma_fb * 100,
                         "mc_pct":    (r.err_fb   / r.sigma_fb) * 100,
                         "kind": ("nom"      if np.isclose(r.mW, MW0) and np.isclose(r.gammaW, GW0)
                                  else "m_only" if np.isclose(r.gammaW, GW0)
                                  else "g_only" if np.isclose(r.mW, MW0)
                                  else "both_off")})
    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax, ax2 = axes
    for kind, col, marker, label in [
        ("nom",      "C0", "o", "nominal"),
        ("m_only",   "C1", "s", r"only $m_W$ off"),
        ("g_only",   "C2", "^", r"only $\Gamma_W$ off"),
        ("both_off", "C3", "x", "both off-axis")]:
        sub = res[res.kind == kind]
        ax.errorbar(sub.sqrts, sub.resid_pct, yerr=sub.mc_pct,
                    fmt=marker, color=col, markersize=4, alpha=0.7,
                    capsize=2, linestyle="none", label=label)
    mc_max = res.mc_pct.max()
    ax.axhspan(-mc_max, mc_max, color="grey", alpha=0.10, label=r"$\pm$MC stat (max)")
    ax.axhline(0, color="grey", alpha=0.5, linewidth=0.7)
    ax.axvspan(sqrts_axis[0], sqrts_axis[0] + 0.6, color="orange", alpha=0.08,
               label="boundary (extrapolation)")
    ax.axvspan(sqrts_axis[-1] - 0.6, sqrts_axis[-1], color="orange", alpha=0.08)
    ax.set_ylabel(r"$(\sigma_{\rm pred} - \sigma_{\rm truth})/\sigma$ [%]")
    ax.set_title("LOO-on-√s residuals: full per-√s morphs + cubic-spline on √s")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    s_unique = sorted(res.sqrts.unique())
    med_v = [float(np.median(np.abs(res[np.isclose(res.sqrts, s)].resid_pct))) for s in s_unique]
    max_v = [float(np.max(np.abs(res[np.isclose(res.sqrts, s)].resid_pct))) for s in s_unique]
    ax2.plot(s_unique, med_v, "o-", color="C0", markersize=5, label="median |resid|")
    ax2.plot(s_unique, max_v, "s-", color="C3", markersize=5, label="max |resid|")
    ax2.axhline(mc_max, color="grey", linestyle="--", linewidth=0.8, label="MC stat (max)")
    ax2.set_ylabel(r"|resid| [%]")
    ax2.set_xlabel(r"held-out $\sqrt{s}$ [GeV]")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS / "4d_morphing_loo.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "4d_morphing_loo.png", dpi=150, bbox_inches="tight")

    interior_mask = (res.sqrts > sqrts_axis[0] + 1.0) & (res.sqrts < sqrts_axis[-1] - 1.0)
    interior = res[interior_mask]
    print(f"  interior √s: median={float(np.median(np.abs(interior.resid_pct))):.4f}%  "
          f"max={float(np.max(np.abs(interior.resid_pct))):.4f}%  n={len(interior)}")
    print(f"wrote {PLOTS / '4d_morphing_loo.png'}")


def main():
    sqrts_axis, splines, _ = build_morph_from_grid(HIGHSTATS_CSV, DENSIFY_CSV)
    print(f"Γ_W^(0)_REF = gamma_W_LO({M_W_BFS_REF}) = {GAMMA_W_LO_REF:.5f} GeV")
    print(f"morph fitted at {len(sqrts_axis)} √s values "
          f"({'highstats + densify' if DENSIFY_CSV.exists() else 'highstats only'})")

    # Raw highstats df is needed by plot_smooth and plot_4d_loo (independent of densify)
    df_fit = filter_uniform_gw(load_grid(HIGHSTATS_CSV))
    plot_smooth(splines, sqrts_axis, df_fit)
    plot_azzurri(splines, sqrts_axis)
    plot_4d_loo(df_fit)


if __name__ == "__main__":
    main()
