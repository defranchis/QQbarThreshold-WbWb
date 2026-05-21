"""√s carrying of the morph quantities — raw per-√s fit vs denoised curve.

Each morph quantity (σ_nom, β, R_m, R_Γ) is fitted independently at
every grid √s. This plots each as the deviation from the *denoised*
curve the operational morph actually uses (the zero line):

  * points     — the raw per-√s fit nodes, scattering with MC noise;
  * thin line  — an interpolating cubic spline through those raw nodes,
                 i.e. the ripple that would be propagated if the grid
                 were not denoised;
  * zero line  — the denoised curve.

denoise_grid removes the ripple by χ²-smoothing the slowly-varying
ratios σ/σ_nom along √s and averaging σ_nom across the (m_W,Γ_W) plane,
so the operational morph carries the smooth (zero-line) curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import build_morph_from_grid, MW0, GW0

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
WHIZARD_TOP = ROOT.parent / "whizard"
HIGHSTATS_CSV = WHIZARD_TOP / "work" / "grid_highstats" / "grid.csv"
DENSIFY_CSV   = WHIZARD_TOP / "work" / "grid_highstats_densify" / "grid.csv"
DENSIFY_SPAN = (157.25, 162.75)


def _R(coef_or_splines, kind, val, ref, s=None):
    """R = polyval(coef, val)/polyval(coef, ref). If `s` is given, read
    the 3 coefficients from a splines dict at √s=s; else `coef_or_splines`
    is the 3-vector directly."""
    if s is None:
        c = coef_or_splines
    else:
        c = np.array([coef_or_splines[f"coef_{kind}_{k}"](s) for k in range(3)])
    return np.polyval(c, val) / np.polyval(c, ref)


def main():
    ax, raw_spl, raw_m = build_morph_from_grid(HIGHSTATS_CSV, DENSIFY_CSV,
                                               denoise=False)
    _,  den_spl, _     = build_morph_from_grid(HIGHSTATS_CSV, DENSIFY_CSV,
                                               denoise=True)
    s_fine = np.linspace(ax[0], ax[-1], 700)
    print(f"{len(ax)} √s nodes, {ax[0]:.1f}–{ax[-1]:.1f} GeV")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    def finish(a, ylabel, title):
        a.axhline(0, color="k", lw=1.6, label="denoised curve (used)")
        a.axvspan(*DENSIFY_SPAN, color="orange", alpha=0.07)
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=12)
        a.grid(alpha=0.25)
        a.legend(loc="best", fontsize=8)

    # σ_nom — relative deviation
    a = axes[0][0]
    raw_node = np.array([m["sigma_nom"] for m in raw_m])
    den_node = den_spl["sigma_nom"](ax)
    a.plot(ax, (raw_node - den_node) / den_node * 100, "o", ms=4, color="C0",
           alpha=0.6, label="raw per-√s node")
    a.plot(s_fine, (raw_spl["sigma_nom"](s_fine) - den_spl["sigma_nom"](s_fine))
                   / den_spl["sigma_nom"](s_fine) * 100, "-", lw=1.0,
           color="C0", alpha=0.7, label="interp. spline on raw nodes")
    finish(a, r"$(\sigma_{\rm nom}-{\rm denoised})/{\rm denoised}$ [%]",
           r"$\sigma_{\rm nom}$ — line-shape normalisation")

    # β — absolute deviation
    a = axes[0][1]
    raw_node = np.array([m["beta"] for m in raw_m])
    a.plot(ax, raw_node - den_spl["beta"](ax), "o", ms=4, color="C3",
           alpha=0.6, label="raw per-√s node")
    a.plot(s_fine, raw_spl["beta"](s_fine) - den_spl["beta"](s_fine), "-",
           lw=1.0, color="C3", alpha=0.7, label="interp. spline on raw nodes")
    finish(a, r"$\beta-\beta_{\rm denoised}$ [GeV$^{-2}$]",
           r"$\beta$ — bilinear cross term")

    # R_m at m_W = ∓100 MeV
    a = axes[1][0]
    for mW, col in [(80.279, "C0"), (80.479, "C1")]:
        raw_node = np.array([_R(m["coef_m"], "m", mW, MW0) for m in raw_m])
        den_node = np.array([_R(den_spl, "m", mW, MW0, s) for s in ax])
        a.plot(ax, (raw_node - den_node) / den_node * 100, "o", ms=4,
               color=col, alpha=0.6, label=rf"$m_W={mW:.3f}$")
        rc = _R(raw_spl, "m", mW, MW0, s_fine)
        dc = _R(den_spl, "m", mW, MW0, s_fine)
        a.plot(s_fine, (rc - dc) / dc * 100, "-", lw=1.0, color=col, alpha=0.7)
    finish(a, r"$(R_m-{\rm denoised})/{\rm denoised}$ [%]",
           r"$R_m$ at $m_W=m_W^0\mp100$ MeV")
    a.set_xlabel(r"$\sqrt{s}$ [GeV]")

    # R_Γ at Γ_W = ∓40 MeV
    a = axes[1][1]
    for gW, col in [(2.045, "C0"), (2.125, "C1")]:
        raw_node = np.array([_R(m["coef_g"], "g", gW, GW0) for m in raw_m])
        den_node = np.array([_R(den_spl, "g", gW, GW0, s) for s in ax])
        a.plot(ax, (raw_node - den_node) / den_node * 100, "o", ms=4,
               color=col, alpha=0.6, label=rf"$\Gamma_W={gW:.3f}$")
        rc = _R(raw_spl, "g", gW, GW0, s_fine)
        dc = _R(den_spl, "g", gW, GW0, s_fine)
        a.plot(s_fine, (rc - dc) / dc * 100, "-", lw=1.0, color=col, alpha=0.7)
    finish(a, r"$(R_\Gamma-{\rm denoised})/{\rm denoised}$ [%]",
           r"$R_\Gamma$ at $\Gamma_W=\Gamma_W^0\mp40$ MeV")
    a.set_xlabel(r"$\sqrt{s}$ [GeV]")

    fig.suptitle("√s carrying of the morph quantities — raw per-√s fit "
                 "vs the denoised curve the morph uses", y=1.00)
    fig.tight_layout()
    out_pdf = PLOTS / "sqrts_interpolation.pdf"
    out_png = PLOTS / "sqrts_interpolation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
