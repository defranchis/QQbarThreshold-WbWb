"""Absolute √s dependence of the morph quantities.

Companion to plot_sqrts_interpolation.py: that script shows the
*deviation* of the raw per-√s fit from the denoised curve; this one
shows the quantities themselves — σ_nom, β, R_m, R_Γ vs √s — so the
magnitude and shape of each are visible. Points are the raw per-√s fit
nodes; the line is the denoised curve the operational morph carries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import build_operational_morph, MW0, GW0

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)
FINE_SPAN = (155.0, 165.0)  # grid_fine √s window (0.1-GeV step)


def _R(coef_or_splines, kind, val, ref, s=None):
    """R = polyval(coef, val)/polyval(coef, ref); read coefficients from
    a splines dict at √s=s if given, else from a 3-vector directly."""
    if s is None:
        c = coef_or_splines
    else:
        c = np.array([coef_or_splines[f"coef_{kind}_{k}"](s) for k in range(3)])
    return np.polyval(c, val) / np.polyval(c, ref)


def main():
    ax, raw_spl, raw_m = build_operational_morph(denoise=False)
    _,  den_spl, _     = build_operational_morph(denoise=True)
    s_fine = np.linspace(ax[0], ax[-1], 700)
    print(f"{len(ax)} √s nodes, {ax[0]:.1f}–{ax[-1]:.1f} GeV")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    def finish(a, ylabel, title):
        a.axvspan(*FINE_SPAN, color="orange", alpha=0.07,
                  label="grid_fine (0.1 GeV)")
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=12)
        a.grid(alpha=0.25)
        a.legend(loc="best", fontsize=8)

    # σ_nom
    a = axes[0][0]
    a.plot(ax, [m["sigma_nom"] for m in raw_m], "o", ms=4, color="C0",
           alpha=0.55, label="raw per-√s node")
    a.plot(s_fine, den_spl["sigma_nom"](s_fine), "-", lw=2.2, color="C0",
           label="denoised curve (used)")
    a.axvline(2 * MW0, color="grey", ls=":", alpha=0.4)
    finish(a, r"$\sigma_{\rm nom}(\mu\nu u\bar d)$ [fb]",
           r"$\sigma_{\rm nom}$ — line shape")

    # β
    a = axes[0][1]
    a.plot(ax, [m["beta"] for m in raw_m], "o", ms=4, color="C3",
           alpha=0.55, label="raw per-√s node")
    a.plot(s_fine, den_spl["beta"](s_fine), "-", lw=2.2, color="C3",
           label="denoised curve (used)")
    a.axhline(0, color="grey", lw=0.7)
    a.axvline(2 * MW0, color="grey", ls=":", alpha=0.4)
    finish(a, r"$\beta$ [GeV$^{-2}$]", r"$\beta$ — bilinear cross term")

    # R_m at m_W = ∓100 MeV
    a = axes[1][0]
    for mW, col in [(80.279, "C0"), (80.479, "C1")]:
        a.plot(ax, [_R(m["coef_m"], "m", mW, MW0) for m in raw_m], "o", ms=4,
               color=col, alpha=0.55, label=rf"$m_W={mW:.3f}$ (node)")
        a.plot(s_fine, _R(den_spl, "m", mW, MW0, s_fine), "-", lw=2.2,
               color=col, label=rf"$m_W={mW:.3f}$ (denoised)")
    a.axhline(1.0, color="grey", lw=0.7)
    finish(a, r"$R_m$", r"$R_m$ at $m_W=m_W^0\mp100$ MeV")
    a.set_xlabel(r"$\sqrt{s}$ [GeV]")

    # R_Γ at Γ_W = ∓40 MeV
    a = axes[1][1]
    for gW, col in [(2.045, "C0"), (2.125, "C1")]:
        a.plot(ax, [_R(m["coef_g"], "g", gW, GW0) for m in raw_m], "o", ms=4,
               color=col, alpha=0.55, label=rf"$\Gamma_W={gW:.3f}$ (node)")
        a.plot(s_fine, _R(den_spl, "g", gW, GW0, s_fine), "-", lw=2.2,
               color=col, label=rf"$\Gamma_W={gW:.3f}$ (denoised)")
    a.axhline(1.0, color="grey", lw=0.7)
    finish(a, r"$R_\Gamma$", r"$R_\Gamma$ at $\Gamma_W=\Gamma_W^0\mp40$ MeV")
    a.set_xlabel(r"$\sqrt{s}$ [GeV]")

    fig.suptitle("Morph quantities vs √s — absolute values "
                 "(raw per-√s nodes and the denoised curve)", y=1.00)
    fig.tight_layout()
    out_pdf = PLOTS / "sqrts_quantities.pdf"
    out_png = PLOTS / "sqrts_quantities.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
