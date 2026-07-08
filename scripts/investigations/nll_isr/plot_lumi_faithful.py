#!/usr/bin/env python3
"""Plot: the 2D convolution's Gauss-Legendre ripple vs the smooth faithful lumi.

Top: sigma_obs(sqrt(s)) for the production 2D (n_quad=128), a ripple-free 2D
reference (mean over high n_quad), and the FAITHFUL grid-based luminosity form.
Bottom: shape residual (normalisation removed at 160 GeV) vs the ripple-free 2D ->
shows the production 2D ripples at the +-0.3..0.7% level while the lumi is smooth
and tracks the reference.  (lnuqq, gf, NLL; the dominant WW channel.)

Publishes PNG+PDF to the EOS web area.

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/plot_lumi_faithful.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
import scripts.investigations.nll_isr.lumi_faithful as LF  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

SQ = np.linspace(157.5, 162.5, 101)
I0 = int(np.argmin(np.abs(SQ - 160.0)))
CH = "lnuqq"


def _shape(c, ref):
    return ((c / c[I0]) / (ref / ref[I0]) - 1.0) * 100.0


def main():
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    grids = load_grids(scheme_alpha="gf")
    nlo = grids[(CH, "nominal")].nlo_fn()
    # ripple-free 2D reference: mean over several high n_quad (ripple phase varies)
    truth = np.mean([isr_beta.convolve_2leg(SQ, nlo, LF.prod_nll(n))
                     for n in (2048, 2560, 3072)], 0) * FB_TO_PB
    p128 = isr_beta.convolve_2leg(SQ, nlo, LF.prod_nll(128)) * FB_TO_PB
    lumi = LF.line_shape_channel(LF.prod_nll(128), SQ, ch=CH, soft_mode="grid",
                                 n_out=256, n_jac=300) * FB_TO_PB
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

    d2 = lambda c: np.abs(np.diff(c / c[I0], 2)).max()  # noqa: E731

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1.0],
                                                  hspace=0.06))
    a0.plot(SQ, truth, "k-", lw=2.2,
            label="2D reference (mean n_quad 2048-3072)")
    a0.plot(SQ, p128, color="tab:red", lw=1.0,
            label="2D production (n_quad=128)")
    a0.plot(SQ, lumi, color="tab:blue", lw=1.5, ls=":",
            label="faithful luminosity (grid, n_jac=300)")
    a0.set_ylabel(r"$\sigma_{\rm obs}$ [pb]   ($e\nu qq$, gf, NLL)")
    a0.legend(frameon=False, fontsize=9, loc="upper left")
    a0.grid(alpha=0.25)
    a0.set_title("ISR convolution: the 2D's Gauss-Legendre ripple "
                 "(straddling the $\\sqrt{\\hat s}=156$ edge) vs the smooth lumi",
                 fontsize=10.5)

    a1.axhline(0, color="k", lw=1.4)
    a1.plot(SQ, _shape(p128, truth), color="tab:red", lw=1.0,
            label="2D-128 production  (ripple, max|2nd-diff|=%.1e)" % d2(p128))
    a1.plot(SQ, _shape(lumi, truth), color="tab:blue", lw=1.7,
            label="faithful lumi  (smooth, max|2nd-diff|=%.1e)" % d2(lumi))
    a1.set_ylabel("shape residual vs\n2D ref (norm rm.) [%]")
    a1.set_xlabel(r"$\sqrt{s}$ [GeV]")
    a1.set_ylim(-0.8, 0.8)
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    a1.grid(alpha=0.25)

    base = "/tmp/lumi_faithful_plot"
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    dest = "/eos/user/m/mdefranc/www/WW_threshold/scheme_variations"
    os.makedirs(dest, exist_ok=True)
    for ext in (".png", ".pdf"):
        shutil.copy(base + ext, os.path.join(dest, "lumi_faithful" + ext))
    os.system(f"chmod -R a+r {dest} 2>/dev/null")
    print("2D-128 max|2nd-diff|=%.2e   lumi max|2nd-diff|=%.2e   ref=%.2e"
          % (d2(p128), d2(lumi), d2(truth)))
    print("published https://mdefranc.web.cern.ch/WW_threshold/scheme_variations/"
          "lumi_faithful.png")


if __name__ == "__main__":
    main()
