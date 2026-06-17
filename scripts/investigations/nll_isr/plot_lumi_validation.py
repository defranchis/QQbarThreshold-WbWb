#!/usr/bin/env python3
"""Validation plot for the luminosity-function convolution prototype.

Top: σ_obs(√s) for the converged 2D (n_quad=1024), production 2D (n_quad=128), and
the luminosity form (radiator built once, n_out=48).  Bottom: shape residual
(normalisation removed at 160 GeV) relative to the 2D-1024 reference — shows the
production 2D's ripple AND that the un-converged 2D-512 is FARTHER from 2D-1024
than the luminosity form is, at ~500× fewer σ̂ evaluations.

Publishes PNG+PDF to the EOS web area.

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/plot_lumi_validation.py
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
import scripts.investigations.nll_isr.luminosity_prototype as L  # noqa: E402
import scripts.investigations.nll_isr.lumi_grid as LG  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

SQ = np.linspace(157.5, 161.5, 161)
I0 = int(np.argmin(np.abs(SQ - 160.0)))


def _lumi():
    """Production luminosity-GRID form: L̃(V;μ_F) built once, single-panel eval."""
    grid = LG.build_lumi_grid(L.prod_nll(128))
    obs, _ = LG.line_shape_grid(grid, SQ, n_out=24)
    return obs


def _shape(c, ref):
    return ((c / c[I0]) / (ref / ref[I0]) - 1.0) * 100.0


def main():
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    truth, _ = L.line_shape_2leg(L.prod_nll(1024), SQ)
    p512, _ = L.line_shape_2leg(L.prod_nll(512), SQ)
    p128, _ = L.line_shape_2leg(L.prod_nll(128), SQ)
    lumi = _lumi()
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1.0],
                                                  hspace=0.06))
    a0.plot(SQ, truth, "k-", lw=2.2, label="2D converged (n_quad=1024)")
    a0.plot(SQ, p128, color="tab:red", lw=1.1, ls="--",
            label="2D production (n_quad=128)")
    a0.plot(SQ, lumi, color="tab:blue", lw=1.4, ls=":",
            label="luminosity form (built once, n_out=48)")
    a0.set_ylabel(r"$\sigma_{\rm obs}$ [pb]  (gf, NLL, pure-WW)")
    a0.legend(frameon=False, fontsize=9, loc="upper left")
    a0.grid(alpha=0.25)
    a0.set_title("Luminosity-function convolution: smoother & more accurate "
                 "than the 2D, ~500× fewer σ̂-evals", fontsize=10.5)

    a1.axhline(0, color="k", lw=1.6)
    a1.plot(SQ, _shape(p128, truth), color="tab:red", lw=1.1,
            label="2D-128 (production)  −  ripple")
    a1.plot(SQ, _shape(p512, truth), color="tab:green", lw=1.1,
            label="2D-512  (still un-converged)")
    a1.plot(SQ, _shape(lumi, truth), color="tab:blue", lw=1.6,
            label="luminosity (closer to converged)")
    a1.set_ylabel("shape residual vs\n2D-1024 (norm rm.) [%]")
    a1.set_xlabel(r"$\sqrt{s}$ [GeV]")
    a1.set_ylim(-0.5, 0.5)
    a1.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    a1.grid(alpha=0.25)

    base = "/tmp/lumi_validation"
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    dest = "/eos/user/m/mdefranc/www/WW_threshold/scheme_variations"
    os.makedirs(dest, exist_ok=True)
    for ext in (".png", ".pdf"):
        shutil.copy(base + ext, os.path.join(dest, "lumi_validation" + ext))
    os.system(f"chmod -R a+r {dest} 2>/dev/null")
    print("published https://mdefranc.web.cern.ch/WW_threshold/scheme_variations/"
          "lumi_validation.png")


if __name__ == "__main__":
    main()
