#!/usr/bin/env python3
"""σ(√s) line-shape comparison across EW (σ̂) and ISR schemes — report figure.

Two panels, side by side:
  LEFT  — hard EW-coupling scheme of σ̂: gf (production) / alphaz / alpha0, with
          the IDENTICAL production ISR (eMELA NLL, DELTA/ALPMZ, α(M_Z)).
  RIGHT — ISR scheme/order at fixed σ̂ (gf) and fixed α(M_Z): LL+exp (BETA),
          NLL DELTA/ALPMZ (production), NLL DELTA/MSBAR-ren.

Top row: absolute σ_obs(√s) [pb] (pure-WW total, nominal m_W=80.379, Γ_W=2.085).
Bottom row: line-shape ratio with the overall NORMALISATION DIVIDED OUT (each curve
and the reference are first normalised to their value at √s₀=161 GeV, then ratioed)
— this isolates the √s-shape difference, which is what biases m_W (a flat
normalisation is absorbed by the luminosity/rate).  Shown as (ratio−1)×100 [%].

Writes report/figs/ww_scheme_variations.pdf (+ a copy under plots/nll_isr_schemes/).

Run:  source setup.sh && PYTHONPATH=$PWD \
      python3 scripts/investigations/nll_isr/plot_scheme_lineshapes.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silence the eMELA C-level banner during grid loads.
_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS
from framework.process.ww.indep.partonic_grid import load_grids
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

ALPHA = isr_beta.ALPHA_MZ_EMELA
GRIDDIR = "/tmp/ww_nll_scheme_scan/grids"
DELTA_GRID = os.path.join(GRIDDIR, "emela_nll_delta_alpmz.npz")
MSBARREN_GRID = os.path.join(GRIDDIR, "emela_nll_delta_msbarren_alpmz.npz")

SQ = np.linspace(157.0, 163.0, 61)
SQ0 = 161.0   # normalisation anchor for the shape ratio

# Gauss-Legendre nodes per leg in the ISR convolution.  Production templates use
# 128; the visible line-shape ripple in the turn-on is a GL quadrature artefact
# that scales as 1/n_quad (the moving σ̂ turn-on kink beating against the nodes).
# Default 512 so the report figure matches its caption (a 128 run puts a
# 0.25-GeV staircase in the shape-ratio panel).  Override with WW_PLOT_NQUAD.
NQUAD = int(os.environ.get("WW_PLOT_NQUAD", "512"))


def _prod_nll():
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", emela_grid=DELTA_GRID,
                              n_quad=NQUAD)


def _assemble(scheme_alpha, cfg):
    """Pure-WW total σ_obs(√s) [pb], nominal varpoint, for one σ̂ scheme + ISR cfg."""
    g = load_grids(scheme_alpha=scheme_alpha)
    tot = np.zeros_like(SQ)
    tot0 = 0.0
    for k, w in dict(PURE_WW_WEIGHTS).items():
        nlo = g[(k, "nominal")].nlo_fn()
        tot = tot + w * isr_beta.sigma_observed(SQ, nlo, cfg)
    return tot * FB_TO_PB


def _shape_pct(curve, ref):
    """(normalised curve / normalised ref − 1)×100, anchored at SQ0."""
    i0 = int(np.argmin(np.abs(SQ - SQ0)))
    return ((curve / curve[i0]) / (ref / ref[i0]) - 1.0) * 100.0


def main():
    # --- EW schemes (production ISR) ---
    ew = {sa: _assemble(sa, _prod_nll()) for sa in ["gf", "alphaz", "alpha0"]}
    # --- ISR schemes (gf σ̂, fixed α(M_Z)) ---
    isr = {
        "LL+exp (BETA)": _assemble("gf", isr_beta.ISRConfig(scheme="LO_beta",
                                                            alpha=ALPHA,
                                                            n_quad=NQUAD)),
        "NLL Δ/ALPMZ (prod.)": _assemble("gf", _prod_nll()),
        "NLL Δ/MSBAR-ren": _assemble("gf", isr_beta.ISRConfig(
            nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
            emela_ren_scheme="MSBAR", emela_grid=MSBARREN_GRID, n_quad=NQUAD)),
    }

    fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex="col",
                           gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.07,
                                            wspace=0.22))
    cEW = {"gf": "k", "alphaz": "tab:red", "alpha0": "tab:blue"}
    lEW = {"gf": r"$G_\mu$ (prod.)", "alphaz": r"$\alpha(M_Z)$",
           "alpha0": r"$\alpha(0)$"}
    cIS = ["tab:green", "k", "tab:purple"]

    # EW column
    for sa in ["gf", "alphaz", "alpha0"]:
        ax[0, 0].plot(SQ, ew[sa], color=cEW[sa], lw=1.7, label=lEW[sa])
        ax[1, 0].plot(SQ, _shape_pct(ew[sa], ew["gf"]), color=cEW[sa], lw=1.7)
    ax[0, 0].set_title(r"Hard EW-coupling scheme of $\hat\sigma$  "
                       r"(same NLL ISR)", fontsize=11)
    ax[0, 0].set_ylabel(r"$\sigma_{\rm obs}$ [pb]")
    ax[1, 0].set_ylabel(r"line-shape ratio to $G_\mu$" "\n"
                        r"(norm. removed) [%]")
    ax[1, 0].axhline(0, color="0.6", lw=0.8, ls=":")

    # ISR column
    for (lab, cur), c in zip(isr.items(), cIS):
        ax[0, 1].plot(SQ, cur, color=c, lw=1.7, label=lab)
        ax[1, 1].plot(SQ, _shape_pct(cur, isr["NLL Δ/ALPMZ (prod.)"]),
                      color=c, lw=1.7)
    ax[0, 1].set_title(r"ISR scheme/order  (same $\hat\sigma_{G_\mu}$, "
                       r"fixed $\alpha(M_Z)$)", fontsize=11)
    ax[1, 1].set_ylabel(r"line-shape ratio to prod." "\n"
                        r"(norm. removed) [%]")
    ax[1, 1].axhline(0, color="0.6", lw=0.8, ls=":")

    for a in ax[0]:
        a.legend(fontsize=9, frameon=False, loc="upper left")
        a.grid(alpha=0.25)
    for a in ax[1]:
        a.set_xlabel(r"$\sqrt{s}$ [GeV]")
        a.grid(alpha=0.25)
    fig.suptitle(r"WW-threshold line shape: EW($\hat\sigma$) and ISR scheme "
                 r"variations (pure-WW, nominal)", fontsize=12, y=0.98)

    outs = [os.path.join(_REPO, "report/figs/ww_scheme_variations.pdf"),
            os.path.join(_REPO, "plots/nll_isr_schemes/ww_scheme_variations.pdf")]
    for o in outs:
        os.makedirs(os.path.dirname(o), exist_ok=True)
        fig.savefig(o, bbox_inches="tight")
        print("wrote", o)
    # A PNG copy (named by n_quad) for quick on-screen viewing where PDF preview
    # is unavailable.
    png = f"/tmp/ww_scheme_variations_nquad{NQUAD}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print("wrote", png)

    # numeric summary (so the shape spread is reported, not just drawn)
    print("\nshape spread (max |line-shape ratio| over 157–163), normalisation removed:")
    for sa in ["alphaz", "alpha0"]:
        print(f"  EW  gf↔{sa:7s}: {np.max(np.abs(_shape_pct(ew[sa], ew['gf']))):.3f} %")
    for lab, cur in isr.items():
        if "prod" in lab:
            continue
        print(f"  ISR prod↔{lab:22s}: "
              f"{np.max(np.abs(_shape_pct(cur, isr['NLL Δ/ALPMZ (prod.)']))):.3f} %")


if __name__ == "__main__":
    main()
