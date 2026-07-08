#!/usr/bin/env python3
"""Diagnostic: the production isr_lumi line shape is SMOOTH where the 2-D ripples.

Three panels (pure-WW = lnuqq+qqqq+mutau, gf NLL, eMELA-grid radiator):
  (top)  σ_obs(√s) for 2D n_quad=128 (production), 2D n_quad=1024, and the lumi;
  (mid)  normalised residual vs the ripple-free 2-D mean → the 2-D's GL ripple
         (∝1/n_quad) vs the lumi's flat residual;
  (bot)  local smoothness |2nd finite-difference| → lumi ≪ 2-D.

Out: plots/nll_isr/lumi_vs_2d_lineshape.png
Run: source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
     python3 scripts/investigations/nll_isr/plot_lumi_production.py
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
import matplotlib.pyplot as plt  # noqa: E402

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from framework.process.ww.indep import isr_beta, isr_lumi as IL  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import FB_TO_PB  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

PROD = os.path.join(_REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
OUT = os.path.join(_REPO, "plots/nll_isr")


def cfg(lumi=False, n_quad=128):
    return isr_beta.ISRConfig(nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
                              emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ",
                              emela_grid=PROD, lumi=lumi, n_quad=n_quad)


def ls_2d(SQ, n_quad):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        tot = tot + wt * isr_beta.convolve_2leg(SQ, grids[(ch, "nominal")].nlo_fn(),
                                                cfg(False, n_quad))
    return tot * FB_TO_PB


def ls_lumi(SQ):
    grids = load_grids(scheme_alpha="gf")
    tot = np.zeros_like(SQ)
    for ch, wt in dict(PURE_WW_WEIGHTS).items():
        tot = tot + wt * IL.sigma_obs(SQ, grids[(ch, "nominal")].nlo_fn(), cfg(True))
    return tot * FB_TO_PB


def main():
    SQ = np.linspace(157.5, 162.5, 101)
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    p128 = ls_2d(SQ, 128)
    lumi = ls_lumi(SQ)
    # ripple-free 2-D reference: mean over many n_quad
    ref = np.mean([ls_2d(SQ, n) for n in range(1024, 2049, 256)], axis=0)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

    i0 = int(np.argmin(np.abs(SQ - 160.0)))
    def sm(c):
        return np.abs(np.diff(c / c[i0], 2))

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1.3, 1.3]))
    ax[0].plot(SQ, p128, lw=1.0, color="C3", label="2-D einsum, n_quad=128 (production)")
    ax[0].plot(SQ, ref, lw=1.6, color="0.4", label="2-D mean (ripple-free)")
    ax[0].plot(SQ, lumi, lw=1.0, color="C0", ls="--", label="luminosity (isr_lumi)")
    ax[0].set_ylabel(r"$\sigma_{\rm obs}$ (pure-WW) [pb]")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("Independent WW NLL ISR: luminosity vs 2-D convolution")

    ax[1].axhline(0, color="0.7", lw=0.7)
    ax[1].plot(SQ, (p128 / ref - 1) * 1e4, color="C3", lw=1.0, label="2-D 128")
    ax[1].plot(SQ, (lumi / ref - 1) * 1e4, color="C0", lw=1.2, ls="--", label="lumi")
    ax[1].set_ylabel(r"residual vs 2-D mean [$10^{-4}$]")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].semilogy(SQ[1:-1], sm(p128), color="C3", lw=1.0, label="2-D 128")
    ax[2].semilogy(SQ[1:-1], sm(lumi), color="C0", lw=1.2, ls="--", label="lumi")
    ax[2].semilogy(SQ[1:-1], sm(ref), color="0.4", lw=1.0, label="2-D mean")
    ax[2].set_ylabel(r"$|\Delta^2(\sigma/\sigma_0)|$")
    ax[2].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT, "lumi_vs_2d_lineshape.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    print(f"max|2nd-diff|:  2D-128={sm(p128).max():.2e}  lumi={sm(lumi).max():.2e}"
          f"  2D-mean={sm(ref).max():.2e}")
    print(f"max|resid vs mean|:  2D-128={np.abs(p128/ref-1).max()*1e4:.1f}e-4  "
          f"lumi={np.abs(lumi/ref-1).max()*1e4:.1f}e-4")


if __name__ == "__main__":
    main()
