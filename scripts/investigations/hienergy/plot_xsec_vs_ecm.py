#!/usr/bin/env python3
"""sigma_WW vs sqrt(s) over 157-365 GeV, NLO+NLL (MoCaNLO NLO-EW (X) eMELA NLL ISR).

The full pure-WW line shape from the independent chain across the whole FCC-ee
WW programme range: threshold scan (existing production grid, 156-164 GeV) plus
the above-threshold grid generated for this plot (results_hienergy, 165-365 GeV,
condor cluster 12874608).  No BFS matching -- plain MoCaNLO NLO-EW partonic
sigma_hat convolved with the eMELA NLL radiator.

We build a COMBINED nominal-only grid dir (symlinks to the production threshold
nominal grids + the high-E nominal grids) and evaluate the nominal line shape
directly via WWGeneratorMoCaNLO._varpoint_lineshape("nominal", .) -- no morph is
needed at the nominal point.  Output: report/figs/xsec_vs_ecm_157_365.pdf.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO

PROD = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results"
HIEN = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_hienergy"
COMBINED = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_curve"
CHANS = ("lnuqq", "qqqq", "mutau")
OUT = os.path.join(_REPO, "report", "figs", "xsec_vs_ecm_157_365.pdf")


def build_combined_dir() -> str:
    """Symlink the NOMINAL *_gf.csv grids for the 3 pure-WW channels from the
    production (threshold) and high-E dirs into one dir for load_grids."""
    os.makedirs(COMBINED, exist_ok=True)
    for f in glob.glob(os.path.join(COMBINED, "*.csv")):
        os.unlink(f)
    n = 0
    for src in (PROD, HIEN):
        for ch in CHANS:
            for path in glob.glob(os.path.join(src, f"{ch}_nominal_ecm*_gf.csv")):
                dst = os.path.join(COMBINED, os.path.basename(path))
                if not os.path.exists(dst):
                    os.symlink(path, dst)
                    n += 1
    print(f"[curve] combined nominal dir: {n} grids linked -> {COMBINED}")
    return COMBINED


def lineshape(nll: bool, ecm: np.ndarray, smooth: float | None) -> np.ndarray:
    gen = WWGeneratorMoCaNLO(
        results_dir=COMBINED, scheme_alpha="gf", lepton_cut=None,
        isr_cfg=isr_beta.ISRConfig(scheme="LO_beta"),
        br_convention="off-shell", isr_nll=nll, smooth=smooth)
    return gen._varpoint_lineshape("nominal", ecm)   # pb, no morph at nominal


def main():
    build_combined_dir()
    ecm = np.linspace(157.0, 365.0, 209)        # 1 GeV steps
    # smooth=None lets each channel spline auto-smooth over its (dense threshold
    # + sparse high-E) support; tune if the curve wiggles.
    y_nll = lineshape(True, ecm, smooth=None)
    y_ll = lineshape(False, ecm, smooth=None)

    print(f"[curve] NLL sigma_WW: {ecm[0]:.0f}->{y_nll[0]:.3f}, "
          f"peak {ecm[np.argmax(y_nll)]:.0f}->{y_nll.max():.3f}, "
          f"{ecm[-1]:.0f}->{y_nll[-1]:.3f} pb")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ecm, y_nll, "-", lw=2.0, color="C0",
            label=r"NLO-EW $\otimes$ NLL ISR")
    ax.plot(ecm, y_ll, "--", lw=1.2, color="C1", alpha=0.8,
            label=r"NLO-EW $\otimes$ LL+exp ISR")
    ax.axvline(2 * 80.379, color="0.7", lw=0.8, ls=":")
    ax.text(2 * 80.379 + 1.5, 0.5, r"$2 m_W$", color="0.5", fontsize=9)
    ax.axvline(240, color="0.8", lw=0.8, ls=":")
    ax.text(241, 14.5, r"$ZH$ run (240)", color="0.5", fontsize=8, rotation=90, va="top")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\sigma_{WW}$ [pb]")
    ax.set_title(r"Independent MoCaNLO pure-WW line shape, $\sqrt{s}=157$--$365\,$GeV")
    ax.set_xlim(157, 365)
    ax.set_ylim(0, max(y_nll.max(), y_ll.max()) * 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"[curve] wrote {OUT}")


if __name__ == "__main__":
    main()
