#!/usr/bin/env python3
"""Figure: the line-shape corrections the BFS-on-MoCaNLO matching adds, vs sqrt(s)
(EXPLORATORY — report sec:match). Ratio to the unmatched-LL line shape at nominal
(m_W,Gamma_W); the BR factor cancels in the ratio so this is convention-independent.
The shape-carrying corrections (delta_NNLO, NLL ISR) are what bias m_W in shape-only;
delta_QCD is a flat +5.4% normalisation (annotated), shape-neutral."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from framework.process.ww.indep.mocanlo_cards import SMInputs

RESULTS = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
           "mocanlo/grid_gen/results_mt174p2")
OUT = os.path.join(_REPO, "report", "figs", "indep_match_lineshape.pdf")


def _obs(ecm, **kw):
    g = WWGeneratorMoCaNLO(results_dir=RESULTS, sm=SMInputs(mt=174.2), **kw)
    return np.asarray(g._morphed(80.379, 2.085, ecm), dtype=float)


def main():
    ecm = np.arange(157.0, 163.01, 0.25)
    ref = _obs(ecm, match_bfs=False)
    d_nnlo = _obs(ecm, match_bfs=True, match_bfs_dqcd=False)            # delta_NNLO
    d_nll = _obs(ecm, match_bfs=False, isr_nll=True)                    # NLL ISR
    d_shape = _obs(ecm, match_bfs=True, match_bfs_dqcd=False, isr_nll=True)  # NNLO+NLL
    d_qcd = _obs(ecm, match_bfs=True, match_bfs_nnlo=False)             # delta_QCD (flat)
    qcd_pct = float(np.mean(100 * (d_qcd / ref - 1)))

    # The NLL/LL ratio uses two different radiators, so the partonic-MC scatter
    # in sigma_hat does NOT cancel (unlike delta_NNLO, same radiator → exact
    # cancellation).  Smooth the ratios in sqrt(s) for display; the underlying
    # physics is smooth and the quantitative result is the fit pull, not this curve.
    def sm(y):
        w = min(9, len(y) - (1 - len(y) % 2))   # odd window <= len
        return savgol_filter(y, w, 2) if w >= 3 else y

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(ecm, sm(100 * (d_nnlo / ref - 1)), lw=2, label=r"$\delta_{\rm NNLO}$ (BFS NNLO block)")
    ax.plot(ecm, sm(100 * (d_nll / ref - 1)), lw=2, label=r"NLL ISR (LL$\to$eMELA-NLL)")
    ax.plot(ecm, sm(100 * (d_shape / ref - 1)), lw=2, ls="--", color="k",
            label=r"matched shape ($\delta_{\rm NNLO}+$NLL)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"line-shape ratio to unmatched-LL  [\%]")
    ax.set_title("BFS-on-MoCaNLO matching: line-shape corrections (nominal)")
    ax.annotate(rf"$\delta_{{\rm QCD}}$: flat ${qcd_pct:+.1f}\%$ (decay normalisation, shape-neutral)",
                xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=9, color="0.35")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"  delta_QCD flat ~ {qcd_pct:+.2f}% ; "
          f"delta_NNLO@163 = {100*(d_nnlo[-1]/ref[-1]-1):+.3f}% ; "
          f"NLL@157 = {100*(d_nll[0]/ref[0]-1):+.3f}%")


if __name__ == "__main__":
    main()
