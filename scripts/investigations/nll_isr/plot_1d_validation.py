#!/usr/bin/env python3
"""Validation plot for the 1D-collapse convolution prototype.

Top: σ_obs(√s) [pb] for the 2D truth (n_quad=512), the production 2D (n_quad=128),
and the 1D collapse (n_panel=64, m_E=512).  Bottom: (method/truth − 1)×100 [%] —
shows the production 2D's GL-quadrature ripple and that the 1D collapse tracks the
truth smoothly.

Writes PNG+PDF to /tmp and (for a shareable link) the EOS web area.

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/plot_1d_validation.py
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

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
import scripts.investigations.nll_isr.convolve_1d_prototype as P  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

SQ = np.linspace(157.5, 161.5, 161)        # 0.025 GeV — resolves the ripple


def main():
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    truth, _ = P.line_shape("2leg", P.prod_nll(512), SQ)
    p128, _ = P.line_shape("2leg", P.prod_nll(128), SQ)
    c1d, _ = P.line_shape("1d", P.prod_nll(128), SQ, n_panel=64, m_E=512)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1.0],
                                                  hspace=0.06))
    a0.plot(SQ, truth, "k-", lw=2.2, label="2D truth (n_quad=512)")
    a0.plot(SQ, p128, color="tab:red", lw=1.2, ls="--",
            label="2D production (n_quad=128)")
    a0.plot(SQ, c1d, color="tab:green", lw=1.2, ls=":",
            label="1D collapse (n_panel=64, m_E=512)")
    a0.set_ylabel(r"$\sigma_{\rm obs}$ [pb]  (gf, NLL, pure-WW)")
    a0.legend(frameon=False, fontsize=9, loc="upper left")
    a0.grid(alpha=0.25)
    a0.set_title("1D-collapse convolution validation vs the 2D quadrature",
                 fontsize=11)

    a1.axhline(0, color="k", lw=1.6)
    a1.plot(SQ, (p128 / truth - 1) * 100, color="tab:red", lw=1.2,
            label=f"2D-128 (max |2nd-diff| {P.smooth_metric(p128, len(SQ)//2)[0]:.1e})")
    a1.plot(SQ, (c1d / truth - 1) * 100, color="tab:green", lw=1.5,
            label=f"1D-64/512 (max |2nd-diff| {P.smooth_metric(c1d, len(SQ)//2)[0]:.1e})")
    a1.set_ylabel(r"$\sigma/\sigma_{\rm truth}-1$ [%]")
    a1.set_xlabel(r"$\sqrt{s}$ [GeV]")
    a1.legend(frameon=False, fontsize=8.5, loc="upper right")
    a1.grid(alpha=0.25)

    base = "/tmp/conv1d_validation"
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print("wrote", base + ".png", base + ".pdf")

    dest = "/eos/user/m/mdefranc/www/WW_threshold/scheme_variations"
    try:
        os.makedirs(dest, exist_ok=True)
        for ext in (".png", ".pdf"):
            import shutil
            shutil.copy(base + ext, os.path.join(dest, "conv1d_validation" + ext))
        os.system(f"chmod -R a+r {dest} 2>/dev/null")
        print("published:",
              "https://mdefranc.web.cern.ch/WW_threshold/scheme_variations/"
              "conv1d_validation.png")
    except Exception as e:
        print("publish failed:", e)


if __name__ == "__main__":
    main()
