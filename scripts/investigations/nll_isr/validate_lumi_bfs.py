"""Validate the BFS luminosity (1-D) NLL ISR convolution against the edge-aware
2-leg tensor rule.

The BFS chain's production NLL ISR is the edge-aware 2-leg double convolution
(``isr.sigma_ISR_2leg_convolution(nll=True, edge_aware=True)``).  ``isr_lumi_bfs``
collapses the same convolution onto the 1-D luminosity variable z = x1 x2 — a
self-contained port of the independent chain's ``indep/isr_lumi.py`` that sources
the per-leg density ρ = x·D from the same ``emela_wrapper.code_pdf`` the 2-leg
path uses, and uses the BFS σ̂ support floor (149 GeV) for V_top.

Expectation: the two SMOOTH quadratures agree to <1 ppm in normalisation and
~0 ppm in shape across the 157-163 GeV scan window — an independent-quadrature
confirmation of the edge-aware line shape.

Run (from repo root, after ``source setup.sh``):
    WW_ISR_NJOBS=8 python3 scripts/investigations/nll_isr/validate_lumi_bfs.py
"""
import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq
from framework.process.ww.xsec_calculator.isr import sigma_ISR_2leg_convolution
from framework.process.ww.xsec_calculator import isr_lumi_bfs as L

MW, GW = 80.379, 2.085
# Production partonic chain, WITHOUT the WHIZARD anchor (isolate the ISR method;
# the anchor multiplies σ̂ identically for both quadratures).
PKW = dict(channel="inclusive", include_coulomb=False, br_convention="pdg-constant",
           include_NLO_hard_decay=True, include_BFS_NNLO=True, apply_delta_QCD=True,
           apply_whizard_anchor=False, decay_uses_full_born=True)


def _shat(sqrt_shat):
    return sigma_partonic_munuqq(np.asarray(sqrt_shat, float) ** 2, MW, GW, **PKW)


def main():
    grid = np.round(np.arange(157.0, 163.01, 0.25), 3)
    ref = sigma_ISR_2leg_convolution(grid, sigma_partonic_munuqq, mW=MW, gammaW=GW,
                                     nll=True, edge_aware=True, **PKW)
    lum = L.sigma_obs(grid, _shat)
    r = np.asarray(lum) / np.asarray(ref)
    print(f"{'sqrt_s':>8} {'2leg_edge':>12} {'lumi':>12} {'rel_ppm':>10}")
    for s, a, b in zip(grid, ref, lum):
        print(f"{s:8.2f} {a:12.6f} {b:12.6f} {1e6 * (b / a - 1):10.1f}")
    print(f"\nmean rel offset = {1e6 * (r.mean() - 1):+.1f} ppm   "
          f"shape spread (max-min) = {1e6 * (r.max() - r.min()):.1f} ppm")
    assert abs(r.mean() - 1) < 5e-6, "normalisation offset too large"
    assert (r.max() - r.min()) < 5e-6, "shape spread too large"
    print("PASS: BFS luminosity reproduces the edge-aware 2-leg line shape.")


if __name__ == "__main__":
    main()
