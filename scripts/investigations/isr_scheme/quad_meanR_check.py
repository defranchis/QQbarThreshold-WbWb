#!/usr/bin/env python3
"""Follow-up: confirm the NORMALISATION (mean R) for ALGMU/ALPMZ is
quadrature-stable, since the per-point R(√s) ratios showed ~5e-4 second-
difference noise at n_quad=128 vs 192. The mean over the grid (= the norm
shift that drives the lumi-prior m_W bias) should be far more stable than
any single ratio point.
"""
from __future__ import annotations
import numpy as np
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
from framework.process.ww.xsec_calculator.eft_xsec import alpha_Gmu, M_W_BFS_REF

MW, GW = 80.379, 2.085
GRID = np.array([157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0])
A_ALPMZ = 1.0 / 128.943
A_ALGMU = alpha_Gmu(M_W_BFS_REF)


def sig(ren, a, nq):
    return np.asarray(sigma_observed_munuqq(
        GRID, mW=MW, gammaW=GW, isr_nll=True,
        isr_emela_ren_scheme=ren, alpha_em_isr=a, n_quad=nq))


print("nq   meanR(ALGMU/ALPMZ)   norm%   slope%/GeV")
for nq in (128, 192, 256):
    R = sig("ALGMU", A_ALGMU, nq) / sig("ALPMZ", A_ALPMZ, nq)
    norm = (np.mean(R) - 1) * 100
    slope = np.polyfit(GRID, R, 1)[0] * 100
    print(f"{nq:4d}  {np.mean(R):.8f}       {norm:+.4f}  {slope:+.5f}")
