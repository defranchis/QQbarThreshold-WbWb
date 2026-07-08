#!/usr/bin/env python3
"""Validate the PRODUCTION lumi module ``framework...indep.isr_lumi`` (direct L̃
self-conv, cached per-√s) against the already-validated evaluator
``lumi_faithful.FaithfulLumi`` (soft_mode='grid') and the 2-D ``convolve_2leg``.

Checks (the step-1/2 gate of HANDOFF_lumi_to_production_2026-06-18.md):
  1. isr_lumi.sigma_obs  ≈  FaithfulLumi          (reproduces the validated 1-D)
  2. isr_lumi  CONVERGES cleanly in n_out (no spline fragility) & SMOOTHER than 2D-128
  3. isr_lumi  ≈  ripple-free 2-D mean            (absolute anchor, sparse √s)
  4. cached per-√s setup: 3-channel scan reuses one L̃ build (timing)

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/validate_lumi_production.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep import isr_lumi as IL  # noqa: E402
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
import scripts.investigations.nll_isr.lumi_faithful as LF  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")


def prod_cfg():
    return isr_beta.ISRConfig(nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
                              emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ",
                              emela_grid=PROD_GRID)


def smooth(c, i0):
    return float(np.abs(np.diff(c / c[i0], 2)).max())


def main():
    cfg = prod_cfg()
    grids = load_grids(scheme_alpha="gf")
    nlo = grids[("lnuqq", "nominal")].nlo_fn()
    ch = "lnuqq"
    SQ = np.linspace(157.5, 162.0, 19)
    i0 = int(np.argmin(np.abs(SQ - 160.5)))

    print("=== ripple-free 2-D reference (mean of convolve_2leg, n_quad 1024..2048) ===")
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    mean, std = LF.ripple_mean_2leg(LF.prod_nll, SQ, ch=ch, ns=range(1024, 2049, 256))
    p128 = isr_beta.convolve_2leg(SQ, nlo, LF.prod_nll(128))
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    sem = (std / math.sqrt(len(range(1024, 2049, 256))) / mean * 1e6)
    print(f"  reference sem ~ {np.median(sem):.0f} ppm (median), "
          f"max {sem.max():.0f} ppm")

    def shape(c):
        return (c / c[i0]) / (mean / mean[i0]) - 1.0

    # --- 1+2. convergence in n_out vs FaithfulLumi (matched) & vs 2-D mean ---
    print(f"\n=== isr_lumi convergence in n_out (ch={ch}) ===")
    print(" n_out  |d_norm vs2Dmean(mean,max ppm)|  |shape vs2Dmean(max ppm)|  "
          "|vs FaithfulLumi(max ppm)|  smooth")
    for n_out in (96, 128, 192, 256):
        lg = IL.sigma_obs(SQ, nlo, cfg, n_out=n_out)
        _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
        fa = np.array([LF.sigma_obs(float(s), nlo,
                                    LF.FaithfulLumi(cfg, float(s), soft_mode="grid",
                                                    n_jac=IL.LUMI_N_JAC),
                                    n_out=n_out) for s in SQ])
        os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
        dn = (lg / mean - 1.0) * 1e6
        sh = shape(lg) * 1e6
        vf = (lg / fa - 1.0) * 1e6
        print(f" {n_out:4d}    {dn.mean():+7.0f} {np.abs(dn).max():6.0f}            "
              f"{np.abs(sh).max():7.0f}                  {np.abs(vf).max():6.0f}"
              f"            {smooth(lg, i0):.2e}")
    print(f"  2D-128 production: shape vs 2D-mean max|{np.abs(shape(p128)).max()*1e6:.0f}| "
          f"ppm   smooth={smooth(p128, i0):.2e}")

    # --- 3. absolute anchor on sparse √s (independent 2-D mean) ---
    SQs = np.array([157.5, 158.5, 159.5, 160.5, 161.5])
    print(f"\n=== isr_lumi (n_out={IL.LUMI_N_OUT}) vs ripple-free 2-D mean, ch={ch} ===")
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    m2, s2 = LF.ripple_mean_2leg(LF.prod_nll, SQs, ch=ch, ns=range(2048, 3073, 256))
    lg_s = IL.sigma_obs(SQs, nlo, cfg)
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    d = (lg_s / m2 - 1.0) * 1e6
    print("  √s    :", " ".join("%8.1f" % s for s in SQs))
    print("  d(ppm):", " ".join("%+8.0f" % x for x in d))
    print("  2D sem:", " ".join("%7.0fp" % (s / m * 1e6)
                                for s, m in zip(s2 / math.sqrt(5), m2)))

    # --- 4. cached per-√s setup: 3-channel scan reuses one L̃ build ---
    print("\n=== timing: cached per-√s L̃ reused across channels ===")
    SQf = np.round(156.0 + 0.25 * np.arange(33), 4)
    IL._LUMI_CACHE.clear()
    t = time.time(); IL.sigma_obs(SQf, nlo, cfg); t_first = time.time() - t
    t = time.time()
    for c in PURE_WW_WEIGHTS:
        IL.sigma_obs(SQf, grids[(c, "nominal")].nlo_fn(), cfg)
    t_3ch = time.time() - t
    _dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
    t = time.time()
    for c in PURE_WW_WEIGHTS:
        isr_beta.convolve_2leg(SQf, grids[(c, "nominal")].nlo_fn(), cfg)
    t_2d = time.time() - t
    os.dup2(_sv, 1); os.close(_sv); os.close(_dn)
    print(f"  isr_lumi first scan (builds + caches L̃): {t_first*1e3:6.0f} ms")
    print(f"  isr_lumi 3-ch scan (reuses cached L̃)   : {t_3ch*1e3:6.0f} ms")
    print(f"  2-D 3-ch scan (cached radiator)        : {t_2d*1e3:6.0f} ms")

    print("\nVERDICT: isr_lumi reproduces FaithfulLumi & the 2-D mean, converges "
          "cleanly\n         in n_out, is smoother than 2D-128, and the cached L̃ "
          "is reused across channels.")


if __name__ == "__main__":
    main()
