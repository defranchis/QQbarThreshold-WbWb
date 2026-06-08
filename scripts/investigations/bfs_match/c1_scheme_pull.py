#!/usr/bin/env python3
"""Δm_W from the C₁ factorisation-scheme choice (β-LL vs exact DELTA O(α)).

The matched-NLL radiator uses eMELA's DELTA-scheme NLL ePDF, but the O(α) matching
subtraction C₁ (isr_beta.oalpha_isr_subtraction) is built from the β-scheme LL
kernel.  Source (Frixione 1909.03886 Eq. G1sol2 + 2105.06688 Eq. Kdelz) shows the
exact DELTA O(α) C₁ shares the SAME kernel and differs ONLY by the β prefactor
(2L−1)→(2L), i.e. a per-√s rescale 2L/(2L−1) ≈ +4.1 % (L = ln(μ_F/m_e)).  This is
NOT a λ₁ soft constant.

This cross-fit measures the m_W bias of the production β-LL C₁ relative to the exact
DELTA C₁: reference templates = matched-NLL with β-LL C₁ (production); injected truth
= matched-NLL with the exact DELTA C₁.  Δm_W is how far the extracted m_W moves if
the C₁ scheme term is neglected.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import uncertainties as unc

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)

import matched_pull as mp                         # reuse _gen/_build_fit/_card_2poi
from framework.process.ww.indep import isr_beta
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from cards import ww_default as card

mp.BR_CONV = "off-shell"

_ORIG_SUB = isr_beta.oalpha_isr_subtraction


def _delta_sub(sqrt_s, born_fn, cfg):
    """Exact DELTA-scheme O(α) C₁ = β-LL C₁ × 2L/(2L−1), L = ln(μ_F/m_e).
    All β components scale together, so the whole subtraction rescales uniformly
    per √s (verified to 5 digits in c1_delta_vs_beta_fixedorder.py)."""
    val = _ORIG_SUB(sqrt_s, born_fn, cfg)
    arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    twoL = np.array([2.0 * np.log(cfg.mu_F(float(s)) / cfg.m_e) for s in arr])
    factor = twoL / (twoL - 1.0)
    out = np.atleast_1d(val) * factor
    return float(out[0]) if np.ndim(sqrt_s) == 0 else out


def _crossfit(ref_gen, truth_nom, shape_only, sub):
    """Build a fresh reference morph from ref_gen, inject truth_nom, fit (m_W,Γ_W)."""
    outdir = os.path.join(mp.BASE, mp.BR_CONV, sub)
    os.makedirs(outdir, exist_ok=True)
    c = mp._card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    for tag in params.tags:
        ref_gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                        mass_scheme="OS", outdir=outdir)
    fit = WWFit(c, ref_gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    if shape_only:
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0
    fit.create_scenario(pseudodata=truth_nom)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    tm = fit.d_params["nominal"]["mass"]
    tw = fit.d_params["nominal"]["width"]
    return {"dmW": (mass.n - tm) * 1e3, "dgW": (width.n - tw) * 1e3,
            "sigmW": mass.s * 1e3, "siggW": width.s * 1e3,
            "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
            "valid": bool(fit.minuit.valid)}


def main():
    print("[c1-pull] BR convention: off-shell, m_t=174.2 rich grid")
    print("[c1-pull] reference = matched-NLL with β-LL C₁ (production)")
    print("[c1-pull] truth     = matched-NLL with exact DELTA C₁  (β-LL × 2L/(2L−1))")

    # Reference generator: production matched-NLL (β-LL C₁, no patch).
    ref_gen = mp._gen(match=True, nnlo=True, dqcd=True, nll=True)

    # Truth generator: same, but build its line shape under the DELTA-C₁ patch.
    isr_beta.oalpha_isr_subtraction = _delta_sub
    try:
        truth_gen = mp._gen(match=True, nnlo=True, dqcd=True, nll=True)
        truth_fit = mp._build_fit(truth_gen, "c1truth_matched_NLL_DELTA")
        truth_nom = truth_fit.template("nominal")
    finally:
        isr_beta.oalpha_isr_subtraction = _ORIG_SUB   # restore for the reference

    # Sanity self-fit: β-truth into β-ref should give ~0.
    beta_truth = mp._build_fit(mp._gen(match=True, nnlo=True, dqcd=True, nll=True),
                               "c1truth_matched_NLL_beta").template("nominal")

    print(f"\n{'fit':36s} {'dm_W':>7s} {'dG_W':>7s} {'s(m_W)':>7s} "
          f"{'s(G_W)':>7s} {'rho':>6s}   [MeV]")
    for shape_only in (True, False):
        mode = "shape-only" if shape_only else "cov-lumi"
        r0 = _crossfit(ref_gen, beta_truth, shape_only, "c1ref_beta_a")
        r1 = _crossfit(ref_gen, truth_nom, shape_only, "c1ref_beta_b")
        flag0 = "" if r0["valid"] else " !INV"
        flag1 = "" if r1["valid"] else " !INV"
        print(f"{'β-self ('+mode+')':36s} {r0['dmW']:7.2f} {r0['dgW']:7.2f} "
              f"{r0['sigmW']:7.2f} {r0['siggW']:7.2f} {r0['rho']:+6.3f}{flag0}")
        print(f"{'DELTA-C₁ truth vs β ref ('+mode+')':36s} {r1['dmW']:7.2f} "
              f"{r1['dgW']:7.2f} {r1['sigmW']:7.2f} {r1['siggW']:7.2f} "
              f"{r1['rho']:+6.3f}{flag1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
