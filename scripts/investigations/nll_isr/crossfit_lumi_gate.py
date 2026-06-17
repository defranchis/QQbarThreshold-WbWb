#!/usr/bin/env python3
"""PROMOTION GATE — cross-fit Δm_W for the 1-D luminosity ISR convolution vs the
2-D einsum (HANDOFF_lumi_to_production_2026-06-18.md, criterion #4).

Both sides use the SAME eMELA-GRID NLL radiator (α(M_Z)/ALPMZ/DELTA); the ONLY
difference is the convolution: ``ISRConfig.lumi=False`` (2-D einsum) vs ``True``
(``isr_lumi`` luminosity self-convolution).  So this isolates the convolution
change — not the grid-vs-direct radiator difference.

Cross-fits (Asimov, full 2-POI factorized morph, pure-WW = lnuqq+qqqq+mutau):
  (a) NULL:   truth=lumi, morph=lumi   → Δ ≈ 0 (construction sanity)
  (b) GATE:   truth=2-D , morph=lumi   → |Δm_W|, |ΔΓ_W| must be < 0.05 MeV
  (b') GATE:  truth=lumi, morph=2-D    → reversed direction, same bar

Reported for shape-only (lumi free) AND production cov-lumi.  Expected ≪0.05 MeV
(the lumi is smoother than the 2-D and converges to its ripple-free mean, so the
coherent shape difference → 0) — but MEASURE it; it is the gate.

Run:  source setup.sh && WW_INDEP_NJOBS=16 PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/crossfit_lumi_gate.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from framework.process.ww.indep.partonic_grid import DEFAULT_RESULTS_DIR
from scripts.investigations.nll_isr.scheme_alpha_scan import _card_2poi

BASE = "/tmp/ww_lumi_gate/templates"
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
GATE_MEV = 0.05


def _cfg(lumi: bool) -> isr_beta.ISRConfig:
    """NLL eMELA-GRID radiator; toggle ONLY the convolution (2-D vs luminosity)."""
    return isr_beta.ISRConfig(
        nll=True, alpha=isr_beta.ALPHA_MZ_EMELA, emela_fac_scheme="DELTA",
        emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID, lumi=lumi)


def _gen(cfg: isr_beta.ISRConfig) -> WWGeneratorMoCaNLO:
    return WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg,
                              results_dir=DEFAULT_RESULTS_DIR)


def gen_full_templates(cfg: isr_beta.ISRConfig, key: str) -> str:
    outdir = os.path.join(BASE, key)
    os.makedirs(outdir, exist_ok=True)
    gen = _gen(cfg)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def fresh_fit(cfg: isr_beta.ISRConfig, outdir: str) -> WWFit:
    c = _card_2poi()
    fit = WWFit(c, _gen(cfg), input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def crossfit(truth_nom, morph_cfg, morph_outdir, shape_only):
    fit = fresh_fit(morph_cfg, morph_outdir)
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
            "smW": mass.s * 1e3,
            "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
            "valid": bool(fit.minuit.valid)}


def main():
    print("PROMOTION GATE — luminosity vs 2-D ISR convolution cross-fit")
    print(f"  same eMELA-grid radiator; toggle ISRConfig.lumi; gate |Δ| < {GATE_MEV} MeV\n")
    print("[gen] 2-D-grid full morph templates ...", flush=True)
    d_2d = gen_full_templates(_cfg(False), "twoD")
    print("[gen] luminosity full morph templates ...", flush=True)
    d_lumi = gen_full_templates(_cfg(True), "lumi")

    truth_2d = fresh_fit(_cfg(False), d_2d).template("nominal")
    truth_lumi = fresh_fit(_cfg(True), d_lumi).template("nominal")

    worst = 0.0
    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"\n=== {mode} ===")
        print(f"{'truth':6s} {'morph':6s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        rows = [("lumi", "lumi", truth_lumi, _cfg(True), d_lumi, "NULL"),
                ("2D",   "lumi", truth_2d,   _cfg(True), d_lumi, "GATE"),
                ("lumi", "2D",   truth_lumi, _cfg(False), d_2d,  "GATE")]
        for tl, ml, truth, mcfg, mdir, kind in rows:
            r = crossfit(truth, mcfg, mdir, shape_only)
            flag = "" if r["valid"] else "  !INVALID"
            print(f"{tl:6s} {ml:6s} {r['dmW']:8.3f} {r['dgW']:8.3f} "
                  f"{r['smW']:8.2f} {r['rho']:+7.3f}   {kind}{flag}")
            if kind == "GATE":
                worst = max(worst, abs(r["dmW"]), abs(r["dgW"]))

    print(f"\nworst |Δ| over GATE cross-fits (m_W & Γ_W, both modes) = {worst:.3f} MeV")
    print(f"GATE ({GATE_MEV} MeV): {'PASS ✓' if worst < GATE_MEV else 'FAIL ✗'}")
    return 0 if worst < GATE_MEV else 1


if __name__ == "__main__":
    raise SystemExit(main())
