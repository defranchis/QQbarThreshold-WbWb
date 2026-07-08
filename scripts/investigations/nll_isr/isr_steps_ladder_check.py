#!/usr/bin/env python3
"""Item 1, decisive check — do the σ̂-knot STEPS bias a DIFFERENT-radiator
(LL↔NLL) cross-fit, i.e. the kind of comparison a theory ladder makes?

Part C of isr_steps_investigation.py showed the EW cross-fit (SAME radiator in
truth & morph) is steps-immune.  The worst case for the steps is a cross-fit
across DIFFERENT radiators — exactly the LL-vs-NLL ratio that shows the steps in
the plot.  This builds that cross-fit:
    truth = gf σ̂ ⊗ LL+exp (BETA)   morph = gf σ̂ ⊗ NLL (production)
giving Δm_W(LL↔NLL), the independent-chain ISR-truncation step.  We run it at the
production σ̂ denoising (s=N) and at a tighter and a looser one, and also with no
denoising (interpolating, s=0).  If Δm_W is stable, the 0.25-GeV σ̂ steps do NOT
bias even the worst-case (radiator-changing) fit — they are a high-frequency
ripple the fit cannot absorb into a smooth m_W shift.  If it moves, the steps
contribute to the ISR-truncation number and must be quantified.

Run:  source setup.sh && WW_INDEP_NJOBS=16 \
      PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/nll_isr/isr_steps_ladder_check.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_dn = os.open(os.devnull, os.O_WRONLY); _sv = os.dup(1); os.dup2(_dn, 1)
import uncertainties as unc  # noqa: E402
from cards import ww_default as card  # noqa: E402
from framework.common.parameters import Parameters  # noqa: E402
from framework.process.ww.fit import WWFit  # noqa: E402
from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO  # noqa: E402
from framework.process.ww.indep.partonic_grid import load_grids  # noqa: E402
from scripts.investigations.nll_isr.scheme_alpha_scan import _card_2poi  # noqa: E402
os.dup2(_sv, 1); os.close(_sv); os.close(_dn)

ALPHA = isr_beta.ALPHA_MZ_EMELA
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")
BASE = "/tmp/ww_isr_steps_ladder"


def cfg_LL():
    return isr_beta.ISRConfig(scheme="LO_beta", alpha=ALPHA)


def cfg_NLL():
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID)


def _gen(cfg, smooth):
    return WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg, smooth=smooth)


def _templates(cfg, key, smooth):
    outdir = os.path.join(BASE, key)
    gen = _gen(cfg, smooth)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def _fit(cfg, outdir, smooth):
    gen = _gen(cfg, smooth)
    c = _card_2poi()
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def _crossfit(truth_nom, morph_cfg, morph_dir, smooth, shape_only=True):
    fit = _fit(morph_cfg, morph_dir, smooth)
    if shape_only:
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0
    fit.create_scenario(pseudodata=truth_nom)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass = res[0]
    return (mass.n - fit.d_params["nominal"]["mass"]) * 1e3


def main():
    print("LL↔NLL cross-fit (DIFFERENT radiators — worst case for σ̂ steps)")
    print("truth = gf σ̂ ⊗ LL+exp ;  morph = gf σ̂ ⊗ NLL  (shape-only Δm_W)\n")
    N = len(load_grids(scheme_alpha="gf")[("lnuqq", "nominal")].ecm)
    variants = [("interp  (s=0)", 0.0), ("tight   (s=0.5N)", 0.5 * N),
                ("default (s=N) PROD", None), ("loose   (s=2N)", 2.0 * N)]
    vals = []
    for lab, sm in variants:
        d_ll = _templates(cfg_LL(), f"ll_{lab.split()[0]}", sm)
        d_nl = _templates(cfg_NLL(), f"nll_{lab.split()[0]}", sm)
        truth_ll = _fit(cfg_LL(), d_ll, sm).template("nominal")
        dmW = _crossfit(truth_ll, cfg_NLL(), d_nl, sm)
        vals.append(dmW)
        print(f"  {lab:22s}: Δm_W(LL↔NLL) = {dmW:+.3f} MeV")
    print(f"\n  spread over σ̂ denoising = {max(vals)-min(vals):.2f} MeV")
    print("  → if ≪1 MeV, the 0.25-GeV σ̂ steps do NOT bias even the worst-case")
    print("    (radiator-changing) cross-fit: the ripple is high-frequency and")
    print("    not degenerate with a smooth m_W shift, so the fit cannot absorb it.")


if __name__ == "__main__":
    main()
