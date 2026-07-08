#!/usr/bin/env python3
"""Decompose the indep-chain LL→NLL m_W/Γ_W pull into its α-value and
NLL-kernel pieces.

The reported "NLL pull" (report sec:match-nll, ~−8.4 MeV shape) is measured by
injecting the NLL truth line shape and fitting it with the production LL
(α_Gμ) morph.  But that swaps TWO things at once:
  (1) the ISR coupling  α_Gμ (1/132.168) → α(M_Z) (1/128.943), and
  (2) the radiator      LL+exp (BETA)    → eMELA NLL (DELTA/ALPMZ).
Only (2) is the physics gain from promoting NLL.  This harness separates them
by holding the σ̂ grid fixed at the production gf scheme and varying ONLY the
ISR radiator (no new MC):

  row 1  α-value     truth = LL(gf, α(M_Z)),  morph = LL(gf, α_Gμ)
  row 2  NLL kernel  truth = NLL(gf, α(M_Z)), morph = LL(gf, α(M_Z))
  row 3  full LL→NLL truth = NLL(gf, α(M_Z)), morph = LL(gf, α_Gμ)   [≈ row1+row2]

Pull convention matches crossfit_scheme.py / the report: truth = variant,
morph = reference, bias = best-fit shift = the m_W/Γ_W you get wrong by modelling
the variant truth with the reference morph.  Reported in shape-only (clean
line-shape bias, the report's headline) and production cov-lumi.

The NLL leg uses the precomputed eMELA grid (built at α(M_Z)=1/128.943 by
scripts/investigations/isr_prewarm_lhapdf/build_production_grid.py), so the whole
run is seconds.  No σ̂ MC: all three configs reuse the production gf grid.

Run:  PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/crossfit_nll_decomp.py
"""
from __future__ import annotations

import os
import sys
import types

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO

BASE = "/tmp/ww_nll_decomp_templates"
GRID = os.path.join(_REPO, "framework", "process", "ww", "indep", "grids",
                    "emela_nll_delta_alpmz.npz")

#: ISR configs, all on the production gf σ̂ grid.  The decomposition holds the
#: hard scattering fixed and changes only the ISR radiator.
CONFIGS = {
    "LL_gmu":  isr_beta.ISRConfig(scheme="LO_beta"),                       # α_Gμ LL (production)
    "LL_amz":  isr_beta.ISRConfig(scheme="LO_beta", alpha=isr_beta.ALPHA_MZ_EMELA),
    "NLL_amz": isr_beta.ISRConfig(nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
                                  ew_scheme="alphaz", emela_fac_scheme="DELTA",
                                  emela_ren_scheme="ALPMZ", emela_grid=GRID),
}

ROWS = [
    ("alpha-value (LL: a_Gmu->a(MZ))", "LL_amz",  "LL_gmu"),
    ("NLL kernel  (a(MZ) fixed)",      "NLL_amz", "LL_amz"),
    ("full LL->NLL (~ row1 + row2)",   "NLL_amz", "LL_gmu"),
]


def _card_2poi():
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def _build_fit(name):
    """Generate the 2-POI morph templates for CONFIGS[name] (once) and return a
    fresh WWFit pointing at them."""
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=CONFIGS[name])
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    outdir = os.path.join(BASE, name)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def _crossfit(truth_nom, morph_name, shape_only):
    fit = _build_fit(morph_name)
    if shape_only:
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0
    fit.create_scenario(pseudodata=truth_nom)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    tm = fit.d_params["nominal"]["mass"]
    tw = fit.d_params["nominal"]["width"]
    return {
        "bias_mW": (mass.n - tm) * 1e3,
        "bias_gW": (width.n - tw) * 1e3,
        "sig_mW": mass.s * 1e3,
        "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
        "valid": bool(fit.minuit.valid),
    }


def main():
    print("alpha(M_Z) eMELA NLL =", isr_beta.ALPHA_MZ_EMELA,
          f"(1/{1.0/isr_beta.ALPHA_MZ_EMELA:.3f})")
    print("alpha_Gmu  LL        =", isr_beta.ALPHA_GMU,
          f"(1/{1.0/isr_beta.ALPHA_GMU:.3f})\n")

    # Build all three configs' nominal truth line shapes once.
    fits = {name: _build_fit(name) for name in CONFIGS}
    truths = {name: fits[name].template("nominal") for name in CONFIGS}

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"=== {mode} ===")
        print(f"{'decomposition step':32s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        for label, truth_name, morph_name in ROWS:
            r = _crossfit(truths[truth_name], morph_name, shape_only)
            flag = "" if r["valid"] else "  !INVALID"
            print(f"{label:32s} {r['bias_mW']:8.2f} {r['bias_gW']:8.2f} "
                  f"{r['sig_mW']:8.2f} {r['rho']:+7.3f}{flag}")
        print()


if __name__ == "__main__":
    main()
