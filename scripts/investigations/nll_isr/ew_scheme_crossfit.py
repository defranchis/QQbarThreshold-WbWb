#!/usr/bin/env python3
"""Task 1 — hard EW-coupling-scheme dependence of σ̂: gf ↔ alphaz ↔ alpha0.

The independent chain runs the MoCaNLO NLO-EW σ̂ in the G_μ (``gf``) EW
renormalisation scheme, then convolves it with the eMELA NLL ISR ePDF, whose
QED coupling is α(M_Z) (ALPMZ).  A recurring concern is whether pairing a
G_μ-renormalised σ̂ (α_Gμ≈1/132) with an α(M_Z)-renormalised ISR (≈1/128.9) is a
hidden mismatch.  It is NOT — the two α's renormalise different objects (hard EW
vertices vs the collinear QED ISR logs) and live in different factors.  This
script *measures* the residual: the genuine hard-EW-scheme spread.

Method: σ̂ grids for the alternative hard-EW schemes ``alphaz`` (hard coupling at
α(M_Z) — the SAME value as the ISR) and ``alpha0`` (Thomson α(0)) already exist
on EOS (1980 inclusive points each, alongside gf).  Build NLL templates for each
scheme with the IDENTICAL production ISR (eMELA NLL, DELTA/ALPMZ, α(M_Z)=
1/128.943) — only σ̂ changes — and cross-fit each against gf:
    truth = NLL[scheme_A] line shape ;  morph = NLL[scheme_B] templates
giving Δm_W(A↔B), shape-only (lumi free) and cov-lumi (production prior).

A small SHAPE spread ⇒ the gf/alpmz coupling choice is harmless (concern
resolved).  A large spread ⇒ a real hard-EW-scheme systematic to carry.
Components are reported SEPARATELY (no quadrature), shape-only and cov-lumi, per
the standing convention.

Run:  source setup.sh && WW_INDEP_NJOBS=8 \
      PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/ew_scheme_crossfit.py
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
from scripts.investigations.nll_isr.scheme_alpha_scan import _card_2poi

BASE = "/tmp/ww_ew_scheme_crossfit/templates"

# Production NLL ISR grid (DELTA / ALPMZ / α(M_Z)=1/128.943) — identical for all
# three EW schemes, so the ONLY difference across template sets is σ̂'s EW scheme.
PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")

SCHEMES = ["gf", "alphaz", "alpha0"]


def nll_cfg() -> isr_beta.ISRConfig:
    return isr_beta.ISRConfig(
        nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
        emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ",
        emela_grid=PROD_GRID)


def _gen(scheme_alpha: str) -> WWGeneratorMoCaNLO:
    return WWGeneratorMoCaNLO(scheme_alpha=scheme_alpha, isr_cfg=nll_cfg())


def gen_templates(scheme_alpha: str) -> str:
    """Generate the 2-POI morph templates for one EW scheme into BASE/<scheme>."""
    outdir = os.path.join(BASE, scheme_alpha)
    gen = _gen(scheme_alpha)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def fresh_fit(scheme_alpha: str, outdir: str) -> WWFit:
    gen = _gen(scheme_alpha)
    c = _card_2poi()
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def crossfit(truth_nom, morph_scheme: str, morph_outdir: str, shape_only: bool):
    fit = fresh_fit(morph_scheme, morph_outdir)
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
    print("Task 1 — hard EW-coupling-scheme dependence of σ̂ (gf/alphaz/alpha0)")
    print(f"ISR (identical across schemes): eMELA NLL, DELTA/ALPMZ, "
          f"α(M_Z)=1/{1.0/isr_beta.ALPHA_MZ_EMELA:.3f}")
    print(f"grid: {PROD_GRID}\n")

    # 1) Generate the 2-POI morph templates once per EW scheme.
    dirs, truth = {}, {}
    for sa in SCHEMES:
        print(f"[gen] scheme_alpha={sa} ...", flush=True)
        dirs[sa] = gen_templates(sa)
        truth[sa] = fresh_fit(sa, dirs[sa]).template("nominal")

    # 2) Cross-fit each alternative against gf (+ a gf↔gf null self-consistency).
    pairs = [("gf", "gf"), ("alphaz", "gf"), ("alpha0", "gf")]
    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"\n=== EW-scheme cross-fit, {mode} ===")
        print(f"{'truth':7s} {'morph':7s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        for a, b in pairs:
            r = crossfit(truth[a], b, dirs[b], shape_only)
            flag = "" if r["valid"] else "  !INVALID"
            print(f"{a:7s} {b:7s} {r['bias_mW']:8.2f} {r['bias_gW']:8.2f} "
                  f"{r['sig_mW']:8.2f} {r['rho']:+7.3f}{flag}")

    print("\nINTERPRETATION: |Δm_W(gf↔alphaz)| is the hard-EW-renorm-scheme spread "
          "with the\nhard coupling moved to the ISR's α(M_Z) value — small ⇒ the "
          "gf/alpmz pairing\nis harmless. Quote shape-only and cov-lumi separately "
          "(no quadrature); cov-lumi\ncarries lumi-degenerate normalisation leakage "
          "(accounting left open).")


if __name__ == "__main__":
    main()
