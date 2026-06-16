#!/usr/bin/env python3
"""Step-1 DECISIVE test: alpha-stability of the NLL-kernel pull, per scheme.

The NLL line-shape m_W pull is alpha-hypersensitive (it swings several MeV for a
sub-percent change in the alpha input).  The hypothesis: that ill-conditioning is
the symptom of an UNMATCHED O(alpha) finite collinear term between the MoCaNLO
NLO-EW grid (CS dipoles -> MSbar IS factorisation) and the eMELA ISR PDF
(convolved in DELTA).  An unmatched term is proportional to alpha, so it produces
exactly a linear-in-alpha swing; running eMELA in the grid's OWN scheme removes
the term and COLLAPSES the swing.

So: for each scheme in {DELTA, MSBAR}, measure the pure NLL-KERNEL pull
    kernel_pull(scheme, alpha) = crossfit( truth = NLL(scheme, alpha),
                                           morph = LL(alpha) )          [shape-only]
at alpha = {1/128.943 (PDG), 1/128.232 (MoCaNLO)} and report the per-scheme swing
    swing(scheme) = kernel_pull(scheme, moca) - kernel_pull(scheme, alpmz).

Using LL(alpha) as the morph (SAME alpha as the NLL truth) removes the trivial
alpha-value/LL piece, so the swing is purely the NLL kernel's O(alpha) content.

VERDICT: the scheme with the SMALLER |swing| is the grid's matching scheme (the
unmatched term has collapsed).  If NEITHER collapses, the alpha-instability lives
elsewhere (ren-scheme / alpha-input / the LL-vs-NLL baseline) -- also decisive.

Grids built by build_scheme_scan_grids.py (run it first).

Run:  PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/scheme_alpha_scan.py
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

BASE = "/tmp/ww_nll_scheme_scan/templates"
GRIDDIR = "/tmp/ww_nll_scheme_scan/grids"

ALPHAS = {"alpmz": isr_beta.ALPHA_MZ_EMELA,   # 1/128.943 (PDG)
          "moca":  isr_beta.ALPHA_MZ}         # 1/128.232 (MoCaNLO)
SCHEMES = ["DELTA", "MSBAR"]


def grid_path(scheme, atag):
    return os.path.join(GRIDDIR, f"emela_nll_{scheme.lower()}_{atag}.npz")


def ll_cfg(atag):
    return isr_beta.ISRConfig(scheme="LO_beta", alpha=ALPHAS[atag])


def nll_cfg(scheme, atag):
    return isr_beta.ISRConfig(nll=True, alpha=ALPHAS[atag],
                              emela_fac_scheme=scheme, emela_ren_scheme="ALPMZ",
                              emela_grid=grid_path(scheme, atag))


def _card_2poi():
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def _gen_templates(cfg, key):
    """Generate the 2-POI morph templates for `cfg` once into BASE/key."""
    outdir = os.path.join(BASE, key)
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def _fresh_fit(cfg, outdir):
    """Construct a WWFit on already-generated templates (no do_scan)."""
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg)
    c = _card_2poi()
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def _crossfit(truth_nom, morph_cfg, morph_outdir, shape_only):
    fit = _fresh_fit(morph_cfg, morph_outdir)
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
    for atag, a in ALPHAS.items():
        print(f"alpha[{atag}] = {a:.8e}  (1/{1.0/a:.3f})")
    print()

    # 1) Generate templates once per config: 2 LL morphs + 4 NLL truths.
    ll_dirs = {atag: _gen_templates(ll_cfg(atag), f"LL_{atag}") for atag in ALPHAS}
    nll_truth = {}
    for scheme in SCHEMES:
        for atag in ALPHAS:
            key = f"NLL_{scheme}_{atag}"
            d = _gen_templates(nll_cfg(scheme, atag), key)
            nll_truth[(scheme, atag)] = _fresh_fit(nll_cfg(scheme, atag),
                                                   d).template("nominal")

    # 2) NLL-kernel pull = crossfit(truth=NLL(scheme,alpha), morph=LL(alpha)).
    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"=== NLL-kernel pull, {mode} ===")
        print(f"{'scheme':7s} {'alpha':6s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        pulls = {}
        for scheme in SCHEMES:
            for atag in ALPHAS:
                r = _crossfit(nll_truth[(scheme, atag)], ll_cfg(atag),
                              ll_dirs[atag], shape_only)
                pulls[(scheme, atag)] = r["bias_mW"]
                flag = "" if r["valid"] else "  !INVALID"
                print(f"{scheme:7s} {atag:6s} {r['bias_mW']:8.2f} "
                      f"{r['bias_gW']:8.2f} {r['sig_mW']:8.2f} "
                      f"{r['rho']:+7.3f}{flag}")
        print(f"\n  {'per-scheme alpha-swing  swing = pull(moca) - pull(alpmz)':s}")
        for scheme in SCHEMES:
            sw = pulls[(scheme, "moca")] - pulls[(scheme, "alpmz")]
            print(f"    {scheme:7s}  swing = {sw:+7.2f} MeV   "
                  f"(alpmz {pulls[(scheme,'alpmz')]:+.2f} -> "
                  f"moca {pulls[(scheme,'moca')]:+.2f})")
        print()
    print("VERDICT: smaller |swing| = grid's matching scheme "
          "(unmatched O(alpha) term collapsed).")


if __name__ == "__main__":
    main()
