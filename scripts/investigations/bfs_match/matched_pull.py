#!/usr/bin/env python3
"""Matched Asimov pull: the m_W / Γ_W shift induced by grafting the BFS
higher-order pieces (δ_NNLO + δ_QCD) and the eMELA NLL ISR onto the independent
MoCaNLO line shape (EXPLORATORY — see report sec:match).

Method (the cross-fit of crossfit_scheme.py): build the UNMATCHED-LL reference
morph; inject, as Asimov pseudodata, the nominal line shape of each *matched* /
*NLL* variant; read off the best-fit (m_W, Γ_W) shift = how many MeV that piece
moves the extracted POI if you neglect it in the templates.

Decomposed truths (all on the m_t=174.2 rich grid, off-shell BR):
  unmatched-LL (self)   sanity: bias ≈ 0
  δ_NNLO only  (LL)     production NNLO threshold block
  δ_QCD  only  (LL)     hadronic-decay QCD correction
  matched-LL           δ_NNLO + δ_QCD
  NLL ISR (unmatched)  LL→NLL initial-state radiation
  matched-NLL          everything

Both lumi modes (shape-only isolates the shape-induced bias; cov-lumi is the
production prior, where a normalisation shift also propagates the BR rate handle).

Usage: matched_pull.py [--skip-nll]
"""
from __future__ import annotations

import argparse
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
from framework.process.ww.indep.mocanlo_cards import SMInputs

RESULTS = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
           "mocanlo/grid_gen/results_mt174p2")
BASE = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
        "mocanlo/grid_gen/matched_pull_templates")
MT_GRID = 174.2


def _card_2poi():
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


BR_CONV = "off-shell"   # set in main() from --br-convention


def _gen(match=False, nnlo=True, dqcd=True, nll=False):
    return WWGeneratorMoCaNLO(
        results_dir=RESULTS, scheme_alpha="gf",
        isr_cfg=isr_beta.ISRConfig(scheme="LO_beta"),
        br_convention=BR_CONV,
        match_bfs=match, match_bfs_nnlo=nnlo, match_bfs_dqcd=dqcd,
        isr_nll=nll, sm=SMInputs(mt=MT_GRID))


def _build_fit(gen, sub):
    outdir = os.path.join(BASE, BR_CONV, sub)
    os.makedirs(outdir, exist_ok=True)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def _crossfit(ref_gen, truth_nom, shape_only):
    fit = _build_fit(ref_gen, "ref_unmatched_LL")     # fresh; ref morph cached on gen
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
        "dmW": (mass.n - tm) * 1e3, "dgW": (width.n - tw) * 1e3,
        "sigmW": mass.s * 1e3, "siggW": width.s * 1e3,
        "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
        "valid": bool(fit.minuit.valid),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nll", action="store_true")
    ap.add_argument("--br-convention", default="off-shell",
                    choices=["off-shell", "pdg-constant"])
    args = ap.parse_args(argv)

    global BR_CONV
    BR_CONV = args.br_convention
    print(f"[pull] BR convention: {BR_CONV}")

    ref_gen = _gen(match=False, nll=False)

    truths = [
        ("unmatched-LL (self)", _gen(match=False, nll=False)),
        ("delta_NNLO only LL",  _gen(match=True, nnlo=True, dqcd=False)),
        ("delta_QCD only LL",   _gen(match=True, nnlo=False, dqcd=True)),
        ("matched-LL",          _gen(match=True, nnlo=True, dqcd=True)),
    ]
    if not args.skip_nll:
        truths += [
            ("NLL ISR (unmatched)", _gen(match=False, nll=True)),
            ("matched-NLL",         _gen(match=True, nnlo=True, dqcd=True, nll=True)),
        ]

    print("[pull] generating truth nominal line shapes (m_t=174.2 grid, off-shell)...")
    truth_noms = []
    for name, g in truths:
        sub = "truth_" + name.split()[0].replace("-", "_")
        f = _build_fit(g, sub)
        truth_noms.append((name, f.template("nominal")))
        print(f"  truth ready: {name}")

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"\n=== matched Asimov pull, {mode} ===")
        print(f"{'truth injected':22s} {'dm_W':>7s} {'dG_W':>7s} "
              f"{'s(m_W)':>7s} {'s(G_W)':>7s} {'rho':>6s}   [MeV]")
        for name, truth in truth_noms:
            r = _crossfit(ref_gen, truth, shape_only)
            flag = "" if r["valid"] else " !INV"
            print(f"{name:22s} {r['dmW']:7.2f} {r['dgW']:7.2f} "
                  f"{r['sigmW']:7.2f} {r['siggW']:7.2f} {r['rho']:+6.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
