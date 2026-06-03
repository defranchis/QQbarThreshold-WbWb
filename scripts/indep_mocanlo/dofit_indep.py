#!/usr/bin/env python3
"""Independent (BFS-free) WW Asimov fit — σ(m_W) / σ(Γ_W) / ρ.

Generates the morph template set with :class:`WWGeneratorMoCaNLO` (MoCaNLO
NLO-EW partonic σ̂ ⊗ decoupled beta-scheme ISR — no BFS code or numbers) and
runs the SAME 2-POI cov-lumi Asimov fit as the BFS theory ladder, so the
resulting sensitivities are directly comparable to the BFS Asimov headline
(σ_mW ≈ 1.2 MeV) while being computed from a fully independent line shape.

This is the STEP-4 entry point of the independent cross-check: it answers
"what σ(m_W)/σ(Γ_W)/ρ does the independent calculation give?", not
"does it match BFS number-for-number".

Usage:
  dofit_indep.py [--scheme gf] [--lepton-cut 0.95] [--isr-scheme single_conv]
                 [--keep] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import types

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO

DEFAULT_OUTDIR = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
                  "mocanlo/grid_gen/fit_templates")


def _card_2poi():
    """SimpleNamespace clone of the WW card, reduced to a 2-POI cov-lumi fit
    (no nuisances/constraints; lumi handled in the covariance) — identical
    scenario to the BFS theory-ladder Asimov column."""
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def _generate_templates(gen, params, outdir):
    os.makedirs(outdir, exist_ok=True)
    mass_scale, width_scale = 80.0, 80.0       # no-op for WW (scaleM/scaleW)
    for tag in params.tags:
        path = gen.do_scan(params.values(tag), mass_scale=mass_scale,
                           width_scale=width_scale, mass_scheme="OS",
                           outdir=outdir)
        print(f"  template {tag:18s} → {os.path.basename(path)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="gf",
                    choices=["gf", "alpha0", "alphaz", "alphamsbar"],
                    help="EW input scheme (the σ̂ grid must exist for it)")
    ap.add_argument("--lepton-cut", type=float, default=None,
                    help="fiducial |cosθ_l|<COS grid (e.g. 0.95); default "
                         "inclusive pure-WW")
    ap.add_argument("--isr-scheme", default="LO_beta",
                    choices=list(isr_beta.ISR_SCHEMES))
    ap.add_argument("--mu-F-factor", type=float, default=1.0,
                    help="ISR factorisation scale μ_F / √s")
    ap.add_argument("--shapeOnly", action="store_true",
                    help="open the lumi prior wide (overall normalisation "
                         "unconstrained) so m_W/Γ_W come from the line-shape "
                         "SHAPE only — removes the off-shell BR²(Γ_W) rate "
                         "handle, giving the apples-to-apples comparison with "
                         "the BFS pdg-constant ρ sign.")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    cfg = isr_beta.ISRConfig(scheme=args.isr_scheme, mu_F_factor=args.mu_F_factor)
    gen = WWGeneratorMoCaNLO(scheme_alpha=args.scheme, lepton_cut=args.lepton_cut,
                             isr_cfg=cfg)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)

    label = ("inclusive pure-WW" if args.lepton_cut is None
             else f"fiducial |cosθ|<{args.lepton_cut}")
    print(f"[indep fit] generator: MoCaNLO {args.scheme}, {label}, "
          f"ISR {args.isr_scheme} (μ_F/√s={args.mu_F_factor})")
    print(f"[indep fit] generating {len(params.tags)} morph templates → {args.outdir}")
    _generate_templates(gen, params, args.outdir)

    fit = WWFit(c, gen, input_dir=args.outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])

    if args.shapeOnly:                      # cov mode: open the normalisation
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0

    fit.fit_parameters()
    res = fit.fit_results(printout=False)   # [mass, width] ufloat, GeV

    mass, width = res[0], res[1]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    lumi_lbl = "shape-only (lumi free)" if args.shapeOnly else "cov-lumi"
    print("\n" + "=" * 64)
    print(f"[indep] INDEPENDENT WW Asimov fit (2-POI, {lumi_lbl}, {S['total_lumi']/1e6:.1f} ab⁻¹)")
    print(f"[indep] scan {S['scan_min']}–{S['scan_max']} GeV step {S['scan_step']}")
    print(f"[indep] σ(m_W)  = {mass.s * 1e3:6.2f} MeV")
    print(f"[indep] σ(Γ_W)  = {width.s * 1e3:6.2f} MeV")
    print(f"[indep] ρ(m,Γ)  = {rho:+.3f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
