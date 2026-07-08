#!/usr/bin/env python3
"""Flat-pedestal sensitivity on the BFS (production) chain — μνqq channel.

Companion to ``scripts/indep_mocanlo/dofit_indep.py --flatConst``: that runs the
study on the INDEPENDENT MoCaNLO line shape; this runs the same clean
2-POI cov-lumi Asimov fit (no nuisances, lumi in the covariance, pdg-constant
BR — NOT byte-identical to the theory-ladder config, see below) on the BFS
production line shape
(``WWGenerator.from_card`` — sigma_observed_munuqq, the single inclusive μνqq
channel the BFS chain fits). So the two flat-const numbers are directly
comparable, scope-for-scope (both μνqq, pdg-constant, cov-lumi).

The free, energy-INDEPENDENT additive σ pedestal c (same constant at every √s,
fully correlated across ECM, no prior) is added via ``FitCore.add_flat_const``.

Baseline vs the report theory-ladder μνqq number (σ(m_W)=1.31): same scenario
but NOT the identical fit config — the ladder fits lumi-CORR-ONLY
(lumi_uncorr=0) while this driver keeps the full per-point card lumi priors,
which fully explains the 1.34-vs-1.31 baseline difference (a ladder-replica on
identical templates reproduces 1.312/2.755/+0.663 — 2026-07-02 review).
"""
from __future__ import annotations

import argparse
import os
import sys
import types

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator

OUTDIR = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
          "mocanlo/grid_gen/fit_templates_bfs_flatconst")


def _card_2poi():
    """Clean 2-POI cov-lumi clone of the WW card — identical to the theory
    ladder's _ladder_card and dofit_indep's _card_2poi (no nuisances; lumi in
    the covariance)."""
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    S = card.SCENARIO
    ap.add_argument("--scan-min", type=float, default=S["scan_min"])
    ap.add_argument("--scan-max", type=float, default=S["scan_max"])
    ap.add_argument("--scan-step", type=float, default=S["scan_step"])
    args = ap.parse_args(argv)

    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    gen = WWGenerator.from_card(card)          # production BFS μνqq chain
    mass_scheme = getattr(card, "MASS_SCHEME", "OS")

    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[bfs flat-const] generator: {gen!r}")
    print(f"[bfs flat-const] generating {len(params.tags)} morph templates → {OUTDIR}")
    for tag in params.tags:
        path, _ = gen.ensure_scan(params.values(tag), mass_scale=80.0,
                                  width_scale=80.0, mass_scheme=mass_scheme,
                                  outdir=OUTDIR)
        print(f"  template {tag:18s} → {os.path.basename(path)}")

    fit = WWFit(c, gen, input_dir=OUTDIR, asimov=True, mass_scheme=mass_scheme)
    fit.init_scenario(scan_min=args.scan_min, scan_max=args.scan_max,
                      scan_step=args.scan_step, total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])

    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    print("\n" + "=" * 64)
    print(f"[bfs] BFS μνqq Asimov fit (2-POI, cov-lumi, {S['total_lumi']/1e6:.1f} ab⁻¹)")
    print(f"[bfs] scan {args.scan_min}–{args.scan_max} GeV step {args.scan_step}")
    print(f"[bfs] σ(m_W)  = {mass.s * 1e3:6.2f} MeV   (report theory-ladder ref: 1.31)")
    print(f"[bfs] σ(Γ_W)  = {width.s * 1e3:6.2f} MeV   (ref: 2.75)")
    print(f"[bfs] ρ(m,Γ)  = {rho:+.3f}              (ref: +0.66)")
    print("=" * 64)

    # + free energy-independent additive pedestal
    fit.add_flat_const()
    fit.fit_parameters()
    res2 = fit.fit_results(printout=False)
    i_cf = fit.param_names.index("cFlat")
    mass2, width2, cflat = res2[0], res2[1], res2[i_cf]
    rho2 = float(unc.correlation_matrix([mass2, width2])[0, 1])
    cm = unc.correlation_matrix([mass2, width2, cflat])
    mean_sigma = float(np.mean(fit.pseudo_data_scenario))   # pb
    print("\n" + "=" * 64)
    print("[bfs] + FREE energy-INDEPENDENT additive σ pedestal c "
          "(correlated across ECM, no prior)")
    print(f"[bfs] mean nominal σ over scan = {mean_sigma:.4f} pb "
          f"(cFlat scale = {fit._flat_const_scale:.4f} pb)")
    print("-" * 64)
    print(f"[bfs] sensitivity σ(c)  = {cflat.s * 1e3:7.3f} fb "
          f"= {cflat.s / mean_sigma * 100:6.3f}% of mean σ "
          f"(c = {cflat.n * 1e3:+.3f} fb)")
    print("-" * 64)
    print(f"[bfs] σ(m_W) : {mass.s * 1e3:6.2f} →{mass2.s * 1e3:6.2f} MeV"
          f"   (Δ = {(mass2.s - mass.s) * 1e3:+.2f} MeV, ×{mass2.s / mass.s:.2f})")
    print(f"[bfs] σ(Γ_W) : {width.s * 1e3:6.2f} →{width2.s * 1e3:6.2f} MeV"
          f"   (Δ = {(width2.s - width.s) * 1e3:+.2f} MeV, ×{width2.s / width.s:.2f})")
    print(f"[bfs] ρ(m,Γ) : {rho:+.3f} →{rho2:+.3f}")
    print(f"[bfs] corr(c, m_W) = {cm[2, 0]:+.3f}   corr(c, Γ_W) = {cm[2, 1]:+.3f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
