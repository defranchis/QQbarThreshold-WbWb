#!/usr/bin/env python3
"""Task 3: strict apples-to-apples vs BFS by MATCHING THE CHANNEL SCOPE.

The headline independent fit uses the full pure-WW assembly
(12 lnuqq + 4 qqqq + 9 mutau), i.e. ~1/B(munuqq) ~ 7x more events than the
single BFS munuqq channel -> it is statistically tighter just from yield.  The
BFS chain instead fits the single mu- vm~ u d~ final state.  The indep ``lnuqq``
block IS exactly ``mu- vm~ u d~``, so running the indep fit with a single
``{lnuqq: 1.0}`` channel matches the BFS event yield and isolates whether the
sigma(m_W) difference is channel scope (yield) or a genuine line-shape
difference.

Runs the SAME 2-POI cov-lumi Asimov fit as dofit_indep.py for {full pure-WW,
single munuqq} x {pdg-constant cov-lumi, shape-only}.
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

OUTDIR = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
          "mocanlo/grid_gen/fit_templates_chanscope")


class ChanScopeGen(WWGeneratorMoCaNLO):
    """Generator whose pure-WW assembly is a fixed lnuqq multiplicity.

    weight=4  -> inclusive munuqq (mu nu + any hadronic): 2 charge x 2 up-types,
                 B ~ 2*BR(W->munu)*BR(W->had) ~ 0.143 = the BFS scope.
    weight=1  -> the single CKM-specific mu- vm~ u d~ final state (B ~ 0.036).
    """
    _lnuqq_weight: float = 4.0

    def _weights(self):
        return {"lnuqq": float(self._lnuqq_weight)}


def _card_2poi():
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def run_fit(gen, params, c, outdir, shape_only):
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    if shape_only:
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    return mass.s * 1e3, width.s * 1e3, rho


def main():
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    cfg = isr_beta.ISRConfig(scheme="LO_beta")

    # (label, generator-class, lnuqq weight, outdir-tag)
    scopes = [
        ("full pure-WW (12 lnuqq+4 qqqq+9 mutau)", WWGeneratorMoCaNLO, None, "full"),
        ("inclusive munuqq, B~0.143 (BFS scope)",  ChanScopeGen,       4.0,  "munuqq"),
        ("specific mu nu u d, B~0.036",            ChanScopeGen,       1.0,  "munuud"),
    ]
    rows = []
    for scope, Gen, w, tag in scopes:
        for mode, shape in (("pdg-constant cov-lumi", False),
                            ("shape-only", True)):
            gen = Gen(scheme_alpha="gf", lepton_cut=None, isr_cfg=cfg,
                      isr_nll=True, br_convention="pdg-constant")
            if w is not None:
                gen._lnuqq_weight = w
            sm, sw, rho = run_fit(gen, params, c, os.path.join(OUTDIR, tag), shape)
            rows.append((scope, mode, sm, sw, rho))
            print(f"  [{scope:42s} | {mode:22s}]  "
                  f"sigma(m_W)={sm:.2f}  sigma(G_W)={sw:.2f}  rho={rho:+.3f}")

    print("\n" + "=" * 78)
    print(f"{'scope':44s} {'mode':22s} {'s(mW)':>7s} {'s(GW)':>7s} {'rho':>7s}")
    print("-" * 78)
    for scope, mode, sm, sw, rho in rows:
        print(f"{scope:44s} {mode:22s} {sm:7.2f} {sw:7.2f} {rho:+7.3f}")
    print("=" * 78)
    print("BFS reference (report sec:fitresults): cov-lumi ~1.2, shape-only ~4.18 MeV")


if __name__ == "__main__":
    main()
