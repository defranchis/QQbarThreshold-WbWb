#!/usr/bin/env python3
"""Independent-WW scheme cross-fit — theory-uncertainty BIAS on m_W / Γ_W.

Inject the REFERENCE-scheme Asimov truth line shape (Gμ EW scheme + LO_beta ISR =
the production default), fit it with a VARIANT scheme's morph templates, and read
off the best-fit (m_W, Γ_W) shift = the scheme-induced bias (the theory
uncertainty).  This is the cross-fit complement to ``dofit_indep.py`` (which
measures the *sensitivity*); the bias is what enters the theory-uncertainty
budget.

Variants:
  • ISR-scheme: LO_eta / LO_mixed / LO_collinear  — reuse the gf σ̂ grid (only the
    ISR convolution differs), so these run with no extra MC.
  • EW-scheme:  alpha0 / alphaz                    — need their own σ̂ grids
    (condor clusters 12816404 / 12816405); skipped automatically until present.

Reported in BOTH lumi modes: shape-only (lumi free → isolates the shape-induced
bias) and production cov-lumi (the realistic prior, where a normalisation shift
also propagates through the BR²(Γ_W) rate handle).
"""
from __future__ import annotations

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
from framework.process.ww.indep.partonic_grid import load_grids

BASE = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
        "mocanlo/grid_gen/crossfit_templates")

#: (label, scheme_alpha, isr_scheme).  First entry is the reference truth.
REFERENCE = ("ref: Gμ + LO_beta", "gf", "LO_beta")
VARIANTS = [
    ("ISR LO_eta",       "gf",     "LO_eta"),
    ("ISR LO_mixed",     "gf",     "LO_mixed"),
    ("ISR LO_collinear", "gf",     "LO_collinear"),
    ("EW α(0)",          "alpha0", "LO_beta"),
    ("EW α(M_Z)",        "alphaz", "LO_beta"),
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


def _has_grid(scheme_alpha):
    try:
        return bool(load_grids(scheme_alpha=scheme_alpha, lepton_cut=None))
    except Exception:
        return False


def _build_fit(scheme_alpha, isr_scheme):
    cfg = isr_beta.ISRConfig(scheme=isr_scheme)
    gen = WWGeneratorMoCaNLO(scheme_alpha=scheme_alpha, isr_cfg=cfg)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    outdir = os.path.join(BASE, f"{scheme_alpha}_{isr_scheme}")
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


def _crossfit(truth_nom, scheme_alpha, isr_scheme, shape_only):
    fit = _build_fit(scheme_alpha, isr_scheme)
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
        "sig_gW": width.s * 1e3,
        "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
        "valid": bool(fit.minuit.valid),
    }


def main():
    _, ref_sa, ref_isr = REFERENCE
    ref = _build_fit(ref_sa, ref_isr)
    truth_nom = ref.template("nominal")
    print(f"reference truth = {REFERENCE[0]}  (m_W=80.379, Γ_W=2.085)\n")

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"=== {mode} ===")
        print(f"{'variant':18s} {'Δm_W':>8s} {'ΔΓ_W':>8s} {'σ(m_W)':>8s} "
              f"{'σ(Γ_W)':>8s} {'ρ':>7s}   [MeV]")
        for vlabel, sa, isr in VARIANTS:
            if not _has_grid(sa):
                print(f"{vlabel:18s}   (σ̂ grid for scheme '{sa}' not present — skipped)")
                continue
            r = _crossfit(truth_nom, sa, isr, shape_only)
            flag = "" if r["valid"] else "  !INVALID"
            print(f"{vlabel:18s} {r['bias_mW']:8.2f} {r['bias_gW']:8.2f} "
                  f"{r['sig_mW']:8.2f} {r['sig_gW']:8.2f} {r['rho']:+7.3f}{flag}")
        print()


if __name__ == "__main__":
    main()
