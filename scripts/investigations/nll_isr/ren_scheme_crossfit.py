#!/usr/bin/env python3
"""Task 3 — ISR coupling-RENORMALISATION-scheme dependence at FIXED α(M_Z):
ALPMZ ↔ MSBAR.

Isolates the QED-coupling SCHEME (how α is renormalised) from the α VALUE.  Both
grids use fac=DELTA (so NO endpoint pathology, unlike Task 2's MS̄ fac) and the
SAME α(M_Z)=1/128.943; only the renormalisation scheme of the running coupling
differs (ALPMZ = α(M_Z) scheme, production; MSBAR = MS̄-renormalised α).  This is
DISTINCT from ALGMU/α(0), which change the α VALUE (the demoted 'ren' diagnostic,
dominated by the non-linear value sensitivity) — those are NOT used here.

Cross-fit: truth = NLL[DELTA / MSBAR-ren], morph = NLL[DELTA / ALPMZ] (production),
σ̂ identical (gf) → Δm_W is purely the coupling-renorm-scheme effect.  Shape-only
(lumi free) and cov-lumi, components separate (no quadrature).

Run:  source setup.sh && WW_INDEP_NJOBS=8 \
      PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/ren_scheme_crossfit.py
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
from framework.process.ww.indep import isr_emela_grid as eg
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from scripts.investigations.nll_isr.scheme_alpha_scan import _card_2poi

BASE = "/tmp/ww_ren_scheme_crossfit/templates"
GRIDDIR = "/tmp/ww_nll_scheme_scan/grids"
ALPHA = isr_beta.ALPHA_MZ_EMELA

ALPMZ_GRID = os.path.join(GRIDDIR, "emela_nll_delta_alpmz.npz")        # production
MSBARREN_GRID = os.path.join(GRIDDIR, "emela_nll_delta_msbarren_alpmz.npz")


def _ensure_grids():
    # omx_lo tracks eg.OMX_FLOOR (deep edge, 2026-07-03 endpoint fix).
    omx = eg.default_omx_knots(omx_hi=0.5, per_decade=16)
    q = eg.default_q_knots(75.0, 350.0, 16)
    if not os.path.exists(ALPMZ_GRID):
        eg.build_and_write(ALPMZ_GRID, fac_scheme="DELTA", ren_scheme="ALPMZ",
                           alpha=ALPHA, omx_knots=omx, q_knots=q,
                           pert_order="NLL", write_lhagrid1=False, verbose=True)
    if not os.path.exists(MSBARREN_GRID):
        print(f"[build] DELTA / MSBAR-ren / α(M_Z) NLL -> {MSBARREN_GRID}")
        eg.build_and_write(MSBARREN_GRID, fac_scheme="DELTA", ren_scheme="MSBAR",
                           alpha=ALPHA, omx_knots=omx, q_knots=q,
                           pert_order="NLL", write_lhagrid1=False, verbose=True)


def cfg(grid):
    # ren_scheme tag here is informational for isr_beta's eMELA init on the grid
    # path; the grid file itself bakes fac/ren/α into its meta.
    ren = "MSBAR" if "msbarren" in grid else "ALPMZ"
    return isr_beta.ISRConfig(nll=True, alpha=ALPHA, emela_fac_scheme="DELTA",
                              emela_ren_scheme=ren, emela_grid=grid)


def gen_templates(grid, key):
    outdir = os.path.join(BASE, key)
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg(grid))
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def fresh_fit(grid, outdir):
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cfg(grid))
    c = _card_2poi()
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def crossfit(truth_nom, morph_grid, morph_outdir, shape_only):
    fit = fresh_fit(morph_grid, morph_outdir)
    if shape_only:
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0
    fit.create_scenario(pseudodata=truth_nom)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    tm = fit.d_params["nominal"]["mass"]; tw = fit.d_params["nominal"]["width"]
    return {"bias_mW": (mass.n - tm) * 1e3, "bias_gW": (width.n - tw) * 1e3,
            "sig_mW": mass.s * 1e3,
            "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
            "valid": bool(fit.minuit.valid)}


def main():
    print("Task 3 — ISR coupling-renorm scheme ALPMZ↔MSBAR at FIXED α(M_Z)="
          f"1/{1.0/ALPHA:.3f}")
    _ensure_grids()
    d_alpmz = gen_templates(ALPMZ_GRID, "alpmz")
    d_msbar = gen_templates(MSBARREN_GRID, "msbarren")
    truth_alpmz = fresh_fit(ALPMZ_GRID, d_alpmz).template("nominal")
    truth_msbar = fresh_fit(MSBARREN_GRID, d_msbar).template("nominal")

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"\n=== ren-scheme cross-fit, {mode} ===")
        print(f"{'truth':9s} {'morph':9s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        r = crossfit(truth_alpmz, ALPMZ_GRID, d_alpmz, shape_only)   # null
        print(f"{'ALPMZ':9s} {'ALPMZ':9s} {r['bias_mW']:8.2f} {r['bias_gW']:8.2f} "
              f"{r['sig_mW']:8.2f} {r['rho']:+7.3f}")
        r = crossfit(truth_msbar, ALPMZ_GRID, d_alpmz, shape_only)   # genuine
        flag = "" if r["valid"] else "  !INVALID"
        print(f"{'MSBARren':9s} {'ALPMZ':9s} {r['bias_mW']:8.2f} {r['bias_gW']:8.2f} "
              f"{r['sig_mW']:8.2f} {r['rho']:+7.3f}{flag}")
    print("\nΔm_W(ALPMZ↔MSBAR) at fixed α = the genuine QED-coupling-renorm-SCHEME")
    print("effect (NOT the α-value sensitivity).  Shape-only + cov-lumi, separate.")


if __name__ == "__main__":
    main()
