#!/usr/bin/env python3
"""Item 4 — G_F-consistency test: is the ~12 MeV hard-EW-scheme spread genuine?

The Task-1 cross-fit gf↔alphaz (≈12 MeV shape) compares the G_μ scheme (α_Gμ=
1/132.18) against the α(M_Z) scheme (α(M_Z)=1/128.23 in Recola).  Those two
schemes use DIFFERENT input α VALUES, which are not mutually SM-consistent, so
part of the 12 MeV may be a spurious input-value difference rather than a genuine
NNLO-EW renormalisation-prescription (truncation) effect.

This test removes the input-value difference: a dedicated grid runs the gf scheme
with G_F TUNED (G_F_adj=1.203449e-5) so the derived α_Gμ EQUALS the alphaz
scheme's hard α(M_Z) (Recola get_alpha_rcl verified α_Gf=0.0077983389 ≈ alphaz
α(M_Z)=0.0077983287, 0.0013 % residual).  With the α VALUE matched, the cross-fit
gf(G_F_adj)↔alphaz isolates the PURE renormalisation-prescription residual.

  * collapses to ~0  ⇒ the 12 MeV was largely the spurious input-α-value
                       difference ⇒ DEMOTE the EW-scheme systematic.
  * stays O(several MeV) ⇒ genuine renorm-prescription (NNLO-EW truncation)
                       ⇒ KEEP it as a real theory systematic.

Comparison is done in a SINGLE cross-fit direction (truth = gf*, morph = alphaz,
which already exists on prod EOS) so gf_prod↔alphaz and gf_adj↔alphaz differ ONLY
by the α value:
    Δ(gf_prod↔alphaz)  = α-value difference + renorm prescription
    Δ(gf_adj ↔alphaz)  =                       renorm prescription   ← the test
    Δ(gf_prod) − Δ(gf_adj) ≈ the (spurious) α-value piece.

gf_adj has ONLY the nominal varpoint (99-pt grid, cluster 12923949,
results_gf_consistency); it is used solely as the truth line shape.  Its morph-
variation template files are cloned from gf_prod (NEVER read — only
template("nominal") is used), satisfying FitCore's all-tags load.

Run (AFTER cluster 12923949 completes):
  source setup.sh && WW_INDEP_NJOBS=16 \
  PYTHONPATH=$PWD:$PYTHONPATH \
  python3 scripts/investigations/nll_isr/gf_consistency_crossfit.py
"""
from __future__ import annotations

import os
import shutil
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep import generator_mocanlo as gm
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from framework.process.ww.indep.partonic_grid import (
    DEFAULT_RESULTS_DIR, load_grids)
from scripts.investigations.nll_isr.scheme_alpha_scan import _card_2poi

BASE = "/tmp/ww_gf_consistency_crossfit/templates"
GF_ADJ_RESULTS = ("/eos/user/m/mdefranc/FCC/QQbar_threshold/"
                  "grid_gen/results_gf_consistency")

PROD_GRID = os.path.join(
    _REPO, "framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz")


def nll_cfg() -> isr_beta.ISRConfig:
    # Explicit alpha → identical ISR across all schemes (generator does NOT
    # re-couple ISR α to scheme_alpha when alpha is set; campaign design).
    return isr_beta.ISRConfig(
        nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
        emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ", emela_grid=PROD_GRID)


def _gen(scheme_alpha: str, results_dir: str = DEFAULT_RESULTS_DIR
         ) -> WWGeneratorMoCaNLO:
    return WWGeneratorMoCaNLO(scheme_alpha=scheme_alpha, isr_cfg=nll_cfg(),
                              results_dir=results_dir)


def gen_full_templates(scheme_alpha: str, key: str,
                       results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    outdir = os.path.join(BASE, key)
    gen = _gen(scheme_alpha, results_dir)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    os.makedirs(outdir, exist_ok=True)
    for tag in params.tags:
        gen.do_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                    mass_scheme="OS", outdir=outdir)
    return outdir


def gen_gf_adj_truthdir(base_dir: str) -> str:
    """gf_adj has only the nominal σ̂ grid.  Clone gf_prod's full template set
    (for the unused morph-variation tags FitCore reads at init), then OVERWRITE
    the nominal template with the REAL gf_adj nominal (POI-keyed filename → same
    name → clean overwrite).  Only template('nominal') is ever read off this."""
    outdir = os.path.join(BASE, "gf_adj")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    shutil.copytree(base_dir, outdir)                 # clone gf_prod full morph
    gen = _gen("gf", GF_ADJ_RESULTS)                  # gf_adj σ̂ (G_F-tuned grid)
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    # gf_adj has only the NOMINAL varpoint, so do_scan's factorized morph (needs
    # all varpoints) is unusable.  The nominal template needs no morph — it is the
    # single nominal varpoint line shape — so write it directly via
    # _varpoint_lineshape (mirrors do_scan's file format), overwriting the cloned
    # gf_prod nominal (POI-keyed filename → same name).  Only this nominal is read.
    ecm = gm._build_fine_grid()
    inside = (ecm >= gm.gridmod.ECM_MIN) & (ecm <= gm.gridmod.ECM_MAX)
    sigma = np.zeros_like(ecm)
    sigma[inside] = gen._varpoint_lineshape("nominal", ecm[inside])
    sigma *= gen._br_factor(gm.MW0, gm.GW0)           # =1 at nominal (off-shell)
    path = gen.file_name(params.values("nominal"), indir=outdir)
    with open(path, "w") as fh:
        fh.write("# generator: gf_adj nominal direct (G_F-consistency, no morph)\n")
        fh.write(f"# mass: {gm.MW0:.4f}  width: {gm.GW0:.4f}  units: pb\n")
        for e, s in zip(ecm, sigma):
            fh.write(f"{e:.4f}, {s:.8f}\n")
    return outdir


def fresh_fit(scheme_alpha: str, outdir: str,
              results_dir: str = DEFAULT_RESULTS_DIR) -> WWFit:
    gen = _gen(scheme_alpha, results_dir)
    c = _card_2poi()
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


def crossfit(truth_nom, morph_scheme, morph_outdir, shape_only):
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
    return {"bias_mW": (mass.n - tm) * 1e3, "bias_gW": (width.n - tw) * 1e3,
            "sig_mW": mass.s * 1e3,
            "rho": float(unc.correlation_matrix([mass, width])[0, 1]),
            "valid": bool(fit.minuit.valid)}


def _check_gf_adj_grid():
    """Verify the 99-point gf_adj nominal grid is present & healthy."""
    g = load_grids(results_dir=GF_ADJ_RESULTS, scheme_alpha="gf")
    print(f"gf_adj grid: {len(g)} (channel,varpoint) groups from {GF_ADJ_RESULTS}")
    nmiss = 0
    for ch in ("lnuqq", "qqqq", "mutau"):
        key = (ch, "nominal")
        if key not in g:
            print(f"  MISSING {key}"); nmiss += 1; continue
        gr = g[key]
        print(f"  {ch:6s} nominal: {len(gr.ecm)} ecm nodes "
              f"[{gr.ecm.min():.1f},{gr.ecm.max():.1f}], "
              f"<err_nlo>={gr.err_nlo.mean():.3f}")
        if len(gr.ecm) < 33:
            print(f"    !! only {len(gr.ecm)}/33 nodes — grid INCOMPLETE")
            nmiss += 1
    if nmiss:
        print(f"\n!! gf_adj grid incomplete ({nmiss} issues) — wait for cluster "
              f"12923949 to finish, then re-run.")
    return nmiss == 0


def main():
    print("Item 4 — G_F-consistency test: gf(G_F_adj)↔alphaz vs gf_prod↔alphaz")
    print(f"  gf_adj α_Gμ tuned to alphaz α(M_Z)=1/128.233 (G_F=1.203449e-5)\n")
    if not _check_gf_adj_grid():
        return 1

    # 1) Morph = alphaz full 2-POI templates (prod EOS); gf_prod full morph (for
    #    the gf_prod truth AND as the clone-base for gf_adj's unused tags).
    print("\n[gen] alphaz full morph (prod EOS) ...", flush=True)
    d_az = gen_full_templates("alphaz", "alphaz")
    print("[gen] gf_prod full morph (prod EOS) ...", flush=True)
    d_gf = gen_full_templates("gf", "gf_prod")
    print("[gen] gf_adj nominal (G_F-tuned grid) over cloned gf_prod morph ...",
          flush=True)
    d_gfadj = gen_gf_adj_truthdir(d_gf)

    truth_gf_prod = fresh_fit("gf", d_gf).template("nominal")
    truth_gf_adj = fresh_fit("gf", d_gfadj).template("nominal")

    for shape_only in (True, False):
        mode = "shape-only (lumi free)" if shape_only else "production cov-lumi"
        print(f"\n=== G_F-consistency cross-fit, {mode} ===")
        print(f"{'truth':9s} {'morph':7s} {'dm_W':>8s} {'dG_W':>8s} "
              f"{'s(m_W)':>8s} {'rho':>7s}   [MeV]")
        rp = crossfit(truth_gf_prod, "alphaz", d_az, shape_only)
        print(f"{'gf_prod':9s} {'alphaz':7s} {rp['bias_mW']:8.2f} "
              f"{rp['bias_gW']:8.2f} {rp['sig_mW']:8.2f} {rp['rho']:+7.3f}"
              f"{'' if rp['valid'] else '  !INVALID'}")
        ra = crossfit(truth_gf_adj, "alphaz", d_az, shape_only)
        print(f"{'gf_ADJ':9s} {'alphaz':7s} {ra['bias_mW']:8.2f} "
              f"{ra['bias_gW']:8.2f} {ra['sig_mW']:8.2f} {ra['rho']:+7.3f}"
              f"{'' if ra['valid'] else '  !INVALID'}")
        dval = rp['bias_mW'] - ra['bias_mW']
        print(f"  → renorm-prescription residual (gf_ADJ↔alphaz) = "
              f"{ra['bias_mW']:+.2f} MeV")
        print(f"  → α-value piece (gf_prod − gf_ADJ)            = "
              f"{dval:+.2f} MeV")
    print("\nINTERPRETATION: if |gf_ADJ↔alphaz| ≪ |gf_prod↔alphaz|, the EW-scheme "
          "spread\nis dominated by the (non-SM-consistent) input-α-value difference "
          "→ demote;\nif it stays several MeV, it is a genuine renorm-prescription "
          "(NNLO-EW) effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
