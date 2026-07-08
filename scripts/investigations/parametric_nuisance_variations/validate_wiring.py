"""Fast LL-template validation of the parametric-prior batch wiring:

  1. aem_isr profiled nuisance appears in the syst breakdown (tiny impact),
  2. per-point lumi-uncorr scaling (uncorr_i = uncorr_ref·√(L_ref/L_i)) is
     applied and shrinks the uncorr lumi for fewer-point scenarios,
  3. channel extrapolation scales the stat term by √(1/B) with lumi flat.

Uses fast LL+exp templates (gen.isr_nll=False) in a scratch dir; WWFit does not
run the freshness check, so production templates are untouched. The fit wiring
is identical LL vs NLL — this validates plumbing, not the production numbers.
"""
import os
import numpy as np

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.common.systematics import compute_syst_breakdown
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.process.ww.scenario_compare import _grouped_syst, _SYST_ROWS
from framework.process.ww.xsec_calculator.eft_xsec import BR_INCLUSIVE_MUNUQQ
from framework.common.fit_core import bec_var_dir

SCRATCH = "/tmp/ww_val"
NOM = os.path.join(SCRATCH, "nominal")
BEC = os.path.join(SCRATCH, "BEC")
# Point the fit's aux-template lookups (BEC) and nominal dir at the scratch set.
card.INPUT_DIRS = dict(card.INPUT_DIRS, nominal=NOM, BEC=BEC)


def gen_ll_templates():
    gen = WWGenerator.from_card(card)
    gen.isr_nll = False           # fast LL+exp analytic chain (~ms/template)
    gen.isr_emela_ll = False
    params = Parameters(card.PARAMETERS, scale_vars=[],
                        cross_terms=getattr(card, "CROSS_TERMS", ()))
    print(f"tags: {params.tags}")
    for tag in params.tags:
        gen.ensure_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                        outdir=NOM, force=True)
    for shift in (+card.INPUT_VAR["BEC"], -card.INPUT_VAR["BEC"]):
        sub = os.path.join(BEC, bec_var_dir(shift))
        for tag in params.tags:
            gen.ensure_scan(params.values(tag), mass_scale=80.0, width_scale=80.0,
                            outdir=sub, ecm_shift_MeV=shift, force=True)
    return gen


def build_fit(gen, *, scan_list=None, lumi_dict=None, total_lumi=None,
              scan_triplet=None, stat_scale=1.0):
    fit = WWFit(card, gen, input_dir=NOM, asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit._extra_stat_scale = stat_scale
    S = card.SCENARIO
    kw = dict(total_lumi=total_lumi if total_lumi is not None else S["total_lumi"],
              last_lumi=S["last_lumi"])
    if scan_list is not None:
        kw["scan_list"] = scan_list
        kw["lumi_dict"] = lumi_dict
    else:
        t = scan_triplet or (S["scan_min"], S["scan_max"], S["scan_step"])
        kw.update(scan_min=t[0], scan_max=t[1], scan_step=t[2])
    fit.init_scenario(**kw)
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.fit_parameters()
    return fit


def main():
    gen = gen_ll_templates()

    print("\n=== (1) baseline 7-point syst breakdown ===")
    fit = build_fit(gen)
    sb = compute_syst_breakdown(fit)
    for src in _SYST_ROWS + ["total"]:
        v = _grouped_syst(sb, "mass", src)
        print(f"  {src:10s} σ(m_W) = {v if v is None else round(v,4)} MeV")
    print(f"  lumi _perbin_scale (should be ~1): {np.round(fit._lumi_perbin_scale,3)}")

    print("\n=== (2) per-point lumi scaling across scenarios (same total lumi) ===")
    for name, t in [("7pt", (157.0, 163.0, 1.0)), ("3pt", (157.0, 163.0, 3.0))]:
        f = build_fit(gen, scan_triplet=t)
        eff = card.PRIORS["lumi"]["uncorr"] * f._lumi_perbin_scale
        print(f"  {name}: N={len(f.scenario)}  scale={np.round(f._lumi_perbin_scale,3)}  "
              f"eff uncorr={np.round(eff*1e4,3)}e-4")

    print("\n=== (3) channel extrapolation (stat × √B, lumi flat) ===")
    f_mu = build_fit(gen, stat_scale=1.0)
    f_in = build_fit(gen, stat_scale=BR_INCLUSIVE_MUNUQQ ** 0.5)
    sb_mu, sb_in = compute_syst_breakdown(f_mu), compute_syst_breakdown(f_in)
    for src in ["stat", "lumi", "aem_isr", "total"]:
        a = _grouped_syst(sb_mu, "mass", src)
        b = _grouped_syst(sb_in, "mass", src)
        ratio = (a / b) if (a and b) else float("nan")
        print(f"  {src:8s} μνqq={a if a is None else round(a,4)}  "
              f"incl={b if b is None else round(b,4)}  ratio={ratio:.3f}")
    print(f"  expected stat ratio √(1/B) = {(1/BR_INCLUSIVE_MUNUQQ)**0.5:.3f}")


if __name__ == "__main__":
    main()
