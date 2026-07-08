"""Build template sets for the old-vs-new chain A/Bs (items 3 + 4) and the
2107.04444 follow-up (item 5).

Three template sets are produced under ``output_xsec/ww_AB_2026-05-29/``:

  * ``prod/``            — current production card as-is.  Used as the NEW
                            reference for both A/Bs *and* for the 2107.04444
                            follow-up.
  * ``legacy_dqcd/``     — δ_QCD on σ instead of in BR.  Differs from prod
                            ONLY in the δ_QCD routing.  OLD for item 3.
  * ``legacy_combined/`` — full pre-2026-05-26 chain: K_C on (FKM),
                            decay_uses_full_born=False, δ_QCD on σ.  OLD for
                            item 4 (combined K_C × decay × δ_QCD).

All other chain knobs (NLO loops, NNLO, anchor, ISR, m_t, M_H, M_Z, α_em,
α_em_isr) are inherited from the current card so the only difference within
each A/B is the targeted change.  Generates ``nominal/`` + BEC variation
subdirs that match the production layout.

USE:  PYTHONPATH=. python3 scripts/investigations/old_new_AB/build_templates.py
"""

from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from cards import ww_default as card
from framework.common.fit_core import bec_var_dir
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator, _build_fine_grid
from framework.process.ww.xsec_calculator.eft_xsec import BFSCorrections
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
from framework.process.ww.xsec_calculator.bfs_eft import delta_QCD_factor
from framework.process.ww.template_metadata import compose_header


BASE_OUT = "output_xsec/ww_AB_2026-05-29"
BEC_VARS_MeV = [10.0, 30.0]


def _emit_template(generator: WWGenerator, values: dict, *,
                   mass_scale: float, width_scale: float, mass_scheme: str,
                   outdir: str, ecm_shift_MeV: float = 0.0,
                   post_multiply_dqcd: bool = False) -> str:
    """Write one template using ``generator``'s chain config.

    With ``post_multiply_dqcd=True`` the σ output is multiplied by
    ``δ_QCD(α_s_eff) / δ_QCD(α_s_ref)`` after generation — together with a
    generator that already has ``apply_delta_QCD=False`` this synthesises
    the legacy "δ_QCD on σ" routing without touching the BR side.  At
    α_s_eff = α_s_ref the ratio is δ_QCD(α_s_ref) (the legacy chain
    normalised σ by the full δ_QCD at the reference, whereas the new chain
    normalises BR by the same factor → the ~4.6 % constant offset between
    the two chains that lumi absorbs).
    """
    mW = float(values["mass"])
    gammaW = float(values["width"])
    alpha_s_eff = generator.alpha_s + float(values.get("alphas", 0.0))

    ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3
    sigma_obs = sigma_observed_munuqq(
        ecm_grid, mW=mW, gammaW=gammaW,
        channel=generator.channel,
        include_coulomb=generator.include_coulomb,
        bfs=generator.bfs,
        include_NLO_hard_decay=generator.include_NLO_hard_decay,
        include_BFS_NNLO=generator.include_BFS_NNLO,
        apply_delta_QCD=generator.apply_delta_QCD,
        alpha_s=alpha_s_eff,
        alpha_s_ref=generator.alpha_s,
        br_convention=generator.br_convention,
        apply_whizard_anchor=generator.apply_whizard_anchor,
        whizard_anchor_source=generator.whizard_anchor_source,
        isr_scheme=generator.isr_scheme,
        isr_nll=generator.isr_nll,
        isr_emela_ll=generator.isr_emela_ll,
        isr_emela_pert_order=generator.isr_emela_pert_order,
        isr_emela_fac_scheme=generator.isr_emela_fac_scheme,
        isr_emela_ren_scheme=generator.isr_emela_ren_scheme,
        isr_scale_factor=generator.isr_scale_factor,
        alpha_em=generator.alpha_em,
        alpha_em_isr=generator.alpha_em_isr,
        coulomb_kc_safe=generator.coulomb_kc_safe,
        decay_uses_full_born=generator.decay_uses_full_born,
        m_t=generator.m_t, M_H=generator.M_H, MZ=generator.MZ,
    )

    if post_multiply_dqcd:
        sigma_obs = sigma_obs * delta_QCD_factor(alpha_s_eff)

    os.makedirs(outdir, exist_ok=True)
    path = generator.file_name(values, mass_scale=mass_scale,
                               width_scale=width_scale,
                               mass_scheme=mass_scheme, indir=outdir)
    header = compose_header(generator.template_fingerprint())
    with open(path, "w") as fh:
        fh.write(header)
        for ecm, sigma in zip(ecm_grid, sigma_obs):
            fh.write(f"{ecm:.4f}, {sigma:.8f}\n")
    return path


def _generate_set(generator: WWGenerator, params: Parameters, *,
                  outdir_root: str, post_multiply_dqcd: bool,
                  mass_scale: float, width_scale: float, mass_scheme: str):
    nominal_dir = os.path.join(outdir_root, "nominal")
    bec_dir     = os.path.join(outdir_root, "BEC")
    print(f"\n[ {outdir_root}/nominal ]")
    for tag in params.tags:
        vals = params.values(tag)
        path = _emit_template(generator, vals,
                              mass_scale=mass_scale, width_scale=width_scale,
                              mass_scheme=mass_scheme, outdir=nominal_dir,
                              post_multiply_dqcd=post_multiply_dqcd)
        print(f"  {tag:14s} → {path}")
    for var in BEC_VARS_MeV:
        for shift in (+var, -var):
            sub = os.path.join(bec_dir, bec_var_dir(shift))
            print(f"\n[ {sub} ]")
            for tag in params.tags:
                vals = params.values(tag)
                path = _emit_template(generator, vals,
                                      mass_scale=mass_scale, width_scale=width_scale,
                                      mass_scheme=mass_scheme, outdir=sub,
                                      ecm_shift_MeV=shift,
                                      post_multiply_dqcd=post_multiply_dqcd)
                print(f"  {tag:14s} → {path}")


def main():
    params = Parameters(card.PARAMETERS, scale_vars=[])
    scales = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0, "vars": []})
    mass_scale = scales["mass"]
    width_scale = scales["width"]
    mass_scheme = getattr(card, "MASS_SCHEME", "OS")

    t0 = time.time()

    # ---- PRODUCTION (NEW) ------------------------------------------------
    gen_prod = WWGenerator.from_card(card)
    print(f"[prod]            {gen_prod.describe()}")
    _generate_set(gen_prod, params,
                  outdir_root=os.path.join(BASE_OUT, "prod"),
                  post_multiply_dqcd=False,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    # ---- LEGACY: δ_QCD on σ (OLD for item 3) -----------------------------
    # Start from production, flip ONLY apply_delta_QCD → False, then
    # post-multiply σ by δ_QCD(α_s_eff).  Net chain = σ_BFS_chain × BR_PDG
    # × δ_QCD(α_s_eff) — the pre-1a6c339 routing.
    gen_dqcd = WWGenerator.from_card(card)
    gen_dqcd.apply_delta_QCD = False
    print(f"\n[legacy_dqcd]    {gen_dqcd.describe()}  +post-mult σ×δ_QCD(α_s)")
    _generate_set(gen_dqcd, params,
                  outdir_root=os.path.join(BASE_OUT, "legacy_dqcd"),
                  post_multiply_dqcd=True,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    # ---- LEGACY K_C alone (decomposition diagnostic) ---------------------
    gen_kc_only = WWGenerator.from_card(card)
    gen_kc_only.include_coulomb = True
    print(f"\n[legacy_kc_only]  {gen_kc_only.describe()}")
    _generate_set(gen_kc_only, params,
                  outdir_root=os.path.join(BASE_OUT, "legacy_kc_only"),
                  post_multiply_dqcd=False,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    # ---- LEGACY decay alone (decomposition diagnostic) -------------------
    gen_decay_only = WWGenerator.from_card(card)
    gen_decay_only.decay_uses_full_born = False
    print(f"\n[legacy_decay_only] {gen_decay_only.describe()}")
    _generate_set(gen_decay_only, params,
                  outdir_root=os.path.join(BASE_OUT, "legacy_decay_only"),
                  post_multiply_dqcd=False,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    # ---- LEGACY K_C × decay (item 4 narrow scope per K_C memo Scope note) -
    # K_C on + decay_uses_full_born=False, BUT with the current δ_QCD
    # routing — measures only the K_C + decay-substitution combined delta.
    gen_kc = WWGenerator.from_card(card)
    gen_kc.include_coulomb = True
    gen_kc.decay_uses_full_born = False
    print(f"\n[legacy_kc_decay] {gen_kc.describe()}")
    _generate_set(gen_kc, params,
                  outdir_root=os.path.join(BASE_OUT, "legacy_kc_decay"),
                  post_multiply_dqcd=False,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    # ---- LEGACY COMBINED: K_C on + decay_uses_full_born=False + δ_QCD on σ
    # (full pre-2026-05-26 "old chain → new chain" delta, bonus measurement)
    gen_comb = WWGenerator.from_card(card)
    gen_comb.apply_delta_QCD = False
    gen_comb.include_coulomb = True           # FKM K_C × σ
    gen_comb.decay_uses_full_born = False     # BFS δ_decay × σ̂^(0)
    print(f"\n[legacy_combined] {gen_comb.describe()}  +post-mult σ×δ_QCD(α_s)")
    _generate_set(gen_comb, params,
                  outdir_root=os.path.join(BASE_OUT, "legacy_combined"),
                  post_multiply_dqcd=True,
                  mass_scale=mass_scale, width_scale=width_scale,
                  mass_scheme=mass_scheme)

    print(f"\nDone in {time.time() - t0:.2f} s.  Output root: {BASE_OUT}/")


if __name__ == "__main__":
    main()
