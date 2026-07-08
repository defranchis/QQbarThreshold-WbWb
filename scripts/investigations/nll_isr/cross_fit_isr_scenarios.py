"""Cross-fit matrix: Asimov pseudodata from scheme A, templates from scheme B.

Reuses the per-scenario template directories produced by
``compare_isr_scenarios.py`` (must run that first).  No new template
generation needed.

For each (Asimov A, templates B) pair:
  1. Build a WWFit with NLO_CONFIG = B's overrides and templates loaded
     from B's output_xsec/ww_isr_scenarios/B/nominal/.
  2. Read A's pseudodata template off disk, smear it with the same BES,
     and feed it to ``fit.create_scenario(pseudodata=...)`` so the fit
     χ² compares (A's pseudodata σ) against (B's morphed templates).
  3. Extract fitted m_W, σ(m_W), σ(Γ_W), ρ.

The off-diagonal entries (A ≠ B) measure the m_W BIAS from using the
wrong ISR scheme — i.e. the ISR systematic on m_W.

Usage:
  source setup.sh
  python scripts/investigations/nll_isr/cross_fit_isr_scenarios.py
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import uncertainties as unc

_dev = os.open(os.devnull, os.O_WRONLY)
_sav = os.dup(1)
os.dup2(_dev, 1)
from cards import ww_default as _card
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit
os.dup2(_sav, 1); os.close(_sav); os.close(_dev)

BASE_OUTDIR = "output_xsec/ww_isr_scenarios"

SCENARIOS = [
    ("baseline_LLexp",  {"isr_scheme": "single_conv", "isr_nll": False, "isr_emela_ll": False}),
    ("emela_LL_2leg",   {"isr_scheme": "2leg",        "isr_nll": False, "isr_emela_ll": True}),
    ("emela_NLL_2leg",  {"isr_scheme": "2leg",        "isr_nll": True,  "isr_emela_ll": False}),
]


def _apply(card, overrides):
    if "_orig_nlo_config" not in card.__dict__:
        card.__dict__["_orig_nlo_config"] = copy.deepcopy(card.NLO_CONFIG)
    card.NLO_CONFIG = copy.deepcopy(card.__dict__["_orig_nlo_config"])
    card.NLO_CONFIG.update(overrides)


def _strip_template_dir_systs(card):
    if "_orig_systematics" not in card.__dict__:
        card.__dict__["_orig_systematics"] = copy.deepcopy(card.SYSTEMATICS)
    card.SYSTEMATICS = {
        k: v for k, v in card.__dict__["_orig_systematics"].items()
        if v.get("source", {}).get("kind") != "template_dir"
    }


def _set_input_dirs(card, scen_tag):
    scen_root = os.path.join(BASE_OUTDIR, scen_tag)
    card.INPUT_DIRS = {
        "nominal":    os.path.join(scen_root, "nominal"),
        "scale_vars": os.path.join(scen_root, "scale_vars"),
        "BEC":        os.path.join(scen_root, "BEC"),
        "pseudo":     os.path.join(scen_root, "pseudo"),
    }


def _build_fit(tmpl_tag, tmpl_overrides):
    """Build a WWFit using templates from tmpl_tag's output dir."""
    _apply(_card, tmpl_overrides)
    _strip_template_dir_systs(_card)
    _set_input_dirs(_card, tmpl_tag)
    generator = WWGenerator.from_card(_card)
    fit = WWFit(
        _card, generator,
        input_dir=_card.INPUT_DIRS["nominal"],
        asimov=True,
        read_scale_vars=False,
        mass_scheme=getattr(_card, "MASS_SCHEME", "OS"),
        debug=False,
    )
    fit.init_scenario(
        scan_min=_card.SCENARIO["scan_min"],
        scan_max=_card.SCENARIO["scan_max"],
        scan_step=_card.SCENARIO["scan_step"],
        total_lumi=_card.SCENARIO["total_lumi"],
        last_lumi=_card.SCENARIO["last_lumi"],
        add_last_ecm=False,
        same_evts=False,
    )
    return fit


def _load_pseudodata_smeared(fit, asimov_tag):
    """Read asimov_tag's pseudodata template, smear with fit's BES."""
    asimov_root = os.path.join(BASE_OUTDIR, asimov_tag, "nominal")
    # Use fit's generator's file_name pattern (filename depends only on
    # POI values + scales, which are scheme-independent).
    nominal_vals = fit.d_params["pseudodata"]
    scales = getattr(_card, "RENORM_SCALES",
                     {"mass": 80.0, "width": 80.0, "vars": []})
    path = fit.generator.file_name(
        nominal_vals,
        mass_scale=scales["mass"], width_scale=scales["width"],
        mass_scheme=getattr(_card, "MASS_SCHEME", "OS"),
        indir=asimov_root,
    )
    raw = fit.read_xsec(path)
    return fit.smear(raw)


def _fit_and_extract(fit):
    fit.fit_parameters()
    params_w_cov = fit.fit_params_with_cov()
    for i, name in enumerate(fit.param_names):
        params_w_cov[i] = fit.value_from_param(params_w_cov[i], name)
    poi = list(fit.parameters.names)
    n_poi = len(poi)
    corr = unc.correlation_matrix(params_w_cov[:n_poi])
    out = {}
    for i, n in enumerate(poi):
        out[f"{n}_n"] = params_w_cov[i].n
        out[f"{n}_s"] = params_w_cov[i].s
    if "mass" in poi and "width" in poi:
        im = poi.index("mass"); iw = poi.index("width")
        out["rho"] = float(corr[im, iw])
    return out


def main():
    rows = []
    for tmpl_tag, tmpl_ov in SCENARIOS:
        for asim_tag, asim_ov in SCENARIOS:
            print(f"\n[Asimov={asim_tag}  |  templates={tmpl_tag}]", flush=True)
            fit = _build_fit(tmpl_tag, tmpl_ov)
            pseudo = _load_pseudodata_smeared(fit, asim_tag)
            fit.create_scenario(pseudodata=pseudo)
            res = _fit_and_extract(fit)
            res["asimov"] = asim_tag
            res["templates"] = tmpl_tag
            rows.append(res)
            print(f"  m_W = {res['mass_n']*1e3:.3f} MeV  ±{res['mass_s']*1e3:.3f}  "
                  f"Γ_W = {res['width_n']*1e3:.3f} MeV  ±{res['width_s']*1e3:.3f}  "
                  f"ρ = {res.get('rho', float('nan')):+.4f}", flush=True)

    tags = [t for t, _ in SCENARIOS]

    # ---- m_W matrix ----
    print(f"\n{'='*80}")
    print("CROSS-FIT m_W [MeV] matrix  (m_W^truth=80384.0 MeV)")
    print(f"{'='*80}")
    header = "  Asimov \\ tmpl  | " + " | ".join(f"{t:^17s}" for t in tags)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asim in tags:
        cells = []
        for tmpl in tags:
            r = next(r for r in rows if r["asimov"] == asim and r["templates"] == tmpl)
            cells.append(f"{r['mass_n']*1e3:>9.3f}±{r['mass_s']*1e3:.2f}")
        print(f"  {asim:16s}| " + " | ".join(f"{c:^17s}" for c in cells))

    # ---- bias relative to closure (diagonal) ----
    print(f"\n{'='*80}")
    print("Δm_W from MIS-MATCHED templates (off-diagonal − closure of same row)")
    print(f"{'='*80}")
    for asim in tags:
        closure = next(r for r in rows if r["asimov"] == asim and r["templates"] == asim)
        for tmpl in tags:
            if tmpl == asim:
                continue
            r = next(r for r in rows if r["asimov"] == asim and r["templates"] == tmpl)
            dmw  = (r["mass_n"]  - closure["mass_n"])  * 1e3
            dgw  = (r["width_n"] - closure["width_n"]) * 1e3
            print(f"  Asimov={asim:18s} tmpl={tmpl:18s}  "
                  f"Δm_W = {dmw:+8.3f} MeV  ΔΓ_W = {dgw:+8.3f} MeV")

    print("\nDone.")


if __name__ == "__main__":
    main()
