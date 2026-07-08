"""Asimov A/B for the alpha_em_isr nuisance.

Quantifies the m_W bias from a +1σ shift of the ISR coupling
α(M_Z) → α(M_Z) + Δα, with Δα = PARAM_UNC["alpha_em_isr"] = 2.4e-7
(FCC-ee A_FB^μμ off-peak projection; Riembau).

Two scenarios, both on the NLL chain (eMELA DELTA+ALPMZ, isr_nll=True):
  - alphaMZ_central     : α_em_isr = 1/128.943      (production default)
  - alphaMZ_plus1sigma  : α_em_isr = 1/128.943 + Δα (+1σ shift)

2×2 cross-fit:
  - (Asimov=central, tmpl=central)   = closure baseline
  - (Asimov=plus1σ,  tmpl=central)   = bias from α(M_Z) uncertainty
  - (Asimov=central, tmpl=plus1σ)    = bias other way (consistency)
  - (Asimov=plus1σ,  tmpl=plus1σ)    = closure at shifted value

The off-diagonal Asimov=plus1σ × tmpl=central is the contribution of
the experimental Δα(M_Z) projection to the m_W systematic budget.

Templates written under output_xsec/ww_isr_alpha_AB/<scenario>/.
Reuses cross_fit_isr_scenarios.py's helper pattern.

Usage (on fcc-ironic-02 or -03 for parallel template gen speedup):
  source setup.sh
  python scripts/investigations/nll_isr/asimov_alpha_em_isr_AB.py
"""
from __future__ import annotations

import contextlib
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import uncertainties as unc

_devnull = os.open(os.devnull, os.O_WRONLY)
_saved   = os.dup(1)
os.dup2(_devnull, 1)
from cards import ww_default as _card
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit
os.dup2(_saved, 1)
os.close(_saved)
os.close(_devnull)


ALPHA_MZ_CENTRAL = 1.0 / 128.943              # PDG 2024
DELTA_ALPHA      = float(_card.PARAM_UNC["alpha_em_isr"])  # 2.4e-7

# Both scenarios are on the NLL chain so eMELA's DGLAP α(M_Z) is genuinely
# exercised.  isr_emela_ren_scheme stays ALPMZ (production default).
COMMON_NLO_OV = {
    "isr_scheme":            "2leg",
    "isr_nll":               True,
    "isr_emela_ll":          False,
    "isr_emela_ren_scheme":  "ALPMZ",
}

SCENARIOS = [
    ("alphaMZ_central",    ALPHA_MZ_CENTRAL,                "α(M_Z) = 1/128.943 (PDG central)"),
    ("alphaMZ_plus1sigma", ALPHA_MZ_CENTRAL + DELTA_ALPHA,  f"α(M_Z) + Δα ; Δα = {DELTA_ALPHA:.2e}"),
]

BASE_OUTDIR = "output_xsec/ww_isr_alpha_AB"


# --------------------------------------------------------------------------
# Card-patching helpers (mirror cross_fit_isr_scenarios.py)
# --------------------------------------------------------------------------

def _apply(card, alpha_value: float) -> None:
    """Patch NLO_CONFIG (NLL chain) + PARAM_INPUTS[alpha_em_isr] in place."""
    if "_orig_nlo_config" not in card.__dict__:
        card.__dict__["_orig_nlo_config"] = copy.deepcopy(card.NLO_CONFIG)
    if "_orig_param_inputs" not in card.__dict__:
        card.__dict__["_orig_param_inputs"] = copy.deepcopy(card.PARAM_INPUTS)
    card.NLO_CONFIG = copy.deepcopy(card.__dict__["_orig_nlo_config"])
    card.NLO_CONFIG.update(COMMON_NLO_OV)
    card.PARAM_INPUTS = copy.deepcopy(card.__dict__["_orig_param_inputs"])
    card.PARAM_INPUTS["alpha_em_isr"] = alpha_value


def _strip_template_dir_systs(card) -> None:
    if "_orig_systematics" not in card.__dict__:
        card.__dict__["_orig_systematics"] = copy.deepcopy(card.SYSTEMATICS)
    card.SYSTEMATICS = {
        k: v for k, v in card.__dict__["_orig_systematics"].items()
        if v.get("source", {}).get("kind") != "template_dir"
    }


def _set_input_dirs(card, scen_tag: str) -> None:
    scen_root = os.path.join(BASE_OUTDIR, scen_tag)
    card.INPUT_DIRS = {
        "nominal":    os.path.join(scen_root, "nominal"),
        "scale_vars": os.path.join(scen_root, "scale_vars"),
        "BEC":        os.path.join(scen_root, "BEC"),
        "pseudo":     os.path.join(scen_root, "pseudo"),
    }


# --------------------------------------------------------------------------
# Template generation
# --------------------------------------------------------------------------

def _generate_templates(card, *, outdir: str) -> None:
    gen = WWGenerator.from_card(card)
    params = Parameters(card.PARAMETERS, scale_vars=[])
    scales = getattr(card, "RENORM_SCALES",
                     {"mass": 80.0, "width": 80.0, "vars": []})
    mass_scheme = getattr(card, "MASS_SCHEME", "OS")
    for tag in params.tags:
        gen.do_scan(
            params.values(tag),
            mass_scale=scales["mass"],
            width_scale=scales["width"],
            mass_scheme=mass_scheme,
            outdir=outdir,
        )


# --------------------------------------------------------------------------
# Fit construction + cross-fit pseudodata loading
# --------------------------------------------------------------------------

def _build_fit(tmpl_tag: str, tmpl_alpha: float) -> WWFit:
    _apply(_card, tmpl_alpha)
    _strip_template_dir_systs(_card)
    _set_input_dirs(_card, tmpl_tag)
    gen = WWGenerator.from_card(_card)
    fit = WWFit(
        _card, gen,
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


def _load_pseudodata_smeared(fit: WWFit, asimov_tag: str):
    asimov_root = os.path.join(BASE_OUTDIR, asimov_tag, "nominal")
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


def _fit_and_extract(fit: WWFit) -> dict:
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


# --------------------------------------------------------------------------
# Main: template gen + 2x2 cross-fit
# --------------------------------------------------------------------------

def main() -> None:
    t_total = time.time()

    # ---- 1) Generate templates for each scenario ----
    for tag, alpha, desc in SCENARIOS:
        print(f"\n{'='*78}\n[{tag}] {desc}\n  α_em_isr = {alpha:.10e}\n{'='*78}",
              flush=True)
        _apply(_card, alpha)
        _strip_template_dir_systs(_card)
        _set_input_dirs(_card, tag)
        os.makedirs(_card.INPUT_DIRS["nominal"], exist_ok=True)

        t0 = time.time()
        log_path = os.path.join(BASE_OUTDIR, tag, "tmplgen.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            _generate_templates(_card, outdir=_card.INPUT_DIRS["nominal"])
        dt = time.time() - t0
        print(f"  [{tag}] templates done in {dt:.1f} s ({dt/60:.1f} min)", flush=True)

    # ---- 2) 2x2 cross-fit ----
    print(f"\n{'='*78}\n2×2 cross-fit matrix\n{'='*78}", flush=True)
    rows = []
    for tmpl_tag, tmpl_alpha, _ in SCENARIOS:
        for asim_tag, _, _ in SCENARIOS:
            print(f"\n[Asimov={asim_tag}  |  templates={tmpl_tag}]", flush=True)
            fit = _build_fit(tmpl_tag, tmpl_alpha)
            pseudo = _load_pseudodata_smeared(fit, asim_tag)
            fit.create_scenario(pseudodata=pseudo)
            res = _fit_and_extract(fit)
            res["asimov"] = asim_tag
            res["templates"] = tmpl_tag
            rows.append(res)
            print(f"  m_W = {res['mass_n']*1e3:.4f} MeV  ±{res['mass_s']*1e3:.4f}  "
                  f"Γ_W = {res['width_n']*1e3:.4f} MeV  ±{res['width_s']*1e3:.4f}  "
                  f"ρ = {res.get('rho', float('nan')):+.4f}", flush=True)

    # ---- 3) Report bias ----
    tags = [t for t, _, _ in SCENARIOS]
    print(f"\n{'='*78}\nm_W CROSS-FIT MATRIX [MeV]  (m_W^truth = 80384.0 MeV)\n{'='*78}")
    header = "  Asimov \\ tmpl  | " + " | ".join(f"{t:^20s}" for t in tags)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for asim in tags:
        cells = []
        for tmpl in tags:
            r = next(r for r in rows if r["asimov"] == asim and r["templates"] == tmpl)
            cells.append(f"{r['mass_n']*1e3:>10.4f}±{r['mass_s']*1e3:.3f}")
        print(f"  {asim:18s}| " + " | ".join(f"{c:^20s}" for c in cells))

    print(f"\n{'='*78}\nΔm_W BIAS (off-diagonal − row closure)\n{'='*78}")
    for asim in tags:
        closure = next(r for r in rows if r["asimov"] == asim and r["templates"] == asim)
        for tmpl in tags:
            if tmpl == asim:
                continue
            r = next(r for r in rows if r["asimov"] == asim and r["templates"] == tmpl)
            dmw = (r["mass_n"] - closure["mass_n"]) * 1e3
            dgw = (r["width_n"] - closure["width_n"]) * 1e3
            print(f"  Asimov={asim:22s} tmpl={tmpl:22s}  "
                  f"Δm_W = {dmw:+8.4f} MeV  ΔΓ_W = {dgw:+8.4f} MeV")

    # Headline: the m_W bias from using the α_em_isr central template
    # when the true α_em_isr is +1σ.
    r_bias = next(r for r in rows if r["asimov"] == "alphaMZ_plus1sigma"
                  and r["templates"] == "alphaMZ_central")
    r_clos = next(r for r in rows if r["asimov"] == "alphaMZ_central"
                  and r["templates"] == "alphaMZ_central")
    dmw_bias = (r_bias["mass_n"] - r_clos["mass_n"]) * 1e3
    print(f"\n{'='*78}\nHEADLINE\n{'='*78}")
    print(f"  α_em_isr +1σ shift  →  Δm_W = {dmw_bias:+.4f} MeV")
    print(f"  Δα = {DELTA_ALPHA:.2e}  (FCC-ee A_FB^μμ off-peak projection, Riembau)")
    print(f"  Reference: statistical target σ(m_W) = 0.25 MeV.")
    print(f"\nTotal wall time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
