"""Run an Asimov fit for each ISR scenario and compare results.

Scenarios (all else equal — chain config inherited from cards.ww_default):
  1. baseline_LLexp  : isr_scheme=single_conv, LL+exp analytic β³-truncated (production default)
  2. emela_LL_2leg   : isr_scheme=2leg, isr_emela_ll=True  (eMELA DGLAP LL, no NLL bracket)
  3. emela_NLL_2leg  : isr_scheme=2leg, isr_nll=True       (full eMELA NLL — target chain)

For each: regen templates to scenario-specific dir, run Asimov fit, capture
m_W central + σ(m_W), σ(Γ_W), ρ(m_W, Γ_W).  Prints a comparison table.

Runs in-process (no subprocess) — each scenario monkey-patches card.NLO_CONFIG
and card.INPUT_DIRS before constructing the generator + fit.

Usage:
  source setup.sh
  python scripts/investigations/nll_isr/compare_isr_scenarios.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import uncertainties as unc

# Suppress eMELA banner (C-level stdout) before any framework imports
_devnull = os.open(os.devnull, os.O_WRONLY)
_saved   = os.dup(1)
os.dup2(_devnull, 1)
from cards import ww_default as _card
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit
from framework.process.ww.xsec_calculator.eft_xsec import BFSCorrections
os.dup2(_saved, 1)
os.close(_saved)
os.close(_devnull)

SCENARIOS = [
    ("baseline_LLexp", {
        "isr_scheme":   "single_conv",
        "isr_nll":      False,
        "isr_emela_ll": False,
    }, "LL+exp analytic (production default)"),
    ("emela_LL_2leg", {
        "isr_scheme":   "2leg",
        "isr_nll":      False,
        "isr_emela_ll": True,
    }, "eMELA DGLAP LL (isolates LL truncation)"),
    ("emela_NLL_2leg", {
        "isr_scheme":   "2leg",
        "isr_nll":      True,
        "isr_emela_ll": False,
    }, "eMELA NLL (production NLL target)"),
]

BASE_OUTDIR = "output_xsec/ww_isr_scenarios"


def _apply_scenario_overrides(card, overrides: dict) -> None:
    """Patch card.NLO_CONFIG in place; deep-copies the dict on first call."""
    if "_orig_nlo_config" not in card.__dict__:
        card.__dict__["_orig_nlo_config"] = copy.deepcopy(card.NLO_CONFIG)
    card.NLO_CONFIG = copy.deepcopy(card.__dict__["_orig_nlo_config"])
    card.NLO_CONFIG.update(overrides)


def _strip_template_dir_systematics(card) -> None:
    """Remove SYSTEMATICS entries that need on-disk aux templates (BEC).
    The comparison fit doesn't activate BEC/BES nuisances; WWFit.__init__
    eagerly reads template_dir sources regardless, so dropping them avoids
    generating BEC templates (~3× compute saving)."""
    if "_orig_systematics" not in card.__dict__:
        card.__dict__["_orig_systematics"] = copy.deepcopy(card.SYSTEMATICS)
    card.SYSTEMATICS = {
        k: v for k, v in card.__dict__["_orig_systematics"].items()
        if not (v.get("source", {}).get("kind") == "template_dir")
    }


def _generate_templates(card, *, outdir: str) -> None:
    """Nominal + parameter-variation templates only (no BEC/BES — the
    Asimov fit below doesn't activate those nuisances)."""
    generator = WWGenerator.from_card(card)
    params = Parameters(card.PARAMETERS, scale_vars=[])
    scales = getattr(card, "RENORM_SCALES",
                     {"mass": 80.0, "width": 80.0, "vars": []})
    mass_scheme = getattr(card, "MASS_SCHEME", "OS")

    for tag in params.tags:
        vals = params.values(tag)
        generator.do_scan(
            vals,
            mass_scale=scales["mass"],
            width_scale=scales["width"],
            mass_scheme=mass_scheme,
            outdir=outdir,
        )


def _run_asimov_fit(card, *, input_dir: str) -> dict:
    """Run a minimal Asimov fit and return (m_W, σ_mW, σ_ΓW, ρ_mW_ΓW, chain)."""
    generator = WWGenerator.from_card(card)
    fit = WWFit(
        card, generator,
        input_dir=input_dir,
        asimov=True,
        read_scale_vars=False,
        mass_scheme=getattr(card, "MASS_SCHEME", "OS"),
        debug=False,
    )
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
        add_last_ecm=False,
        same_evts=False,
    )
    fit.fit_parameters()
    params_w_cov = fit.fit_params_with_cov()
    for i, name in enumerate(fit.param_names):
        params_w_cov[i] = fit.value_from_param(params_w_cov[i], name)

    poi_names = list(fit.parameters.names)
    n_poi = len(poi_names)
    corr = unc.correlation_matrix(params_w_cov[:n_poi])

    out = {"chain": fit.template_metadata().get("chain", "?")}
    for i, name in enumerate(poi_names):
        v = params_w_cov[i]
        out[f"{name}_central"] = v.n
        out[f"{name}_sigma"]   = v.s
    # ρ between mass and width if both are POIs
    if "mass" in poi_names and "width" in poi_names:
        im = poi_names.index("mass")
        iw = poi_names.index("width")
        out["rho_mW_GW"] = float(corr[im, iw])
    return out


def main() -> None:
    results = []
    for tag, overrides, desc in SCENARIOS:
        print(f"\n{'='*78}\n[{tag}] {desc}\n  overrides: {overrides}\n{'='*78}",
              flush=True)
        # Per-scenario INPUT_DIRS so templates don't clash
        scen_root = os.path.join(BASE_OUTDIR, tag)
        _card.INPUT_DIRS = {
            "nominal":    os.path.join(scen_root, "nominal"),
            "scale_vars": os.path.join(scen_root, "scale_vars"),
            "BEC":        os.path.join(scen_root, "BEC"),
            "pseudo":     os.path.join(scen_root, "pseudo"),
        }
        _apply_scenario_overrides(_card, overrides)
        _strip_template_dir_systematics(_card)

        t0 = time.time()
        print(f"  [{tag}] generating templates → {scen_root}/", flush=True)
        _generate_templates(_card, outdir=_card.INPUT_DIRS["nominal"])
        dt_gen = time.time() - t0
        print(f"  [{tag}] templates done in {dt_gen:.1f} s", flush=True)

        t0 = time.time()
        # Capture fit's own stdout (verbose) — write to log file, keep
        # the summary print clean.
        log_path = os.path.join(scen_root, "fit.log")
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            res = _run_asimov_fit(_card, input_dir=_card.INPUT_DIRS["nominal"])
        dt_fit = time.time() - t0
        res["wall_gen_s"] = dt_gen
        res["wall_fit_s"] = dt_fit
        res["tag"] = tag
        res["desc"] = desc
        results.append(res)
        print(f"  [{tag}] fit done in {dt_fit:.1f} s — m_W={res['mass_central']*1e3:.3f} MeV ±{res['mass_sigma']*1e3:.2f}",
              flush=True)

    # ---- Comparison table ----
    print(f"\n{'='*78}\nISR scenario comparison (Asimov)\n{'='*78}")
    print(f"  {'scenario':22s}  {'m_W [GeV]':>12s}  {'σ_mW [MeV]':>11s}  "
          f"{'σ_ΓW [MeV]':>11s}  {'ρ(m,Γ)':>7s}  {'gen+fit [s]':>11s}")
    for r in results:
        gen_fit_s = r["wall_gen_s"] + r["wall_fit_s"]
        sigma_w_MeV = r.get("width_sigma", 0.0) * 1e3
        rho = r.get("rho_mW_GW", float("nan"))
        print(f"  {r['tag']:22s}  {r['mass_central']:>12.5f}  "
              f"{r['mass_sigma']*1e3:>11.3f}  {sigma_w_MeV:>11.3f}  "
              f"{rho:>+7.3f}  {gen_fit_s:>11.1f}")

    # ΔmW relative to baseline
    if results:
        ref = results[0]
        print(f"\n  Δ relative to {ref['tag']}:")
        print(f"  {'scenario':22s}  {'ΔmW [MeV]':>11s}  {'Δσ_mW [MeV]':>13s}  {'Δρ':>10s}")
        for r in results:
            dmw = (r["mass_central"] - ref["mass_central"]) * 1e3
            dsig = (r["mass_sigma"] - ref["mass_sigma"]) * 1e3
            drho = r.get("rho_mW_GW", 0) - ref.get("rho_mW_GW", 0)
            print(f"  {r['tag']:22s}  {dmw:>+11.3f}  {dsig:>+13.3f}  {drho:>+10.4f}")

    # Print chain labels for traceability
    print(f"\n  chain labels (from template metadata):")
    for r in results:
        print(f"    [{r['tag']:22s}] {r['chain']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
