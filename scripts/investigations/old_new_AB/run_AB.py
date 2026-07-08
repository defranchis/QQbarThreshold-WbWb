"""Asimov A/B for the chain-routing changes (items 3 and 4).

Reads the three template sets built by ``build_templates.py`` under
``output_xsec/ww_AB_2026-05-29/`` and runs four-point Asimov cross-fits:

  Item 3 (δ_QCD routing): prod  vs legacy_dqcd
  Item 4 (full old chain): prod vs legacy_combined  (K_C × decay × δ_QCD)

For each pair (NEW vs OLD):
  truth=NEW + fit=NEW  → consistency  (m_W = SM exactly)
  truth=NEW + fit=OLD  → Δm_W bias of the OLD chain
  truth=OLD + fit=OLD  → consistency
  truth=OLD + fit=NEW  → symmetric bias

USE:  PYTHONPATH=. python3 scripts/investigations/old_new_AB/run_AB.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from cards import ww_default as card
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit


BASE = "output_xsec/ww_AB_2026-05-29"

CHAIN_OVERRIDES = {
    # Each entry monkey-patches NLO_CONFIG so the in-memory generator matches
    # the template fingerprint and freshness check passes.
    "prod":               {},
    "legacy_dqcd":        {"apply_delta_QCD": False},
    "legacy_kc_only":     {"include_coulomb": True},
    "legacy_decay_only":  {"decay_uses_full_born": False},
    "legacy_kc_decay":    {"include_coulomb": True,
                           "decay_uses_full_born": False},
    "legacy_combined":    {"apply_delta_QCD": False,
                           "include_coulomb": True,
                           "decay_uses_full_born": False},
}


def _build_fit(chain: str):
    """Construct a WWFit reading templates from ``BASE/<chain>/`` with the
    matching chain overrides monkey-patched onto the card so the freshness
    check passes.  Returns (fit, restorer) where ``restorer()`` rolls the
    card back to its original state."""
    overrides = CHAIN_OVERRIDES[chain]
    nlo = card.NLO_CONFIG
    orig = {k: nlo[k] for k in overrides}
    for k, v in overrides.items():
        nlo[k] = v
    orig_dirs = dict(card.INPUT_DIRS)
    card.INPUT_DIRS["nominal"] = os.path.join(BASE, chain, "nominal")
    card.INPUT_DIRS["BEC"]     = os.path.join(BASE, chain, "BEC")
    card.INPUT_DIRS["pseudo"]  = os.path.join(BASE, chain, "nominal")

    def restorer():
        for k, v in orig.items():
            nlo[k] = v
        card.INPUT_DIRS.update(orig_dirs)

    try:
        gen = WWGenerator.from_card(card)
        fit = WWFit(card, gen, asimov=True,
                    mass_scheme=getattr(card, "MASS_SCHEME", "OS"),
                    debug=False)
        sc = card.SCENARIO
        fit.init_scenario(
            scan_min=sc["scan_min"], scan_max=sc["scan_max"],
            scan_step=sc["scan_step"], total_lumi=sc["total_lumi"],
            last_lumi=sc["last_lumi"], add_last_ecm=False, same_evts=False,
        )
        # Note: do NOT call restorer() here — keep the card pinned to this
        # chain for the lifetime of the fit, so subsequent operations (Asimov
        # data fetch, template reads on update) keep using the right dirs.
        return fit, restorer
    except Exception:
        restorer()
        raise


def _fit_and_extract(fit, *, pseudo_data=None, label: str = ""):
    if pseudo_data is None:
        fit.fit_parameters()
    else:
        fit.update(pseudo_data=pseudo_data, init_minuit=True,
                   update_scenario=True)
    res = fit.fit_results(printout=False)
    by_name = dict(zip(fit.param_names, res))
    mW = by_name["mass"]
    gW = by_name["width"]
    asx = by_name.get("alphas", None)
    sm_mW = card.PARAMETERS["mass"]["nominal"]
    sm_gW = card.PARAMETERS["width"]["nominal"]
    print(f"  [{label}]")
    print(f"     m_W = {mW.n:.6f}  ±  {mW.s*1e3:.3f} MeV   "
          f"(Δ from SM {sm_mW} = {(mW.n-sm_mW)*1e3:+.3f} MeV)")
    print(f"     Γ_W = {gW.n:.6f}  ±  {gW.s*1e3:.3f} MeV   "
          f"(Δ from SM {sm_gW} = {(gW.n-sm_gW)*1e3:+.3f} MeV)")
    if asx is not None:
        print(f"     α_s = {asx.n:+.5f}  ±  {asx.s:.5f}")
    # Correlation
    try:
        cov = fit.fit_results_covariance()
        # fit_results_covariance may not exist; use fit object internals.
    except Exception:
        cov = None
    return mW, gW, asx


def _one_AB(new_tag: str, old_tag: str, title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print(f"    NEW = {new_tag}    OLD = {old_tag}")
    print("=" * 78)

    fit_new, restore_new = _build_fit(new_tag)
    pseudo_new = fit_new.template(fit_new.pseudodata_tag)
    print(f"\n  Pseudo-data (NEW) generated from {new_tag}.")
    restore_new()

    fit_old, restore_old = _build_fit(old_tag)
    pseudo_old = fit_old.template(fit_old.pseudodata_tag)
    print(f"  Pseudo-data (OLD) generated from {old_tag}.")
    restore_old()

    # Rebuild both fits cleanly with their own pseudodata.
    fit_new, restore_new = _build_fit(new_tag)
    print(f"\n-- truth=NEW, fit=NEW --")
    mW_nn, gW_nn, _ = _fit_and_extract(fit_new, pseudo_data=pseudo_new,
                                       label="truth=NEW, fit=NEW")
    rho_nn = _correlation(fit_new)
    print(f"\n-- truth=OLD, fit=NEW   (← bias the OLD chain carried) --")
    mW_on, gW_on, _ = _fit_and_extract(fit_new, pseudo_data=pseudo_old,
                                       label="truth=OLD, fit=NEW")
    restore_new()

    fit_old, restore_old = _build_fit(old_tag)
    print(f"\n-- truth=OLD, fit=OLD --")
    mW_oo, gW_oo, _ = _fit_and_extract(fit_old, pseudo_data=pseudo_old,
                                       label="truth=OLD, fit=OLD")
    rho_oo = _correlation(fit_old)
    print(f"\n-- truth=NEW, fit=OLD   (← symmetric) --")
    mW_no, gW_no, _ = _fit_and_extract(fit_old, pseudo_data=pseudo_new,
                                       label="truth=NEW, fit=OLD")
    restore_old()

    print("\n" + "-" * 78)
    print(f"  Δm_W shift (OLD-chain bias relative to NEW chain):")
    bias_truthOLD_fitNEW = (mW_on.n - mW_nn.n) * 1e3
    bias_truthNEW_fitOLD = (mW_no.n - mW_oo.n) * 1e3
    print(f"    truth=OLD, fit=NEW: Δm_W = {bias_truthOLD_fitNEW:+.3f} MeV")
    print(f"    truth=NEW, fit=OLD: Δm_W = {bias_truthNEW_fitOLD:+.3f} MeV")
    print(f"\n  ΔΓ_W shift:")
    biasG_truthOLD_fitNEW = (gW_on.n - gW_nn.n) * 1e3
    biasG_truthNEW_fitOLD = (gW_no.n - gW_oo.n) * 1e3
    print(f"    truth=OLD, fit=NEW: ΔΓ_W = {biasG_truthOLD_fitNEW:+.3f} MeV")
    print(f"    truth=NEW, fit=OLD: ΔΓ_W = {biasG_truthNEW_fitOLD:+.3f} MeV")
    print(f"\n  σ_mW (stat-only, MeV): NEW={mW_nn.s*1e3:.2f}  OLD={mW_oo.s*1e3:.2f}")
    print(f"  ρ(m_W, Γ_W):           NEW={rho_nn:+.3f}        OLD={rho_oo:+.3f}")
    return {
        "title": title,
        "bias_m_W_truthOLD_fitNEW_MeV": bias_truthOLD_fitNEW,
        "bias_m_W_truthNEW_fitOLD_MeV": bias_truthNEW_fitOLD,
        "bias_G_W_truthOLD_fitNEW_MeV": biasG_truthOLD_fitNEW,
        "bias_G_W_truthNEW_fitOLD_MeV": biasG_truthNEW_fitOLD,
        "sigma_m_W_NEW_MeV": mW_nn.s * 1e3,
        "sigma_m_W_OLD_MeV": mW_oo.s * 1e3,
        "rho_NEW": rho_nn,
        "rho_OLD": rho_oo,
    }


def _correlation(fit) -> float:
    """Pearson ρ(mass, width) from the Minuit covariance, or NaN if unavailable."""
    try:
        cov = fit.minuit.covariance
        i = list(fit.param_names).index("mass")
        j = list(fit.param_names).index("width")
        return float(cov[i, j] / (cov[i, i] ** 0.5 * cov[j, j] ** 0.5))
    except Exception:
        return float("nan")


def main():
    results = []
    results.append(_one_AB("prod", "legacy_dqcd",
                           "ITEM 3: δ_QCD routing  (in_br  vs  on_sigma)"))
    results.append(_one_AB("prod", "legacy_kc_only",
                           "decomp K_C alone (include_coulomb=True only)"))
    results.append(_one_AB("prod", "legacy_decay_only",
                           "decomp decay alone (decay_uses_full_born=False only)"))
    results.append(_one_AB("prod", "legacy_kc_decay",
                           "ITEM 4 (narrow): K_C + decay_full_born=False vs current"))
    results.append(_one_AB("prod", "legacy_combined",
                           "ITEM 4 (full): K_C + decay_full_born=False + δ_QCD on σ vs current"))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"\n  {r['title']}")
        print(f"    σ_mW (stat): NEW={r['sigma_m_W_NEW_MeV']:.2f}  OLD={r['sigma_m_W_OLD_MeV']:.2f}  MeV")
        print(f"    Δm_W bias (truth=OLD, fit=NEW): {r['bias_m_W_truthOLD_fitNEW_MeV']:+.3f} MeV")
        print(f"    Δm_W bias (truth=NEW, fit=OLD): {r['bias_m_W_truthNEW_fitOLD_MeV']:+.3f} MeV")
        print(f"    ΔΓ_W bias (truth=OLD, fit=NEW): {r['bias_G_W_truthOLD_fitNEW_MeV']:+.3f} MeV")
        print(f"    ρ(m,Γ):  NEW={r['rho_NEW']:+.3f}   OLD={r['rho_OLD']:+.3f}")


if __name__ == "__main__":
    main()
