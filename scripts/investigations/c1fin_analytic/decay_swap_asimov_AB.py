"""Asimov A/B: m_W central-value shift induced by the BFS decay substitution.

Two template sets, identical chain except for ``decay_uses_full_born``:
  * nominal_decay_OFF/  — δ_decay × σ̂_LR^(0) (historical chain)
  * nominal_decay_ON/   — δ_decay × σ_Born   (BFS recipe, production default)

For each, run the standard FCC-ee threshold-scan Asimov fit twice:
  (a) own templates + own data — sanity check (should give SM m_W exactly)
  (b) own templates + the OTHER set's pseudodata — quantifies the m_W bias

The four numbers spell out the A/B:

  Truth=ON, fit with ON  → m_W = SM           (consistency)
  Truth=ON, fit with OFF → m_W = SM + Δ_bias  (← interesting: "what we'd get
                                                if the world is BFS-correct
                                                but we fit with the broken
                                                pre-substitution chain")

  Truth=OFF, fit with OFF → m_W = SM          (consistency)
  Truth=OFF, fit with ON  → m_W = SM − Δ_bias (symmetric)

USE: PYTHONPATH=. python3 scripts/investigations/c1fin_analytic/decay_swap_asimov_AB.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from cards import ww_default as card
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit


# The card has decay_uses_full_born=True (production). For the OFF generator
# we monkey-patch the card dict — we only need the generator's flag, since
# the templates we're reading are already pre-computed.
def _build_fit(input_dir: str, knob_value: bool):
    """Construct a WWFit with the WWGenerator's decay knob set to
    ``knob_value``, reading templates from ``input_dir``. Disables the
    template-freshness check by stashing both possible BEC dirs to match."""
    # Build a generator that *claims* the value of knob_value, so the
    # freshness check on the matching template set passes.
    orig = card.NLO_CONFIG["decay_uses_full_born"]
    card.NLO_CONFIG["decay_uses_full_born"] = knob_value
    # Also point BEC dir at the matching backup so freshness check on
    # binned-nuisance reads doesn't fire (we don't actually need them for
    # the base fit, but ww_fit may inspect them).
    orig_bec = card.INPUT_DIRS["BEC"]
    card.INPUT_DIRS["BEC"] = ("output_xsec/ww/BEC_decay_OFF"
                              if knob_value is False else
                              "output_xsec/ww/BEC_decay_ON")
    try:
        gen = WWGenerator.from_card(card)
        fit = WWFit(card, gen, input_dir=input_dir, asimov=True,
                    read_scale_vars=False,
                    mass_scheme=card.MASS_SCHEME, debug=False)
    finally:
        # Restore card so the next call sees a clean state
        card.NLO_CONFIG["decay_uses_full_born"] = orig
        card.INPUT_DIRS["BEC"] = orig_bec
    return fit


def _init_scenario(fit):
    sc = card.SCENARIO
    fit.init_scenario(
        scan_min=sc["scan_min"], scan_max=sc["scan_max"],
        scan_step=sc["scan_step"], total_lumi=sc["total_lumi"],
        last_lumi=sc["last_lumi"], add_last_ecm=False, same_evts=False,
    )


def _fit_and_extract(fit, *, pseudo_data=None, label: str = ""):
    """Run minuit with the given pseudodata (None → own Asimov), print +
    return (m_W central, m_W uncertainty in MeV, Γ_W central in GeV)."""
    if pseudo_data is None:
        fit.fit_parameters()
    else:
        fit.update(pseudo_data=pseudo_data, init_minuit=True,
                   update_scenario=True)
    res = fit.fit_results(printout=False)
    names = fit.param_names
    # Map name → ufloat
    by_name = dict(zip(names, res))
    mW = by_name["mass"]
    gW = by_name["width"]
    asx = by_name.get("alphas", None)
    print(f"  [{label}]")
    print(f"     m_W = {mW.n:.6f}  ±  {mW.s*1e3:.3f} MeV  "
          f"(Δ from card SM 80.379 = {(mW.n-80.379)*1e3:+.3f} MeV)")
    print(f"     Γ_W = {gW.n:.6f}  ±  {gW.s*1e3:.3f} MeV  "
          f"(Δ from card SM 2.085 = {(gW.n-2.085)*1e3:+.3f} MeV)")
    if asx is not None:
        print(f"     α_s = {asx.n:.5f}  ±  {asx.s:.5f}")
    return mW.n, mW.s, gW.n, gW.s


def main():
    print("=" * 72)
    print("Asimov A/B: BFS decay substitution (δ_decay × σ_Born  vs  × σ^(0))")
    print("=" * 72)
    print()

    print("Building OFF fit (knob=False) ...")
    fit_off = _build_fit("output_xsec/ww/nominal_decay_OFF", knob_value=False)
    _init_scenario(fit_off)

    print("Building ON  fit (knob=True)  ...")
    fit_on  = _build_fit("output_xsec/ww/nominal_decay_ON",  knob_value=True)
    _init_scenario(fit_on)

    # Capture each set's pseudodata template — same SM point so it IS the
    # Asimov data for that chain. Use the dedicated 'pseudodata' tag, which
    # is what create_scenario reads by default.
    pseudo_off = fit_off.template(fit_off.pseudodata_tag)
    pseudo_on  = fit_on.template(fit_on.pseudodata_tag)

    print()
    print("-" * 72)
    print("Consistency checks (each chain fitting its OWN Asimov data):")
    print("-" * 72)
    mw_off_own, smw_off_own, _, _ = _fit_and_extract(
        fit_off, pseudo_data=pseudo_off, label="truth=OFF, fit=OFF")
    mw_on_own,  smw_on_own,  _, _ = _fit_and_extract(
        fit_on,  pseudo_data=pseudo_on,  label="truth=ON,  fit=ON")

    print()
    print("-" * 72)
    print("Cross fits (Asimov from one chain, templates from the other):")
    print("-" * 72)
    mw_off_xs, smw_off_xs, _, _ = _fit_and_extract(
        fit_off, pseudo_data=pseudo_on,
        label="truth=ON,  fit=OFF  ← the bias we were carrying")
    mw_on_xs,  smw_on_xs,  _, _ = _fit_and_extract(
        fit_on,  pseudo_data=pseudo_off,
        label="truth=OFF, fit=ON   ← symmetric")

    print()
    print("=" * 72)
    print("Δm_W shift attributable to the BFS decay substitution:")
    print("=" * 72)
    bias_ON_world_OFF_fit = (mw_off_xs - mw_off_own) * 1e3
    bias_OFF_world_ON_fit = (mw_on_xs  - mw_on_own ) * 1e3
    print(f"  truth=ON,  fit=OFF: Δm_W = {bias_ON_world_OFF_fit:+.3f} MeV")
    print(f"  truth=OFF, fit=ON : Δm_W = {bias_OFF_world_ON_fit:+.3f} MeV")
    print()
    print(f"  m_W stat-uncertainty (one bin example, knob=ON): {smw_on_own*1e3:.2f} MeV")
    print()
    print("Reading:")
    print("  If the world is BFS-correct (recipe = ON), and we had not fixed")
    print("  the chain, we would have biased m_W by the first number.")
    print("  After fixing, no bias — the central value moves by that amount.")


if __name__ == "__main__":
    main()
