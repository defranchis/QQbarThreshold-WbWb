"""Validate the aemEW EW-coupling normalization nuisance.

Checks:
  1. aemEW is auto-activated as a global nuisance (always_on).
  2. Its morph is a FLAT relative shift = INPUT_VAR['aemEW'] across all sqrt(s).
  3. Asimov no-bias: fitted m_W / Gamma_W are unchanged with vs without aemEW.
  4. Rough sigma_mW impact (quadrature-subtracted), expected ~0.02 MeV (Riembau).

Run: python scripts/investigations/aemEW_nuisance/validate_aemEW.py
"""
import numpy as np
from cards import ww_default as card
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.common.systematics import OFF

POIS = ("mass", "width")
SCALE = {p: card.POI_DISPLAY[p]["scale"] for p in POIS}  # GeV -> MeV


def build_fit():
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"], scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"], total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"], add_last_ecm=False, same_evts=False,
    )
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    return fit


def fit_pois(fit):
    fit.fit_parameters()
    res = fit.results_from_minuit(fit.minuit)
    return {p: (res[fit._idx[p]].n, res[fit._idx[p]].s) for p in POIS}


def main():
    fit = build_fit()

    print("=" * 64)
    print("1. ACTIVATION")
    print(f"   active global nuisances : {sorted(fit._active_global_nuisances)}")
    print(f"   aemEW in param_names    : {'aemEW' in fit.param_names}")
    assert "aemEW" in fit._active_global_nuisances, "aemEW not auto-activated!"

    print("\n2. FLAT MORPH (should equal INPUT_VAR['aemEW'] = "
          f"{card.INPUT_VAR['aemEW']} at every sqrt(s))")
    morph = np.asarray(fit.morph_scenario["aemEW"]["xsec"])
    ecm = np.asarray(fit.morph_scenario["aemEW"]["ecm"])
    for e, m in zip(ecm, morph):
        print(f"   sqrt(s)={e:7.3f}  morph={m:+.6e}")
    assert np.allclose(morph, card.INPUT_VAR["aemEW"], rtol=0, atol=1e-12), \
        "aemEW morph is not flat = INPUT_VAR!"
    print("   -> FLAT and equal to INPUT_VAR. OK")

    print("\n3. ASIMOV NO-BIAS (central values with vs without aemEW)")
    full = fit_pois(fit)
    prior_saved = fit._nuisance_priors["aemEW"]["prior"]
    fit._nuisance_priors["aemEW"]["prior"] = OFF       # pin aemEW at centre
    off = fit_pois(fit)
    fit._nuisance_priors["aemEW"]["prior"] = prior_saved
    for p in POIS:
        d = (full[p][0] - off[p][0]) * SCALE[p]
        print(f"   {p:6s}: central with={full[p][0]:.6f}  without={off[p][0]:.6f}"
              f"   |Δ|={abs(d):.2e} MeV")
        assert abs(d) < 1e-3, f"{p} central value moved by aemEW (bias)!"
    print("   -> central values unchanged (no bias). OK")

    print("\n4. sigma_mW / sigma_GammaW IMPACT (quadrature-subtracted)")
    for p in POIS:
        s_full = full[p][1] * SCALE[p]
        s_off = off[p][1] * SCALE[p]
        imp = np.sqrt(max(s_full**2 - s_off**2, 0.0))
        print(f"   {p:6s}: sigma_full={s_full:.4f}  sigma_noAemEW={s_off:.4f}"
              f"   aemEW impact={imp:.4f} MeV")
    print("=" * 64)


if __name__ == "__main__":
    main()
