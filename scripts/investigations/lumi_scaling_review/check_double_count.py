"""Check for double-counting of the last-bin lumi tightening when
add_last_ecm=True AND LUMI_UNCORR_SCALES=True in NUISANCE mode.

In nuisance mode, create_scenario shrinks morph_scenario['lumi'] last row by
1/sqrt(factor_above) (lines 775-782) AND _lumi_perbin_scale shrinks the prior
at the last bin (sqrt(L_ref/L_i)). Both encode "last point has more lumi ->
tighter lumi". If both apply, the last-bin constraint is too tight vs cov mode.

We compare the effective last-bin lumi morph * prior product against cov mode.
"""
import numpy as np
from cards import ww_default as card
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.common.fit_core import ecm_to_str

gen = WWGenerator.from_card(card)

def make(mode):
    f = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
              mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    f.lumi_mode = mode
    scan3 = [ecm_to_str(e) for e in (157.0, 160.0, 163.0)]
    f.init_scenario(scan_list=scan3, total_lumi=card.SCENARIO["total_lumi"],
                    last_lumi=card.SCENARIO["last_lumi"], add_last_ecm=True)
    if mode == "nuisance":
        f.add_binned_nuisance("lumi")
    f.add_binned_nuisance("BEC")
    f.add_binned_nuisance("BES")
    f.init_minuit()
    return f

fn = make("nuisance")
fc = make("cov")

L_i = np.array(list(fn.scenario.values()), dtype=float)
factor_above = fn.scenario[ecm_to_str(fn.last_ecm)] / fn.scenario[list(fn.scenario.keys())[0]]
print("scenario:", list(fn.scenario.keys()))
print("L_i:", L_i, " factor_above:", factor_above)

# nuisance-mode last-bin morph response (should equal nominal/sqrt(factor_above))
lumi_morph = np.asarray(fn.morph_scenario["lumi"]["xsec"])
print("\nNUISANCE-mode morph_scenario['lumi'] (relative response per bin):")
print(lumi_morph)
print("  -> last bin shrunk by 1/sqrt(factor_above)? ratio last/first =",
      lumi_morph[-1] / lumi_morph[0], " expected", 1/np.sqrt(factor_above))

# nuisance-mode prior at last bin
prior_u = fn._nuisance_priors["lumi"]["uncorr"]
eff_prior = prior_u * fn._lumi_perbin_scale
print("\nNUISANCE-mode eff prior per bin:", eff_prior)
print("  last-bin prior scale (vs first):", eff_prior[-1] / eff_prior[0],
      " expected sqrt(L0/Llast) =", np.sqrt(L_i[0]/L_i[-1]))

# The "effective last-bin lumi uncertainty as seen by the data" in nuisance
# mode is morph_response * prior (a unit change in sigma needs param =
# 1/morph; the prior on that param is eff_prior -> effective sigma_lumi seen
# by data = morph_response_first-bin-normalised * eff_prior). Combine both:
# A unit lumi shift (relative dsigma) at the last bin requires
# delta_param = 1 / morph_last, penalised by (delta_param / eff_prior_last)^2.
# Effective constraint strength on the last-bin lumi:
sigma_last_nuis = eff_prior[-1] * lumi_morph[-1] / fn.input_var["lumi"]
sigma_first_nuis = eff_prior[0] * lumi_morph[0] / fn.input_var["lumi"]
# Wait: morph already = (relative dsigma per unit param). prior is in internal
# param units. eff sigma on relative-sigma = morph * prior.
print("\nEffective per-bin lumi sigma (relative-xsec) NUISANCE mode:")
eff_sigma_nuis = lumi_morph * eff_prior / fn.input_var["lumi"]
# morph is per *input_var-scaled* param; input_var['lumi']=0.01, and the flat
# morph response for +1 internal param = input_var (1%). Normalise out:
print(eff_sigma_nuis)

# cov-mode per-bin lumi uncorr sigma directly:
print("\nCOV-mode lumi_uncorr_ecm (the per-bin uncorr lumi sigma):")
print(fc.lumi_uncorr_ecm)
print("\nRatio NUISANCE eff_sigma / COV lumi_uncorr_ecm (should be ~1 each bin):")
print(eff_sigma_nuis / fc.lumi_uncorr_ecm)
