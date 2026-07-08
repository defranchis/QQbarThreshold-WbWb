"""Focused review of per-point lumi uncorr scaling in fit_core.py.

Exercises: 7pt baseline, 3pt, Azzurri 60/40 uneven split, add_last_ecm,
same_evts, and the reinit-to-stat path. Verifies:
  (a) DIRECTION: larger L_i -> smaller effective prior.
  (b) BIN ORDER: _lumi_perbin_scale order matches _per_kind_bin_idx['lumi']
      order matches scenario ecm order; lengths equal.
  (c) UNEVEN split gives distinct per-point priors.
  (d) cov-mode lumi_uncorr_ecm scaling matches nuisance-mode.
  (e) reinit-to-stat leaves the lumi penalty effectively off.
"""
import numpy as np
from cards import ww_default as card
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.common.fit_core import ecm_to_str

def build(**init_kw):
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(**init_kw)
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.init_minuit()
    return fit

def report(fit, tag):
    print(f"\n===== {tag} =====")
    sc = fit.scenario
    L_i = np.array(list(sc.values()), dtype=float)
    ecms = list(sc.keys())
    print("scenario ecm order:", ecms)
    print("L_i:", L_i)
    print("_lumi_perbin_scale:", fit._lumi_perbin_scale)
    # bin index order for lumi
    bin_idx = fit._per_kind_bin_idx["lumi"]
    print("lumi bin param idx:", bin_idx)
    print("n scenario points:", len(sc), " n lumi bins:", len(bin_idx),
          " n perbin_scale:", None if fit._lumi_perbin_scale is None else len(fit._lumi_perbin_scale))
    # which ecm does each lumi bin map to? via _per_bin_meta and morph_scenario['lumi']
    morph_ecms = [ecm_to_str(e) for e in fit.morph_scenario["lumi"]["ecm"]]
    print("morph_scenario['lumi'] ecm order:", morph_ecms)
    bin_to_morphrow = [fit._per_bin_meta[i][1] for i in bin_idx]
    print("bin -> morph row idx:", bin_to_morphrow)
    bin_ecms = [morph_ecms[r] for r in bin_to_morphrow]
    print("lumi bin ecm order:", bin_ecms)
    assert bin_ecms == ecms, f"ORDER MISMATCH: bins {bin_ecms} vs scenario {ecms}"
    # effective per-bin prior (display in nuisance internal units * scale)
    prior_u = fit._nuisance_priors["lumi"]["uncorr"]
    eff = prior_u * fit._lumi_perbin_scale
    print(f"prior_u (internal)={prior_u:.6g}; eff per-bin prior={eff}")
    # direction check: larger L_i -> smaller eff prior
    # eff should be DECREASING in L_i: larger L -> smaller eff. Robust to ties:
    # check eff == prior_u*sqrt(Lref/L_i) elementwise and anti-correlation.
    assert np.all(np.diff(eff[np.argsort(L_i)]) <= 1e-15), \
        "DIRECTION WRONG: eff prior not non-increasing in L_i"
    assert np.allclose(eff, prior_u * np.sqrt(card.LUMI_UNCORR_CALIB_LUMI / L_i)), \
        "eff per-bin prior != prior_u*sqrt(Lref/L_i)"
    # cov-mode comparison
    fit_cov = WWFit(card, fit.generator, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                    mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit_cov.lumi_mode = "cov"
    fit_cov.scenario_dict = fit.scenario_dict
    fit_cov.scenario = fit.scenario
    fit_cov.pseudo_data_scenario = fit.pseudo_data_scenario
    fit_cov.unc_pseudodata_scenario = fit.unc_pseudodata_scenario
    fit_cov.lumi_uncorr = card.PRIORS["lumi"]["uncorr"]
    fit_cov.lumi_corr = card.PRIORS["lumi"]["corr"]
    fit_cov._build_cov()
    cov_ecm = fit_cov.lumi_uncorr_ecm
    expect = card.PRIORS["lumi"]["uncorr"] * np.sqrt(card.LUMI_UNCORR_CALIB_LUMI / L_i)
    print("cov lumi_uncorr_ecm:", cov_ecm)
    print("expected:          ", expect)
    assert np.allclose(cov_ecm, expect), "COV-MODE scaling mismatch"
    return fit

def reinit_check(fit):
    # Call the REAL _nuisance_prior so we exercise the floor at line 881.
    n = len(fit.param_names)
    # nominal: small displacement on each lumi bin -> nonzero penalty
    fit.reinitialise_to_nominal()
    fit._build_chi2_caches()
    p = np.zeros(n)
    for i in fit._per_kind_bin_idx["lumi"]:
        p[i] = 1.0   # 1 internal-unit displacement on each lumi bin
    nom = fit._nuisance_prior(p, "lumi")
    # stat (off): same displacement should be HEAVILY penalised (pins param->0),
    # i.e. the nuisance no longer floats -> contributes no uncertainty.
    fit.reinitialise_to_stat()
    off = fit._nuisance_prior(p, "lumi")
    print(f"\nreinit: _nuisance_prior(lumi) nominal={nom:.4g}  stat/off={off:.4g}")
    print("  off >> nominal (param pinned when off):", off > 1e6 * max(nom, 1e-30))
    assert off > nom, "reinit-to-stat did not tighten the lumi prior (should pin)"
    fit.reinitialise_to_nominal()
    restored = fit._nuisance_priors["lumi"]["uncorr"]
    expect = card.PRIORS["lumi"]["uncorr"] / fit.input_var["lumi"]
    print(f"  reinit-to-nominal restored uncorr={restored:.6g} expect={expect:.6g}")
    assert np.isclose(restored, expect), "reinit-to-nominal did not restore lumi uncorr"

L = card.SCENARIO["total_lumi"]
scan7 = [ecm_to_str(e) for e in np.arange(157.0, 163.0 + 1e-9, 1.0)]
scan3 = [ecm_to_str(e) for e in (157.0, 160.0, 163.0)]

f7 = build(scan_list=scan7, total_lumi=L, last_lumi=card.SCENARIO["last_lumi"])
report(f7, "7pt baseline (equal split)")

f3 = build(scan_list=scan3, total_lumi=L, last_lumi=card.SCENARIO["last_lumi"])
report(f3, "3pt (equal split)")

# Azzurri-like 60/40 uneven split
from framework.process.ww.scenario_compare import gamma_crossing_ecm
xc = ecm_to_str(gamma_crossing_ecm())
lo = ecm_to_str(157.0)
azz = {lo: 0.6 * L, xc: 0.4 * L}
fa = build(scan_list=[lo, xc], lumi_dict=azz, total_lumi=L, last_lumi=card.SCENARIO["last_lumi"])
report(fa, "Azzurri 60/40 uneven")

# add_last_ecm
fl = build(scan_list=scan3, total_lumi=L, last_lumi=card.SCENARIO["last_lumi"], add_last_ecm=True)
report(fl, "3pt + add_last_ecm")

# same_evts
fe = build(scan_list=scan7, total_lumi=L, last_lumi=card.SCENARIO["last_lumi"], same_evts=True)
report(fe, "7pt same_evts")

reinit_check(f7)
print("\nALL ORDER/DIRECTION/COV ASSERTIONS PASSED" )
