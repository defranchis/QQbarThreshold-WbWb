"""Verify the BES uncorr counting-measurement rescale (2026-06-02).

Builds the production-config Asimov fit (POIs + BES/BEC binned nuisances) at the
7-point baseline and a 3-point FCC layout, and checks:

  1. BES is in fit._counting_uncorr_kinds (lumi always is).
  2. fit._uncorr_perbin_scale = √(L_ref/L_i) has one entry per scan point and
     aligns with the BES per-bin nuisance count.
  3. The BES uncorr per-point prior is now tightened for the 3-point layout
     (more lumi/point), exactly like lumi — and the m_W impact stays ≈0.

Run from the repo root:  python scripts/investigations/bes_uncorr_scaling/verify_bes_scaling.py
"""
import numpy as np

import cards.ww_default as card
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit


def build(scan_list, label):
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(scan_list=scan_list,
                      total_lumi=card.SCENARIO["total_lumi"],
                      last_lumi=card.SCENARIO["last_lumi"])
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.fit_parameters()
    print(f"\n=== {label}  ({len(scan_list)} points) ===")
    print("  _counting_uncorr_kinds :", sorted(fit._counting_uncorr_kinds))
    print("  _uncorr_perbin_scale   :", np.round(fit._uncorr_perbin_scale, 4))
    n_bes_bins = len(fit._per_kind_bin_idx["BES"])
    print(f"  BES per-bin nuisances  : {n_bes_bins}  (scale len {len(fit._uncorr_perbin_scale)})")
    bes_prior = card.PRIORS["BES"]["uncorr"]
    print(f"  BES uncorr per point   : {bes_prior} -> {np.round(bes_prior*fit._uncorr_perbin_scale,5)}")
    res = fit.fit_results(printout=False)
    by_name = dict(zip(fit.param_names, res))
    print(f"  sigma(m_W)     = {by_name['mass'].std_dev*1000:.3f} MeV")
    print(f"  sigma(Gamma_W) = {by_name['width'].std_dev*1000:.3f} MeV")
    return fit


SEVEN = [f"{e:.1f}" for e in np.arange(157.0, 163.0 + 0.5, 1.0)]
THREE = ["158.0", "161.0", "162.0"]  # FCC-like 3-point geometry

build(SEVEN, "7-point baseline")
build(THREE, "3-point FCC")
