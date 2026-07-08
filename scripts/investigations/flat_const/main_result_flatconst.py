"""Main-result sensitivity under shape / shape+norm / free flat pedestal.

Companion to isr_variant_flatconst.py, but for the MAIN result (the production
mu-nu-qq fit), not an ISR cross-fit. Fits the production template (BFS NNLO +
NLL ISR) against itself (Asimov) in three configs and reports the SENSITIVITY:
  - shape-only         : correlated lumi floated (norm free) -> pure shape info
  - shape+norm         : realistic correlated-lumi prior (the headline)
  - shape+norm + free c : also float an energy-INDEPENDENT additive pedestal c

MOTIVATION for the free c: it stands in for ANY possible higher-order
NON-RESONANT contribution -- a flat continuum sigma the calculation has not
included, fully correlated across sqrt(s), with no prior. Letting it float
measures how robust sigma(m_W)/sigma(Gamma_W) are to such an uncalculated term
(and reports its own constraint sigma(c)).

Reuses the cached theory-ladder production template (no regeneration).
Run (from repo root, after 'source setup.sh'):
    WW_ISR_NJOBS=1 python3 scripts/investigations/flat_const/main_result_flatconst.py
"""
import os
import sys

import uncertainties as unc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from cards import ww_default as card  # noqa: E402
from framework.process.ww import theory_ladder as TL  # noqa: E402

LUMI_FREE = TL.LUMI_CORR_FREE                 # 1.0  -> shape-only
LUMI_PRIOR = card.PRIORS["lumi"]["corr"]      # 1e-4 -> shape+norm


def _fit(card_ov, truth_nom, lumi_corr, flat=False):
    """Production-chain 2-POI Asimov fit; optionally float a free flat pedestal.
    Returns (sig_mW_MeV, sig_gW_MeV, rho, c_fb)."""
    fit = TL._build_fit(card_ov, TL.TRUTH_HARD, TL.TRUTH_ISR, TL.LADDER_CACHE_DIR)
    fit.lumi_uncorr = 0.0
    fit.lumi_corr = lumi_corr
    fit.create_scenario(pseudodata=truth_nom)
    if flat:
        fit.add_flat_const()
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    c_fb = None
    if flat:
        idx = fit.param_names.index(fit._flat_const_name)
        # sigma(c) = constraint on the pedestal: sigma(cFlat) * scale (pb -> fb).
        c_fb = res[idx].s * fit._flat_const_scale * 1e3
    return mass.s * 1e3, width.s * 1e3, rho, c_fb


def main():
    card_ov = TL._ladder_card()
    truth_fit = TL._build_fit(card_ov, TL.TRUTH_HARD, TL.TRUTH_ISR, TL.LADDER_CACHE_DIR)
    truth_nom = truth_fit.template("nominal")

    sS, gS, rS, _ = _fit(card_ov, truth_nom, LUMI_FREE)
    sP, gP, rP, _ = _fit(card_ov, truth_nom, LUMI_PRIOR)
    sF, gF, rF, cF = _fit(card_ov, truth_nom, LUMI_PRIOR, flat=True)

    print(f"{'config':26} {'sig_mW':>7} {'sig_gW':>7} {'rho':>6} {'sig(c)[fb]':>10}")
    print("-" * 62)
    print(f"{'shape-only (lumi free)':26} {sS:7.2f} {gS:7.2f} {rS:6.2f} {'-':>10}")
    print(f"{'shape+norm (lumi prior)':26} {sP:7.2f} {gP:7.2f} {rP:6.2f} {'-':>10}")
    print(f"{'shape+norm + free c':26} {sF:7.2f} {gF:7.2f} {rF:6.2f} {abs(cF):10.2f}")
    print("-" * 62)
    print(f"free-c inflation vs shape+norm:  sigma(m_W) x{sF/sP:.2f}   "
          f"sigma(Gamma_W) x{gF/gP:.2f}")


if __name__ == "__main__":
    main()
