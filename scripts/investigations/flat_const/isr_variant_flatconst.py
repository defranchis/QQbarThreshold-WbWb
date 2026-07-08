"""ISR-scheme cross-fit under a free sqrt(s)-independent (flat) pedestal.

For the talk (slide 15): the ISR-variation Delta m_W is quoted in two fit
configs — shape-only (correlated-lumi floated, pure shape) and shape+norm
(realistic correlated-lumi prior, where a flat/normalisation-like part of the
variation LEAKS into m_W). This adds a THIRD config: shape+norm PLUS a free,
energy-independent additive term c (FitCore.add_flat_const), to see how much of
the leaked ISR-normalisation bias a free flat pedestal can absorb.

Physics expectation (from the code semantics): the LL<->NLL ISR change is a
nearly PURE MULTIPLICATIVE flux change (delta_sigma ~ sigma(sqrt s)); the free
flat term is ADDITIVE (constant c, not c*sigma). Over the narrow 157-163 window
the two are only PARTIALLY degenerate, so c absorbs SOME but not all of the
leaked bias — the correlated-lumi term is the correct normalisation handle.

Reuses the cached theory-ladder ISR-variant templates (no regeneration).
Run (from repo root, after 'source setup.sh'):
    WW_ISR_NJOBS=1 python3 scripts/investigations/flat_const/isr_variant_flatconst.py
"""
import os
import sys

import uncertainties as unc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from cards import ww_default as card  # noqa: E402
from framework.process.ww.fit import WWFit  # noqa: E402
from framework.process.ww import theory_ladder as TL  # noqa: E402

LUMI_FREE = TL.LUMI_CORR_FREE                 # 1.0  -> shape-only
LUMI_PRIOR = card.PRIORS["lumi"]["corr"]      # 1e-4 -> shape+norm


def _fit_variant(card_ov, label, overrides, truth_nom, lumi_corr, flat=False):
    """Cross-fit one ISR variant vs the production (NLL) truth; optionally float
    a free flat pedestal.  Returns (bias_mW_MeV, sig_mW_MeV, rho, c_pb)."""
    gen = TL._make_generator(dict(TL.HARD_RUNGS_BY_KEY[TL.TRUTH_HARD]),
                             overrides, TL.ORDER_OF[TL.TRUTH_HARD])
    fit = WWFit(card_ov, gen, input_dir=TL._isr_var_dir(TL.LADDER_CACHE_DIR, label),
                asimov=True)
    TL._init_scenario(fit, None)
    fit.lumi_uncorr = 0.0
    fit.lumi_corr = lumi_corr
    fit.create_scenario(pseudodata=truth_nom)
    if flat:
        fit.add_flat_const()
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[0], res[1]
    truth_mass = fit.d_params["nominal"]["mass"]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    c_pb = None
    if flat:
        # physical pedestal c = cFlat * scale (pb), in the template sigma units.
        idx = fit.param_names.index(fit._flat_const_name)
        c_pb = res[idx].n * fit._flat_const_scale
    return (mass.n - truth_mass) * 1e3, mass.s * 1e3, rho, c_pb


def main():
    card_ov = TL._ladder_card()
    truth_fit = TL._build_fit(card_ov, TL.TRUTH_HARD, TL.TRUTH_ISR, TL.LADDER_CACHE_DIR)
    truth_nom = truth_fit.template("nominal")

    variants = {lbl: ov for lbl, _kind, ov in TL._isr_scheme_variants()}
    show = ["eMELA-LL", "alpha(0)"]   # NNLL-truncation component + biggest leaker

    print(f"{'variant':10} {'config':22} {'dmW[MeV]':>9} {'sig_mW':>7} "
          f"{'rho':>6} {'c[fb]':>8}")
    print("-" * 70)
    for lbl in show:
        ov = variants[lbl]
        dS, sS, rS, _ = _fit_variant(card_ov, lbl, ov, truth_nom, LUMI_FREE)
        dP, sP, rP, _ = _fit_variant(card_ov, lbl, ov, truth_nom, LUMI_PRIOR)
        dF, sF, rF, cF = _fit_variant(card_ov, lbl, ov, truth_nom, LUMI_PRIOR, flat=True)
        print(f"{lbl:10} {'shape-only (lumi free)':22} {dS:9.2f} {sS:7.2f} {rS:6.2f} {'-':>8}")
        print(f"{lbl:10} {'shape+norm (lumi prior)':22} {dP:9.2f} {sP:7.2f} {rP:6.2f} {'-':>8}")
        print(f"{lbl:10} {'shape+norm + free c':22} {dF:9.2f} {sF:7.2f} {rF:6.2f} "
              f"{cF*1e3:8.2f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
