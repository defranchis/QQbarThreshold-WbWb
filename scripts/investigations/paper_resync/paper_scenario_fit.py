"""Re-derive the paper's (paper/main.tex) headline numbers on the FIXED chain.

The paper draft (commit 0c2d82b, 2026-05-29) was generated with its OWN
scenario — NOT the current production card:

  * total_lumi        = 12.0 ab^-1 (equal split over the 7 points)  [placeholder]
  * selection_efficiency = 1.0  (knob did not exist yet)
  * PRIORS lumi       = {uncorr 1.0e-3, corr 5.0e-4}   (0.1 % / 0.05 %)
  * PRIORS BEC        = {uncorr 2.0,   corr 1.0} MeV   (era-card "PLACEHOLDER";
                          the paper TEXT says 10/5 MeV — inconsistent with its
                          own Table 1, see RESULTS_paper_prep.md)
  * PRIORS BES        = {uncorr 0.01,  corr 5.0e-3}    (1 % / 0.5 %, unchanged)
  * alphas prior      = 1.0e-4                          (unchanged)
  * no XSEC_SYST placeholder cross-section systematics
  * (new since the paper: aem_isr + aemEW nuisances stay ON — both are
    always_on in the production chain and contribute < 0.05 MeV)

This script monkeypatches those values onto the live card IN MEMORY (the card
file is untouched), then runs the exact `doFit_ww.py --systTable` fit
configuration on the current production templates:

  1. full Asimov fit -> sigma(m_W), sigma(Gamma_W), rho
  2. compute_syst_breakdown -> per-source impacts (paper Table 1)
  3. theory ladder + ISR scheme-variation block under the same scenario
     (--paper-ladder) -> the Section 5 ISR-uncertainty replacement numbers

Usage:  PYTHONPATH=. python3 scripts/investigations/paper_resync/paper_scenario_fit.py
        [--skip-ladder] [--workers N]
"""

import argparse
import os
import sys

import numpy as np

from cards import ww_default as card


# ---------------------------------------------------------------------------
# Paper-era (0c2d82b) scenario + priors, applied in memory
# ---------------------------------------------------------------------------
def apply_paper_scenario():
    card.SCENARIO["total_lumi"] = 12.0e6          # /pb — the paper's 12 ab^-1
    card.SCENARIO["selection_efficiency"] = 1.0   # knob absent in the era card
    n_pts = round((card.SCENARIO["scan_max"] - card.SCENARIO["scan_min"])
                  / card.SCENARIO["scan_step"]) + 1
    card.LUMI_UNCORR_CALIB_LUMI = card.SCENARIO["total_lumi"] / n_pts
    card.PRIORS["lumi"] = {"uncorr": 1.0e-3, "corr": 5.0e-4}
    card.PRIORS["BEC"] = {"uncorr": 2.0, "corr": 1.0}
    # BES + alphas priors identical in both eras; aem_isr/aemEW stay ON.
    print("[paper-scenario] L=12/ab equal split, eff=1.0, "
          "lumi 1e-3/5e-4, BEC 2/1 MeV, BES 0.01/5e-3, no xsec-syst")


def build_fit():
    from framework.process.ww.fit import WWFit
    from framework.process.ww.generator import WWGenerator
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
        add_last_ecm=False,
    )
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    # NO set_xsec_systematics: the paper predates the XSEC_SYST placeholders.
    return fit


def full_fit_and_breakdown():
    import uncertainties as unc
    from framework.common.systematics import compute_syst_breakdown

    fit = build_fit()
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    mass, width = res[fit._idx["mass"]], res[fit._idx["width"]]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    print(f"\n[paper-scenario] FULL fit:  sigma(m_W) = {mass.s*1e3:.2f} MeV   "
          f"sigma(Gamma_W) = {width.s*1e3:.2f} MeV   rho = {rho:+.3f}")

    syst, totals, centrals = compute_syst_breakdown(fit)
    print("\n[paper-scenario] systematics breakdown (paper Table 1 layout, MeV):")
    pois = list(syst.keys())
    hdr = f"{'source':22s}" + "".join(f"{p:>12s}" for p in pois)
    print(hdr)
    print("-" * len(hdr))
    rows = list(next(iter(syst.values())).keys())
    for s in rows:
        cells = "".join(
            f"{syst[p][s]:12.2f}" if np.isfinite(syst[p][s]) else f"{'--':>12s}"
            for p in pois)
        print(f"{s:22s}{cells}")
    print("-" * len(hdr))
    print(f"{'total exp':22s}" + "".join(f"{totals[p]:12.2f}" for p in pois))

    # Paper also quotes the stat-only rho implicitly via the contour; record
    # the stat-only fit for completeness.
    fit.reinitialise_to_stat()
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    m2, w2 = res[fit._idx["mass"]], res[fit._idx["width"]]
    rho2 = float(unc.correlation_matrix([m2, w2])[0, 1])
    print(f"\n[paper-scenario] STAT-ONLY fit: sigma(m_W) = {m2.s*1e3:.2f} MeV   "
          f"sigma(Gamma_W) = {w2.s*1e3:.2f} MeV   rho = {rho2:+.3f}")
    fit.reinitialise_to_nominal()
    return syst, totals


def paper_ladder(workers):
    """Theory ladder + ISR scheme-variation block under the paper scenario
    (templates come from the shared persistent cache — fit step only)."""
    from framework.common.fit_core import ecm_to_str
    from framework.process.ww.theory_ladder import LADDER_CACHE_DIR, run_theory_ladder
    scan7 = [ecm_to_str(e) for e in np.arange(157.0, 163.0 + 1e-9, 1.0)]
    scenario = {"scan_list": scan7, "total_lumi": card.SCENARIO["total_lumi"]}
    out = os.path.join("plots", "paper_resync", "paper_ladder")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_theory_ladder(isr="both", workers=workers, out=out, scheme_var=True,
                      base=LADDER_CACHE_DIR, scenario=scenario)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    apply_paper_scenario()
    full_fit_and_breakdown()
    if not args.skip_ladder:
        paper_ladder(args.workers)


if __name__ == "__main__":
    sys.exit(main())
