"""Morph-closure validation under each ISR scheme.

Question: does linear morphing of σ_obs(m_W, Γ_W) — built from nominal +
single-step variation templates — close to the directly-computed σ_obs
at intermediate points, AFTER the full ISR convolution has been applied?

If yes (residual << σ_stat), then the fit-side linear template morph
is self-consistent under any ISR scheme, and no re-morphing per scheme
is required.

For each ISR scenario:
  - Compute σ_obs(nominal)            at (m_W^0,        Γ_W^0)
  - Compute σ_obs(mass_var)           at (m_W^0 + δm,   Γ_W^0)
  - Compute σ_obs(mass_half)          at (m_W^0 + δm/2, Γ_W^0)
  - Form linear-morph prediction:
      σ_morph = σ(nominal) + 0.5 · [σ(mass_var) - σ(nominal)]
  - Residual = (σ_morph - σ_direct) / σ_direct
  - Report max |residual| over the scan √s grid

A residual ≪ 10⁻⁴ confirms linearity; ≳ 10⁻³ would indicate ISR-induced
curvature large enough to need a finer morph grid.

Usage:
  source setup.sh
  python scripts/investigations/nll_isr/morph_closure_post_isr.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np

_dev = os.open(os.devnull, os.O_WRONLY)
_sav = os.dup(1)
os.dup2(_dev, 1)
from cards import ww_default as _card
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
os.dup2(_sav, 1); os.close(_sav); os.close(_dev)

SCENARIOS = [
    ("baseline_LLexp",  {"isr_scheme": "single_conv", "isr_nll": False, "isr_emela_ll": False}),
    ("emela_LL_2leg",   {"isr_scheme": "2leg",        "isr_nll": False, "isr_emela_ll": True}),
    ("emela_NLL_2leg",  {"isr_scheme": "2leg",        "isr_nll": True,  "isr_emela_ll": False}),
]

# Use the same √s grid the FIT actually sees (its scan grid 157–163 GeV step 1.0)
SCAN_ECM = np.arange(_card.SCENARIO["scan_min"],
                     _card.SCENARIO["scan_max"] + 0.5 * _card.SCENARIO["scan_step"],
                     _card.SCENARIO["scan_step"])

MW0  = _card.PARAMETERS["mass"]["nominal"]   # 80.379
GW0  = _card.PARAMETERS["width"]["nominal"]  # 2.085
DMW  = _card.PARAMETERS["mass"]["variation"] # 0.010 (10 MeV)
DGW  = _card.PARAMETERS["width"]["variation"]


def _isr_kwargs(scen_overrides: dict) -> dict:
    """Take the card's full NLO_CONFIG, override with the scenario, then
    extract only the kwargs sigma_observed_munuqq accepts."""
    cfg = dict(_card.NLO_CONFIG)
    cfg.update(scen_overrides)
    # Forward both σ-chain and ISR knobs.
    return dict(
        channel=cfg.get("channel", "inclusive"),
        br_convention=cfg.get("br_convention", "pdg-constant"),
        include_coulomb=cfg.get("include_coulomb", False),
        include_NLO_hard_decay=cfg.get("include_NLO_hard_decay", True),
        include_BFS_NNLO=cfg.get("include_BFS_NNLO", True),
        apply_delta_QCD=cfg.get("apply_delta_QCD", True),
        apply_whizard_anchor=cfg.get("apply_whizard_anchor", True),
        whizard_anchor_source=cfg.get("whizard_anchor_source", "morph"),
        isr_scheme=cfg["isr_scheme"],
        isr_nll=cfg["isr_nll"],
        isr_emela_ll=cfg.get("isr_emela_ll", False),
        isr_emela_pert_order=cfg.get("isr_emela_pert_order", "NLL"),
        isr_emela_fac_scheme=cfg.get("isr_emela_fac_scheme", "DELTA"),
        isr_emela_ren_scheme=cfg.get("isr_emela_ren_scheme", "ALGMU"),
        coulomb_kc_safe=cfg.get("coulomb_kc_safe", False),
        decay_uses_full_born=cfg.get("decay_uses_full_born", True),
        m_t=_card.PARAM_INPUTS["m_t"],
        M_H=_card.PARAM_INPUTS["M_H"],
        MZ=_card.PARAM_INPUTS["M_Z"],
        alpha_s=_card.PARAM_INPUTS["alpha_s_MW"],
        alpha_s_ref=_card.PARAM_INPUTS["alpha_s_MW"],
        alpha_em=_card.PARAM_INPUTS["alpha_em"],
        alpha_em_isr=_card.PARAM_INPUTS["alpha_em_isr"],
    )


def _sigma_grid(mW, GW, isr_kw):
    return sigma_observed_munuqq(SCAN_ECM, mW=mW, gammaW=GW, **isr_kw)


def main():
    print(f"Morph closure check (linear interp at m_W^0 + δm/2):")
    print(f"  m_W^0 = {MW0:.3f} GeV   δm = ±{DMW*1e3:.0f} MeV (variation step)")
    print(f"  Γ_W^0 = {GW0:.3f} GeV   δΓ = ±{DGW*1e3:.0f} MeV")
    print(f"  scan √s: {SCAN_ECM[0]:.1f}–{SCAN_ECM[-1]:.1f} step "
          f"{_card.SCENARIO['scan_step']:.1f} GeV ({len(SCAN_ECM)} bins)")

    results = []
    for tag, overrides in SCENARIOS:
        kw = _isr_kwargs(overrides)
        t0 = time.time()

        # --- m_W axis ---
        sig_nom    = _sigma_grid(MW0,            GW0, kw)
        sig_mvar   = _sigma_grid(MW0 + DMW,      GW0, kw)
        sig_mhalf  = _sigma_grid(MW0 + DMW * 0.5, GW0, kw)
        morph_m    = sig_nom + 0.5 * (sig_mvar - sig_nom)
        resid_m    = (morph_m - sig_mhalf) / sig_mhalf
        max_m      = float(np.max(np.abs(resid_m)))

        # --- Γ_W axis ---
        sig_wvar   = _sigma_grid(MW0, GW0 + DGW,        kw)
        sig_whalf  = _sigma_grid(MW0, GW0 + DGW * 0.5,  kw)
        morph_w    = sig_nom + 0.5 * (sig_wvar - sig_nom)
        resid_w    = (morph_w - sig_whalf) / sig_whalf
        max_w      = float(np.max(np.abs(resid_w)))

        dt = time.time() - t0
        results.append((tag, max_m, max_w, dt))
        # Translate residual to Δm_W via local slope (assume σ_stat ~ √(σ_nom · L))
        # We just print residual in ppm and σ_obs at peak as a scale reference.
        print(f"\n[{tag}]  ({dt:.1f} s)")
        print(f"  m_W morph residual:  max|Δσ/σ| = {max_m*1e6:9.3f} ppm  "
              f"({max_m:.2e})")
        print(f"  Γ_W morph residual:  max|Δσ/σ| = {max_w*1e6:9.3f} ppm  "
              f"({max_w:.2e})")
        # Per-bin breakdown
        print(f"  Per-bin |Δσ/σ| [ppm] (m_W axis):")
        print(f"    " + "  ".join(f"{sq:5.1f}" for sq in SCAN_ECM))
        print(f"    " + "  ".join(f"{r*1e6:5.2f}" for r in resid_m))

    # Summary
    print(f"\n{'='*70}")
    print("Summary: linear-morph closure residual (max over scan grid)")
    print(f"{'='*70}")
    print(f"  {'scenario':22s} {'m_W axis':>14s} {'Γ_W axis':>14s} {'time':>8s}")
    for tag, mm, mw, dt in results:
        print(f"  {tag:22s} {mm*1e6:>10.2f} ppm {mw*1e6:>10.2f} ppm "
              f"{dt:>7.1f}s")

    print("\nInterpretation:")
    print("  • residual ≪ 10⁻⁴ (100 ppm) ⇒ linear morph closes, no re-morphing needed")
    print("  • per-bin Δσ/σ × |∂σ/∂m_W|⁻¹ gives the equivalent ΔmW per bin (typ. ~ keV)")
    print("\nDone.")


if __name__ == "__main__":
    main()
