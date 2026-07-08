"""Asimov closure test for the bilinear (m_W × Γ_W) cross-term morph.

Setup
-----
1. Build pseudo-data at the corner point (m_W + variation, Γ_W + variation)
   by reading the cross_mass_width template directly.
2. Run two fits on that pseudo-data:
     (a) bilinear chain — CROSS_TERMS=[("mass","width")] (production default)
     (b) linear-only chain — CROSS_TERMS=() (override)
   For both, MIGRAD is supposed to converge to a value near
   (80.379+0.010, 2.085+0.010) = (80.389, 2.095). The bilinear chain
   should land within numerical noise (sub-eV); the linear chain has
   the residual bilinear bias.

Reports |Δm_W^{linear} - Δm_W^{bilinear}| and same for Γ_W. That
difference IS the bilinear bias at the (10, 10) MeV corner — the
worst-case 2σ excursion on either POI in the FCC-ee fit.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

# Allow running from anywhere
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cards import ww_default as _ww_card


def run_fit_at_corner(cross_terms):
    """Initialise the WW fit with given cross_terms, Asimov on the corner,
    fit and return (m_W, Γ_W, σ_m, σ_Γ, ρ).

    Note: mutates _ww_card.CROSS_TERMS in place; caller is responsible
    for snapshotting/restoring around calls.
    """
    _ww_card.CROSS_TERMS = list(cross_terms)
    from framework.common.fit_core import FitCore
    from framework.process.ww.generator import WWGenerator
    gen = WWGenerator.from_card(_ww_card)
    fit = FitCore(_ww_card, gen, asimov=True)
    fit.init_scenario(
        total_lumi=_ww_card.SCENARIO["total_lumi"],
        last_lumi=_ww_card.SCENARIO["last_lumi"],
        scan_min=_ww_card.SCENARIO["scan_min"],
        scan_max=_ww_card.SCENARIO["scan_max"],
        scan_step=_ww_card.SCENARIO["scan_step"],
    )
    fit.init_minuit()
    fit.minuit.migrad()
    fit.minuit.hesse()
    m  = fit.value_from_param(fit.minuit.values["mass"],  "mass")
    g  = fit.value_from_param(fit.minuit.values["width"], "width")
    sm = fit.minuit.errors["mass"]  * fit.parameters.step("mass")
    sg = fit.minuit.errors["width"] * fit.parameters.step("width")
    rho = float(fit.minuit.covariance.correlation()[
        fit._idx["mass"], fit._idx["width"]
    ])
    return m, g, sm, sg, rho


def main():
    # Patch the pseudo offsets to land exactly at the corner template
    # (m_W+δm, Γ_W+δΓ, α_s_0).  alphas pseudo zeroed out so the
    # pseudodata file_name matches the corner template (which is
    # generated at α_s offset = 0).
    original = copy.deepcopy(_ww_card.PARAMETERS)
    _ww_card.PARAMETERS["mass"]["pseudo"]   = _ww_card.PARAMETERS["mass"]["variation"]
    _ww_card.PARAMETERS["width"]["pseudo"]  = _ww_card.PARAMETERS["width"]["variation"]
    _ww_card.PARAMETERS["alphas"]["pseudo"] = 0.0
    try:
        m_b, g_b, sm_b, sg_b, rho_b = run_fit_at_corner(
            cross_terms=_ww_card.CROSS_TERMS)
        m_l, g_l, sm_l, sg_l, rho_l = run_fit_at_corner(cross_terms=())
    finally:
        _ww_card.PARAMETERS = original

    m_true = _ww_card.PARAMETERS["mass"]["nominal"]  + _ww_card.PARAMETERS["mass"]["variation"]
    g_true = _ww_card.PARAMETERS["width"]["nominal"] + _ww_card.PARAMETERS["width"]["variation"]

    print(f"\n  truth at corner:  m_W = {m_true:.6f} GeV   Γ_W = {g_true:.6f} GeV")
    print(f"\n  bilinear chain:  m_W = {m_b:.6f}  Γ_W = {g_b:.6f}"
          f"   σ_m = {sm_b*1e3:.3f} MeV  σ_Γ = {sg_b*1e3:.3f} MeV  ρ = {rho_b:+.3f}")
    print(f"  linear  chain:  m_W = {m_l:.6f}  Γ_W = {g_l:.6f}"
          f"   σ_m = {sm_l*1e3:.3f} MeV  σ_Γ = {sg_l*1e3:.3f} MeV  ρ = {rho_l:+.3f}")
    print(f"\n  bias bilinear→truth:  Δm_W = {(m_b-m_true)*1e3:+.3f} MeV"
          f"   ΔΓ_W = {(g_b-g_true)*1e3:+.3f} MeV   (closure check)")
    print(f"  bias linear  →truth:  Δm_W = {(m_l-m_true)*1e3:+.3f} MeV"
          f"   ΔΓ_W = {(g_l-g_true)*1e3:+.3f} MeV   (linear-morph residual)")


if __name__ == "__main__":
    main()
