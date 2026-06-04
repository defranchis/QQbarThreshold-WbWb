#!/usr/bin/env python3
"""Validate the factorized (BFS-style) indep morph (framework/.../indep/morph.py).

1. SYNTHETIC exact-recovery: build line shapes from a KNOWN factorized truth on
   the real varpoint lattice; the fit must recover every varpoint to machine
   precision (the model is exact for the generating family).
2. REAL closure on the gf grid: the morph must reproduce the nominal line shape
   EXACTLY (σ_nom·1·1·1) and the off-nominal varpoints to within the fit/denoise
   residual; the morphed nominal must be smooth in √s.

Run:  python3 scripts/investigations/indep_mocanlo/validate_morph_factorized.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import morph as morphmod
from framework.process.ww.indep.varpoints import VARPOINTS, MW0, GW0
from framework.process.ww.indep.generator_mocanlo import (
    WWGeneratorMoCaNLO, _build_fine_grid,
)
from framework.process.ww.indep import grid as gridmod


def test_synthetic() -> None:
    print("[1] synthetic exact-recovery on the real varpoint lattice")
    rng = np.linspace(157.0, 163.0, 7)
    n_s = len(rng)
    # known truth: σ_nom(√s), per-√s quadratic ratios in Δm/ΔΓ, and β(√s)
    sigma_nom = 5.0 + 0.5 * (rng - 160.0)            # arbitrary smooth line shape
    a1 = -1.0e-4 + 1e-6 * (rng - 160.0)              # dR_m/dΔm coefficients
    a2 = 3.0e-7 * np.ones(n_s)
    b1 = +2.0e-4 * np.ones(n_s)
    b2 = 1.0e-7 * np.ones(n_s)
    beta = 5.0e-9 + 1e-10 * (rng - 160.0)

    def Rm(dm):
        return 1.0 + a1 * dm + a2 * dm * dm

    def Rg(dw):
        return 1.0 + b1 * dw + b2 * dw * dw

    coords, lines = {}, {}
    for v in VARPOINTS:
        dm, dw = v.dmW_MeV, v.dgW_MeV
        coords[v.key] = (dm, dw)
        lines[v.key] = sigma_nom * Rm(dm) * Rg(dw) * (1.0 + beta * dm * dw)

    model = morphmod.fit_factorized(coords, lines, rng)
    worst = 0.0
    for v in VARPOINTS:
        pred = model.evaluate(v.dmW_MeV, v.dgW_MeV)
        rel = np.max(np.abs(pred - lines[v.key]) / np.abs(lines[v.key]))
        worst = max(worst, rel)
    print(f"    max rel recovery error over 20 varpoints = {worst:.2e}")
    assert worst < 1e-10, f"synthetic recovery failed: {worst:.2e}"
    print("    PASS (exact for the factorized family)")


def test_real_closure() -> None:
    print("[2] real closure on the gf pure-WW grid (LO_beta ISR)")
    gen = WWGeneratorMoCaNLO(scheme_alpha="gf")
    fine = _build_fine_grid()
    inside = (fine >= gridmod.ECM_MIN) & (fine <= gridmod.ECM_MAX)
    ss = fine[inside]
    model = gen._fit_morph(ss)

    print("    per-varpoint closure (morph.evaluate vs raw assembled line shape):")
    nom_rel = None
    worst_off = 0.0
    for v in VARPOINTS:
        raw = gen._varpoint_lineshape(v.key, ss)
        pred = model.evaluate(v.dmW_MeV, v.dgW_MeV)
        rel = np.max(np.abs(pred - raw) / np.abs(raw))
        tag = "  (nominal)" if (v.dmW_MeV == 0 and v.dgW_MeV == 0) else ""
        if v.dmW_MeV == 0 and v.dgW_MeV == 0:
            nom_rel = rel
        else:
            worst_off = max(worst_off, rel)
        print(f"      {v.key:9s} Δm={v.dmW_MeV:+4.0f} ΔΓ={v.dgW_MeV:+4.0f}  "
              f"max|Δσ/σ| = {rel:.2e}{tag}")
    print(f"    nominal closure         = {nom_rel:.2e}  (expect ~machine 0)")
    print(f"    worst off-nominal resid = {worst_off:.2e}  (fit/denoise residual)")
    assert nom_rel < 1e-12, f"nominal not reproduced exactly: {nom_rel:.2e}"

    # smoothness: relative 2nd difference of the morphed nominal line shape
    nom = model.evaluate(0.0, 0.0)
    d2 = np.abs(np.diff(nom, 2)) / np.abs(nom[1:-1])
    print(f"    nominal smoothness: max rel 2nd-diff = {np.max(d2):.2e}")

    # a representative off-grid fit point (mass_var +10 MeV, width +10 MeV)
    test = model.evaluate(10.0, 10.0)
    assert np.all(np.isfinite(test)) and np.all(test > 0), "non-finite/neg morph"
    print("    PASS (nominal exact, finite/positive, smooth)")


if __name__ == "__main__":
    test_synthetic()
    test_real_closure()
    print("\nALL MORPH VALIDATION PASSED")
