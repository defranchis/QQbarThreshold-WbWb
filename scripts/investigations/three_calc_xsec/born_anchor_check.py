#!/usr/bin/env python3
"""Locate the ×1.5 normalization split seen in munuqq_163.py.

Compares, per √s, the same final state e⁺e⁻ → μ⁻ν̄_μ u d̄ (single assignment):

    WHIZARD   — full-4f tree Born from the anchor grid           [fb]
    MoCaNLO   — partonic grid Born and NLO-EW (lnuqq, inclusive) [fb]
    BFS-EFT   — partonic N^(3/2)LO + NNLO + δ_QCD (no ISR)       [fb]

All three are ISR-free partonic quantities, so they must agree at the
few-% level if the bookkeeping is consistent.  Also prints the ISR-folded
BFS-NLL observed total σ_WW at the card scan points (157–163), the
flat-lumi ⟨σ⟩ and N_WW for 19.2 ab⁻¹, and the value at 161.3 GeV where
LEP measured 3.69 ± 0.45 pb.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cards import ww_default as card
from framework.process.ww.xsec_calculator.bfs_eft import sigma_BFS_specific_munuud_pb
from framework.process.ww.indep.partonic_grid import load_grids

MW, GW = 80.379, 2.085
ECMS = np.array([157.0, 158.0, 159.0, 160.0, 161.0, 161.3, 162.0, 163.0])


def whizard_fb(ecms: np.ndarray) -> np.ndarray:
    try:
        from framework.process.ww.xsec_calculator.whizard_grid import whizard_sigma
        return np.array([float(whizard_sigma(e**2, MW, GW)) for e in ecms])
    except Exception as exc:                                  # grid file absent
        print(f"[whizard_grid failed: {exc}; falling back to morph]")
        from framework.process.ww.xsec_calculator.grid_morph import whizard_sigma_morph
        return np.array([float(whizard_sigma_morph(e**2, MW, GW)) for e in ecms])


def main() -> None:
    wh = whizard_fb(ECMS)

    grids = load_grids()
    lnu_keys = sorted(k for k in grids if k[0] == "lnuqq")
    print("lnuqq varpoints:", [k[1] for k in lnu_keys])
    nom_key = next(k for k in lnu_keys if k[1] in ("nom", "nominal", "n0"))
    g = grids[nom_key]
    born = g.born_fn()
    nlo = g.nlo_fn()
    mo_b = np.array([float(born(e)) for e in ECMS])
    mo_n = np.array([float(nlo(e)) for e in ECMS])

    bfs = 1e3 * np.asarray(sigma_BFS_specific_munuud_pb(
        ECMS**2, mW=MW, gammaW=GW,
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, alpha_s=0.1199,
        mt=172.5, MH=125.25, MZ=91.1876))

    print(f"\nσ(e+e- → μ⁻ν̄_μ u d̄), single assignment, NO ISR  [fb]   "
          f"(m_W={MW}, Γ_W={GW})")
    print(f"{'√s':>7} {'WHIZARD':>9} {'MoCa Born':>10} {'MoCa NLO':>10} "
          f"{'BFS part.':>10} {'MoCaB/WH':>9} {'BFS/WH':>8}")
    for i, e in enumerate(ECMS):
        print(f"{e:7.1f} {wh[i]:9.2f} {mo_b[i]:10.2f} {mo_n[i]:10.2f} "
              f"{bfs[i]:10.2f} {mo_b[i]/wh[i]:9.4f} {bfs[i]/wh[i]:8.4f}")

    # ---- ISR-folded BFS-NLL observed totals at the card scan points ----
    import munuqq_163 as m
    BR_INCL = 2.0 * card.BR_W_MUNU * card.BR_W_HAD
    scan = np.arange(card.SCENARIO["scan_min"], card.SCENARIO["scan_max"] + 0.5,
                     card.SCENARIO["scan_step"])
    print(f"\nBFS chain (production NLL ISR) observed totals  "
          f"σ_WW = σ(μνqq̄)/{BR_INCL:.5f}:")
    tots = []
    for e in list(scan) + [161.3]:
        m.ECM = float(e)
        tot = m.bfs_sigma(isr_nll=True) / BR_INCL
        tots.append((e, tot))
        print(f"  √s = {e:7.2f} GeV   σ_WW = {tot:7.4f} pb")
    scan_tots = np.array([t for e, t in tots if e in scan])
    avg = scan_tots.mean()
    lumi = card.SCENARIO["total_lumi"]  # /pb
    print(f"\nflat-lumi ⟨σ_WW⟩ over {len(scan)} points = {avg:.4f} pb")
    print(f"N_WW @ {lumi/1e6:.1f} ab⁻¹ = {avg*lumi:.3e}  "
          f"(LEP @161.3: 3.69 ± 0.45 pb measured)")


if __name__ == "__main__":
    main()
