#!/usr/bin/env python3
"""3×3 summary at √s=163: {BFS, MoCaNLO, matched} × {no ISR, LL+exp, NLL}.

Totals σ_WW in pb at m_W=80.379, Γ_W=2.085.  BFS converts μνqq̄ → total via
PDG BRs (pdg-constant); MoCaNLO/matched use the native channel assembly
12·lnuqq + 4·qqqq + 9·mutau (off-shell convention; =1 at nominal).
The BFS no-ISR entry carries the production WHIZARD anchor so all three BFS
entries share the same Born normalisation.
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
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from framework.process.ww.indep.partonic_grid import DEFAULT_RESULTS_DIR, load_grids
from framework.process.ww.indep.channels import PURE_WW_WEIGHTS, BLOCKS_BY_KEY
from framework.process.ww.indep import isr_beta, match_bfs
from framework.process.ww.indep.mocanlo_cards import SMInputs

import munuqq_163 as m

ECM = 163.0
MW, GW = 80.379, 2.085
BR_INCL = 2.0 * card.BR_W_MUNU * card.BR_W_HAD
MT, MH, MZ, ALPS = 172.5, 125.25, 91.1876, 0.1199


def bfs_no_isr(anchor: bool) -> float:
    val = sigma_BFS_specific_munuud_pb(
        np.array([ECM**2]), mW=MW, gammaW=GW,
        include_NLO_hard_decay=True, include_BFS_NNLO=True,
        apply_delta_QCD=True, alpha_s=ALPS,
        apply_whizard_anchor=anchor, whizard_anchor_source="morph",
        mt=MT, MH=MH, MZ=MZ)
    return float(np.atleast_1d(val)[0]) * 4.0 / BR_INCL


def indep_no_isr(matched: bool) -> float:
    """Assembled partonic total at √ŝ=163, mirroring _varpoint_lineshape
    with the ISR convolution replaced by a pointwise eval."""
    grids = load_grids()
    dnnlo = (match_bfs.delta_nnlo_interp(MW, GW, mt=MT, MH=MH, MZ=MZ)
             if matched else None)
    tot = 0.0
    for key, w in PURE_WW_WEIGHTS.items():
        g = grids[(key, "nominal")]
        sig = float(g.nlo_fn()(ECM))
        if dnnlo is not None:
            sig += float(dnnlo(np.array([ECM]))[0]) * float(g.born_fn()(ECM))
            sig *= match_bfs.delta_qcd_channel_factor(
                BLOCKS_BY_KEY[key].outgoing, ALPS)
        tot += w * sig
    return tot * 1e-3      # fb → pb


def indep_gen(matched: bool, nll: bool) -> float:
    gen = WWGeneratorMoCaNLO(
        results_dir=DEFAULT_RESULTS_DIR, scheme_alpha="gf", lepton_cut=None,
        isr_cfg=isr_beta.ISRConfig(scheme="LO_beta"),
        br_convention="off-shell", match_bfs=matched,
        match_bfs_nnlo=True, match_bfs_dqcd=True,
        alpha_s=ALPS, sm=SMInputs(mt=MT), isr_nll=nll)
    return float(gen._morphed(MW, GW, np.array([ECM]))[0])


if __name__ == "__main__":
    rows = []
    m.ECM = ECM
    rows.append(("BFS-EFT (NNLO+δQCD, anchored)",
                 bfs_no_isr(anchor=True),
                 m.bfs_sigma(isr_nll=False) / BR_INCL,
                 m.bfs_sigma(isr_nll=True) / BR_INCL))
    rows.append(("MoCaNLO (NLO-EW off-shell)",
                 indep_no_isr(matched=False),
                 indep_gen(matched=False, nll=False),
                 indep_gen(matched=False, nll=True)))
    rows.append(("matched (MoCa+δNNLO+δQCD)",
                 indep_no_isr(matched=True),
                 indep_gen(matched=True, nll=False),
                 indep_gen(matched=True, nll=True)))

    print(f"\ntotal σ_WW [pb] at √s = {ECM} GeV, m_W={MW}, Γ_W={GW}")
    print(f"{'calculation':<32} {'no ISR':>8} {'LL+exp':>8} {'NLL':>8} "
          f"{'NLL/noISR':>10}")
    for name, a, b, c in rows:
        print(f"{name:<32} {a:8.3f} {b:8.3f} {c:8.3f} {c/a:10.4f}")
    print(f"\n(BFS no-ISR without WHIZARD anchor: "
          f"{bfs_no_isr(anchor=False):.3f} pb)")
