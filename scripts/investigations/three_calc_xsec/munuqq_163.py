#!/usr/bin/env python3
"""σ(e+e- → μν qq̄) at √s = 163 GeV, nominal parameters, for the three WW
calculations implemented in this repo:

    BFS      — BFS-EFT N^(3/2)LO Born + NLO + NNLO + δ_QCD + Whizard anchor,
               eMELA NLL ISR (production config via WWGenerator.from_card).
    MoCaNLO  — independent MoCaNLO NLO-EW ⊗ analytic LL+exp ISR (no BFS).
    matched  — MoCaNLO Born + grafted BFS δ_NNLO threshold K-factor + δ_QCD
               (decay), same LL+exp ISR as MoCaNLO.

The OBSERVABLE is made identical across all three: the inclusive μνqq̄ channel
(both W charge orderings × both up-type hadronic combos ud̄/cs̄, single μ
family).  In the BFS chain this is channel="inclusive" (multiplicity 4 over the
μ⁻ν̄_μ ud̄ representative; BR_INCLUSIVE = 2·BR(μν)·BR(had)).  In the MoCaNLO
chain the same content is 4·(lnuqq block) — see framework/.../indep/channels.py
(lnuqq = μ⁻ν̄_μ u d̄, no t-channel).  We restrict the MoCaNLO morph to that one
block with multiplicity 4 so the two definitions count the same final states.

At the nominal point (m_W = MW0, Γ_W = GW0) the MoCaNLO BR convention factor is
1, so off-shell vs pdg-constant is immaterial here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cards import ww_default as card
from framework.process.ww.generator import WWGenerator
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
from framework.process.ww.indep.partonic_grid import DEFAULT_RESULTS_DIR
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.mocanlo_cards import SMInputs

ECM = 163.0
MW = float(card.PARAMETERS["mass"]["nominal"])   # 80.379
GW = float(card.PARAMETERS["width"]["nominal"])  # 2.085
MT_GRID = 172.5   # production MoCaNLO grid (EOS grid_gen/results)


# ---------------------------------------------------------------------------
# 1) BFS chain — single √s point, exactly the kwargs WWGenerator.do_scan uses.
# ---------------------------------------------------------------------------
def bfs_sigma(isr_nll: bool, *, include_BFS_NNLO=None, apply_delta_QCD=None) -> float:
    g = WWGenerator.from_card(card)
    # Allow stripping the higher-order pieces for a "pure NLO + LL ISR" baseline.
    nnlo = g.include_BFS_NNLO if include_BFS_NNLO is None else include_BFS_NNLO
    dqcd = g.apply_delta_QCD  if apply_delta_QCD  is None else apply_delta_QCD
    val = sigma_observed_munuqq(
        np.array([ECM]),
        mW=MW, gammaW=GW,
        channel=g.channel,                 # "inclusive" = μνqq̄ (mult 4)
        include_coulomb=g.include_coulomb,
        bfs=g.bfs,
        include_NLO_hard_decay=g.include_NLO_hard_decay,
        include_BFS_NNLO=nnlo,
        apply_delta_QCD=dqcd,
        alpha_s=g.alpha_s, alpha_s_ref=g.alpha_s,
        br_convention=g.br_convention,
        apply_whizard_anchor=g.apply_whizard_anchor,
        whizard_anchor_source=g.whizard_anchor_source,
        isr_scheme=g.isr_scheme,
        isr_nll=isr_nll,                   # production True; False = LL+exp bridge
        isr_emela_ll=g.isr_emela_ll,
        isr_emela_pert_order=g.isr_emela_pert_order,
        isr_emela_fac_scheme=g.isr_emela_fac_scheme,
        isr_emela_ren_scheme=g.isr_emela_ren_scheme,
        isr_scale_factor=g.isr_scale_factor,
        alpha_em=g.alpha_em, alpha_em_isr=g.alpha_em_isr,
        coulomb_kc_safe=g.coulomb_kc_safe,
        decay_uses_full_born=g.decay_uses_full_born,
        m_t=g.m_t, M_H=g.M_H, MZ=g.MZ,
    )
    return float(np.atleast_1d(val)[0])


# ---------------------------------------------------------------------------
# 2/3) MoCaNLO + matched — restrict the morph to the μνqq̄ block (4·lnuqq).
# ---------------------------------------------------------------------------
class _MunuqqMoCaNLO(WWGeneratorMoCaNLO):
    """Same MoCaNLO line-shape machinery, but the inclusive channel set is
    pinned to the single μνqq̄ block with its μ-family multiplicity (4 =
    2 charge orderings × 2 up-type combos).  This is the MoCaNLO counterpart of
    the BFS channel="inclusive" observable."""

    def _weights(self):
        return {"lnuqq": 4.0}


def _moca_gen(cls, match_bfs: bool):
    return cls(
        results_dir=DEFAULT_RESULTS_DIR,
        scheme_alpha="gf",
        lepton_cut=None,                            # inclusive (no fiducial cut)
        isr_cfg=isr_beta.ISRConfig(scheme="LO_beta"),  # analytic LL+exp
        br_convention="off-shell",                  # =pdg-constant at nominal
        match_bfs=match_bfs,
        match_bfs_nnlo=True, match_bfs_dqcd=True,
        alpha_s=0.1199,
        sm=SMInputs(mt=MT_GRID),
        isr_nll=False,
    )


def moca_sigma(match_bfs: bool) -> float:
    """σ(μνqq̄) — morph restricted to the 4·lnuqq block."""
    gen = _moca_gen(_MunuqqMoCaNLO, match_bfs)
    return float(gen._morphed(MW, GW, np.array([ECM]))[0])


def moca_total(match_bfs: bool) -> float:
    """Total σ_WW — standard assembly 12·lnuqq + 4·qqqq + 9·mutau."""
    gen = _moca_gen(WWGeneratorMoCaNLO, match_bfs)
    return float(gen._morphed(MW, GW, np.array([ECM]))[0])


if __name__ == "__main__":
    # PDG inclusive μνqq̄ branching fraction used by the BFS pdg-constant chain
    # (σ_munuqq = σ_WW_total × BR_INCLUSIVE).  Invert it to read off BFS σ_WW.
    BR_INCL = 2.0 * card.BR_W_MUNU * card.BR_W_HAD

    print(f"σ(e+e- → μν qq̄) at √s = {ECM} GeV, "
          f"m_W = {MW} GeV, Γ_W = {GW} GeV")
    print(f"BR(μνqq̄, inclusive) = 2·BR(μν)·BR(had) = {BR_INCL:.5f}\n")

    bfs_nll = bfs_sigma(isr_nll=True)
    bfs_ll  = bfs_sigma(isr_nll=False)
    moca    = moca_sigma(match_bfs=False)
    matched = moca_sigma(match_bfs=True)

    moca_tot    = moca_total(match_bfs=False)
    matched_tot = moca_total(match_bfs=True)
    bfs_nll_tot = bfs_nll / BR_INCL          # pdg-constant: σ_WW = σ_munuqq / BR
    bfs_ll_tot  = bfs_ll  / BR_INCL

    print("  --- σ(μν qq̄)  [pb] ---")
    print(f"  BFS      (NLL ISR, production)     : {bfs_nll:10.5f}")
    print(f"  BFS      (LL+exp ISR, bridge)      : {bfs_ll:10.5f}")
    print(f"  MoCaNLO  (LL+exp ISR)              : {moca:10.5f}")
    print(f"  matched  (LL+exp ISR + δNNLO+δQCD) : {matched:10.5f}")
    print()
    print("  --- total σ_WW  [pb]  (μνqq̄ ÷ BR for BFS; full assembly for MoCaNLO) ---")
    print(f"  BFS      (NLL ISR, production)     : {bfs_nll_tot:10.5f}")
    print(f"  BFS      (LL+exp ISR, bridge)      : {bfs_ll_tot:10.5f}")
    print(f"  MoCaNLO  (LL+exp ISR)              : {moca_tot:10.5f}")
    print(f"  matched  (LL+exp ISR + δNNLO+δQCD) : {matched_tot:10.5f}")
    print()
    print("  --- cross-checks ---")
    print(f"  MoCaNLO μνqq̄ / total      = {moca/moca_tot:8.5f}  "
          f"(cf. PDG BR_incl = {BR_INCL:.5f})")
    print(f"  matched / MoCaNLO (μνqq̄)  = {matched/moca:8.5f}  "
          f"(BFS higher-order graft: {100*(matched/moca-1):+.2f}%)")
    print(f"  MoCaNLO / BFS-LL (σ_WW)   = {moca_tot/bfs_ll_tot:8.5f}  "
          f"(off-shell 4f vs on-shell EFT×BR: {100*(moca_tot/bfs_ll_tot-1):+.1f}%)")
    print(f"  BFS-NLL / BFS-LL          = {bfs_nll/bfs_ll:8.5f}  "
          f"(ISR NLL vs LL: {100*(bfs_nll/bfs_ll-1):+.2f}%)")
