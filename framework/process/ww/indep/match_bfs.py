"""Exploratory BFS-on-MoCaNLO matching layer (NOT production).

MoCaNLO's partonic σ̂ is the *complete* fixed-order NLO-EW result for the full
off-shell e+e- → 4f process — it already contains every O(α) threshold effect
(the O(α) Coulomb, soft/collinear, non-factorizable, EW-decay).  So the only
BFS content that is genuinely *missing* from MoCaNLO, and can be added without
double counting, is:

  1. ``delta_nnlo_relative`` — the BFS NNLO threshold block (the α²/v² Coulomb
     + the NNLO hard/decay terms, BFS arXiv:0807.0102 eq. 49), taken as a
     RELATIVE correction to the BFS Born *production* cross section.  The
     relative form cancels the WHIZARD anchor and the BR convention, so it
     transplants cleanly onto MoCaNLO's own Born.

     CHANNEL UNIVERSALITY (why it multiplies every channel equally): the
     dominant NNLO terms — C×[S+H], NLO-C, C×res, triple-Coulomb C3 — are
     PRODUCTION-side Coulomb corrections to e+e- → W+W-, applied *before* the
     W's decay, hence blind to the final state and genuinely universal.  The
     ONE channel-dependent NNLO term is C×decay (Coulomb × the EW decay
     correction, which differs lep vs had).  We compute δ_NNLO from the total
     σ_WW (sigma_BFS_LO_total_WW_pb = 27× the μνud specific channel), so its
     C×decay piece carries the μνud decay flavour, applied universally.  That
     is an approximation, but C×decay ~ Coulomb(few %) × δ_decay^EW(~1 %) is a
     <0.1 % piece and its channel SPREAD (lep vs had) is a small fraction of
     that → <<0.01 %, negligible vs the 0.25 MeV m_W target.  Contrast with
     δ_QCD below, a LEADING ~4 %/hadronic-W effect that IS treated per-channel.
     TODO(validate): bound the channel spread numerically; split into
     universal-production + per-channel-decay if it ever exceeds ~1 keV on m_W.

  2. ``delta_qcd_channel_factor`` — the QCD correction to hadronic W decay.
     MoCaNLO runs at QCD order 0 (EW only), so this is entirely absent.  It is
     a per-channel multiplicative factor δ_QCD(α_s)^(n_hadronic_W), orthogonal
     to the EW production/Coulomb sector.

The strictly-NLO BFS pieces (HSC / c_p,LR^(1,fin), EW-decay, the O(α) Coulomb)
are deliberately NOT exposed here: they are already inside MoCaNLO's full NLO
and adding them would double-count.

Bookkeeping note (the matching is correct because):
  σ̂_comb(ŝ) = σ̂_MoCaNLO,NLO(ŝ)  +  δ_NNLO(ŝ)·σ̂_MoCaNLO,Born(ŝ)
The NNLO term rides through the SAME single ISR convolution as the NLO term,
but the O(α) ISR matching subtraction stays keyed to σ̂_MoCaNLO,Born ONLY —
because MoCaNLO's σ̂_NLO carries the explicit O(α) ISR log while the BFS-derived
δ_NNLO·Born term is ISR-naked.  See generator_mocanlo.WWGeneratorMoCaNLO.

This module imports the BFS σ-chain on purpose: the "independent" MoCaNLO path
stays independent only in its default (match_bfs=False) mode; the matched mode
is an explicit, flagged cross-combination, never the production default.
"""

from __future__ import annotations

import numpy as np

# BFS σ-chain (production calculator).  Imported lazily-ish at module load; the
# matched path is opt-in so this only costs when someone asks for it.
from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_BFS_LO_total_WW_pb,
    delta_QCD_factor,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    ALPHA_S_MW_DEFAULT,
    M_T_DEFAULT,
    M_H_DEFAULT,
    M_Z as M_Z_DEFAULT,
)

#: Quark final-state tokens (MoCaNLO <outgoing> strings), with/without bar.
_QUARK_FLAVOURS = ("u", "d", "s", "c", "b")


def n_hadronic_w(outgoing: str) -> int:
    """Number of hadronically-decaying W's in a channel block.

    Counts quark tokens in the MoCaNLO ``<outgoing>`` string (each hadronic W
    contributes a qq̄ pair → 2 quark tokens).  e.g. ``"mu- vm~ u d~"`` → 1,
    ``"u d~ s c~"`` → 2, ``"mu- vm~ ta+ vt"`` → 0.
    """
    n_quark_tokens = 0
    for tok in outgoing.split():
        base = tok.rstrip("~")
        if base in _QUARK_FLAVOURS:
            n_quark_tokens += 1
    if n_quark_tokens % 2 != 0:
        raise ValueError(f"odd number of quark tokens in {outgoing!r}")
    return n_quark_tokens // 2


def delta_qcd_channel_factor(outgoing: str,
                             alpha_s: float = ALPHA_S_MW_DEFAULT) -> float:
    """Per-channel QCD-on-decay factor δ_QCD(α_s)^(n_hadronic_W).

    MoCaNLO has NO QCD correction (card QCD order 0), so the *full* δ_QCD
    correction is missing and is applied here (not a ratio to a reference α_s,
    unlike the BFS pdg-constant path where δ_QCD/δ_QCD(α_s_ref) avoids a
    double-count against an already-QCD-corrected BR).

    δ_QCD(α_s) = 1 + α_s/π + 1.409 (α_s/π)²  (BFS eq. delta_qcd, per hadronic W).
    """
    n_had = n_hadronic_w(outgoing)
    if n_had == 0:
        return 1.0
    return float(delta_QCD_factor(alpha_s)) ** n_had


def delta_nnlo_relative(sqrt_shat,
                        mW: float,
                        gW: float,
                        *,
                        mt: float = M_T_DEFAULT,
                        MH: float = M_H_DEFAULT,
                        MZ: float = M_Z_DEFAULT,
                        alpha_em: float | None = None,
                        denom_floor_rel: float = 1.0e-3):
    """Relative BFS NNLO correction δ_NNLO(√ŝ) = [σ_NNLO − σ_NLO] / σ_Born.

    All three σ's are the *total* e+e- → W+W- production cross section (no BR,
    no δ_QCD, no WHIZARD anchor) from the BFS chain, differing ONLY by which
    correction blocks are switched on.  The difference (σ_NNLO − σ_NLO) is
    exactly the BFS NNLO threshold block (eq. 49); dividing by σ_Born makes it a
    production K-factor that multiplies MoCaNLO's own Born.

    δ_NNLO is computed with the WHIZARD anchor and the BR correction switched
    OFF (``apply_whizard_anchor=False``, ``apply_BR_correction=False``) and the
    channel multiplicity divided out, so they never enter.  This makes δ_NNLO a
    clean, essentially convention-independent number whose only inputs are
    (m_W, Γ_W) and the SM masses in the matching coefficients (mt/MH weakly; the
    dominant NNLO is Coulomb ∝ α_em, m_W).  NB the anchor/BR factors do not
    cancel *exactly* in the ratio (they are mildly √ŝ-dependent → a ~1% residual
    if turned on); stripping them at the source is what makes the definition
    convention-clean.  The ~1% would in any case be only ~1%×δ_NNLO ≈ 2e-3 % on
    the line shape (≈0.02 MeV on m_W).

    ``sqrt_shat`` in GeV (scalar or array).  Returns δ_NNLO (dimensionless).
    Where σ_Born is negligible (deep sub-threshold tail) the ratio is set to 0;
    MoCaNLO's Born is ~0 there too, so δ_NNLO·Born → 0 regardless.
    """
    s = np.asarray(sqrt_shat, dtype=float) ** 2
    common = dict(order="N3/2LO",
                  apply_BR_correction=False,
                  apply_delta_QCD=False,
                  apply_whizard_anchor=False,
                  decay_uses_full_born=True,
                  mt=mt, MH=MH, MZ=MZ,
                  alpha_em=alpha_em)
    born = np.atleast_1d(np.asarray(sigma_BFS_LO_total_WW_pb(
        s, mW, gW, include_NLO_hard_decay=False, include_BFS_NNLO=False,
        **common), dtype=float))
    nlo = np.atleast_1d(np.asarray(sigma_BFS_LO_total_WW_pb(
        s, mW, gW, include_NLO_hard_decay=True, include_BFS_NNLO=False,
        **common), dtype=float))
    nnlo = np.atleast_1d(np.asarray(sigma_BFS_LO_total_WW_pb(
        s, mW, gW, include_NLO_hard_decay=True, include_BFS_NNLO=True,
        **common), dtype=float))

    delta = nnlo - nlo
    ref = float(np.nanmax(born)) if born.size else 1.0
    floor = denom_floor_rel * max(ref, 1e-300)
    out = np.zeros_like(born)
    safe = born > floor
    out[safe] = delta[safe] / born[safe]
    return float(out[0]) if np.ndim(sqrt_shat) == 0 else out


def delta_nnlo_interp(mW: float,
                      gW: float,
                      *,
                      mt: float = M_T_DEFAULT,
                      MH: float = M_H_DEFAULT,
                      MZ: float = M_Z_DEFAULT,
                      alpha_em: float | None = None,
                      grid_lo: float = 150.0,
                      grid_hi: float = 169.0,
                      grid_step: float = 0.1):
    """Smooth δ_NNLO(√ŝ) callable, precomputed on a √ŝ grid.

    The BFS chain is ~10–20 ms per call and the ISR convolution would otherwise
    evaluate δ_NNLO on its ~n_quad² mesh per output point.  We instead sample
    δ_NNLO on a fine grid once and return a linear interpolator that is 0
    outside [grid_lo, grid_hi] (the convolution only multiplies it by MoCaNLO's
    Born, which is 0 below ~156 GeV anyway).
    """
    grid = np.round(np.arange(grid_lo, grid_hi + 0.5 * grid_step, grid_step), 4)
    vals = delta_nnlo_relative(grid, mW, gW, mt=mt, MH=MH, MZ=MZ,
                               alpha_em=alpha_em)
    vals = np.atleast_1d(np.asarray(vals, dtype=float))

    def fn(sqrt_shat):
        x = np.asarray(sqrt_shat, dtype=float)
        y = np.interp(x, grid, vals, left=0.0, right=0.0)
        return float(y) if np.ndim(sqrt_shat) == 0 else y

    return fn


if __name__ == "__main__":
    # Quick self-check at the nominal point.
    mW, gW = 80.379, 2.085
    grid = np.array([157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0])
    d = delta_nnlo_relative(grid, mW, gW)
    print("δ_NNLO(√ŝ) [relative to BFS Born], nominal (m_W, Γ_W):")
    for e, dd in zip(grid, np.atleast_1d(d)):
        print(f"  √ŝ={e:6.1f}  δ_NNLO={dd:+.5f}  ({dd*100:+.3f} %)")
    print()
    for outg, lbl in (("mu- vm~ u d~", "lnuqq"), ("u d~ s c~", "qqqq"),
                      ("mu- vm~ ta+ vt", "mutau")):
        print(f"δ_QCD factor {lbl:6s} (n_had={n_hadronic_w(outg)}): "
              f"{delta_qcd_channel_factor(outg):.5f}")
