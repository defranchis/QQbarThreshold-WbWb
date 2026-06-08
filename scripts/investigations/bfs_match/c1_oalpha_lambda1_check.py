"""Is the λ₁ soft constant the ONLY O(α) piece missing from the matched-NLL C₁?

Context.  The matched line shape (framework/process/ww/indep/isr_beta.py) is

    σ_obs = convolve_2leg(σ̂_NLO)  −  C₁[σ̂_Born]

with the radiator (convolve_2leg) using eMELA's full NLL DELTA ePDF but the C₁
subtraction (oalpha_isr_subtraction) using only the analytic β-scheme LL D₁.  The
proposed O(α)-exact refinement adds the BCFS soft constant
    Δsoft = (α/π) β_e (λ₁/4),   λ₁ = 3/8 − π²/2 + 6ζ₃ ≈ +2.6525
to C₁'s δ(1−x) term.  The question this script answers: does the *full* eMELA NLL
ePDF differ from the β-LL form by ONLY that soft (x→1) constant, or are there
x-dependent NLL pieces too — and if so, do they survive weighting by the
threshold-peaked σ̂_Born that C₁ actually integrates?

Method.  eMELA exposes only resummed LL/NLL (no fixed-order O(α) mode), so we read
off the resummed per-leg difference
    R(x) = [x D_NLL(x,Q)] / [x D_LL,β(x,Q)] − 1
(NLL DELTA/ALPMZ vs LL BETA, same α(M_Z)).  A pure-λ₁ (soft-only) story predicts
R(x) → const = (α/π) β_e λ₁/4 as x→1 and that the Born-weighted C₁ correction is
dominated by x≈1.  Any flat-in-bulk offset or x-slope beyond the soft constant is a
non-λ₁ O(α) NLL piece.  We then weight by a real σ̂_Born(√ŝ) grid to see what C₁
actually samples.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold"
sys.path.insert(0, REPO)

from framework.process.ww.xsec_calculator import emela_wrapper as emela
from framework.process.ww.xsec_calculator.isr import LAMBDA1_NF0
from framework.process.ww.indep import partonic_grid as pg

ALPHA_MZ = 1.0 / 128.943
M_E = 0.51099895e-3
PI = math.pi


def beta_e(mu_F: float, alpha: float = ALPHA_MZ, m_e: float = M_E) -> float:
    return (alpha / PI) * (2.0 * math.log(mu_F / m_e) - 1.0)


def ratio_curve(Q: float, x: np.ndarray) -> np.ndarray:
    """R(x) = D_NLL,DELTA / D_LL,BETA − 1, per leg, at scale Q."""
    emela.initialize(pert_order="NLL", fac_scheme="DELTA",
                     ren_scheme="ALPMZ", alpha=ALPHA_MZ)
    out = np.empty_like(x)
    for i, xv in enumerate(x):
        omx = 1.0 - xv
        d_nll = emela.code_pdf(xv, omx, Q)        # x*D NLL DELTA
        d_ll = emela.ll_pdf(1, xv, omx, Q)        # x*D LL BETA
        out[i] = d_nll / d_ll - 1.0 if d_ll != 0.0 else np.nan
    return out


def main() -> None:
    outdir = os.path.join(REPO, "plots", "bfs_match")
    os.makedirs(outdir, exist_ok=True)

    for Q in (157.0, 161.0, 163.0):
        be = beta_e(Q)
        c_soft = (ALPHA_MZ / PI) * be * LAMBDA1_NF0 / 4.0   # per-leg λ₁ soft const
        print(f"\n==== Q = {Q:.1f} GeV ====")
        print(f"  β_e            = {be:.6e}")
        print(f"  λ₁ soft const  = (α/π) β_e λ₁/4 = {c_soft:.6e}  "
              f"(+{c_soft*100:.4f}% per leg)")

        # R(x) over the radiator window, with a fine endpoint cluster.
        xb = np.concatenate([
            np.linspace(0.55, 0.99, 45),
            1.0 - np.logspace(-2, -10, 40),     # x = 1-1e-2 … 1-1e-10
        ])
        xb = np.unique(np.clip(xb, 0.55, 1 - 1e-10))
        R = ratio_curve(Q, xb)

        # Endpoint limit vs the λ₁ prediction.
        tail = R[xb > 1 - 1e-6]
        print(f"  R(x→1) mean    = {np.nanmean(tail):.6e}   "
              f"(λ₁ predicts {c_soft:.6e}; ratio {np.nanmean(tail)/c_soft:.3f})")
        # Bulk behaviour: is R flat at the soft value, or x-dependent?
        for xq in (0.60, 0.80, 0.95, 0.99, 1 - 1e-4, 1 - 1e-8):
            j = int(np.argmin(np.abs(xb - xq)))
            print(f"    R(x={xb[j]:.8f}) = {R[j]:+.6e}   "
                  f"(R−c_soft = {R[j]-c_soft:+.3e})")

    # ---- Born-weighted view: what does C₁ actually sample? ----
    print("\n==== Born-weighted C₁ support (inclusive gf grid) ====")
    grids = pg.load_grids(scheme_alpha="gf", lepton_cut=None)
    # pick the nominal varpoint of a representative channel
    keys = sorted(grids)
    # nominal = the varpoint appearing for all channels with no shift token; just
    # take the lexicographically central one per channel for a sanity weighting.
    chosen = None
    for k in keys:
        if k[1].lower() in ("nom", "nominal", "central") or "nom" in k[1].lower():
            chosen = k
            break
    if chosen is None:
        chosen = keys[len(keys) // 2]
    grid = grids[chosen]
    born = grid.born_fn()
    print(f"  using channel/varpoint = {chosen}")
    print(f"  grid √ŝ range          = [{grid.ecm.min():.2f}, {grid.ecm.max():.2f}] GeV")

    Q = 161.0
    be = beta_e(Q)
    c_soft = (ALPHA_MZ / PI) * be * LAMBDA1_NF0 / 4.0
    # x grid where Born has support at √s=Q
    xs = np.linspace((grid.ecm.min() / Q) ** 2, 1 - 1e-10, 4000)
    sqrt_shat = np.sqrt(xs) * Q
    wB = born(sqrt_shat)                     # σ̂_Born(xs) weight
    R = ratio_curve(Q, xs)
    dens = wB * R                            # ∝ the NLL−LL correction integrand to C₁
    # fraction of the |correction| coming from the soft tail x>0.99
    soft = xs > 0.99
    num_soft = np.trapz(np.abs(dens[soft]), xs[soft])
    num_tot = np.trapz(np.abs(dens), xs)
    print(f"  ∫|σ̂_Born·R| dx total           = {num_tot:.4e}")
    print(f"  fraction from x>0.99 (soft)     = {num_soft/num_tot*100:.1f} %")
    # effective Born-weighted R vs the pure soft constant
    Reff = np.trapz(wB * R, xs) / np.trapz(wB, xs)
    print(f"  Born-weighted ⟨R⟩               = {Reff:+.6e}")
    print(f"  pure-λ₁ soft constant c_soft    = {c_soft:+.6e}")
    print(f"  ⟨R⟩ / c_soft                    = {Reff/c_soft:.3f}")
    print(f"  non-λ₁ remainder ⟨R⟩−c_soft     = {Reff-c_soft:+.3e}  "
          f"({(Reff-c_soft)/Reff*100:+.1f}% of ⟨R⟩)")


if __name__ == "__main__":
    main()
