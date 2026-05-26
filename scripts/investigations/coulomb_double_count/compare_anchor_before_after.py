"""Diagnose σ shape impact of the K_C-safe Coulomb fix.

NOTE on naming: the WHIZARD Born anchor f(δ, Γ_W) = σ_WHIZARD / σ_BFS_Born
is a Born-level matching factor — it does NOT depend on K_C or any NLO
loops, so the anchor itself is INVARIANT under ``coulomb_kc_safe``. The
quantity that changes is what multiplies the anchor in the chain (NLO
loops × K_C × δ_QCD).

Four diagnostic panels written to PDF + PNG:

  1. σ_partonic(√s) absolute, with kcsafe={False, True}; WHIZARD Born
     overlaid for reference (the chain target before NLO loops).
  2. Relative shift σ(kcsafe=True) / σ(kcsafe=False) − 1 vs √s.
     Quantifies the Coulomb double-count footprint on σ.
  3. dσ/dm_W vs √s for both options. The shape that drives m_W
     sensitivity in the fit; we want the new combination to leave it
     smooth and not introduce kinks.
  4. dσ/dΓ_W vs √s for both options. Same logic, for Γ_W.

Plus a printed table of the implied m_W central-value shift at the
canonical scan points (Δm_W ≈ −Δσ / (∂σ/∂m_W)). This is a
quick-and-dirty estimate; a real shift comes from rerunning doFit_ww.py.

Run from the repo root:

    python3 -m scripts.investigations.coulomb_double_count.compare_anchor_before_after
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq
from framework.process.ww.xsec_calculator.bfs_eft import (
    sigma_BFS_LO_total_WW_pb,
)


OUT_DIR = "plots/coulomb_double_count"
os.makedirs(OUT_DIR, exist_ok=True)

MW = 80.379
GW = 2.085
DM = 5e-3       # ±5 MeV finite diff for ∂σ/∂m_W
DG = 5e-3       # ±5 MeV finite diff for ∂σ/∂Γ_W

CHAIN_KWARGS = dict(
    mW=MW, gammaW=GW,
    channel="inclusive", include_coulomb=True, br_convention="pdg-constant",
    include_NLO_hard_decay=True, include_BFS_NNLO=True,
    apply_delta_QCD=True, apply_whizard_anchor=True, whizard_anchor_source="spline",
)


def chain_sigma(s, *, coulomb_kc_safe, **overrides):
    """Full partonic σ at the chain's PDG-constant config."""
    kw = {**CHAIN_KWARGS, **overrides}
    return sigma_partonic_munuqq(s, coulomb_kc_safe=coulomb_kc_safe, **kw)


def dsigma_dx(s, *, axis, h, **chain_kw):
    """Central finite difference of σ wrt 'mass' or 'width'."""
    overrides_p = {"mW" if axis == "mass" else "gammaW": (MW if axis == "mass" else GW) + h}
    overrides_m = {"mW" if axis == "mass" else "gammaW": (MW if axis == "mass" else GW) - h}
    sp = chain_sigma(s, **chain_kw, **overrides_p)
    sm = chain_sigma(s, **chain_kw, **overrides_m)
    return (sp - sm) / (2.0 * h)


def main():
    sqrts = np.linspace(154.0, 172.0, 181)
    s = sqrts ** 2

    print("Computing chain σ at kcsafe={False, True} on the full grid...")
    sig_unsafe = chain_sigma(s, coulomb_kc_safe=False)
    sig_kcsafe = chain_sigma(s, coulomb_kc_safe=True)

    # WHIZARD Born reference: σ_BFS_LO × spline anchor, no NLO loops, no
    # K_C, no BR — pure 4f Born × BR_PDG for the chain channel.
    from framework.process.ww.xsec_calculator.eft_xsec import BR_INCLUSIVE_MUNUQQ
    sig_born_4f = sigma_BFS_LO_total_WW_pb(
        s, MW, GW, order="N3/2LO",
        include_NLO_hard_decay=False,
        include_BFS_NNLO=False,
        apply_delta_QCD=False,
        apply_whizard_anchor=True,
        whizard_anchor_source="spline",
    ) * BR_INCLUSIVE_MUNUQQ

    print("Computing dσ/dm_W and dσ/dΓ_W ...")
    dsig_dm_unsafe = dsigma_dx(s, axis="mass",  h=DM, coulomb_kc_safe=False)
    dsig_dm_kcsafe = dsigma_dx(s, axis="mass",  h=DM, coulomb_kc_safe=True)
    dsig_dg_unsafe = dsigma_dx(s, axis="width", h=DG, coulomb_kc_safe=False)
    dsig_dg_kcsafe = dsigma_dx(s, axis="width", h=DG, coulomb_kc_safe=True)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # 1. absolute σ
    ax[0, 0].plot(sqrts, sig_unsafe, lw=1.5, color="C0",
                  label=r"current chain (kcsafe=False)")
    ax[0, 0].plot(sqrts, sig_kcsafe, lw=1.5, color="C1", ls="--",
                  label=r"K$_C$-safe (kcsafe=True)")
    ax[0, 0].plot(sqrts, sig_born_4f, lw=1.0, color="grey", alpha=0.7,
                  label=r"WHIZARD 4f Born $\times$ BR$_{\rm PDG}$ (reference)")
    ax[0, 0].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[0, 0].set_ylabel(r"$\sigma_{\rm partonic}(\mu\nu q\bar q)$  [pb]")
    ax[0, 0].set_title("Absolute partonic σ")
    ax[0, 0].legend(loc="upper left", fontsize=9)
    ax[0, 0].grid(True, alpha=0.3)

    # 2. relative shift
    rel = (sig_kcsafe - sig_unsafe) / sig_unsafe * 100
    ax[0, 1].plot(sqrts, rel, lw=1.5, color="C3")
    ax[0, 1].axhline(0, color="k", lw=0.5, alpha=0.5)
    ax[0, 1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[0, 1].set_ylabel(r"$\Delta\sigma / \sigma$  [%]")
    ax[0, 1].set_title("σ shift from K$_C$-safe fix (kcsafe=True vs False)")
    ax[0, 1].grid(True, alpha=0.3)

    # 3. dσ/dm_W
    ax[1, 0].plot(sqrts, dsig_dm_unsafe * 1e3, lw=1.5, color="C0",
                  label=r"current chain")
    ax[1, 0].plot(sqrts, dsig_dm_kcsafe * 1e3, lw=1.5, color="C1", ls="--",
                  label=r"K$_C$-safe")
    ax[1, 0].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[1, 0].set_ylabel(r"$d\sigma/dm_W$  [fb / MeV]")
    ax[1, 0].set_title(r"Mass sensitivity $\partial\sigma/\partial m_W$")
    ax[1, 0].legend(loc="upper left", fontsize=9)
    ax[1, 0].grid(True, alpha=0.3)

    # 4. dσ/dΓ_W
    ax[1, 1].plot(sqrts, dsig_dg_unsafe * 1e3, lw=1.5, color="C0",
                  label=r"current chain")
    ax[1, 1].plot(sqrts, dsig_dg_kcsafe * 1e3, lw=1.5, color="C1", ls="--",
                  label=r"K$_C$-safe")
    ax[1, 1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax[1, 1].set_ylabel(r"$d\sigma/d\Gamma_W$  [fb / MeV]")
    ax[1, 1].set_title(r"Width sensitivity $\partial\sigma/\partial\Gamma_W$")
    ax[1, 1].legend(loc="upper left", fontsize=9)
    ax[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "compare_kc_safe_chain.pdf")
    png = os.path.join(OUT_DIR, "compare_kc_safe_chain.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # ------------- console summary -------------
    print()
    print("=" * 78)
    print("Implied m_W shift at scan points (quick estimate, NOT a fit)")
    print("=" * 78)
    print("  Δm_W ≈ −Δσ / (∂σ/∂m_W).   Real shift from doFit_ww.py.")
    print()
    scan = np.array([157.0, 161.0, 162.5, 163.0])
    for sq in scan:
        s_sc = sq ** 2
        a = float(chain_sigma(s_sc, coulomb_kc_safe=False))
        b = float(chain_sigma(s_sc, coulomb_kc_safe=True))
        dsigma_dm = float(dsigma_dx(s_sc, axis="mass", h=DM, coulomb_kc_safe=False))
        delta_sig = b - a
        # Sign: dσ/dm_W < 0 → if Δσ < 0 (the fix), m_W must INCREASE to recover σ
        dmW_MeV = -delta_sig / dsigma_dm * 1e3 if abs(dsigma_dm) > 0 else float("nan")
        print(f"  √s = {sq:6.2f}  σ_unsafe={a:9.5f}  σ_kcsafe={b:9.5f}  "
              f"Δσ/σ={(b-a)/a*100:+6.2f}%   "
              f"∂σ/∂m_W={dsigma_dm*1e3:+8.4f} fb/MeV  "
              f"⇒ Δm_W ≈ {dmW_MeV:+7.1f} MeV")
    print()
    print("(Single-energy ΔmW is illustrative — the actual fit shift comes")
    print("from the joint likelihood across the scan; many-MeV shifts at)")
    print("one √s often cancel partially against other scan points.)")


if __name__ == "__main__":
    main()
