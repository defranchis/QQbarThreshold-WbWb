"""Validation closure plots for the BFS implementation report.

One PDF per validation subsection of ``report/ww_bfs_implementation.tex``,
written to ``report/figs/``. Each figure has two panels (sharex):

  * top    — code vs paper σ as a function of √s
  * bottom — residual (code − paper) in fb

Run from the repository root with ``PYTHONPATH=.``:

    PYTHONPATH=. python -m scripts.make_report_plots
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from framework.process.ww.xsec_calculator.bfs_eft import (
    TABLE_1 as BFS_TABLE_1,                 # paper Table 1 (LO width)
    TABLE_2 as BFS_TABLE_2,                 # paper Table 2 (NLO+QCD width)
    sigma_BFS_specific_munuud_pb,
    sigma_BFS_LO_total_WW_pb,
    whizard_anchor_factor,
    delta_sigma_NNLO_C_soft_hard_specific_pb,
    delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb,
    delta_sigma_NNLO_C_decay_specific_pb,
    delta_sigma_NNLO_C_residue_specific_pb,
    delta_sigma_NNLO_triple_Coulomb_specific_pb,
)
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figs")
SCAN_LO, SCAN_HI = 157.0, 163.0      # FCC-ee scan window shading

# --- BFS NNLO paper Table 1 (helicity-averaged, fb) ----------------------
BFSNNLO_T1 = {
    "sqrts_GeV":  np.array([158.0, 161.0, 164.0, 167.0, 170.0]),
    "sumN32_fb":  np.array([-0.001, 0.147, 0.811, 1.287, 1.577]),
    "CxSH_fb":    np.array([-0.116, -0.321, -0.417, -0.389, -0.354]),
    "NLO_C_fb":   np.array([0.104,  0.226,  0.393,  0.473,  0.511]),
    "Cxdecay_fb": np.array([-0.037, -0.091, -0.134, -0.142, -0.142]),
    "Cxres_fb":   np.array([0.044,  0.324,  0.965,  1.345,  1.561]),
    "C3_fb":      np.array([0.004,  0.010,  0.003,  0.001,  0.000]),
}

# --- BFS NNLO paper Table 2 (helicity-averaged, fb) ----------------------
BFSNNLO_T2 = {
    "sqrts_GeV":  np.array([158.0, 161.0, 164.0, 167.0, 170.0]),
    "N32_noISR":  np.array([-0.001, 0.147, 0.811, 1.287, 1.577]),
    "N32_ISR":    np.array([+0.000, 0.087, 0.544, 0.936, 1.207]),
}

# --- BFS NLO paper Table 4 columns (specific channel μ⁻ν̄_μ ud̄, fb) ------
# (Born, Born+ISR, NLO_with_ISR, NLO_ISR_tree). m_W = 80.377, Γ_W = 2.09201.
BFSNLO_T4 = {
    "sqrts_GeV": np.array([158.0, 161.0, 164.0, 167.0, 170.0]),
    "born_isr":  np.array([45.64, 108.60, 219.70, 310.20, 378.40]),
    "nlo":       np.array([49.19, 117.81, 234.90, 328.20, 398.00]),
}


def _setup_axes(title, ylabel_top, ylabel_bot, *, height=5.5):
    fig, (ax_t, ax_b) = plt.subplots(
        2, 1, figsize=(7, height), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.4], hspace=0.05),
    )
    ax_t.set_title(title)
    ax_t.set_ylabel(ylabel_top)
    ax_b.set_ylabel(ylabel_bot)
    ax_b.set_xlabel(r"$\sqrt{s}$ [GeV]")
    for ax in (ax_t, ax_b):
        ax.axvspan(SCAN_LO, SCAN_HI, color="0.92", zorder=0)
        ax.grid(alpha=0.3)
    ax_b.axhline(0, color="0.5", lw=0.7)
    return fig, ax_t, ax_b


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, name + ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf}")


# ============================================================================
# Plot 1 — BFS NLO Tables 1 and 2 (Born expansion closure)
# ============================================================================
def plot_bfs_born_tables():
    sqrts = BFS_TABLE_1["sqrts_GeV"]
    s = sqrts ** 2

    # Table 1: m_W = 80.377, Γ_W = Γ_W^(0) = 2.04483, BR correction off.
    # The bfs_eft self-test in __main__ uses the per-piece accumulator with
    # apply_BR_correction=False — but the public function
    # sigma_BFS_specific_munuud_pb always applies the BR correction. With
    # Γ_W = Γ_W^(0), the correction factor evaluates to 1 (trivial), so the
    # public entry point gives the right Table-1 number for these inputs.
    mine_T1 = sigma_BFS_specific_munuud_pb(
        s, mW=80.377, gammaW=2.04483, order="N3/2LO") * 1e3
    # Table 2 keeps Table 1's pole m_W (80.377); only Γ_W changes. BFS §6.1
    # is explicit: "replacing Γ_W^(0) by Γ_W wherever it appears". The
    # string "80.379" does not appear in the BFS paper.
    mine_T2 = sigma_BFS_specific_munuud_pb(
        s, mW=80.377, gammaW=2.09201, order="N3/2LO") * 1e3

    fig, ax_t, ax_b = _setup_axes(
        r"BFS NLO Tables 1 \& 2: $N^{3/2}\mathrm{LO}_\mathrm{EFT}$ Born for "
        r"$e^+e^-\to\mu^-\bar\nu_\mu u\bar d$",
        ylabel_top=r"$\sigma$ [fb]",
        ylabel_bot=r"code $-$ paper [fb]",
        height=6.0,
    )
    ax_t.plot(sqrts, mine_T1, "-", color="C0",
              label=r"Table 1 (LO width, no BR corr): code")
    ax_t.plot(sqrts, BFS_TABLE_1["eft_N32LO_fb"], "s",
              mfc="none", color="C0", label=r"Table 1: paper")
    ax_t.plot(sqrts, mine_T2, "-", color="C1",
              label=r"Table 2 (NLO+QCD width, BR on): code")
    ax_t.plot(sqrts, BFS_TABLE_2["eft_N32LO_fb"], "s",
              mfc="none", color="C1", label=r"Table 2: paper")
    ax_t.legend(fontsize=9, loc="upper left")

    ax_b.plot(sqrts, mine_T1 - BFS_TABLE_1["eft_N32LO_fb"], "o-", color="C0",
              label="Table 1")
    ax_b.plot(sqrts, mine_T2 - BFS_TABLE_2["eft_N32LO_fb"], "o-", color="C1",
              label="Table 2")
    ax_b.legend(fontsize=8, loc="lower right")
    _save(fig, "val_bfs_born_tables")


# ============================================================================
# Plot 2 — BFS NLO Table 4 (Born+ISR and NLO closure)
# ============================================================================
def plot_bfs_table4():
    mW, gammaW = 80.377, 2.09201
    sqrts = BFSNLO_T4["sqrts_GeV"]

    # Apples-to-Table-4 settings (see validate_bfs_nlo.py:scenario_F docstring):
    #   * include_coulomb=False — fixed-order, no K_C resummation (overlaps
    #     with σ^(1)_pot and eq.62 leading-α);
    #   * apply_whizard_anchor=True — Table 4 col "Born" = Whizard 4f Born
    #     (BFS caption: identical to Table 2's last column), reached by
    #     σ_BFS·f rather than σ_BFS itself;
    #   * apply_delta_QCD = False (Born) / True (NLO) — BFS §6.1: δ_QCD
    #     multiplies only the entire NLO electroweak cross section;
    #   * isr_scheme="2leg" — paper LL+exp two-leg ISR.
    common = dict(
        mW=mW, gammaW=gammaW, channel="munuud",
        br_convention="bfs-eft",
        include_BFS_NNLO=False,                # Table 4 is NLO, no NNLO
        include_coulomb=False,
        apply_whizard_anchor=True,
        isr_scheme="2leg",
    )
    born_isr_code = np.asarray(sigma_observed_munuqq(
        sqrts, include_NLO_hard_decay=False,
        apply_delta_QCD=False, **common)) * 1e3
    nlo_code      = np.asarray(sigma_observed_munuqq(
        sqrts, include_NLO_hard_decay=True,
        apply_delta_QCD=True, alpha_s=0.1199, **common)) * 1e3

    fig, ax_t, ax_b = _setup_axes(
        r"BFS NLO Table 4: Born$\otimes$ISR and NLO for $\mu^-\bar\nu_\mu u\bar d$",
        ylabel_top=r"$\sigma$ [fb]",
        ylabel_bot=r"code $-$ paper [fb]",
        height=6.0,
    )
    ax_t.plot(sqrts, born_isr_code, "-", color="C0",
              label=r"Born$\otimes$ISR: code")
    ax_t.plot(sqrts, BFSNLO_T4["born_isr"], "s",
              mfc="none", color="C0", label=r"Born$\otimes$ISR: paper")
    ax_t.plot(sqrts, nlo_code, "-", color="C3",
              label=r"NLO$\otimes$ISR: code")
    ax_t.plot(sqrts, BFSNLO_T4["nlo"], "s",
              mfc="none", color="C3", label=r"NLO$\otimes$ISR: paper")
    ax_t.legend(fontsize=9, loc="upper left")

    ax_b.plot(sqrts, born_isr_code - BFSNLO_T4["born_isr"], "o-", color="C0",
              label=r"Born$\otimes$ISR")
    ax_b.plot(sqrts, nlo_code - BFSNLO_T4["nlo"], "o-", color="C3",
              label=r"NLO$\otimes$ISR")
    ax_b.legend(fontsize=8, loc="lower right")
    _save(fig, "val_bfs_table4")


# ============================================================================
# Plot 2b — BFS NLO Table 4 decay-substitution A/B
# ============================================================================
def plot_bfs_table4_decay_swap():
    """NLO σ for μ⁻ν̄_μ ud̄ at BFS Table 4 reference points, both chain
    settings of the BFS σ̂_LR^(0)→σ_Born substitution (BFS line 2255),
    overlaid on the paper. Shows the closure improvement at every √s
    and the residual at 158 GeV that is NOT this knob (NLL ISR territory)."""
    mW, gammaW = 80.377, 2.09201
    sqrts = BFSNLO_T4["sqrts_GeV"]
    common = dict(
        mW=mW, gammaW=gammaW, channel="munuud",
        br_convention="bfs-eft",
        include_BFS_NNLO=False,
        include_coulomb=False,
        apply_whizard_anchor=True,
        isr_scheme="2leg",
        include_NLO_hard_decay=True,
        apply_delta_QCD=True, alpha_s=0.1199,
    )
    nlo_off = np.asarray(sigma_observed_munuqq(
        sqrts, decay_uses_full_born=False, **common)) * 1e3
    nlo_on  = np.asarray(sigma_observed_munuqq(
        sqrts, decay_uses_full_born=True,  **common)) * 1e3

    fig, ax_t, ax_b = _setup_axes(
        r"BFS Table~4 NLO$\otimes$ISR closure: decay substitution A/B "
        r"($\mu^-\bar\nu_\mu u\bar d$)",
        ylabel_top=r"$\sigma_{\rm NLO}\otimes$ISR [fb]",
        ylabel_bot=r"code/paper $-$ 1 [\%]",
        height=6.0,
    )
    ax_t.plot(sqrts, BFSNLO_T4["nlo"], "s",
              mfc="none", color="0.2", ms=8, mew=1.4,
              label=r"BFS Table 4 NLO$\otimes$ISR (paper)")
    ax_t.plot(sqrts, nlo_off, "o--", color="C1",
              label=r"code NLO$\otimes$ISR, "
                    r"$\delta_{\rm dec}\!\times\!\hat\sigma^{(0)}$ (historical)")
    ax_t.plot(sqrts, nlo_on, "o-", color="C2",
              label=r"code NLO$\otimes$ISR, "
                    r"$\delta_{\rm dec}\!\times\!\hat\sigma_{\rm Born}$ "
                    r"(BFS recipe, default)")
    ax_t.legend(fontsize=9, loc="upper left")

    ax_b.plot(sqrts, 100 * (nlo_off / BFSNLO_T4["nlo"] - 1), "o--", color="C1",
              label="knob OFF")
    ax_b.plot(sqrts, 100 * (nlo_on  / BFSNLO_T4["nlo"] - 1), "o-",  color="C2",
              label="knob ON")
    ax_b.axhspan(-0.1, 0.1, color="0.85", zorder=0,
                 label=r"BFS MC stat $\pm 0.1\,\%$")
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.set_ylim(-3.5, 1.5)
    _save(fig, "val_bfs_table4_decay_swap")


# ============================================================================
# Plot 3 — Whizard-anchor closure (BFS Table 1 inputs)
# ============================================================================
def plot_whizard_anchor():
    sqrts = BFS_TABLE_1["sqrts_GeV"]
    s = sqrts ** 2
    mW, gammaW = 80.377, 2.04483    # Table 1 inputs, BR correction trivial

    # sigma_BFS_specific_munuud_pb × whizard_anchor_factor
    # (the public function does this internally when apply_whizard_anchor=True).
    code_anchored = sigma_BFS_specific_munuud_pb(
        s, mW=mW, gammaW=gammaW, order="N3/2LO",
        apply_whizard_anchor=True) * 1e3
    paper_whizard = BFS_TABLE_1["exact_Born_fb"]

    fig, ax_t, ax_b = _setup_axes(
        "Whizard-anchor closure (BFS NLO Table 1 inputs)",
        ylabel_top=r"$\sigma_\mathrm{BFS}\cdot f$ and $\sigma_\mathrm{Whizard}$ [fb]",
        ylabel_bot=r"code $-$ paper [fb]",
        height=5.5,
    )
    ax_t.plot(sqrts, code_anchored, "-", color="C2",
              label=r"$\sigma_\mathrm{BFS}\cdot f$ (code)")
    ax_t.plot(sqrts, paper_whizard, "s",
              mfc="none", color="C2", label="BFS Whizard reference")
    ax_t.legend(fontsize=9, loc="upper left")

    ax_b.plot(sqrts, code_anchored - paper_whizard, "o-", color="C2")
    _save(fig, "val_whizard_anchor")


# ============================================================================
# Plot 4 — BFS NNLO Table 1 closed-form pieces + sum
# ============================================================================
def plot_bfs_nnlo_pieces():
    sqrts = BFSNNLO_T1["sqrts_GeV"]
    s = sqrts ** 2
    mW, gammaW = 80.377, 2.09201

    # Each closed-form piece: code returns σ_LR-specific in pb; the paper
    # quotes Δσ_LR/4 (helicity-averaged) in fb → factor 1e3 / 4.
    pieces_code = {
        "CxSH_fb":     delta_sigma_NNLO_C_soft_hard_specific_pb(
                          s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0,
        "NLO_C_fb":    delta_sigma_NNLO_NLO_Coulomb_potential_specific_pb(
                          s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0,
        "Cxdecay_fb":  delta_sigma_NNLO_C_decay_specific_pb(
                          s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0,
        "Cxres_fb":    delta_sigma_NNLO_C_residue_specific_pb(
                          s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0,
        "C3_fb":       delta_sigma_NNLO_triple_Coulomb_specific_pb(
                          s, mW, gammaW, apply_BR_correction=False) * 1e3 / 4.0,
    }
    sum_code = sum(pieces_code.values())

    labels = {
        "CxSH_fb":    r"$C\!\times\![S\!+\!H]$ (eq. 34)",
        "NLO_C_fb":   r"NLO-$C$ (eq. 39)",
        "Cxdecay_fb": r"$C\!\times\!\mathrm{decay}$ (eq. 40)",
        "Cxres_fb":   r"$C\!\times\!\mathrm{res}$ (eq. 48)",
        "C3_fb":      r"$C3$ (eq. 11)",
    }
    colours = {"CxSH_fb": "C0", "NLO_C_fb": "C1", "Cxdecay_fb": "C2",
               "Cxres_fb": "C3", "C3_fb": "C4"}

    fig, ax_t, ax_b = _setup_axes(
        "BFS NNLO Table 1: closed-form pieces (helicity-averaged)",
        ylabel_top=r"$\Delta\sigma$ [fb]",
        ylabel_bot=r"code $-$ paper [fb]",
        height=6.5,
    )
    for k, code_vals in pieces_code.items():
        ax_t.plot(sqrts, code_vals, "-", color=colours[k], label=labels[k])
        ax_t.plot(sqrts, BFSNNLO_T1[k], "s", mfc="none", color=colours[k])
        ax_b.plot(sqrts, code_vals - BFSNNLO_T1[k], "o-",
                  color=colours[k], lw=0.8, ms=3)
    # Sum on top of the pieces
    ax_t.plot(sqrts, sum_code, "k-", lw=1.6, label=r"sum $\hat\sigma^{(3/2)}$ (code)")
    ax_t.plot(sqrts, BFSNNLO_T1["sumN32_fb"], "ko", mfc="none",
              ms=6, label=r"sum $\hat\sigma^{(3/2)}$ (paper)")
    ax_b.plot(sqrts, sum_code - BFSNNLO_T1["sumN32_fb"], "ko-",
              ms=4, label="sum")

    ax_t.legend(fontsize=8, loc="lower right", ncol=2)
    _save(fig, "val_bfs_nnlo_pieces")


# ============================================================================
# Plot 5 — BFS NNLO Table 2 (NNLO partonic vs ISR-improved)
# ============================================================================
def plot_bfs_nnlo_isr():
    sqrts = BFSNNLO_T2["sqrts_GeV"]
    mW, gammaW = 80.377, 2.09201

    # Code: σ_obs(NNLO on) − σ_obs(NNLO off), specific channel μνud̄,
    # BFS-EFT BR, no anchor, no δ_QCD (paper baseline for Table 2).
    common = dict(
        mW=mW, gammaW=gammaW, channel="munuud",
        br_convention="bfs-eft",
        apply_whizard_anchor=False,
        apply_delta_QCD=False,
        include_NLO_hard_decay=True,
        isr_scheme="single_conv",
    )
    off = np.asarray(sigma_observed_munuqq(sqrts, include_BFS_NNLO=False, **common))
    on  = np.asarray(sigma_observed_munuqq(sqrts, include_BFS_NNLO=True,  **common))
    code_isr = (on - off) * 1e3
    paper_partonic = BFSNNLO_T2["N32_noISR"]
    paper_isr      = BFSNNLO_T2["N32_ISR"]

    fig, ax_t, ax_b = _setup_axes(
        "BFS NNLO Table 2: partonic vs ISR-improved $\\hat\\sigma^{(3/2)}$",
        ylabel_top=r"$\hat\sigma^{(3/2)}$ [fb] (helicity-averaged)",
        ylabel_bot=r"code $-$ paper [fb]",
        height=5.5,
    )
    ax_t.plot(sqrts, paper_partonic, "k--", lw=0.9,
              label=r"$\hat\sigma^{(3/2)}$ partonic (paper)")
    ax_t.plot(sqrts, code_isr,  "-", color="C3",
              label=r"$\sigma_\mathrm{ISR}^{(3/2)}$ (code)")
    ax_t.plot(sqrts, paper_isr, "s", mfc="none", color="C3",
              label=r"$\sigma_\mathrm{ISR}^{(3/2)}$ (paper)")
    ax_t.legend(fontsize=9, loc="upper left")

    ax_b.plot(sqrts, code_isr - paper_isr, "o-", color="C3")
    _save(fig, "val_bfs_nnlo_isr")


def main():
    print(f"Writing validation plots to {OUT_DIR}")
    plot_bfs_born_tables()
    plot_bfs_table4()
    plot_bfs_table4_decay_swap()
    plot_whizard_anchor()
    plot_bfs_nnlo_pieces()
    plot_bfs_nnlo_isr()
    print("done.")


if __name__ == "__main__":
    main()
