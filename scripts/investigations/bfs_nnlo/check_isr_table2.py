"""Δσ̂^(3/2) shift after ISR convolution — round-trip vs Table 2 of
arXiv:0807.0102.

Table 2 of the NNLO paper tabulates the partonic combined σ̂^(3/2) AND
the ISR-convoluted σ_ISR^(3/2). Both are quoted helicity-averaged in fb.
According to the paper text, the ISR convolution reduces the NNLO shift
by ~40% at threshold and almost not at all at 170 GeV.

We compute Δσ_NNLO,ISR = σ_obs(NNLO on) − σ_obs(NNLO off) in the
specific channel μ⁻ν̄_μ ud̄ with BFS-EFT BR (per-component). The
``channel='munuud'`` path returns (σ_LR + σ_RL)/4 already, so the result
is helicity-averaged and directly comparable to Table 2.

Paper inputs (eqs. 50–51):  M_W = 80.377, Γ_W = 2.09201 GeV.
``apply_whizard_anchor=False`` so our Born matches the BFS-EFT N^(3/2)LO
baseline used in the paper for the NNLO comparison (the anchor would
swap in a Whizard-tuned Born and bias the comparison).
"""

import numpy as np

from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq


# Table 2 of arXiv:0807.0102 (helicity-averaged, fb)
TABLE_2 = {
    "sqrts_GeV":  np.array([158.0, 161.0, 164.0, 167.0, 170.0]),
    "N32_noISR":  np.array([-0.001, 0.147, 0.811, 1.287, 1.577]),
    "N32_ISR":    np.array([+0.000, 0.087, 0.544, 0.936, 1.207]),
}


def _run(mW: float, gammaW: float, sqrt_s):
    common = dict(
        mW=mW, gammaW=gammaW,
        channel="munuud",
        br_convention="bfs-eft",
        apply_whizard_anchor=False,
        apply_delta_QCD=False,    # Table 2 Born/NLO columns are pure EW
        include_NLO_hard_decay=True,
        isr_scheme="single_conv",
    )
    off = sigma_observed_munuqq(sqrt_s, include_BFS_NNLO=False, **common)
    on  = sigma_observed_munuqq(sqrt_s, include_BFS_NNLO=True,  **common)
    # σ_observed in 'munuud' channel returns the helicity-averaged specific
    # cross section (σ_LR + σ_RL)/4 in pb → convert to fb for Table 2.
    return (np.asarray(on) - np.asarray(off)) * 1e3


if __name__ == "__main__":
    mW, gammaW = 80.377, 2.09201
    sqrts = TABLE_2["sqrts_GeV"]

    print("=" * 72)
    print("BFS NNLO ISR-improved — round-trip vs Table 2 of arXiv:0807.0102")
    print(f"  M_W={mW}, Γ_W={gammaW}, anchor=off, δ_QCD=off, helicity-averaged")
    print("=" * 72)

    mine = _run(mW, gammaW, sqrts)
    paper = TABLE_2["N32_ISR"]
    print(f"\n{'√s':>5}  {'mine':>9}  {'paper':>9}  {'mine-paper':>11}")
    for i, sq in enumerate(sqrts):
        print(f"  {sq:5.1f}  {mine[i]:8.4f}  {paper[i]:8.4f}  "
              f"{mine[i] - paper[i]:+10.4f}")

    # Also report the reduction factor (NNLO_ISR / NNLO_partonic) — paper
    # text says ~40% reduction at threshold falling to nothing at 170 GeV.
    print("\n  paper ratio σ_ISR^(3/2) / σ̂^(3/2):")
    for i, sq in enumerate(sqrts):
        denom = TABLE_2["N32_noISR"][i]
        if abs(denom) < 1e-3:
            print(f"  {sq:5.1f}    (partonic too small to ratio)")
        else:
            print(f"  {sq:5.1f}    {paper[i]/denom:+6.3f}")
