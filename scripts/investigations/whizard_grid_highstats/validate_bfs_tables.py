"""BFS-paper closure of the smoothly-interpolated (morphed) WHIZARD grid.

The morph is evaluated at the two BFS arXiv:0707.0773 reference points
and compared against the paper's WHIZARD 4f Born column:

  Table 1  m_W = 80.377, Γ_W = 2.04483  (LO width)
  Table 2  m_W = 80.377, Γ_W = 2.09201  (NLO+QCD width — BFS §6.1 keeps the
                                          Table 1 pole m_W and only swaps Γ_W)

This is the morph analogue of validate_bfs_nlo.py:scenario_I_anchor (the
"old anchor" closure). Two things are tested at once:

  σ_morph / σ_BFS   end-to-end: does the smoothly interpolated WHIZARD
                    3.1.5 grid reproduce the BFS-era WHIZARD numbers.
  σ_grid  / σ_BFS   raw on-grid WHIZARD 3.1.5 point at the same Γ_W
                    column (the BFS Γ_W values 2.04483 / 2.09201 ARE
                    grid columns, but are EXCLUDED from the morph fit
                    by filter_uniform_gw) — isolates the genuine
                    WHIZARD-version offset from the morph interpolation
                    error.

  morph input: grid_fine + outer 0.5-GeV wings (load_operational_grid)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from morph import (build_operational_morph, load_operational_grid,
                    sigma_morph)

plt.style.use(hep.style.CMS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLOTS = ROOT / "plots" / "whizard_grid_highstats"
PLOTS.mkdir(parents=True, exist_ok=True)

# BFS arXiv:0707.0773 Tables 1 & 2 — WHIZARD 4f Born column, specific
# μ⁻ν̄_μ ud̄ channel, σ in fb. Identical to validate_bfs_nlo.py's
# table1_whiz / table2_whiz dicts (committed, cross-checked there).
BFS_TABLES = {
    "Table 1": {
        "mW": 80.377, "gammaW": 2.04483,
        "label": r"BFS Table 1 ($m_W=80.377$, $\Gamma_W=2.04483$)",
        "sigma": {155: 34.43, 158: 63.39, 161: 160.62,
                  164: 318.30, 167: 428.60, 170: 505.10},
    },
    "Table 2": {
        "mW": 80.377, "gammaW": 2.09201,
        "label": r"BFS Table 2 ($m_W=80.377$, $\Gamma_W=2.09201$)",
        "sigma": {155: 33.58, 158: 61.67, 161: 154.19,
                  164: 303.00, 167: 408.80, 170: 481.70},
    },
}


def grid_lookup(df, sqrts, mW, gammaW):
    """Raw on-grid WHIZARD value at the nearest grid m_W and the exact
    (√s, Γ_W). Returns (sigma, err, mW_used) or (None, None, None)."""
    mw_grid = np.array(sorted(df.mW.unique()))
    mw_used = float(mw_grid[np.argmin(np.abs(mw_grid - mW))])
    sel = df[np.isclose(df.sqrts, sqrts, atol=1e-3)
             & np.isclose(df.mW, mw_used, atol=1e-4)
             & np.isclose(df.gammaW, gammaW, atol=1e-4)]
    if len(sel) == 0:
        return None, None, None
    return float(sel.sigma_fb.iloc[0]), float(sel.err_fb.iloc[0]), mw_used


def main():
    sqrts_axis, splines, _ = build_operational_morph()
    print(f"morph fitted at {len(sqrts_axis)} √s values "
          f"(grid_fine + outer 0.5-GeV wings)")
    df_full = load_operational_grid()  # unfiltered: keeps the BFS Γ_W columns

    grid_sqrts = np.array(sorted(df_full.sqrts.unique()))
    s_fine = np.linspace(grid_sqrts[0], grid_sqrts[-1], 400)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex="col",
                             gridspec_kw={"height_ratios": [3, 2]})
    results = {}
    for col, (name, spec) in enumerate(BFS_TABLES.items()):
        ax_abs, ax_rat = axes[0][col], axes[1][col]
        mW, gammaW = spec["mW"], spec["gammaW"]
        sqrts = sorted(spec["sigma"])
        print(f"\n{name}:  m_W={mW}, Γ_W={gammaW}")
        print(f"  {'√s':>6} {'σ_BFS':>10} {'σ_morph':>10} {'morph/BFS':>11}"
              f" {'σ_grid':>10} {'grid/BFS':>10}  (grid m_W)")
        rows = []
        for s in sqrts:
            s_bfs   = spec["sigma"][s]
            s_morph = float(sigma_morph(s, mW, gammaW, splines=splines))
            s_grid, e_grid, mw_used = grid_lookup(df_full, s, mW, gammaW)
            r_morph = s_morph / s_bfs
            r_grid  = (s_grid / s_bfs) if s_grid is not None else float("nan")
            print(f"  {s:>6.0f} {s_bfs:>10.2f} {s_morph:>10.3f} {r_morph:>11.5f}"
                  f" {s_grid:>10.3f} {r_grid:>10.5f}  ({mw_used:.3f})")
            rows.append((s, s_bfs, s_morph, r_morph, s_grid, r_grid))
        results[name] = rows

        s_arr   = np.array([r[0] for r in rows])
        rm_arr  = np.array([r[3] for r in rows])
        rg_arr  = np.array([r[5] for r in rows])

        # absolute σ(√s): smooth morph curve + raw grid + BFS reference points
        sig_morph_fine = sigma_morph(s_fine, mW, gammaW, splines=splines)
        grid_sig = [grid_lookup(df_full, s, mW, gammaW)[0] for s in grid_sqrts]
        ax_abs.plot(s_fine, sig_morph_fine, "-", color="#1f5fa8", lw=2.0,
                    label=r"$\sigma_{\rm morph}$ (smooth)")
        ax_abs.plot(grid_sqrts, grid_sig, "s", ms=5, mfc="white",
                    color="#888888", label="raw WHIZARD-3.1.5 grid")
        ax_abs.plot(s_arr, [r[1] for r in rows], "*", ms=18, color="#d62728",
                    label="BFS Tables 1/2 4f Born")
        ax_abs.axvline(2 * mW, color="grey", ls=":", alpha=0.5)
        ax_abs.set_title(spec["label"], fontsize=13)
        ax_abs.grid(alpha=0.25)
        ax_abs.legend(loc="upper left", fontsize=10)

        # ratio panel
        ax_rat.axhspan(0.999, 1.001, color="grey", alpha=0.15,
                       label=r"$\pm0.1\%$")
        ax_rat.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
        ax_rat.plot(s_arr, rg_arr, "s", ms=9, color="#888888",
                    label=r"raw grid / BFS")
        ax_rat.plot(s_arr, rm_arr, "o", ms=10, color="#1f5fa8",
                    label=r"$\sigma_{\rm morph}\,/\,\sigma_{\rm BFS}$")
        ax_rat.set_ylim(0.9980, 1.0020)
        ax_rat.set_xlabel(r"$\sqrt{s}$ [GeV]")
        ax_rat.grid(alpha=0.25)
        ax_rat.legend(loc="lower right", fontsize=10)
    axes[0][0].set_ylabel(r"$\sigma(\mu\nu u\bar d)$ [fb]")
    axes[1][0].set_ylabel(r"ratio to BFS WHIZARD 4f Born")

    fig.suptitle("Morphed WHIZARD-3.1.5 grid vs BFS arXiv:0707.0773 "
                 "Tables 1 & 2", y=1.00)
    fig.tight_layout()
    out_pdf = PLOTS / "validate_bfs_tables.pdf"
    out_png = PLOTS / "validate_bfs_tables.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    print(f"\n{'campaign':>10}  {'max|morph/BFS-1|':>17} {'max|grid/BFS-1|':>16}")
    for name, rows in results.items():
        mm = max(abs(r[3] - 1) for r in rows) * 100
        mg = max(abs(r[5] - 1) for r in rows) * 100
        print(f"{name:>10}  {mm:>15.3f}%  {mg:>14.3f}%")
    print(f"\nwrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
