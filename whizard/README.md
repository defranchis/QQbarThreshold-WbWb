# WHIZARD grid generation

End-to-end recipe to (re)produce the channel-specific 4f Born grid
σ(e⁺e⁻ → μ⁻ν̄_μ ud̄) on the (√s, m_W, Γ_W) cube that anchors the WW EFT
calculation. Final products:

```
../whizard/work/grid/grid.csv                       # 1295 rows = 37 √s × 5 m_W × 7 Γ_W (0.05% MC)
../whizard/work/grid_highstats/grid.csv             # 2331 rows = 37 √s × 9 m_W × 7 Γ_W (~0.016% MC)
../whizard/work/grid_highstats_densify/grid.csv     # 756 rows  = 12 √s × 9 m_W × 7 Γ_W (extra 0.25-GeV-step √s)
../whizard/work/grid_validate/grid.csv              # 75 rows   = 3 √s × 5 m_W × 5 Γ_W (1-MeV-step held-out test set)
../whizard/work/grid_fine/grid.csv                  # 6363 rows = 101 √s × 9 m_W × 7 Γ_W (0.1-GeV-step √s in [155,165], ~0.008% MC)
../whizard/work/grid_validate_fine/grid.csv         # 243 rows  = 81 (m_W,Γ_W) 1D-scan pts × 3 √s (0.1-MeV steps, ~0.005% MC)
```

The fit currently anchors on `grid/grid.csv` (the 1295-pt grid, trilinear
interpolation in `framework/.../whizard_grid.py`). The highstats / densify
/ fine campaigns feed the morphing scheme (below), which is validated but
not yet wired into the fit.

`../whizard/` (sibling of `WW_threshold/`) is the install + workspace; the
scripts here orchestrate everything.

## Layout

```
WW_threshold/whizard/
├── install.sh           ← bootstrap WHIZARD 3.1.5 into ../whizard/install/
├── job.sh               ← per-job condor wrapper (runs WHIZARD on a worker node)
├── submit.py            ← emits all *.sub files (bfs_table, grid, grid_augment, grid_highstats,
│                          grid_highstats_densify, grid_validate, grid_fine, grid_validate_fine)
├── parse.py             ← extracts RESULT lines from each whizard.log → results.csv
├── aggregate.py         ← merges results.csv per campaign → grid.csv + provenance.json
├── fixup.py             ← grid: emit grid_fixup.sub for any missing (m_W, Γ_W) pairs
└── highstats_fixup.py   ← grid_highstats: emit grid_highstats_retry.sub for batches that
                           missed the wall-clock cap (uses tomorrow queue)
```

## Cold-start workflow (≈ 30 min wall on lxplus + HTCondor)

```bash
# 1. Build WHIZARD locally (lxplus, ~15 min).
./whizard/install.sh

# 2. Generate ALL the HTCondor submit files.
python3 whizard/submit.py
#   → whizard/bfs_table.sub                2 jobs, BFS Tables 1+2 validation
#   → whizard/grid.sub                     245 jobs, full production grid (0.5 GeV step, 0.05% MC)
#   → whizard/grid_augment.sub             70 jobs, BFS reference Γ_W subset of grid.sub
#   → whizard/grid_highstats.sub           441 jobs, 9 m_W × 7 Γ_W × 7 √s blocks (~0.016% MC)
#   → whizard/grid_highstats_densify.sub   126 jobs, 12 extra √s at 0.25 GeV in [157.25, 162.75]
#   → whizard/grid_validate.sub            25 jobs, 1-MeV (m_W, Γ_W) plane at √s ∈ {161, 162, 163}

# 3. Submit (lxplus). Multiple clusters run in parallel.
cd whizard
condor_submit bfs_table.sub
condor_submit grid.sub                  # production grid (~3 h on longlunch queue)
condor_submit grid_highstats.sub        # ~8 h on workday (use tomorrow if any wall-clock retries needed)
condor_submit grid_highstats_densify.sub  # ~8 h on tomorrow
condor_submit grid_validate.sub         # ~3 h on tomorrow
cd ..

# 4. Wait for queues to drain.
condor_q

# 5. Parse RESULT lines, then aggregate (one CSV + provenance per campaign).
python3 whizard/parse.py
python3 whizard/aggregate.py                       # all campaigns, strict (raises on gaps)
# or, while jobs are still landing:
python3 whizard/aggregate.py --mode grid_highstats --allow-gaps

# 6. Gap-fill if necessary.
python3 whizard/fixup.py             # production grid: missing (m_W, Γ_W) pairs → grid_fixup.sub
python3 whizard/highstats_fixup.py   # highstats grid: missing b-batches      → grid_highstats_retry.sub
# Re-submit if any retry sub is emitted, then re-run parse.py + aggregate.py.
```

The aggregator keys on the in-row (m_W, Γ_W, √s) values, not directory
names, so re-runs are idempotent — re-aggregating after a fixup overwrites
the file with the fuller data.

## What lives where

| Artifact                                | Lives in                                       |
| --------------------------------------- | ---------------------------------------------- |
| WHIZARD binary                          | `../whizard/install/bin/whizard`               |
| Setup script (sources LCG view + paths) | `../whizard/setup.sh`                          |
| Per-job condor outputs                  | `../whizard/work/condor/{bfs_table_*,grid}/`   |
| Per-job condor logs (.out/.err/.log)    | `../whizard/work/condor/logs/`                 |
| **Final grid + provenance**             | `../whizard/work/grid/{grid.csv,provenance.json}` |

## Validation expected

Phase 2 (`bfs_table.sub` cluster) reproduces BFS arXiv:0707.0773 Tables 1
and 2 at the exact reference (m_W, Γ_W) values:

| √s [GeV] | BFS T1  | this  | Δ%    | BFS T2  | this  | Δ%    |
| -------- | ------- | ----- | ----- | ------- | ----- | ----- |
| 155      | 34.43   | 34.45 | +0.05 | 33.58   | 33.60 | +0.05 |
| 158      | 63.39   | 63.43 | +0.06 | 61.67   | 61.64 | −0.05 |
| 161      | 160.62  | 160.49 | −0.08 | 154.19  | 153.93 | −0.17 |
| 164      | 318.30  | 318.37 | +0.02 | 303.00  | 302.87 | −0.04 |
| 167      | 428.60  | 429.37 | +0.18 | 408.80  | 408.87 | +0.02 |
| 170      | 505.10  | 505.46 | +0.07 | 481.70  | 481.56 | −0.03 |

All within ±0.2%, median ±0.06%. Modern WHIZARD 3.1.5 reproduces the
BFS-2007 numbers essentially bit-for-bit.

## Grid axes (canonical, hard-coded in `submit.py`)

| Axis      | Values                                                                 | Step          |
| --------- | ---------------------------------------------------------------------- | ------------- |
| √s [GeV]  | 154.0 → 172.0                                                          | 0.5 GeV (37 pts) |
| m_W [GeV] | 80.279, 80.329, **80.379**, 80.429, 80.479                             | 50 MeV (5 pts)   |
| Γ_W [GeV] | **2.04483**, 2.045, 2.065, **2.085**, **2.09201**, 2.105, 2.125         | irregular (7 pts) |

Γ_W includes the two BFS reference values (2.04483 = LO-width Table 1,
2.09201 = NLO+QCD-width Table 2) so the BFS validation is byte-exact
(no Γ_W interpolation needed).

## Tuning knobs

* **Iteration spec** — fixed in `job.sh` per mode:
  * `bfs`:        `8:200000:"gw",5:1000000`  (target ≤ 0.1% MC stat)
  * `grid`:       `6:100000:"gw",3:300000`   (target ≈ 0.05% MC stat)
  * `highstats`:  `6:500000:"gw",5:5000000`  (target ≈ 0.016% MC stat; needs ≥ tomorrow queue)
  * `fine`:       `6:500000:"gw",5:20000000` (target ≈ 0.008%; 4× highstats; needs nextweek queue)
  * `ultra`:      `6:500000:"gw",5:50000000` (target ≈ 0.005%; 10× highstats; fine (m_W,Γ_W) validation)
* **Cores per job** — `request_cpus = 4` in submit files; `OMP_NUM_THREADS=4` in `job.sh`.
* **WHIZARD version** — pinned in `install.sh` (`VERSION=3.1.5`).
* **Channel** — `e1, E1 => e2, N2, u, D` in `job.sh` (μνqq specific 4f).
  Multiply σ by 27 for the all-flavour 4f sum (BFS convention).

## Downstream — morphing scheme

The operational morph under
`scripts/investigations/whizard_grid_highstats/` is built from
`grid_fine` plus the 0.5-GeV outer wings of `grid_highstats`
(supplied as one DataFrame by `morph.load_operational_grid()`).
`grid_validate` (1-MeV plane) and `grid_validate_fine` (sub-MeV 1D
scans at ~0.005% MC) are its held-out test sets. The predictor is

    σ_pred(√s, m_W, Γ_W) = σ_nom(√s) × R_m(√s, m_W) × R_Γ(√s, Γ_W)
                                    × [1 + β(√s)·(m_W−m_W₀)(Γ_W−Γ_W₀)]

with per-√s quadratic m_W and Γ_W morphs + one bilinear cross-term
coefficient β(s) absorbing the joint coupling at the threshold rise.
The 8 √s-dependent quantities are cubic-spline-interpolated along √s,
on a grid first denoised in √s (`denoise_grid` — χ²-smoothed σ/σ_nom
ratios + (m_W,Γ_W)-plane-averaged σ_nom) so MC ripple is not
propagated. β additionally receives its own χ²-targeted spline pass
(`UnivariateSpline` weighted by 1/σ_β from the bilinear LSQ) because
it is a single scalar, not part of the anti-correlated R_m/R_Γ
coefficient group. Closure on the held-out `grid_validate_fine`
sub-MeV scans: max 0.020%, median 0.004% — sub-MeV-safe.
Construction + validation are documented in report Appendix B.

**Not yet wired into the fit** — `framework/.../whizard_grid.py` still
does trilinear interpolation on `grid/grid.csv`. Plumbing the morph in
(replacing the trilinear `grid` anchor) is the next step; see
`project_followup_morphing_scheme.md` in the project memory.
