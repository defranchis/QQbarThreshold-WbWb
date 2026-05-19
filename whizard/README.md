# WHIZARD grid generation

End-to-end recipe to (re)produce the channel-specific 4f Born grid
σ(e⁺e⁻ → μ⁻ν̄_μ ud̄) on the (√s, m_W, Γ_W) cube that anchors the WW EFT
calculation. Final product:

```
../whizard/work/grid/grid.csv      # 1295 rows = 37 √s × 5 m_W × 7 Γ_W
../whizard/work/grid/provenance.json
```

`../whizard/` (sibling of `WW_threshold/`) is the install + workspace; the
scripts here orchestrate everything.

## Layout

```
WW_threshold/whizard/
├── install.sh   ← bootstrap WHIZARD 3.1.5 into ../whizard/install/
├── job.sh       ← per-job condor wrapper (runs WHIZARD on a worker node)
├── submit.py    ← generates bfs_table.sub + grid.sub + grid_augment.sub
├── parse.py     ← extracts RESULT lines from each whizard.log → results.csv
├── aggregate.py ← merges all results.csv → grid.csv + provenance.json
└── fixup.py     ← scans existing data for gaps; emits grid_fixup.sub
```

## Cold-start workflow (≈ 30 min wall on lxplus + HTCondor)

```bash
# 1. Build WHIZARD locally (lxplus, ~15 min).
./whizard/install.sh

# 2. Generate the HTCondor submit files.
python3 whizard/submit.py
#   → whizard/bfs_table.sub   2 jobs, BFS Tables 1+2 validation
#   → whizard/grid.sub        245 jobs, full production grid
#   → whizard/grid_augment.sub  70 jobs, BFS reference Γ_W only (subset of grid.sub)

# 3. Submit to HTCondor (lxplus). Both clusters run in parallel.
cd whizard
condor_submit bfs_table.sub
condor_submit grid.sub
cd ..

# 4. Wait for the queue to drain (~10-15 min for the grid, ~30 min for BFS).
condor_q                  # or: condor_wait whizard/work/condor/logs/grid.log

# 5. Parse RESULT lines from each job's whizard.log, then aggregate.
python3 whizard/parse.py
python3 whizard/aggregate.py

# 6. Verify the grid is gap-free. If not, submit the suggested fixup.
python3 whizard/fixup.py
# If output reports "Grid is complete", you're done. Otherwise:
cd whizard && condor_submit grid_fixup.sub && cd ..
# Then re-run parse.py + aggregate.py.
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
  * `bfs`: `8:200000:"gw",5:1000000` (target ≤ 0.1% MC stat)
  * `grid`: `6:100000:"gw",3:300000` (target ≈ 0.05% MC stat)
* **Cores per job** — `request_cpus = 4` in submit files; `OMP_NUM_THREADS=4` in `job.sh`.
* **WHIZARD version** — pinned in `install.sh` (`VERSION=3.1.5`).
* **Channel** — `e1, E1 => e2, N2, u, D` in `job.sh` (μνqq specific 4f).
  Multiply σ by 27 for the all-flavour 4f sum (BFS convention).
