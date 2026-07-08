#!/usr/bin/env python3
"""Scan existing grid/ data, identify (m_W, Γ_W, √s) tuples missing from
the target grid, and emit a grid_fixup.sub with the minimum jobs needed
to fill in the holes.

Always re-runs ALL 37 √s for any (m_W, Γ_W) pair that has any missing point —
saves us from authoring a per-sqrt(s) condor job and lets VAMP reuse its
adapted phase-space grid across adjacent points. Uses the 5-digit gw_label
from submit.py so the new dirs never collide with the old.

Run from WW_threshold/:
    python3 whizard/fixup.py
    cd whizard && condor_submit grid_fixup.sub
"""

import sys
from pathlib import Path

# Reuse axes + labelling helpers from submit.py (works from any cwd).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import (MW_VALS, GW_VALS, SQRTS_VALS, SQRTS_PER_BLOCK,
                    sqrts_blocks, csv, mw_label, gw_label,
                    HEADER, CONDOR_DIR, write_submit, SCRIPTS_DIR)

GRID_DIR = CONDOR_DIR / "grid"


def existing_tuples():
    """Build the set of (m_W, Γ_W, √s) already present in any results.csv."""
    have = set()
    for csv_path in GRID_DIR.rglob("results.csv"):
        with csv_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                sqrts, _, _, mw, gw = (float(p) for p in parts)
                have.add((round(mw, 5), round(gw, 5), round(sqrts, 2)))
    return have


def main():
    have = existing_tuples()
    blocks = sqrts_blocks(SQRTS_VALS, SQRTS_PER_BLOCK)

    fixup_jobs = []
    for mw in MW_VALS:
        for gw in GW_VALS:
            missing = [s for s in SQRTS_VALS
                       if (round(mw, 5), round(gw, 5), round(s, 2)) not in have]
            if not missing:
                continue
            # Re-run all 7 blocks for this (m_W, Γ_W) pair.
            for b_idx, block in enumerate(blocks):
                label = f"fixup_{mw_label(mw)}_{gw_label(gw)}_b{b_idx}"
                subdir = f"grid/{mw_label(mw)}_{gw_label(gw)}/b{b_idx}"
                fixup_jobs.append((label, mw, gw, csv(block), "grid", subdir))

    if not fixup_jobs:
        print("Grid is complete — no fixup needed.")
        return

    sub_path = SCRIPTS_DIR / "grid_fixup.sub"
    write_submit(sub_path,
                 HEADER.format(flavour="longlunch", cpus=4, memory=2048),
                 fixup_jobs)

    print(f"Wrote {sub_path} ({len(fixup_jobs)} jobs).")
    pairs = {(j[1], j[2]) for j in fixup_jobs}
    print(f"Covers {len(pairs)} (m_W, Γ_W) pairs:")
    for mw, gw in sorted(pairs):
        n_miss = sum(1 for s in SQRTS_VALS
                     if (round(mw, 5), round(gw, 5), round(s, 2)) not in have)
        print(f"  m_W={mw:7.3f}, Γ_W={gw:7.5f}: {n_miss}/37 √s missing")


if __name__ == "__main__":
    main()
