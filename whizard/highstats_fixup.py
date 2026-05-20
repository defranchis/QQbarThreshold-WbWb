#!/usr/bin/env python3
"""Scan grid_highstats/ and re-emit the b-batches that have no results.csv.

The first highstats campaign (workday, 8 h wall) hit the wall-clock cap on
5 of 441 batches. Those need a longer queue — this generator submits them
on `tomorrow` (24 h) without touching the 436 batches that already
finished. b_idx + √s sub-block layout matches submit.py exactly so the new
results land in the same directory tree.

Run from WW_threshold/:
    python3 whizard/highstats_fixup.py
    cd whizard && condor_submit grid_highstats_retry.sub
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import (HIGHSTATS_MW_VALS, HIGHSTATS_GW_VALS,
                    SQRTS_VALS, SQRTS_PER_BLOCK,
                    sqrts_blocks, csv, mw_label, gw_label,
                    HEADER, CONDOR_DIR, write_submit, SCRIPTS_DIR)

HIGHSTATS_DIR = CONDOR_DIR / "grid_highstats"


def main():
    blocks = sqrts_blocks(SQRTS_VALS, SQRTS_PER_BLOCK)
    retry_jobs = []
    for mw in HIGHSTATS_MW_VALS:
        for gw in HIGHSTATS_GW_VALS:
            for b_idx, block in enumerate(blocks):
                subdir = f"grid_highstats/{mw_label(mw)}_{gw_label(gw)}/b{b_idx}"
                results = CONDOR_DIR / subdir / "results.csv"
                if results.exists():
                    continue
                label = f"hs_{mw_label(mw)}_{gw_label(gw)}_b{b_idx}"
                retry_jobs.append((label, mw, gw, csv(block), "highstats", subdir))

    if not retry_jobs:
        print("grid_highstats is complete — no retry needed.")
        return

    sub_path = SCRIPTS_DIR / "grid_highstats_retry.sub"
    # tomorrow (24 h) — workday (8 h) is the exact wall-clock that killed these.
    write_submit(sub_path,
                 HEADER.format(flavour="tomorrow", cpus=4, memory=2048),
                 retry_jobs)

    print(f"Wrote {sub_path} ({len(retry_jobs)} jobs, tomorrow queue).")
    for label, mw, gw, sqrts_csv, _, _ in retry_jobs:
        print(f"  {label}  m_W={mw:.5f}  Γ_W={gw:.5f}  √s=[{sqrts_csv}]")


if __name__ == "__main__":
    main()
