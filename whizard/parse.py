#!/usr/bin/env python3
"""Re-extract RESULT lines from every condor job's whizard.log and write
results.csv next to it. Idempotent: overwrites only when new RESULT lines
were found.

Usage:
    parse_results.py [whizard/work/condor]   # root dir; default if omitted
"""

import re
import sys
from pathlib import Path

CONDOR_DIR_DEFAULT = Path(
    "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard/work/condor"
)


def parse_one(log_path: Path) -> list[str]:
    """Return the list of RESULT lines (without trailing newline) from a log."""
    out = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("RESULT "):
                out.append(line[len("RESULT "):])
    # WHIZARD prints the same printf twice (once inside scan loop, once in
    # the "Time estimate" tail block). De-duplicate while preserving order.
    seen = set()
    deduped = []
    for r in out:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else CONDOR_DIR_DEFAULT
    n_ok = n_empty = n_missing = 0
    for log in root.rglob("whizard.log"):
        results = parse_one(log)
        out = log.with_name("results.csv")
        with out.open("w") as f:
            f.write("# sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV\n")
            for r in results:
                f.write(r + "\n")
        if results:
            n_ok += 1
        else:
            n_empty += 1
    # Look for output dirs without whizard.log at all (failed jobs).
    for d in root.rglob("b*"):
        if d.is_dir() and not (d / "whizard.log").exists():
            n_missing += 1
    for d in (root / "bfs_table_1", root / "bfs_table_2"):
        if d.is_dir() and not (d / "whizard.log").exists():
            n_missing += 1
    print(f"results.csv written: {n_ok} non-empty, {n_empty} empty (job not yet finished or RESULT missing)")
    print(f"output dirs missing whizard.log: {n_missing}")


if __name__ == "__main__":
    main()
