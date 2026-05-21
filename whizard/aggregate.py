#!/usr/bin/env python3
"""Aggregate per-job results.csv from the WHIZARD condor campaigns into
sorted grid.csv + provenance.json files, one per campaign.

Campaigns (sources → outputs):
    condor/grid              → work/grid/{grid.csv,provenance.json}
                               1295 pts = 37 √s × 5 m_W × 7 Γ_W
    condor/grid_highstats    → work/grid_highstats/{grid.csv,provenance.json}
                               2331 pts = 37 √s × 9 m_W × 7 Γ_W

Channel: σ(e+e- → μ⁻ν̄_μ ud̄) in fb (BFS specific 4f). Multiply by 27 for
the all-flavour sum (matches BFS Tables 1+2 convention).

Usage:
    python3 whizard/aggregate.py                    # both, fail if non-rectangular
    python3 whizard/aggregate.py --mode highstats   # only the highstats campaign
    python3 whizard/aggregate.py --allow-gaps       # warn instead of raise
"""

import argparse
import datetime as dt
import getpass
import json
import platform
import subprocess
from pathlib import Path

WHIZARD_TOP = Path("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard")
CONDOR_DIR  = WHIZARD_TOP / "work" / "condor"
WORK_DIR    = WHIZARD_TOP / "work"

# Campaign specs: (name, condor_subdir, output_subdir, expected_sqrts_per_pair,
#                  iter_spec). expected_n_pairs is the cross-product count of
#                  the m_W × Γ_W axes — used for the gap report.
CAMPAIGNS = {
    "grid": {
        "condor_subdir": "grid",
        "output_subdir": "grid",
        "expected_n_sqrts_per_pair": 37,
        "expected_n_pairs": 5 * 7,
        "iter_spec": '6:100000:"gw",3:300000',
    },
    "grid_highstats": {
        "condor_subdir": "grid_highstats",
        "output_subdir": "grid_highstats",
        "expected_n_sqrts_per_pair": 37,
        "expected_n_pairs": 9 * 7,
        "iter_spec": '6:500000:"gw",5:5000000',
    },
    # 12 extra √s pts at 0.25 GeV step in [157.25, 162.75] for every (m_W,
    # Γ_W) pair on the highstats axis. Used to tighten the cubic-spline-
    # along-√s of σ_nom and β at the peak rise.
    "grid_highstats_densify": {
        "condor_subdir": "grid_highstats_densify",
        "output_subdir": "grid_highstats_densify",
        "expected_n_sqrts_per_pair": 12,
        "expected_n_pairs": 9 * 7,
        "iter_spec": '6:500000:"gw",5:5000000',
    },
    # 1-MeV-step (m_W, Γ_W) plane at 3 √s slices for sub-step morph
    # validation. Held out from any morph fit.
    "grid_validate": {
        "condor_subdir": "grid_validate",
        "output_subdir": "grid_validate",
        "expected_n_sqrts_per_pair": 3,
        "expected_n_pairs": 5 * 5,
        "iter_spec": '6:500000:"gw",5:5000000',
    },
    # Fine √s grid: 0.1 GeV step in [155, 165], full 9 m_W × 7 Γ_W plane,
    # at 4× highstats MC (~0.008%). Operational successor to the 0.5 GeV
    # highstats + 0.25 GeV densify grids in the analysis window.
    "grid_fine": {
        "condor_subdir": "grid_fine",
        "output_subdir": "grid_fine",
        "expected_n_sqrts_per_pair": 101,
        "expected_n_pairs": 9 * 7,
        "iter_spec": '6:500000:"gw",5:20000000',
    },
    # Fine (m_W, Γ_W) 1D scans (0.1-MeV steps to ±1 MeV, 0.2-MeV to
    # ±3 MeV) at 3 √s, ~0.005% MC. Held out — bounds the morph's sub-MeV
    # interpolation bias. 81 1D-scan points, not a rectangular plane.
    "grid_validate_fine": {
        "condor_subdir": "grid_validate_fine",
        "output_subdir": "grid_validate_fine",
        "expected_n_sqrts_per_pair": 3,
        "expected_n_pairs": 81,
        "iter_spec": '6:500000:"gw",5:50000000',
    },
}


def parse_csv(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed row in {path}: {line!r}")
            rows.append(tuple(float(p) for p in parts))
    return rows


def whizard_version():
    """Run `whizard --version` after sourcing setup.sh so LD_LIBRARY_PATH is set."""
    out = subprocess.run(
        ["bash", "-c", f"source {WHIZARD_TOP / 'setup.sh'} && whizard --version 2>&1 | head -1"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() or "unknown"


def aggregate_one(name: str, spec: dict, *, allow_gaps: bool) -> None:
    src = CONDOR_DIR / spec["condor_subdir"]
    if not src.exists():
        print(f"[{name}] source tree {src} not present — skipping.")
        return

    rows = []
    for r in sorted(src.rglob("results.csv")):
        rows.extend(parse_csv(r))
    if not rows:
        print(f"[{name}] no rows found in {src} — skipping.")
        return

    rows.sort(key=lambda t: (t[3], t[4], t[0]))   # (mW, gW, sqrts)

    # Rectangular check.
    pair_counts = {}
    for sqrts, sigma, err, mw, gw in rows:
        pair_counts[(mw, gw)] = pair_counts.get((mw, gw), 0) + 1
    n_per_pair = spec["expected_n_sqrts_per_pair"]
    short = {k: v for k, v in pair_counts.items() if v != n_per_pair}
    n_pairs = len(pair_counts)
    expected_pairs = spec["expected_n_pairs"]
    msg_parts = []
    if short:
        msg_parts.append(f"{len(short)} (m_W, Γ_W) pairs are short of {n_per_pair} √s rows")
    if n_pairs != expected_pairs:
        msg_parts.append(f"found {n_pairs} (m_W, Γ_W) pairs, expected {expected_pairs}")
    if msg_parts:
        msg = f"[{name}] grid is not rectangular — " + "; ".join(msg_parts)
        if allow_gaps:
            print(f"WARNING: {msg}")
            for k, v in sorted(short.items()):
                print(f"  short pair m_W={k[0]:.5f}, Γ_W={k[1]:.5f}: {v}/{n_per_pair} √s")
        else:
            raise RuntimeError(msg + " (pass --allow-gaps to write anyway)")

    out_dir = WORK_DIR / spec["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = out_dir / "grid.csv"
    with grid_csv.open("w") as f:
        f.write("# sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV\n")
        f.write("# Channel: e+ e- -> mu- nu_mu_bar u d_bar (specific 4f, BFS reference)\n")
        f.write("# Multiply sigma by 27 for the all-flavour 4f sum (BFS Tables 1+2 convention)\n")
        for r in rows:
            f.write(" ".join(f"{x:g}" for x in r) + "\n")

    sw_sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                             "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    sqrts_axis = sorted({r[0] for r in rows})
    mW_axis    = sorted({r[3] for r in rows})
    gW_axis    = sorted({r[4] for r in rows})
    provenance = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generated_by": getpass.getuser(),
        "host": platform.node(),
        "whizard_version": whizard_version(),
        "channel": "e+ e- -> mu- nu_mu_bar u d_bar (specific 4f)",
        "model_params": {
            "M_Z": 91.188, "m_t": 174.2, "M_H": 115,
            "G_F": 1.16637e-5,
        },
        "grid_axes": {
            "sqrts_GeV": sqrts_axis,
            "mW_GeV":    mW_axis,
            "gammaW_GeV":gW_axis,
        },
        "n_points":          len(rows),
        "n_points_expected": expected_pairs * n_per_pair,
        "missing_pairs": [
            {"mW": k[0], "gammaW": k[1], "n_sqrts": v}
            for k, v in sorted(short.items())
        ],
        "iter_spec": spec["iter_spec"],
        "ww_threshold_sha": sw_sha,
        "source_dir": str(src),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"[{name}] wrote {grid_csv} ({len(rows)} points, "
          f"axes {len(sqrts_axis)} √s × {len(mW_axis)} m_W × {len(gW_axis)} Γ_W)")
    print(f"[{name}] wrote {out_dir / 'provenance.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   choices=["grid", "grid_highstats", "grid_highstats_densify",
                            "grid_validate", "all"],
                   default="all",
                   help="Which campaign to aggregate (default: all available)")
    p.add_argument("--allow-gaps", action="store_true",
                   help="Warn (don't raise) when (m_W, Γ_W) pairs are short of "
                        "their expected √s rows. Use for partial campaigns "
                        "still missing some condor jobs.")
    args = p.parse_args()

    names = list(CAMPAIGNS) if args.mode == "all" else [args.mode]
    for name in names:
        aggregate_one(name, CAMPAIGNS[name], allow_gaps=args.allow_gaps)


if __name__ == "__main__":
    main()
