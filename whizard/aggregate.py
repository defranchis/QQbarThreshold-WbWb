#!/usr/bin/env python3
"""Aggregate the per-job results.csv from the Phase-3 WHIZARD grid into a
single sorted grid.csv + provenance.json next to it.

Output: ../../whizard/work/grid/{grid.csv,provenance.json}

The grid is the channel-specific 4f Born cross section
σ(e+e- → μ⁻ν̄_μ ud̄)  in fb, vs (√s, m_W, Γ_W). Multiply by 27 for the
all-flavour sum (matches BFS Tables 1+2 convention).
"""

import datetime as dt
import getpass
import json
import platform
import subprocess
from pathlib import Path

WHIZARD_TOP = Path("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard")
CONDOR_DIR  = WHIZARD_TOP / "work" / "condor"
GRID_DIR    = WHIZARD_TOP / "work" / "grid"


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


def main():
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for r in sorted((CONDOR_DIR / "grid").rglob("results.csv")):
        all_rows.extend(parse_csv(r))

    # Sort by (m_W, Γ_W, √s) so the file is readable + an interpolator can
    # treat it as a 3D regular grid.
    all_rows.sort(key=lambda t: (t[3], t[4], t[0]))   # (mW, gW, sqrts)

    # Sanity check: every (m_W, Γ_W) pair has 37 √s rows.
    pair_counts = {}
    for sqrts, sigma, err, mw, gw in all_rows:
        pair_counts[(mw, gw)] = pair_counts.get((mw, gw), 0) + 1
    bad = {k: v for k, v in pair_counts.items() if v != 37}
    if bad:
        raise RuntimeError(f"Grid is not rectangular — short pairs: {bad}")

    # Write grid.csv (one line per point).
    grid_csv = GRID_DIR / "grid.csv"
    with grid_csv.open("w") as f:
        f.write("# sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV\n")
        f.write("# Channel: e+ e- -> mu- nu_mu_bar u d_bar (specific 4f, BFS reference)\n")
        f.write("# Multiply sigma by 27 for the all-flavour 4f sum (BFS Tables 1+2 convention)\n")
        for r in all_rows:
            f.write(" ".join(f"{x:g}" for x in r) + "\n")

    # Provenance.
    sw_sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                             "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
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
            "sqrts_GeV": sorted({r[0] for r in all_rows}),
            "mW_GeV":    sorted({r[3] for r in all_rows}),
            "gammaW_GeV":sorted({r[4] for r in all_rows}),
        },
        "n_points": len(all_rows),
        "iter_spec": '6:100000:"gw",3:300000',
        "ww_threshold_sha": sw_sha,
        "source_dir": str(CONDOR_DIR / "grid"),
    }
    (GRID_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"Wrote {grid_csv} ({len(all_rows)} points)")
    print(f"Wrote {GRID_DIR / 'provenance.json'}")
    print(f"Grid axes: "
          f"{len(provenance['grid_axes']['sqrts_GeV'])} √s × "
          f"{len(provenance['grid_axes']['mW_GeV'])} m_W × "
          f"{len(provenance['grid_axes']['gammaW_GeV'])} Γ_W")


if __name__ == "__main__":
    main()
