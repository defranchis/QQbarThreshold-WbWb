"""Aggregate the grid_lowext WHIZARD campaign into grid.csv (grid_fine format).

Writes <campaign>/grid.csv with the same header/columns as grid_fine/grid.csv
(sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV) plus a provenance.json, so
grid_morph.load_operational_grid can splice the sub-154 slices in front of the
wings.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

CAMPAIGN = "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard/work/grid_lowext"

rows = []
for d in sorted(os.listdir(CAMPAIGN)):
    if not d.startswith("p_"):
        continue
    sq, mw, gw = (float(x) for x in d[2:].split("_"))
    log = os.path.join(CAMPAIGN, d, "run.log")
    text = open(log, errors="replace").read()
    integ = re.findall(r"integral\(mnuqq\) =\s*([0-9.Ee+-]+)", text)
    err = re.findall(r"error\(mnuqq\) =\s*([0-9.Ee+-]+)", text)
    code = open(os.path.join(CAMPAIGN, d, "exit.code")).read().strip()
    if code != "0" or not integ:
        sys.exit(f"FAILED point {d} (exit {code})")
    rows.append((sq, float(integ[-1]), float(err[-1]), mw, gw))

rows.sort()
out = os.path.join(CAMPAIGN, "grid.csv")
with open(out, "w") as fh:
    fh.write("# sqrts_GeV sigma_fb err_fb mW_GeV gammaW_GeV\n")
    fh.write("# Channel: e+ e- -> mu- nu_mu_bar u d_bar (specific 4f, BFS reference)\n")
    fh.write("# Multiply sigma by 27 for the all-flavour 4f sum (BFS Tables 1+2 convention)\n")
    for r in rows:
        fh.write(f"{r[0]:.1f} {r[1]:.6e} {r[2]:.3e} {r[3]:.3f} {r[4]:.5f}\n")

sq = sorted({r[0] for r in rows})
prov = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "generated_by": "mdefranc",
    "purpose": "sub-154 GeV extension of the morph anchor grid (2026-07-02 "
               "morph-edge fix follow-up); uniform-gammaW sublattice only",
    "whizard_version": "WHIZARD 3.1.5",
    "channel": "e+ e- -> mu- nu_mu_bar u d_bar (specific 4f)",
    "model_params": {"M_Z": 91.188, "m_t": 174.2, "M_H": 115, "G_F": 1.16637e-05},
    "grid_axes": {"sqrts_GeV": sq,
                  "mW_GeV": sorted({r[3] for r in rows}),
                  "gammaW_GeV": sorted({r[4] for r in rows})},
    "n_points": len(rows),
    "integration": "iterations 5:30000 + 3:60000 (lighter than grid_fine; "
                    "sub-threshold anchor precision demand is <1e-3)",
}
with open(os.path.join(CAMPAIGN, "provenance.json"), "w") as fh:
    json.dump(prov, fh, indent=2)

print(f"wrote {out}: {len(rows)} rows, sqrts {sq[0]}-{sq[-1]}")
