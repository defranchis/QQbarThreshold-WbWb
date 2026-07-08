#!/usr/bin/env python3
"""Born-level ZZ/NC-overlap cross-check for the hadronic WW channel.

The independent calc builds σ_hadronic = 4·σ(u d̄ s c̄), i.e. it represents ALL
four CKM-favoured hadronic W⁺W⁻ flavour combos by the single mixed-flavour
state.  Two of those four combos are SAME-flavour-pair final states reachable by
BOTH W⁺W⁻ *and* neutral-current ZZ/Zγ/γγ (which interfere):

    u ū d d̄   (W⁺→ud̄,W⁻→ūd   |   Z→uū,Z→dd̄)
    c c̄ s s̄   (W⁺→cs̄,W⁻→c̄s   |   Z→cc̄,Z→ss̄)

MoCaNLO/Recola generates the COMPLETE EW diagram set per final state (only QCD
order is fixed to 0), so running `u u~ d d~` / `c c~ s s~` automatically includes
the ZZ/Zγ/γγ diagrams + their interference with WW, while `u d~ s c~` does not.

The neglected piece is therefore exactly

    Δσ_NC+int  =  ½[σ(uūdd̄) + σ(cc̄ss̄)] − σ(ud̄sc̄)        (per √ŝ, Born)

This is a TREE-level effect, so we run Born only (cheap → high statistics) and
isolate it cleanly.  No POI variation, nominal m_W/Γ_W, scheme gf, no cut.

Runs on fcc-ironic (needs $MOCANLO_BIN via mocanlo_env.sh).  Scratch is
node-local (/tmp); only the small summary CSV is written to AFS.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep.channels import ChannelBlock, BLOCKS_BY_KEY
from framework.process.ww.indep.varpoints import VARPOINTS_BY_KEY
from framework.process.ww.indep.mocanlo_cards import (
    SMInputs, IntegrationSettings, write_cards,
)
from framework.process.ww.indep.parse_xsec import (
    find_latest_result, parse_cross_section_dat,
)

#: The three Born blocks: mixed-flavour reference + the two same-flavour
#: NC-overlap states.  qqqq is the production block (CC-only at Born); uudd/ccss
#: are NEW and carry the ZZ/Zγ/γγ diagrams + WW×ZZ interference.
BLOCKS = {
    "qqqq": BLOCKS_BY_KEY["qqqq"],                       # u d~ s c~  (reference, |W|²)
    "uudd": ChannelBlock("uudd", "u u~ d d~", 0.0, False, "u ū d d̄ (WW+NC)"),
    "ccss": ChannelBlock("ccss", "c c~ s s~", 0.0, False, "c c̄ s s̄ (WW+NC)"),
    # NC-ONLY: no down-type+up-type W pair possible → no WW, pure |N|² (ZZ/Zγ/γγ).
    "uucc": ChannelBlock("uucc", "u u~ c c~", 0.0, False, "u ū c c̄ (NC-only |N|²)"),
    "ddss": ChannelBlock("ddss", "d d~ s s~", 0.0, False, "d d̄ s s̄ (NC-only |N|²)"),
}


def _seed_for(channel: str, ecm: float, base: int) -> int:
    key = f"{channel}|{ecm:.4f}".encode()
    return base + (zlib.crc32(key) % 1_000_000)


def run_born(channel: str, ecm: float, events: int, workdir: str,
             base_seed: int) -> dict:
    """Write cards, run the Born sub-run only, parse σ̂_Born [fb]."""
    block = BLOCKS[channel]
    vp = VARPOINTS_BY_KEY["nominal"]
    sm = SMInputs(scheme_alpha="gf")
    integ = IntegrationSettings(n_target_accepted=events,
                                target_rel_precision_pct=2.0,
                                time_wall="0-08:00")
    tag = f"{channel}_ecm{ecm:.4f}_born"
    procdir = os.path.join(workdir, tag)
    os.makedirs(procdir, exist_ok=True)
    write_cards(procdir, block, vp.mW, vp.gW, ecm, sm, integ)

    mocanlo_bin = os.environ["MOCANLO_BIN"]
    seed = _seed_for(channel, ecm, base_seed) + 1   # run id 1
    t0 = time.time()
    log_path = os.path.join(procdir, "run_1_born.log")
    with open(log_path, "w") as log:
        rc = subprocess.call([mocanlo_bin, procdir, "1", str(seed)],
                             cwd=procdir, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    try:
        path = find_latest_result(procdir, "born")
        sig, err = parse_cross_section_dat(path)
    except Exception as exc:                      # noqa: BLE001
        sig, err = float("nan"), float("nan")
        print(f"  [{tag}] PARSE FAIL rc={rc}: {exc}", file=sys.stderr)
    shutil.rmtree(procdir, ignore_errors=True)
    print(f"  [{tag}] rc={rc} {dt:6.0f}s  σ_born={sig:.4f} ± {err:.4f} fb",
          flush=True)
    return dict(channel=channel, ecm=ecm, sigma_born=sig, err_born=err,
                rc=rc, wall_s=round(dt, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecms", default="157,158,159,160,161,162,163",
                    help="comma list of √ŝ [GeV]")
    ap.add_argument("--channels", default="qqqq,uudd,ccss")
    ap.add_argument("--events", type=int, default=10_000_000,
                    help="n_target_accepted Born events (high → tree precision)")
    ap.add_argument("--jobs", type=int, default=21,
                    help="parallel MoCaNLO workers (≤48 on ironic)")
    ap.add_argument("--base-seed", type=int, default=2000)
    ap.add_argument("--csv-name", default="born_compare.csv",
                    help="output CSV filename")
    ap.add_argument("--workdir", default="/tmp/zz_overlap",
                    help="NODE-LOCAL scratch (never AFS/EOS)")
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args(argv)

    if not os.environ.get("MOCANLO_BIN"):
        print("ERROR: $MOCANLO_BIN unset — source mocanlo_env.sh first",
              file=sys.stderr)
        return 2

    ecms = [float(x) for x in args.ecms.split(",")]
    chans = args.channels.split(",")
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    tasks = [(c, e) for c in chans for e in ecms]
    print(f"[zz_overlap] {len(tasks)} Born points "
          f"({len(chans)} chan × {len(ecms)} √ŝ), events={args.events:,}, "
          f"jobs={args.jobs}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_born, c, e, args.events, args.workdir,
                          args.base_seed) for c, e in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda r: (r["channel"], r["ecm"]))
    out_csv = os.path.join(args.outdir, args.csv_name)
    cols = ["channel", "ecm", "sigma_born", "err_born", "rc", "wall_s"]
    with open(out_csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]}" for c in cols) + "\n")
    print(f"[zz_overlap] wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
