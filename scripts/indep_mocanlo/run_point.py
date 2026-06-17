#!/usr/bin/env python3
"""Generate one independent-WW partonic grid point with MoCaNLO.

Writes the 5 cards, runs the 4 NLO sub-runs (born/virt/real/idip) at a given
seed, parses the cross sections, and writes a one-line result CSV.  Designed to
be the unit of work for a condor job (one point per job) or an ironic test.

Requires ``mocanlo_env.sh`` to have been sourced (sets ``$MOCANLO_BIN``).

Usage:
  run_point.py CHANNEL VARPOINT ECM [options]

  CHANNEL   one of: enuqq lnuqq qqqq enuenu emu lnulnu
  VARPOINT  one of: nominal massUp massDn widthUp widthDn cross
  ECM       partonic √ŝ in GeV (= cms_beam_energy, since pdf_set=none)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zlib

# repo root on sys.path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep.channels import BLOCKS_BY_KEY
from framework.process.ww.indep.varpoints import VARPOINTS_BY_KEY
from framework.process.ww.indep.mocanlo_cards import (
    SMInputs, IntegrationSettings, write_cards,
)
from framework.process.ww.indep.parse_xsec import read_point

_RUN_TYPES = (("1", "born"), ("2", "virt"), ("3", "real"), ("4", "idip"))


def _seed_for(channel: str, ecm: float, base: int) -> int:
    """Correlated-sampling seed: depends on (channel, ecm) but NOT the varpoint,
    so nominal and every POI variation at the same √ŝ draw the same MC sequence
    → the statistical fluctuation cancels in the morph differences."""
    key = f"{channel}|{ecm:.4f}".encode()
    return base + (zlib.crc32(key) % 1_000_000)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channel", choices=sorted(BLOCKS_BY_KEY))
    ap.add_argument("varpoint", choices=sorted(VARPOINTS_BY_KEY))
    ap.add_argument("ecm", type=float, help="partonic √ŝ [GeV]")
    ap.add_argument("--scheme-alpha", default="gf",
                    choices=["gf", "alpha0", "alphaz", "alphamsbar"])
    ap.add_argument("--m-t", type=float, default=None,
                    help="override top mass [GeV] (default: SMInputs 174.2)")
    ap.add_argument("--m-h", type=float, default=None,
                    help="override Higgs mass [GeV] (default: SMInputs 125.0)")
    ap.add_argument("--m-z", type=float, default=None,
                    help="override Z mass [GeV] (default: SMInputs 91.188)")
    ap.add_argument("--fermi-constant", type=float, default=None,
                    help="override Fermi constant G_F [GeV^-2] (default: SMInputs "
                         "1.1663787e-5). In the 'gf' scheme this sets the derived "
                         "α_Gμ=√2 G_F M_W² sw²/π — used by the G_F-consistency test "
                         "to pin α_Gμ to α(M_Z) (G_F=1.1957623e-5 → 1/128.936).")
    ap.add_argument("--precision-pct", type=float, default=0.5,
                    help="MoCaNLO target_relative_precision per run [%%]")
    ap.add_argument("--events", type=int, default=200000,
                    help="n_target_accepted_events per run")
    ap.add_argument("--time-wall", default="0-02:00", help="per-run wall cap D-HH:MM")
    ap.add_argument("--workdir", default=None,
                    help="MoCaNLO scratch dir (default: NODE-LOCAL "
                         "$_CONDOR_SCRATCH_DIR/$TMPDIR/tmp — never shared AFS/EOS; "
                         "deleted after the run unless --keep-rundir)")
    ap.add_argument("--outdir",
                    default="/eos/user/m/mdefranc/FCC/QQbar_threshold/"
                            "grid_gen/results",
                    help="result CSV dir (default: EOS — small, append-only)")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--decorrelate", action="store_true",
                    help="A/B diagnostic: include the varpoint in the seed (old "
                         "behaviour) so varpoints are NOT correlated; tags output "
                         "'_decorr'. Default is correlated seeds (varpoint-independent).")
    ap.add_argument("--lepton-cut", type=float, default=None,
                    help="fiducial charged-lepton |cosθ|<COS acceptance cut "
                         "(e.g. 0.95); omit for inclusive no-cut")
    ap.add_argument("--lepton-pt-min", type=float, default=None,
                    help="fiducial charged-lepton p_T>PT GeV detector-floor cut "
                         "(e.g. 10)")
    ap.add_argument("--lepton-mll-min", type=float, default=None,
                    help="fiducial m_ℓℓ>MLL GeV cut on same-flavour OS pairs "
                         "(e.g. 10); the physical tool for the γ*→ℓℓ NC pole")
    ap.add_argument("--keep-rundir", action="store_true",
                    help="keep the MoCaNLO scratch dir (default: DELETE it after "
                         "writing the result CSV — only results/*.csv is needed)")
    args = ap.parse_args(argv)

    mocanlo_bin = os.environ.get("MOCANLO_BIN")
    if not mocanlo_bin or not os.path.exists(mocanlo_bin):
        print("ERROR: $MOCANLO_BIN unset/missing — source mocanlo_env.sh first",
              file=sys.stderr)
        return 2

    block = BLOCKS_BY_KEY[args.channel]
    vp = VARPOINTS_BY_KEY[args.varpoint]
    cut_tag = "" if args.lepton_cut is None else f"_cut{int(round(args.lepton_cut*100))}"
    if args.lepton_pt_min is not None:
        cut_tag += f"pt{int(round(args.lepton_pt_min))}"
    if args.lepton_mll_min is not None:
        cut_tag += f"mll{int(round(args.lepton_mll_min))}"
    if args.decorrelate:
        cut_tag += "_decorr"
    tag = f"{args.channel}_{args.varpoint}_ecm{args.ecm:.4f}_{args.scheme_alpha}{cut_tag}"
    # Node-local scratch by default (NOT shared AFS/EOS): MoCaNLO does heavy
    # many-small-file random I/O, which only belongs on fast local disk and must
    # never accumulate on AFS (it filled the work volume) or thrash EOS.
    workdir = (args.workdir or os.environ.get("_CONDOR_SCRATCH_DIR")
               or os.environ.get("TMPDIR") or "/tmp")
    procdir = os.path.join(workdir, tag)
    os.makedirs(procdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    sm_over = {k: v for k, v in (("mt", args.m_t), ("mH", args.m_h),
                                 ("mZ", args.m_z),
                                 ("fermi_constant", args.fermi_constant))
               if v is not None}
    sm = SMInputs(scheme_alpha=args.scheme_alpha, **sm_over)
    integ = IntegrationSettings(n_target_accepted=args.events,
                                target_rel_precision_pct=args.precision_pct,
                                time_wall=args.time_wall)
    write_cards(procdir, block, vp.mW, vp.gW, args.ecm, sm, integ,
                cos_theta_max=args.lepton_cut, pt_min=args.lepton_pt_min,
                mll_min=args.lepton_mll_min)

    if args.decorrelate:
        seed0 = args.base_seed + (zlib.crc32(
            f"{args.channel}|{args.varpoint}|{args.ecm:.4f}".encode()) % 1_000_000)
    else:
        seed0 = _seed_for(args.channel, args.ecm, args.base_seed)
    print(f"[{tag}] mW={vp.mW} GW={vp.gW} ecm={args.ecm} seed0={seed0}")

    t_start = time.time()
    for rid, rtype in _RUN_TYPES:
        seed = seed0 + int(rid)
        t0 = time.time()
        log_path = os.path.join(procdir, f"run_{rid}_{rtype}.log")
        with open(log_path, "w") as log:
            rc = subprocess.call([mocanlo_bin, procdir, rid, str(seed)],
                                 cwd=procdir, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        print(f"  run {rid} {rtype:5s}: rc={rc}  {dt:6.1f}s")
        if rc != 0:
            print(f"  WARNING: mocanlo rc={rc} for {rtype} (see {log_path})")

    res = read_point(procdir)
    res.update(channel=args.channel, varpoint=args.varpoint, ecm=args.ecm,
               mW=vp.mW, gW=vp.gW, scheme_alpha=args.scheme_alpha,
               lepton_cut=("none" if args.lepton_cut is None else args.lepton_cut),
               wall_s=round(time.time() - t_start, 1))

    out_csv = os.path.join(args.outdir, f"{tag}.csv")
    cols = ["channel", "varpoint", "ecm", "mW", "gW", "scheme_alpha", "lepton_cut",
            "sigma_born", "err_born", "sigma_virt", "err_virt",
            "sigma_real", "err_real", "sigma_idip", "err_idip",
            "sigma_nlo", "err_nlo", "wall_s"]
    with open(out_csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        fh.write(",".join(f"{res[c]}" for c in cols) + "\n")

    print(f"  σ̂_Born = {res['sigma_born']:.6f} ± {res['err_born']:.6f} fb")
    print(f"  σ̂_NLO  = {res['sigma_nlo']:.6f} ± {res['err_nlo']:.6f} fb")
    print(f"  wrote {out_csv}")

    # Drop the (node-local) scratch now that the result CSV is safely written —
    # only results/*.csv is consumed downstream (partonic_grid.load_grids).
    if not args.keep_rundir:
        shutil.rmtree(procdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
