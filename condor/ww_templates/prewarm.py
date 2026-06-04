"""Pre-build the eMELA-radiator disk cache for the WW NLL template regen.

The σ̂-independent per-leg eMELA radiator (see
``framework/process/ww/xsec_calculator/isr.py``) is shared by every template:
the whole tag set (mass/width/α_s/cross/pseudodata variations) is σ̂-side and
reuses ONE radiator per (√s, ISR-cfg).  This driver builds that shared set ONCE,
in parallel, so the subsequent generation only ever *reads* it --- whether the
templates are produced on a single multi-core node (``compute_xsec_ww.py`` with
``WW_ISR_NJOBS`` high, the radiators loaded from disk) or fanned out over condor
(this script as a DAG-parent, so cross-node jobs never race to rebuild identical
arrays on the lock-less AFS cache).

Single source of truth: the cfg + grids are derived from the SAME card and
``WWGenerator`` the jobs use, so a drift cannot silently make the prewarm miss.
The radiator fingerprint the generator's ``sigma_observed_munuqq`` call keys on
is (α_isr, ξ, x_min, n_quad, nll, emela_ll, pert/fac/ren scheme); we reproduce
each field from the generator, the two card-default knobs that the generator
does NOT pass through (``z_min=0.30 → x_min=√0.30`` and ``n_quad`` auto-mapped
200→128 on the 2-leg path) are pinned here and re-checked end-to-end by
``--verify``.

Usage:
    PYTHONPATH=$PWD python3 condor/ww_templates/prewarm.py [--n-workers N] [--verify]

Exit non-zero (halting a DAG before the fan-out) if the disk cache is disabled
or the prewarm did not build/find every file it queued.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator, _build_fine_grid
from framework.process.ww.xsec_calculator import isr

#: sigma_observed_munuqq defaults the generator does NOT override: z_min=0.30
#: (→ x_min=√0.30) and n_quad=200 which the 2-leg path auto-maps to 128.  Pinned
#: here; --verify proves they still match the live generator end-to-end.
_N_QUAD_2LEG = 128


def _radiator_cfgs_and_grids():
    """Return (cfgs, grids, gen) byte-matching what the condor jobs look up.

    grids: _build_fine_grid() shifted by the SAME BEC steps submit.py fans out.
    cfgs:  the nominal radiator cfg (all fields from the generator) plus, if any
           PARAMETERS tag carries a non-zero aem_isr offset (the always-on ISR-α
           nuisance), the α-shifted cfg --- because alpha_em_isr IS in the
           radiator fingerprint.
    """
    gen = WWGenerator.from_card(card)

    # BEC shifts: identical to submit.py / compute_xsec_ww.py (single source).
    bec = float(card.INPUT_VAR["BEC"])
    shifts_MeV = [0.0, +bec, -bec]
    base_grid = _build_fine_grid()
    grids = [base_grid + s * 1e-3 for s in shifts_MeV]      # do_scan: +ecm_shift_MeV*1e-3

    base = isr.radiator_cfg(
        x_min=isr._X_MIN_2LEG_DEFAULT, n_quad=_N_QUAD_2LEG,
        nll=gen.isr_nll, emela_ll=gen.isr_emela_ll,
        emela_pert_order=gen.isr_emela_pert_order,
        emela_fac_scheme=gen.isr_emela_fac_scheme,
        emela_ren_scheme=gen.isr_emela_ren_scheme,
        isr_scale_factor=gen.isr_scale_factor,
        alpha_em_isr=gen.alpha_em_isr,
    )
    cfgs = [base]

    # aem_isr is an OFFSET from the nominal ISR α; the aem_isr_var tag shifts it
    # → a distinct radiator.  Read the offsets from the card so they track it.
    params = Parameters(card.PARAMETERS, scale_vars=[],
                        cross_terms=getattr(card, "CROSS_TERMS", ()))
    offsets = sorted({float(params.values(t).get("aem_isr", 0.0)) for t in params.tags}
                     - {0.0})
    if offsets and gen.alpha_em_isr is None:
        raise SystemExit(
            "aem_isr offset present in the card but generator.alpha_em_isr is None: "
            "do_scan would raise. Set PARAM_INPUTS['alpha_em_isr'] in the card.")
    for off in offsets:
        cfgs.append({**base, "alpha_em_isr": gen.alpha_em_isr + off})

    return cfgs, grids, gen


def _verify_end_to_end(gen, params, n_workers):
    """Confirm the prewarm actually matches a REAL generator template lookup:
    count rad_bfs_* files, run one full do_scan (nominal tag, shift 0) which
    loads its 152 radiators, and assert NO new radiator file was written.  This
    is the drift catch for the two pinned knobs (x_min/n_quad) — if the
    generator's σ_obs call changed them, do_scan would build files outside the
    prewarmed set and the count would grow."""
    import tempfile
    cache_dir = isr._radiator_cache_dir()
    if not cache_dir or not os.path.isdir(cache_dir):
        print("[verify] no cache dir — skipped"); return True

    def _n_pkls():
        return sum(1 for f in os.listdir(cache_dir) if f.startswith("rad_bfs_"))

    isr._RADIATOR_CACHE.clear()              # force the disk path, not the warm L1
    before = _n_pkls()
    scales = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0})
    with tempfile.TemporaryDirectory(prefix="ww_prewarm_verify_") as td:
        gen.do_scan(params.values("nominal"),
                    mass_scale=scales["mass"], width_scale=scales["width"],
                    mass_scheme=getattr(card, "MASS_SCHEME", "OS"),
                    outdir=td, ecm_shift_MeV=0.0)
    after = _n_pkls()
    ok = (after == before)
    print(f"[verify] real do_scan(nominal, shift 0): rad_bfs_* {before} → {after} "
          f"({'all loaded from cache — prewarm matches the jobs' if ok else 'NEW FILES BUILT — prewarm cfg/grid DRIFTED from the generator'})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-workers", type=int, default=int(os.environ.get("WW_PREWARM_NWORKERS", "8")),
                    help="parallel build processes (default 8 / $WW_PREWARM_NWORKERS; "
                         "match the slot's request_cpus, cap 48 on ironic)")
    ap.add_argument("--verify", action="store_true",
                    help="after prewarming, run one real do_scan and assert it loads "
                         "the cache (catches an x_min/n_quad drift from the generator)")
    args = ap.parse_args()

    cfgs, grids, gen = _radiator_cfgs_and_grids()

    # Guard: only the eMELA paths disk-cache.  Analytic LL+exp has nothing to do.
    if not (gen.isr_nll or gen.isr_emela_ll):
        print("[prewarm] ISR is analytic LL+exp (no eMELA) — nothing to prewarm.")
        return 0

    npts = sum(len(g) for g in grids)
    print(f"[prewarm] {len(grids)} √s grids × {len(cfgs)} radiator cfg(s) "
          f"= {npts}×{len(cfgs)} candidate (√s,cfg); n_workers={args.n_workers}")
    print(f"[prewarm] cfgs: " + "; ".join(
        f"α_isr={c['alpha_em_isr']!r} {c['emela_fac_scheme']}/{c['emela_ren_scheme']} "
        f"nll={c['nll']} ll={c['emela_ll']} nq={c['n_quad']}" for c in cfgs))

    rep = isr.prewarm(grids, cfgs, n_workers=args.n_workers, verbose=True)

    # Fail loud (halts a DAG before the fan-out) on a disabled cache or an
    # incomplete build — either means the fan-out would stampede anyway.
    if rep["no_disk"]:
        print("[prewarm] ERROR: disk cache disabled ($WW_ISR_RADIATOR_CACHE empty) "
              "— prewarm is pointless; the jobs would each rebuild in memory.",
              file=sys.stderr)
        return 2
    if rep["built"] + rep["exists"] != rep["n_unique"]:
        print(f"[prewarm] ERROR: incomplete — built+exists="
              f"{rep['built']+rep['exists']} != n_unique={rep['n_unique']}.",
              file=sys.stderr)
        return 3

    if args.verify:
        params = Parameters(card.PARAMETERS, scale_vars=[],
                            cross_terms=getattr(card, "CROSS_TERMS", ()))
        if not _verify_end_to_end(gen, params, args.n_workers):
            return 4

    print(f"[prewarm] OK: {rep['n_unique']} radiators ready in {rep['cache_dir']} "
          f"(built {rep['built']}, reused {rep['exists']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
