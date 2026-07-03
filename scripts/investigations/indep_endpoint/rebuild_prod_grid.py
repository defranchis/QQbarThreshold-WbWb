#!/usr/bin/env python3
"""Rebuild the PRODUCTION indep-chain eMELA NLL grid with the deep omx edge.

2026-07-03 deep-endpoint fix (mirror of BFS 9a8862b): the omx<1e-15 norm_nll
substitution was removed from isr_beta/isr_lumi, so the grid must carry the
genuine NLL soft drift down to the deep edge (eg.OMX_FLOOR = 1e-70; below it
EmelaGrid.xfxQ continues log-linearly).  Knot pattern otherwise IDENTICAL to
the previous production build (16/decade in omx, Q in [75, 350] x 16 - wide
enough for the mu_F xi variations), provenance meta baked by build_and_write.

Also prints a deep-edge sanity block: the last decades of ln(x*D) vs ln(omx)
must be log-linear (the genuine NLL soft exponent; slope = beta_e - 1 - delta,
delta ~ +4e-4) and every sample positive (EmelaGrid raises otherwise).

Run:  source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH \
      python3 scripts/investigations/indep_endpoint/rebuild_prod_grid.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from framework.process.ww.indep import isr_beta  # noqa: E402
from framework.process.ww.indep import isr_emela_grid as eg  # noqa: E402

PROD = os.path.join(_REPO, "framework/process/ww/indep/grids/"
                           "emela_nll_delta_alpmz.npz")

N_WORKERS = 32          # fork pool; eMELA-under-fork is the proven prewarm pattern


def _sample_chunk(arg):
    """Worker: rows of x*D(x,Q) for a chunk of omx knots (eMELA state inherited
    from the parent's initialize() via fork COW, as in isr_beta.prewarm)."""
    omx_c, q = arg
    from framework.process.ww.xsec_calculator import emela_wrapper as _emela
    table = np.empty((omx_c.size, q.size), dtype=float)
    for i, om in enumerate(omx_c):
        x = 1.0 - om                      # underflows to exactly 1.0 deep in (fine)
        for j, Q in enumerate(q):
            table[i, j] = _emela.code_pdf(x, float(om), float(Q))
    return table


def _sample_parallel(omx, q, n_workers=N_WORKERS):
    """sample_emela_table, fork-parallel over omx rows (identical output)."""
    import multiprocessing as mp
    from framework.process.ww.xsec_calculator import emela_wrapper as _emela
    _emela.initialize(pert_order="NLL", fac_scheme="DELTA", ren_scheme="ALPMZ",
                      alpha=isr_beta.ALPHA_MZ_EMELA)
    chunks = np.array_split(np.arange(omx.size), n_workers * 4)
    chunks = [c for c in chunks if c.size]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        rows = pool.map(_sample_chunk, [(omx[c], q) for c in chunks])
    return np.vstack(rows)


def main():
    omx = eg.default_omx_knots(omx_hi=0.5, per_decade=16)   # omx_lo = OMX_FLOOR
    q = eg.default_q_knots(75.0, 350.0, 16)
    print(f"[rebuild] {omx.size} omx knots [{omx[0]:.1e}, {omx[-1]:.1e}] "
          f"x {q.size} Q knots [{q[0]:.0f}, {q[-1]:.0f}] -> {PROD} "
          f"({N_WORKERS} workers)")
    table = _sample_parallel(omx, q)
    # Same meta build_grid() bakes (the _per_leg_grid_nll/_check_grid_provenance
    # guard REQUIRES alpha/fac_scheme/ren_scheme).
    meta = dict(fac_scheme="DELTA", ren_scheme="ALPMZ",
                alpha=isr_beta.ALPHA_MZ_EMELA, pert_order="NLL",
                omx_lo=float(omx.min()), omx_hi=float(omx.max()),
                q_lo=float(q.min()), q_hi=float(q.max()))
    g = eg.EmelaGrid(omx, q, table, meta)   # raises if any sample <= 0
    g.save_npz(PROD)
    setdir = os.path.splitext(PROD)[0] + "_lha"
    eg.write_lhapdf_set(setdir, g)
    print(f"[rebuild] wrote {PROD} ({os.path.getsize(PROD)/1024:.0f} KiB) + {setdir}")
    print(f"[rebuild] meta: {g.meta}")

    # Deep-edge sanity: log-linearity of ln(x*D) in ln(omx) over the last decades.
    Q = 161.0
    iq = int(np.argmin(np.abs(g.q - Q)))
    ln_omx = np.log(g.omx)
    ln_xd = np.log(g.table[:, iq])
    deep = g.omx < 1e-40
    slope, icpt = np.polyfit(ln_omx[deep], ln_xd[deep], 1)
    resid = ln_xd[deep] - (slope * ln_omx[deep] + icpt)
    be = isr_beta.beta_components(g.q[iq], alpha=isr_beta.ALPHA_MZ_EMELA)[0]
    print(f"[rebuild] deep region (omx<1e-40, Q={g.q[iq]:.1f}): "
          f"slope={slope:.8f}  (beta_e-1={be-1.0:.8f}  ->  delta={be-1.0-slope:+.3e})")
    print(f"[rebuild] log-linearity: max|resid| = {np.max(np.abs(resid)):.2e} "
          f"(must be ~<1e-6 for the edge continuation to be exact)")
    print(f"[rebuild] deep_slope(Q=161) accessor: {g.deep_slope(161.0):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
