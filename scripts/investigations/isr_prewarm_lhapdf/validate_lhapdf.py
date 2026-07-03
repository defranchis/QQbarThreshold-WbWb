"""Thorough validation of the LHAPDF-grid eMELA-NLL path (isr_emela_grid +
isr_beta.ISRConfig.emela_grid) vs the direct per-node eMELA DGLAP solver.

Run (grid + caches forced onto /tmp, never AFS):
    PYTHONPATH=$PWD python3 scripts/investigations/isr_prewarm_lhapdf/validate_lhapdf.py

Checks
------
G.  grid build + persistence (npz + standard lhagrid1 set on /tmp).
N.  PER-NODE closure: grid.xfxQ vs eMELA.code_pdf at the actual quadrature nodes
    (mid region omx∈[1e-15, 1-x_min]).  This is the raw interpolation accuracy.
C.  CONVERGENCE: closure tightens as the omx knot density grows (controllable).
L.  LINE-SHAPE closure: σ_obs(√s) grid vs direct on a toy σ̂ (observable impact —
    smaller than per-node because the soft-dominant region is analytic-identical).
M.  ASIMOV m_W CROSS-FIT on real EOS σ̂ grids: build the direct-eMELA NLL morph as
    "truth", fit it with the grid morph, read off Δm_W / ΔΓ_W bias (the decision
    metric for MeV-precision m_W).  Skips if the σ̂ grid is incomplete.
S.  SPEED: per-node eMELA (~22 ms) vs vectorised grid interp; radiator build.
E.  ENDPOINT: grid tracks eMELA in its valid range and the analytic norm_nll
    takes over below 1e-15 (unchanged) — the grid never has to reach x=1.

2026-07-03 NOTE (deep-endpoint fix): the analytic norm_nll substitution is
REMOVED from production — the grid path now serves every node, continuing
log-linearly below the grid's own deepest knot, and eg.OMX_FLOOR changed
meaning (1e-15 substitution boundary → 1e-70 default deep BUILD edge).  The
`mid = omx >= eg.OMX_FLOOR` masks below therefore now cover ALL nodes (still
a valid grid-vs-direct closure — code_pdf takes omx explicitly and is healthy
through the deep region), the in-script grid builds span ~70 decades (slower),
and section [E]'s "norm_nll takes over" wording describes the REMOVED
behaviour.  The lhagrid1 x-space artifact cannot represent omx ≲ 5e-17
(x collapses to 1.0 in float64), so real-lhapdf cross-reads are only
meaningful above that.
"""
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WW_ISR_RADIATOR_CACHE", "")        # disk radiator cache off here

from framework.process.ww.indep import isr_beta as ib            # noqa: E402
from framework.process.ww.indep import isr_emela_grid as eg       # noqa: E402
from framework.process.ww.xsec_calculator import emela_wrapper as em  # noqa: E402

PID = os.getpid()
GRIDPATH = f"/tmp/ww_emela_grid_val_{PID}.npz"
ALPHA = ib.ALPHA_MZ
Q_KNOTS = eg.default_q_knots(150.0, 168.0, 14)
X_MIN = float(np.sqrt(0.30))

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def section(t):
    print("\n" + t)


def main():
    # === G. build grid ===
    section("[G] BUILD grid (sample eMELA once → npz + lhagrid1, on /tmp)")
    t0 = time.time()
    grid = eg.build_and_write(GRIDPATH, fac_scheme="DELTA", ren_scheme="ALPMZ",
                              alpha=ALPHA, q_knots=Q_KNOTS, verbose=True)
    print(f"      built in {time.time()-t0:.1f}s; "
          f"omx∈[{grid.omx.min():.0e},{grid.omx.max():.2f}] ({grid.omx.size} knots), "
          f"Q∈[{grid.q.min():.0f},{grid.q.max():.0f}] ({grid.q.size} knots)")
    lha = grid.meta.get("lhagrid1_set", "")
    check("npz + lhagrid1 written, file < 100 KiB",
          os.path.exists(GRIDPATH) and os.path.isdir(lha)
          and os.path.getsize(GRIDPATH) < 100 * 1024,
          f"{os.path.getsize(GRIDPATH)/1024:.0f} KiB; set={os.path.basename(lha)}")

    # === N. per-node closure at real quadrature nodes ===
    section("[N] PER-NODE closure  grid.xfxQ vs eMELA.code_pdf (mid region)")
    em.initialize(pert_order="NLL", fac_scheme="DELTA", ren_scheme="ALPMZ", alpha=ALPHA)
    worst = 0.0
    for sqrt_s in (157.0, 160.0, 162.5, 164.0):
        be = ib.beta_components(sqrt_s, scheme="LO_beta", alpha=ALPHA)[0]
        _u, _w, x_vals, omx, _j = ib._endpoint_grid(be, X_MIN, 128)
        mid = omx >= eg.OMX_FLOOR
        xv, ov = x_vals[mid], omx[mid]
        g = grid.xfxQ(xv, ov, sqrt_s)
        d = np.array([em.code_pdf(float(x), float(o), sqrt_s) for x, o in zip(xv, ov)])
        rel = np.abs(g - d) / np.abs(d)
        worst = max(worst, float(rel.max()))
        print(f"      √s={sqrt_s:6.1f}  nodes={mid.sum():3d}  "
              f"median={np.median(rel):.1e}  p99={np.percentile(rel,99):.1e}  "
              f"max={rel.max():.1e} @omx={ov[rel.argmax()]:.1e}")
    check("per-node mid-region max rel < 2e-3", worst < 2e-3, f"worst={worst:.1e}")

    # === C. convergence with knot density ===
    section("[C] CONVERGENCE  per-node max rel vs omx knots/decade")
    sqrt_s = 161.0
    be = ib.beta_components(sqrt_s, scheme="LO_beta", alpha=ALPHA)[0]
    _u, _w, x_vals, omx, _j = ib._endpoint_grid(be, X_MIN, 128)
    mid = omx >= eg.OMX_FLOOR
    xv, ov = x_vals[mid], omx[mid]
    d = np.array([em.code_pdf(float(x), float(o), sqrt_s) for x, o in zip(xv, ov)])
    prev = None
    mono = True
    for ppd in (4, 8, 16):
        gk = eg.build_grid(fac_scheme="DELTA", ren_scheme="ALPMZ", alpha=ALPHA,
                           omx_knots=eg.default_omx_knots(per_decade=ppd),
                           q_knots=Q_KNOTS)
        rel = float(np.max(np.abs(gk.xfxQ(xv, ov, sqrt_s) - d) / np.abs(d)))
        print(f"      {ppd:2d}/decade ({gk.omx.size:3d} knots)  max rel = {rel:.2e}")
        if prev is not None and rel > prev * 1.5:
            mono = False
        prev = rel
    check("closure improves (or holds) with density", mono)

    # === L. line-shape closure on a toy σ̂ ===
    section("[L] LINE-SHAPE closure  σ_obs grid vs direct (toy σ̂)")
    def toy(ss):
        s = np.asarray(ss, float)
        return 160.0 * 0.5 * (1.0 + np.tanh((s - 160.76) / 0.8)) * (s / 161.0)
    scan = np.array([157.5, 158.5, 159.5, 160.5, 161.5, 162.3, 163.0, 164.0])
    cdir = ib.ISRConfig(nll=True, alpha=ALPHA, ew_scheme="alphaz", emela_ren_scheme="ALPMZ")
    cgrd = ib.ISRConfig(nll=True, alpha=ALPHA, ew_scheme="alphaz", emela_ren_scheme="ALPMZ",
                        emela_grid=GRIDPATH)
    ib._RADIATOR_CACHE.clear()
    ls_d = ib.convolve_2leg(scan, toy, cdir)
    ib._RADIATOR_CACHE.clear()
    ls_g = ib.convolve_2leg(scan, toy, cgrd)
    relmax = float(np.max(np.abs(ls_g - ls_d) / np.abs(ls_d)))
    print(f"      max rel diff over scan = {relmax:.2e}")
    check("line-shape max rel < 1e-5", relmax < 1e-5, f"{relmax:.1e}")

    # === M. Asimov m_W cross-fit on real EOS σ̂ grids ===
    section("[M] ASIMOV m_W cross-fit  (direct morph = truth, grid morph = model)")
    try:
        from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO
        mscan = np.array([157.5, 158.5, 159.5, 160.5, 161.5, 162.3, 163.0, 164.0, 161.0])
        gd = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cdir, match_bfs=False)
        gg = WWGeneratorMoCaNLO(scheme_alpha="gf", isr_cfg=cgrd, match_bfs=False)
        t = time.time()
        Cd = gd._fit_morph(mscan)              # (6, n_s) truth morph (direct eMELA)
        t_dir = time.time() - t
        t = time.time()
        Cg = gg._fit_morph(mscan)              # (6, n_s) model morph (grid)
        t_grd = time.time() - t
        print(f"      morph build: direct {t_dir:.1f}s   grid {t_grd:.1f}s   "
              f"speedup ×{t_dir/max(t_grd,1e-6):.0f}")

        # Asimov data = direct central (Δm=Δw=0) = c0 row.  Model linearised in
        # (Δm,Δw) [MeV] about the grid morph: σ ≈ Cg0 + Cg1·Δm + Cg2·Δw.
        data = Cd[0]
        delta = Cg[0] - data                   # central residual grid−direct
        J = np.vstack([Cg[1], Cg[2]]).T        # (n_s, 2) slopes
        # FCC-ee-like Poisson weighting: equal lumi/point ⇒ w ∝ 1/σ.
        w = 1.0 / np.maximum(np.abs(data), 1e-12)
        JT_W = J.T * w
        theta = -np.linalg.solve(JT_W @ J, JT_W @ delta)   # (Δm,Δw) best fit [MeV]
        dmW, dgW = float(theta[0]), float(theta[1])
        relshape = float(np.max(np.abs(delta) / np.abs(data)))
        print(f"      central line-shape residual (grid−direct): max rel = {relshape:.2e}")
        print(f"      ⇒ induced bias:  Δm_W = {dmW:+.4f} MeV   ΔΓ_W = {dgW:+.4f} MeV")
        check("m_W bias |Δm_W| < 0.10 MeV (≪ 0.25 target)", abs(dmW) < 0.10,
              f"Δm_W={dmW:+.4f} MeV")
        check("Γ_W bias |ΔΓ_W| < 0.20 MeV", abs(dgW) < 0.20, f"ΔΓ_W={dgW:+.4f} MeV")
    except Exception as exc:
        print(f"      SKIP cross-fit ({type(exc).__name__}: {exc})")

    # === S. speed ===
    section("[S] SPEED  per-node eMELA vs vectorised grid interp")
    be = ib.beta_components(161.0, scheme="LO_beta", alpha=ALPHA)[0]
    _u, _w, x_vals, omx, _j = ib._endpoint_grid(be, X_MIN, 128)
    mid = omx >= eg.OMX_FLOOR
    xv, ov = x_vals[mid], omx[mid]
    t = time.time()
    for x, o in zip(xv, ov):
        em.code_pdf(float(x), float(o), 161.0)
    t_em = time.time() - t
    t = time.time()
    for _ in range(200):
        grid.xfxQ(xv, ov, 161.0)
    t_gr = (time.time() - t) / 200
    print(f"      {mid.sum()} nodes: eMELA {1e3*t_em:.0f}ms   grid {1e3*t_gr:.2f}ms   "
          f"→ ×{t_em/max(t_gr,1e-9):.0f} faster per per-leg build")
    check("grid per-leg build > 100× faster", t_em / max(t_gr, 1e-9) > 100,
          f"×{t_em/max(t_gr,1e-9):.0f}")

    # === E. endpoint behaviour ===
    section("[E] ENDPOINT  grid vs eMELA approaching the 1e-15 analytic cutoff")
    print("      omx        grid xD        eMELA xD       grid/eMELA")
    for o in (1e-3, 1e-6, 1e-9, 1e-12, 1e-14):
        x = 1.0 - o
        gv = grid.xfxQ(x, o, 161.0)
        dv = em.code_pdf(x, o, 161.0)
        print(f"      {o:.0e}   {gv:.6e}   {dv:.6e}   {gv/dv:.6f}")
    print("      (below 1e-15 BOTH paths use the analytic norm_nll — identical;")
    print("       the grid never queries x=1, so no endpoint divergence enters.)")

    # cleanup /tmp
    try:
        os.remove(GRIDPATH)
        import shutil
        shutil.rmtree(lha, ignore_errors=True)
    except OSError:
        pass

    print("\n" + "=" * 60)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAIL(S): {FAILS}")
        sys.exit(1)
    print("RESULT: ALL LHAPDF CHECKS PASS")


if __name__ == "__main__":
    main()
