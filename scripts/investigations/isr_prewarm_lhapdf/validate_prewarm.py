"""Thorough validation of isr_beta.prewarm() — the eMELA-NLL radiator disk-cache
pre-build that de-stampedes a parallel campaign fan-out.

Run (cache forced onto /tmp, never AFS):
    WW_ISR_RADIATOR_CACHE=/tmp/ww_prewarm_val \
      PYTHONPATH=$PWD python3 scripts/investigations/isr_prewarm_lhapdf/validate_prewarm.py

Checks
------
A. EXACTNESS      prewarmed→disk-load arrays are byte-identical to a fresh
                  in-process build with disk caching OFF (the cache must be a
                  pure accelerator, never perturb a single bit).
B. IDEMPOTENCY    a second prewarm() builds 0 files, reports all 'exists'.
C. ACCOUNTING     {grids}×{NLL cfgs} unique files; LL cfgs are skipped, not cached.
D. DISK-LOAD      a cold process (empty L1) loads from disk and reproduces the
                  exact line shape of a no-disk fresh build.
E. CONCURRENCY    N cold processes hitting the SAME (grid,cfg) at once:
                    E1 no prewarm  → race is corruption-safe (identical results,
                       exactly one valid .pkl, no leftover .tmp) but every worker
                       pays the full build (the stampede prewarm removes);
                    E2 prewarm-first → every worker LOADS (ms), zero rebuild.
                  Same numeric checksum in E1 and E2 (built ≡ loaded).
"""
import hashlib
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

CACHE = os.environ.setdefault("WW_ISR_RADIATOR_CACHE", "/tmp/ww_prewarm_val")

from framework.process.ww.indep import isr_beta as ib   # noqa: E402

X_MIN = float(np.sqrt(0.30))
GRID4 = np.array([157.0, 159.0, 161.0, 162.5])
NLL = dict(scheme="LO_beta", alpha=ib.ALPHA_MZ, mu_F_factor=1.0, x_min=X_MIN,
           n_quad=128, nll=True, emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ")


def _setup_checksum(setups):
    """sha1 over every (x_vals, w, per_leg) array — a byte-level fingerprint."""
    h = hashlib.sha1()
    for x_vals, w, per_leg in setups:
        for a in (x_vals, w, per_leg):
            h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()


def _fresh_no_disk(grid, cfg_kwargs):
    """Build the radiator with disk caching OFF and a clean in-memory cache."""
    old = os.environ.get("WW_ISR_RADIATOR_CACHE")
    os.environ["WW_ISR_RADIATOR_CACHE"] = ""        # disable disk
    ib._RADIATOR_CACHE.clear()
    try:
        return ib._radiator_setup(np.asarray(grid, float), ib.ISRConfig(**cfg_kwargs))
    finally:
        if old is None:
            os.environ.pop("WW_ISR_RADIATOR_CACHE", None)
        else:
            os.environ["WW_ISR_RADIATOR_CACHE"] = old


# ----- worker for the concurrency test (must be top-level: fork pool) ----------
def _cold_worker(args):
    idx, grid, cfg_kwargs, cache_dir = args
    os.environ["WW_ISR_RADIATOR_CACHE"] = cache_dir
    from framework.process.ww.indep import isr_beta as _ib
    _ib._RADIATOR_CACHE.clear()                     # simulate a cold cross-process start
    t0 = time.time()
    setups = _ib._radiator_setup(np.asarray(grid, float), _ib.ISRConfig(**cfg_kwargs))
    dt = time.time() - t0
    return idx, dt, _setup_checksum(setups)


def main():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    shutil.rmtree(CACHE, ignore_errors=True)
    print(f"cache dir = {CACHE!r}\n")

    # --- reference: fresh build, no disk ---
    print("[ref] fresh in-process build (disk OFF)")
    t0 = time.time()
    ref = _fresh_no_disk(GRID4, NLL)
    ref_ck = _setup_checksum(ref)
    print(f"      {len(ref)} √s points, {time.time()-t0:.1f}s, checksum={ref_ck[:12]}")

    # === A. exactness: prewarm→disk-load == fresh-no-disk, byte-for-byte ===
    print("\n[A] EXACTNESS  prewarm→disk vs fresh-no-disk")
    os.environ["WW_ISR_RADIATOR_CACHE"] = CACHE
    ib._RADIATOR_CACHE.clear()
    rep = ib.prewarm(GRID4, ib.ISRConfig(**NLL), verbose=False)
    ib._RADIATOR_CACHE.clear()                       # force the disk-load path
    loaded = ib._radiator_setup(np.asarray(GRID4, float), ib.ISRConfig(**NLL))
    load_ck = _setup_checksum(loaded)
    byte_id = all(
        a.tobytes() == b.tobytes()
        for (xa, wa, pa), (xb, wb, pb) in zip(ref, loaded)
        for a, b in ((xa, xb), (wa, wb), (pa, pb)))
    check("byte-identical to fresh build", byte_id and load_ck == ref_ck,
          f"checksum {load_ck[:12]} == {ref_ck[:12]}")
    check("prewarm built exactly 1 file", rep["built"] == 1 and rep["n_unique"] == 1,
          f"built={rep['built']} n_unique={rep['n_unique']}")

    # === B. idempotency ===
    print("\n[B] IDEMPOTENCY  second prewarm builds nothing")
    rep2 = ib.prewarm(GRID4, ib.ISRConfig(**NLL), verbose=False)
    check("re-run builds 0, all 'exists'", rep2["built"] == 0 and rep2["exists"] == 1,
          f"built={rep2['built']} exists={rep2['exists']}")

    # === C. accounting: {grids}×{NLL cfgs}, LL skipped ===
    print("\n[C] ACCOUNTING  cross-product + LL skip")
    grids = [GRID4, np.array([158.0, 160.0, 162.0])]
    nll_a = ib.ISRConfig(**NLL)
    nll_b = ib.ISRConfig(**{**NLL, "emela_ren_scheme": "ALGMU"})   # distinct file
    ll_c = ib.ISRConfig(**{**NLL, "nll": False})                   # never cached
    rep3 = ib.prewarm(grids, [nll_a, nll_b, ll_c], verbose=False)
    # 2 grids × 2 NLL cfgs = 4 unique; one (GRID4,nll_a) already on disk from [A]
    check("n_unique == grids×NLLcfgs (4)", rep3["n_unique"] == 4, f"n_unique={rep3['n_unique']}")
    check("skipped_LL == grids×LLcfgs (2)", rep3["skipped_LL"] == 2, f"skipped_LL={rep3['skipped_LL']}")
    check("built==3, exists==1 (GRID4×nll_a reused)",
          rep3["built"] == 3 and rep3["exists"] == 1,
          f"built={rep3['built']} exists={rep3['exists']}")

    # === D. cold-process disk load reproduces the line shape ===
    print("\n[D] DISK-LOAD  cold process reproduces line shape")
    def toy(sqrt_shat):
        s = np.asarray(sqrt_shat, float)
        return 160.0 * 0.5 * (1.0 + np.tanh((s - 160.76) / 0.8)) * (s / 161.0)
    ib._RADIATOR_CACHE.clear()                       # cold L1; disk has it
    ls_disk = ib.convolve_2leg(GRID4, toy, ib.ISRConfig(**NLL))
    ls_fresh_setups = _fresh_no_disk(GRID4, NLL)     # rebuild no-disk
    # reproduce convolve by hand from fresh setups for an apples-to-apples shape
    os.environ["WW_ISR_RADIATOR_CACHE"] = ""
    ib._RADIATOR_CACHE.clear()
    ls_fresh = ib.convolve_2leg(GRID4, toy, ib.ISRConfig(**NLL))
    os.environ["WW_ISR_RADIATOR_CACHE"] = CACHE
    relmax = float(np.max(np.abs((ls_disk - ls_fresh) / ls_fresh)))
    check("line shape rel-diff (disk vs no-disk) == 0", relmax == 0.0, f"max rel={relmax:.1e}")

    # === E. concurrency: N cold processes on the SAME (grid,cfg) ===
    import multiprocessing as mp
    N = 4

    print(f"\n[E1] CONCURRENCY no-prewarm  ({N} cold workers, empty cache)")
    shutil.rmtree(CACHE, ignore_errors=True)
    os.makedirs(CACHE, exist_ok=True)
    ib._RADIATOR_CACHE.clear()
    ctx = mp.get_context("fork")
    args = [(i, GRID4, NLL, CACHE) for i in range(N)]
    with ctx.Pool(N) as pool:
        res1 = pool.map(_cold_worker, args)
    cks1 = {ck for _, _, ck in res1}
    times1 = [dt for _, dt, _ in res1]
    pkls = [f for f in os.listdir(CACHE) if f.endswith(".pkl")]
    tmps = [f for f in os.listdir(CACHE) if ".tmp." in f]
    check("all workers identical result (race-safe)", len(cks1) == 1, f"checksums={len(cks1)}")
    check("result matches reference checksum", cks1 == {ref_ck})
    check("exactly one valid .pkl, no .tmp leftover", len(pkls) == 1 and not tmps,
          f"pkls={len(pkls)} tmps={len(tmps)}")
    print(f"       worker build times: {[f'{t:.1f}s' for t in times1]} "
          f"(min {min(times1):.1f}s → each paid a full build)")

    print(f"\n[E2] CONCURRENCY prewarm-first  ({N} cold workers)")
    shutil.rmtree(CACHE, ignore_errors=True)
    os.makedirs(CACHE, exist_ok=True)
    ib._RADIATOR_CACHE.clear()
    ib.prewarm(GRID4, ib.ISRConfig(**NLL), verbose=False)   # one build up front
    with ctx.Pool(N) as pool:
        res2 = pool.map(_cold_worker, args)
    cks2 = {ck for _, _, ck in res2}
    times2 = [dt for _, dt, _ in res2]
    check("all workers identical result", len(cks2) == 1, f"checksums={len(cks2)}")
    check("loaded result == built result (E1)", cks2 == cks1)
    slow1, fast2 = min(times1), max(times2)
    check("workers LOAD not rebuild (>20x faster than E1 build)", fast2 * 20 < slow1,
          f"slowest load {fast2:.3f}s vs fastest build {slow1:.1f}s "
          f"→ {slow1/max(fast2,1e-6):.0f}x")
    print(f"       worker load times: {[f'{t*1e3:.0f}ms' for t in times2]}")

    print("\n" + "=" * 60)
    if fails:
        print(f"RESULT: {len(fails)} FAIL(S): {fails}")
        sys.exit(1)
    print("RESULT: ALL PREWARM CHECKS PASS")


if __name__ == "__main__":
    main()
