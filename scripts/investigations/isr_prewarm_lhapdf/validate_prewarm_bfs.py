"""Validate the eMELA-radiator cache + prewarm ported onto the BFS chain
(framework/process/ww/xsec_calculator/isr.py) vs the independent-chain pattern.

Run (cache forced onto /tmp, never AFS):
    WW_ISR_RADIATOR_CACHE=/tmp/ww_bfs_prewarm_val \
      PYTHONPATH=$PWD python3 scripts/investigations/isr_prewarm_lhapdf/validate_prewarm_bfs.py

Checks
------
R.  REGRESSION (the decisive one): the refactored isr.py is BYTE-IDENTICAL to the
    committed (HEAD) version — NLL (code_pdf), eMELA-LL (ll_pdf) AND analytic
    LL+exp — on a toy σ̂ and through the real sigma_observed_munuqq(isr_nll=True).
A.  In-memory cache: warm == cold (byte-identical), cache off == cache on.
B.  prewarm idempotency: 2nd run builds 0, all 'exists'.
C.  Accounting: n_unique == grids×eMELA-cfgs; analytic cfg skipped.
D.  Cold-process disk load → line shape byte-identical to a fresh build.
E.  Concurrency: N cold workers race-safe (identical, 1 pkl/√s, no .tmp); a
    prewarm-first fan-out only LOADS (≫ faster than a cold build).
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WW_ISR_RADIATOR_CACHE", "/tmp/ww_bfs_prewarm_val")
CACHE = os.environ["WW_ISR_RADIATOR_CACHE"]

from framework.process.ww.xsec_calculator import isr            # noqa: E402
from framework.process.ww.xsec_calculator.eft_xsec import (      # noqa: E402
    sigma_partonic_munuqq)

GRID = np.array([158.0, 160.0, 161.0, 162.5])
NQ = 48                              # modest per-leg count keeps the suite quick
_n_fail = 0


def section(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def check(label, ok, detail=""):
    global _n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        _n_fail += 1


def csum(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()[:12]


def toy_sigma(s, mW, gammaW, **kw):
    s = np.asarray(s, dtype=float)
    return np.exp(-(np.sqrt(s) - 161.0) ** 2 / 4.0)


def clear_caches():
    isr._RADIATOR_CACHE.clear()
    if os.path.isdir(CACHE):
        for f in os.listdir(CACHE):
            os.remove(os.path.join(CACHE, f))


def load_head_reference():
    """Load HEAD's isr.py as a sibling module so we can diff new-vs-old in one
    process (relative imports resolve via __package__)."""
    src = subprocess.check_output(
        ["git", "-C", REPO, "show", "HEAD:framework/process/ww/xsec_calculator/isr.py"],
        text=True)
    path = "/tmp/_isr_head_ref.py"
    with open(path, "w") as fh:
        fh.write(src)
    name = "framework.process.ww.xsec_calculator._isr_head_ref"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "framework.process.ww.xsec_calculator"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- R
section("[R] REGRESSION  new isr.py vs HEAD — byte-identical")
clear_caches()
old = load_head_reference()
os.environ["WW_ISR_RADIATOR_CACHE"] = ""        # old has no cache; compare apples-to-apples
isr._RADIATOR_CACHE.clear()

for tag, kw in [("NLL code_pdf", dict(nll=True)),
                ("eMELA-LL ll_pdf", dict(emela_ll=True)),
                ("analytic LL+exp", dict(nll=False, emela_ll=False))]:
    a_new = isr.sigma_ISR_2leg_convolution(GRID, toy_sigma, n_jobs=1, n_quad=NQ, **kw)
    a_old = old.sigma_ISR_2leg_convolution(GRID, toy_sigma, n_jobs=1, n_quad=NQ, **kw)
    check(f"toy σ̂  {tag}: new == HEAD", np.array_equal(a_new, a_old),
          f"{csum(a_new)} vs {csum(a_old)}  max|Δ|={float(np.max(np.abs(a_new-a_old))):.2e}")

# real σ̂ through the public API (the production NLL 2-leg path)
from framework.process.ww.xsec_calculator import isr as _isr_new  # noqa: E402
clear_caches()
o_new = _isr_new.sigma_observed_munuqq(GRID, isr_nll=True, n_quad=NQ)
isr._RADIATOR_CACHE.clear()
o_old = old.sigma_observed_munuqq(GRID, isr_nll=True, n_quad=NQ)
check("real σ̂  sigma_observed_munuqq(isr_nll=True): new == HEAD",
      np.array_equal(o_new, o_old),
      f"{csum(o_new)} vs {csum(o_old)}  max|Δ|={float(np.max(np.abs(o_new-o_old))):.2e}")

os.environ["WW_ISR_RADIATOR_CACHE"] = CACHE

# --------------------------------------------------------------------------- A
section("[A] in-memory cache  warm == cold,  cache-off == cache-on")
clear_caches()
os.environ["WW_ISR_RADIATOR_CACHE"] = ""
isr._RADIATOR_CACHE.clear()
cold = isr.sigma_ISR_2leg_convolution(GRID, toy_sigma, nll=True, n_jobs=1, n_quad=NQ)
warm = isr.sigma_ISR_2leg_convolution(GRID, toy_sigma, nll=True, n_jobs=1, n_quad=NQ)
check("warm (in-mem hit) == cold", np.array_equal(cold, warm), f"{csum(cold)} == {csum(warm)}")
os.environ["WW_ISR_RADIATOR_CACHE"] = CACHE
clear_caches()
on = isr.sigma_ISR_2leg_convolution(GRID, toy_sigma, nll=True, n_jobs=1, n_quad=NQ)
check("cache-on == cache-off", np.array_equal(cold, on), f"{csum(cold)} == {csum(on)}")

# --------------------------------------------------------------------------- B,C
section("[B,C] prewarm idempotency + accounting")
clear_caches()
cfg_nll = isr.radiator_cfg(n_quad=NQ)                      # production NLL
cfg_ll = isr.radiator_cfg(n_quad=NQ, nll=False, emela_ll=True)
cfg_analytic = isr.radiator_cfg(n_quad=NQ, nll=False, emela_ll=False)
rep1 = isr.prewarm(GRID, [cfg_nll, cfg_ll, cfg_analytic], verbose=True)
check("n_unique == grids × eMELA-cfgs (4×2=8)", rep1["n_unique"] == 2 * len(GRID),
      f"n_unique={rep1['n_unique']}")
check("built == n_unique on cold run", rep1["built"] == rep1["n_unique"],
      f"built={rep1['built']}")
check("analytic cfg skipped (== len(grid))", rep1["skipped_analytic"] == len(GRID),
      f"skipped_analytic={rep1['skipped_analytic']}")
rep2 = isr.prewarm(GRID, [cfg_nll, cfg_ll, cfg_analytic], verbose=False)
check("re-run builds 0, all 'exists'", rep2["built"] == 0 and rep2["exists"] == rep1["n_unique"],
      f"built={rep2['built']} exists={rep2['exists']}")

# --------------------------------------------------------------------------- D
section("[D] cold-process disk load → byte-identical line shape")
# Two fresh processes sharing the disk cache: proc A builds + writes, proc B
# must LOAD the same bits.  A clean script file avoids -c quoting pitfalls.
helper = "/tmp/_bfs_prewarm_lineshape.py"
with open(helper, "w") as fh:
    fh.write(
        "import os, sys, hashlib\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "from framework.process.ww.xsec_calculator import isr\n"
        "g = np.array([158., 160., 161., 162.5])\n"
        "def f(s, mW, gW, **k):\n"
        "    s = np.asarray(s, float)\n"
        "    return np.exp(-(np.sqrt(s) - 161.) ** 2 / 4.)\n"
        f"o = isr.sigma_ISR_2leg_convolution(g, f, nll=True, n_jobs=1, n_quad={NQ})\n"
        "print(hashlib.sha1(np.ascontiguousarray(o, float).tobytes()).hexdigest()[:12])\n"
    )
clear_caches()
env = dict(os.environ, WW_ISR_RADIATOR_CACHE=CACHE)
h_build = subprocess.check_output([sys.executable, helper], env=env, text=True).strip().splitlines()[-1]
h_load = subprocess.check_output([sys.executable, helper], env=env, text=True).strip().splitlines()[-1]
check("cold build (proc A) == disk load (proc B)", h_build == h_load, f"{h_build} == {h_load}")

# --------------------------------------------------------------------------- E
section("[E] concurrency  race-safe + prewarm makes the fan-out only LOAD")


def _worker(_i):
    import numpy as np
    from framework.process.ww.xsec_calculator import isr as _i_isr
    t0 = time.time()

    def f(s, mW, gW, **k):
        s = np.asarray(s, float)
        return np.exp(-(np.sqrt(s) - 161.0) ** 2 / 4.0)
    o = _i_isr.sigma_ISR_2leg_convolution(GRID, f, nll=True, n_jobs=1, n_quad=NQ)
    return (csum(o), time.time() - t0)


import multiprocessing as mp                                       # noqa: E402
ctx = mp.get_context("fork")

clear_caches()                                                    # 4 COLD workers race
with ctx.Pool(4) as pool:
    cold_res = pool.map(_worker, range(4))
cold_sums = {s for s, _ in cold_res}
pkls = [f for f in os.listdir(CACHE) if f.endswith(".pkl")]
tmps = [f for f in os.listdir(CACHE) if ".tmp" in f]
check("all cold workers identical (race-safe)", len(cold_sums) == 1, f"checksums={len(cold_sums)}")
check("one .pkl per √s, no .tmp leftover", len(pkls) == len(GRID) and not tmps,
      f"pkls={len(pkls)} tmps={len(tmps)}")
slow_cold = max(dt for _, dt in cold_res)

clear_caches()                                                   # prewarm THEN fan out
isr.prewarm(GRID, isr.radiator_cfg(n_quad=NQ), verbose=False)
with ctx.Pool(4) as pool:
    warm_res = pool.map(_worker, range(4))
warm_sums = {s for s, _ in warm_res}
fast_warm = max(dt for _, dt in warm_res)
check("prewarmed workers identical to cold", warm_sums == cold_sums, f"{warm_sums} == {cold_sums}")
check("prewarmed fan-out LOADS (≥5× faster than cold build)",
      fast_warm < slow_cold / 5.0, f"cold {slow_cold:.2f}s → warm {fast_warm:.3f}s "
      f"(×{slow_cold/max(fast_warm,1e-6):.0f})")

print("\n" + "=" * 64)
print("RESULT:", "ALL BFS PREWARM CHECKS PASS" if _n_fail == 0 else f"{_n_fail} CHECK(S) FAILED")
print("=" * 64)
sys.exit(1 if _n_fail else 0)
