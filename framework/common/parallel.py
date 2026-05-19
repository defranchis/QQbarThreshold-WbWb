"""Fork-based parallel dispatcher for the scan helpers.

Each scan helper is self-contained — it reads from ``fit`` without mutating
it (see ``scripts/audit_scans.py`` for the per-scan invariant check) and
writes a plot file under ``fit.plot_dir``. That makes parallelisation
trivial under ``fork``: child processes inherit ``fit`` from the parent's
memory, do their work, exit. No pickling, no shared mutable state.

Why ``multiprocessing.Process`` rather than ``Pool`` / ``ProcessPoolExecutor``:
both of those route tasks through a queue that pickles its arguments,
which would force two awkward changes here — the inline ``lambda``\\s in
``doFit_wbwb.py`` that capture ``fit`` aren't picklable, and the
``FitCore`` itself carries enough state (smeared DataFrames, the cached
morph matrix, the Minuit object) that round-tripping it through a
queue per dispatch would dwarf the scan work. With ``fork`` and a bare
``Process``, ``self._target`` lives in inherited memory and is never
serialised.
"""

import multiprocessing as mp
from multiprocessing.connection import wait


def run_parallel(jobs, max_workers=6):
    """Run a list of zero-argument callables in parallel via fork.

    Parameters
    ----------
    jobs : sequence of callables
        Each callable should produce its own side effects (typically a
        plot file written under ``fit.plot_dir``). Return values are
        ignored.
    max_workers : int
        Hard ceiling on the number of concurrent child processes. New jobs
        are launched as earlier ones finish, so a long list of short jobs
        still completes promptly.

    Returns
    -------
    None — the function blocks until every job has exited.

    Raises
    ------
    RuntimeError if any child exited with a non-zero status; the rest of
    the jobs are still allowed to finish first so partial output is
    preserved.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    ctx = mp.get_context("fork")
    procs = []
    active = []

    for job in jobs:
        while len(active) >= max_workers:
            # Block until at least one child exits — no polling needed,
            # the kernel wakes us via the sentinel pipes.
            wait([p.sentinel for p in active])
            active = [p for p in active if p.is_alive()]
            for p in procs:
                if p not in active and p.exitcode is None:
                    p.join()
        p = ctx.Process(target=job)
        p.start()
        procs.append(p)
        active.append(p)

    for p in procs:
        p.join()

    failures = [p for p in procs if p.exitcode != 0]
    if failures:
        names = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in failures)
        raise RuntimeError(f"{len(failures)} scan worker(s) failed: {names}")
