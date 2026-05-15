"""Fork-based parallel dispatcher for the scan helpers.

Each scan helper is self-contained — it reads from ``fit`` (without mutating
it, see the audit in /tmp/audit_scans.py) and writes a plot file under
``fit.plot_dir``. That makes parallelisation trivial under ``fork``: child
processes inherit ``fit`` from the parent's memory, do their work, exit.
No pickling, no shared mutable state.

Why ``multiprocessing.Process`` rather than ``Pool`` / ``ProcessPoolExecutor``:
the latter route tasks through a queue, which serialises arguments via
``pickle`` — and ``iminuit.Minuit`` (held on ``fit.minuit``) wraps a C++
object whose pickling is fragile. With ``fork`` and a bare ``Process``,
``self._target`` lives in inherited memory and is never pickled.
"""

import multiprocessing as mp
import time


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
    procs = []      # all spawned, for final join + exit-code check
    active = []     # currently running

    for job in jobs:
        while len(active) >= max_workers:
            _reap_finished(active)
            if len(active) >= max_workers:
                time.sleep(0.05)
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


def _reap_finished(active):
    for p in list(active):
        if not p.is_alive():
            p.join()
            active.remove(p)
