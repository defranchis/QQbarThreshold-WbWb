"""Per-scan plot-data dumper for numerical bit-equivalence checks.

Monkey-patches ``matplotlib.pyplot.plot`` so every ``(x, y)`` pair fed in
is captured and flushed to stdout at the end. Companion to
``audit_scans.py`` (which checks state isolation) — this script checks
that the *numerical* output of every scan is preserved across a refactor.

Usage (compare before vs after some change)::

    # 1. dump current ("refactored") code
    cd WW_threshold && python scripts/scan_dump.py > /tmp/after.txt

    # 2. stash the change and dump the baseline
    git stash push -m "validation" common/ scripts/
    python scripts/scan_dump.py > /tmp/before.txt
    git stash pop

    # 3. compare
    diff /tmp/before.txt /tmp/after.txt

A zero-line diff means the refactor preserved scan output exactly. A
non-empty diff means migrad found a measurably different minimum;
inspect case-by-case (cold-start vs warm-start migrad gives ~1e-4
hesse differences when the cov surface changes substantially between
scan iterations — see ``common/scans.py:scan_lumi`` for the rationale).

The pseudo-data subset is picked near the nominal mass so
``scan_true_value``'s bias filter doesn't reject every entry.
"""
import contextlib
import io
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _audit_common import (
    base_card,
    build_fit,
    make_pseudo_subset,
    make_scan_specs,
)


PLOT_TMP = tempfile.mkdtemp(prefix="scan_dump_plots_")
PSEUDO_TMP = make_pseudo_subset(
    "scan_dump_pseudo_",
    target_mass=base_card.PARAMETERS["mass"]["nominal"],
)
SCAN_SPECS = make_scan_specs(PSEUDO_TMP)


_real_plot = plt.plot
_current_scan = ""
_DUMP = []  # collected lines, flushed once at the end so scans' own stdout
            # (e.g. print_syst_table) can be redirected away without losing
            # our captures.


def _dump_plot(*args, **kwargs):
    if len(args) >= 2:
        try:
            x = np.asarray(args[0], dtype=float).flatten()
            y = np.asarray(args[1], dtype=float).flatten()
            for xi, yi in zip(x, y):
                _DUMP.append(f"{_current_scan}\t{xi:.12e}\t{yi:.12e}")
        except (TypeError, ValueError):
            pass
    return _real_plot(*args, **kwargs)


def main():
    global _current_scan
    plt.plot = _dump_plot
    try:
        for name, kw, fn in SCAN_SPECS:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    fit = build_fit(PLOT_TMP, **kw)
                except Exception as exc:
                    _DUMP.append(f"{name}\tBUILD_FAILED\t{exc}")
                    continue
                _current_scan = name
                try:
                    fn(fit)
                except Exception as exc:
                    _DUMP.append(f"{name}\tSCAN_FAILED\t{exc}")
    finally:
        plt.plot = _real_plot
    sys.stdout.write("\n".join(_DUMP) + ("\n" if _DUMP else ""))


if __name__ == "__main__":
    main()
