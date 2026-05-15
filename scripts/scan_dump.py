"""Per-scan plot-data dumper for numerical bit-equivalence checks.

Monkey-patches ``matplotlib.pyplot.plot`` so every ``(x, y)`` pair fed in is
also written to stdout, prefixed by the current scan name. Companion to
``audit_scans.py`` (which checks state isolation) — this script checks that
the *numerical* output of every scan is preserved across a refactor.

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
non-empty diff means migrad found a measurably different minimum; inspect
case-by-case (see e.g. the cold-start vs warm-start discussion when this
script was first written for the deepcopy-elimination refactor — warm-start
gave ~1e-4 differences in ``hesse`` cov when the chi2 cov matrix changed
substantially between scan iterations).

Each scan is invoked with small-grid arguments so the whole sweep runs in
under a minute on lxplus. The pseudo-data subset (``PSEUDO_TMP``) is
picked so that at least one file passes the bias filter in
``scan_true_value`` — otherwise that scan would produce zero plot data
in both runs and the dump would silently miss any regression there.
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow the script to be invoked from anywhere as long as the project
# root sits on the parent directory of this file's parent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import wbwb_default as base_card
from common import scans
from common.systematics import print_syst_table
from process.wbwb.fit import WbWbFit
from process.wbwb.generator import WbWbGenerator


PLOT_TMP = tempfile.mkdtemp(prefix="scan_dump_plots_")
PSEUDO_TMP = tempfile.mkdtemp(prefix="scan_dump_pseudo_")


def _pick_pseudo_subset(src, n=3):
    """Symlink ``n`` pseudo files near the nominal mass (171.5 GeV) into
    PSEUDO_TMP so ``scan_true_value`` actually fits and plots something."""
    if not os.path.isdir(src):
        return
    target_mass = base_card.PARAMETERS["mass"]["nominal"]
    entries = []
    for f in os.listdir(src):
        # filename: ..._mass<value>_width...
        try:
            mass_str = f.split("_mass")[1].split("_")[0]
            entries.append((abs(float(mass_str) - target_mass), f))
        except (IndexError, ValueError):
            continue
    for _, f in sorted(entries)[:n]:
        os.symlink(os.path.abspath(os.path.join(src, f)),
                   os.path.join(PSEUDO_TMP, f))


_pick_pseudo_subset(base_card.INPUT_DIRS["pseudo"])


def build_fit(*, with_bec=False, with_bes=False, with_sw2=False,
              sm_width=False, constrain_yukawa=True,
              last_ecm=False, scale_vars=False, shift_scan=False):
    card = types.ModuleType("card_audit")
    card.__dict__.update(base_card.__dict__)
    card.PLOT_DIR = PLOT_TMP
    gen = WbWbGenerator(order=card.ORDER, isr=True)
    fit = WbWbFit(
        card, gen,
        sm_width=sm_width, asimov=True,
        constrain_yukawa=constrain_yukawa, read_scale_vars=scale_vars,
        shift_scan=shift_scan,
    )
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"], scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"], total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"], add_last_ecm=last_ecm, same_evts=False,
    )
    if with_bec:
        fit.add_bec_nuisances()
    if with_bes:
        fit.add_bes_nuisances()
    if with_sw2:
        fit.add_sw2_nuisance()
    fit.fit_parameters()
    with contextlib.redirect_stdout(io.StringIO()):
        fit.fit_results()
    return fit


_real_plot = plt.plot
_current_scan = ""
_DUMP = []  # collected lines, flushed once at the end so that stdout written
            # by scans themselves (e.g. print_syst_table) can be redirected
            # away without losing our captures.


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


plt.plot = _dump_plot


def _scan_true_value_subset(fit):
    """Drive ``scan_true_value`` against the symlinked PSEUDO_TMP subset
    rather than the full 100+ pseudodata files in the card's INPUT_DIRS."""
    saved = fit.card.INPUT_DIRS["pseudo"]
    fit.card.INPUT_DIRS = dict(fit.card.INPUT_DIRS, pseudo=PSEUDO_TMP)
    try:
        scans.scan_true_value(fit)
    finally:
        fit.card.INPUT_DIRS = dict(fit.card.INPUT_DIRS, pseudo=saved)


# (scan name, build_fit kwargs, callable). Each callable runs a small-grid
# version of the scan against the configured fit. New scans / generators
# should be added here.
SPECS = [
    ("scan_beam_resolution",   {},
        lambda f: scans.scan_beam_resolution(f, lo=0.1, hi=0.2, step=0.05)),
    ("scan_bec",               {"with_bec": True},
        lambda f: scans.scan_bec(f, lo=0, hi=2, step=1.0)),
    ("scan_bes",               {"with_bes": True},
        lambda f: scans.scan_bes(f, lo=0, hi=0.01, step=0.005)),
    ("scan_lumi",              {},
        lambda f: scans.scan_lumi(f, lo=0, hi=2, points=3)),
    ("scan_alphas",            {},
        lambda f: scans.scan_alphas(f, hi=2e-4, step=1e-4)),
    ("scan_yukawa_constraint", {},
        lambda f: scans.scan_yukawa_constraint(f, hi=0.02, step=0.01)),
    ("scan_yukawa_theory",     {"last_ecm": True, "constrain_yukawa": False},
        lambda f: scans.scan_yukawa_theory(f, max_shift=0.005, step=0.005)),
    ("scan_width",             {"sm_width": True},
        lambda f: scans.scan_width(f, hi=5, step=2.5)),
    ("scan_chi2",              {},
        lambda f: scans.scan_chi2(f)),
    ("scan_scale_vars",        {"scale_vars": True},
        lambda f: scans.scan_scale_vars(f)),
    ("scan_shift",             {"shift_scan": True},
        lambda f: scans.scan_shift(f, max_abs_shift_neg=0, max_abs_shift_pos=0.5, step=0.5)),
    ("scan_true_value",        {},
        _scan_true_value_subset),
    ("print_syst_table",       {"with_bec": True, "with_bes": True},
        lambda f: print_syst_table(f, latex_path="")),
]


def main():
    global _current_scan
    for name, kw, fn in SPECS:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                fit = build_fit(**kw)
            except Exception as exc:
                _DUMP.append(f"{name}\tBUILD_FAILED\t{exc}")
                continue
            _current_scan = name
            try:
                fn(fit)
            except Exception as exc:
                _DUMP.append(f"{name}\tSCAN_FAILED\t{exc}")
    sys.stdout.write("\n".join(_DUMP) + ("\n" if _DUMP else ""))


if __name__ == "__main__":
    main()
