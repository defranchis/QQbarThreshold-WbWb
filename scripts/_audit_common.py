"""Shared scaffolding for the harness scripts (``audit_scans.py``,
``scan_dump.py``): card-cloning ``build_fit``, pseudo-data subset picker,
and the canonical 13-entry SCAN_SPECS covering every scan helper plus
``print_syst_table``.

Each harness owns its own observer (state snapshot+diff vs. plot-data dump)
and main loop; this module only carries the parts they previously
copy-pasted between them.
"""

import contextlib
import io
import os
import sys
import tempfile
import types

# Allow imports of cards/ common/ process/ when this module is imported
# from scripts/ (which it always is).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import wbwb_default as base_card
from common import scans
from common.systematics import print_syst_table
from process.wbwb.fit import WbWbFit
from process.wbwb.generator import WbWbGenerator


def make_pseudo_subset(prefix, *, n=3, target_mass=None):
    """Create a tempdir under ``{tempdir}/{prefix}*`` and symlink ``n``
    pseudo-data files into it.

    If ``target_mass`` is given, pick the ``n`` files with mass closest to
    it (used by ``scan_dump`` so ``scan_true_value``'s bias filter doesn't
    reject every entry); otherwise the first ``n`` alphabetically (which
    is fine for the audit — it only needs no crash, not fit success).

    Returns the tempdir path.
    """
    dst = tempfile.mkdtemp(prefix=prefix)
    src = base_card.INPUT_DIRS["pseudo"]
    if not os.path.isdir(src):
        return dst
    if target_mass is None:
        chosen = sorted(os.listdir(src))[:n]
    else:
        entries = []
        for f in os.listdir(src):
            try:
                mass = float(f.split("_mass")[1].split("_")[0])
                entries.append((abs(mass - target_mass), f))
            except (IndexError, ValueError):
                continue
        chosen = [f for _, f in sorted(entries)[:n]]
    for f in chosen:
        os.symlink(os.path.abspath(os.path.join(src, f)),
                   os.path.join(dst, f))
    return dst


def build_fit(plot_dir, *, with_bec=False, with_bes=False, with_sw2=False,
              sm_width=False, constrain_yukawa=True,
              last_ecm=False, scale_vars=False, shift_scan=False):
    """Build a fully-initialised WbWbFit for the audit/dump harness.

    ``plot_dir`` is set as the card's ``PLOT_DIR`` so neither harness
    clobbers the real ``plots/`` tree. The nominal fit and ``fit_results``
    are run so downstream consumers see realistic state."""
    card = types.ModuleType("card_audit")
    card.__dict__.update(base_card.__dict__)
    card.PLOT_DIR = plot_dir

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
        fit.add_binned_nuisance("BEC")
    if with_bes:
        fit.add_binned_nuisance("BES")
    if with_sw2:
        fit.add_global_nuisance("sw2")
    fit.fit_parameters()
    with contextlib.redirect_stdout(io.StringIO()):
        fit.fit_results()
    return fit


def scan_true_value_with_subset(fit, pseudo_dir):
    """Drive ``scan_true_value`` against ``pseudo_dir`` rather than the
    full ``INPUT_DIRS["pseudo"]`` (~100+ files in production)."""
    saved = fit.card.INPUT_DIRS["pseudo"]
    fit.card.INPUT_DIRS = dict(fit.card.INPUT_DIRS, pseudo=pseudo_dir)
    try:
        scans.scan_true_value(fit)
    finally:
        fit.card.INPUT_DIRS = dict(fit.card.INPUT_DIRS, pseudo=saved)


def make_scan_specs(pseudo_dir):
    """Standard list of ``(name, build_fit_kwargs, scan_callable)`` specs
    covering every scan helper plus ``print_syst_table``. ``pseudo_dir``
    redirects ``scan_true_value``'s input directory to a small subset."""
    return [
        ("scan_beam_resolution", {},
            lambda f: scans.scan_beam_resolution(f, lo=0.1, hi=0.2, step=0.05)),
        ("scan_bec",             {"with_bec": True},
            lambda f: scans.scan_bec(f, lo=0, hi=2, step=1.0)),
        ("scan_bes",             {"with_bes": True},
            lambda f: scans.scan_bes(f, lo=0, hi=0.01, step=0.005)),
        ("scan_lumi",            {},
            lambda f: scans.scan_lumi(f, lo=0, hi=2, points=3)),
        ("scan_alphas",          {},
            lambda f: scans.scan_alphas(f, hi=2e-4, step=1e-4)),
        ("scan_yukawa_constraint", {},
            lambda f: scans.scan_yukawa_constraint(f, hi=0.02, step=0.01)),
        ("scan_yukawa_theory",   {"last_ecm": True, "constrain_yukawa": False},
            lambda f: scans.scan_yukawa_theory(f, max_shift=0.005, step=0.005)),
        ("scan_width",           {"sm_width": True},
            lambda f: scans.scan_width(f, hi=5, step=2.5)),
        ("scan_chi2",            {},
            lambda f: scans.scan_chi2(f)),
        ("scan_scale_vars",      {"scale_vars": True},
            lambda f: scans.scan_scale_vars(f)),
        ("scan_shift",           {"shift_scan": True},
            lambda f: scans.scan_shift(f, max_abs_shift_neg=0, max_abs_shift_pos=0.5, step=0.5)),
        ("scan_true_value",      {},
            lambda f: scan_true_value_with_subset(f, pseudo_dir)),
        ("print_syst_table",     {"with_bec": True, "with_bes": True},
            lambda f: print_syst_table(f, latex_path="")),
    ]
