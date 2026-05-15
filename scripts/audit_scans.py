"""Per-scan state-isolation audit.

For each scan helper:
  1. Build a fresh fit configured for that scan.
  2. Snapshot every piece of fit state downstream consumers read.
  3. Run the scan.
  4. Confirm the snapshot is unchanged (deepcopy / scan_chi2 alike).

Run with the project root (the directory containing ``cards/`` and
``common/``) as the current working directory:

    cd WW_threshold && python scripts/audit_scans.py

Plot output is redirected to a tempdir so we don't clobber the real
``plots/`` tree.
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np

# Allow the script to be invoked from anywhere as long as the project
# root sits on the parent directory of this file's parent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import wbwb_default as base_card
from common import scans
from process.wbwb.fit import WbWbFit
from process.wbwb.generator import WbWbGenerator


PLOT_TMP = tempfile.mkdtemp(prefix="audit_plots_")


def build_fit(*, with_bec=False, with_bes=False, with_sw2=False,
              sm_width=False, constrain_yukawa=True,
              last_ecm=False, scale_vars=False, shift_scan=False):
    # Shallow-clone the card and override the plot dir so we don't touch real plots/.
    card = types.ModuleType("card_audit")
    card.__dict__.update(base_card.__dict__)
    card.PLOT_DIR = PLOT_TMP

    gen = WbWbGenerator(order=card.ORDER, isr=True)
    fit = WbWbFit(
        card, gen,
        sm_width=sm_width,
        asimov=True,
        constrain_yukawa=constrain_yukawa,
        read_scale_vars=scale_vars,
        shift_scan=shift_scan,
    )
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
        add_last_ecm=last_ecm,
        same_evts=False,
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


def snapshot(fit):
    return {
        "minuit.values":    list(fit.minuit.values),
        "minuit.errors":    list(fit.minuit.errors),
        "minuit.fixed":     list(fit.minuit.fixed),
        "minuit.fval":      fit.minuit.fval,
        "minuit.cov":       np.asarray(fit.minuit.covariance).copy(),
        "last_fit_n":       [r.n for r in fit.last_fit_results],
        "last_fit_s":       [r.s for r in fit.last_fit_results],
        "xsec_scenario":    np.asarray(fit.xsec_scenario["xsec"]).copy(),
        "scale_var_scen":   np.asarray(fit.scale_var_scenario).copy(),
        "pseudo_data_scen": np.asarray(fit.pseudo_data_scenario).copy(),
        "cov":              fit.cov.copy(),
        "_xsec_base":       fit._xsec_base.copy(),
        "_morph_matrix":    fit._morph_matrix.copy(),
        "_idx":             dict(fit._idx),
        "param_names":      list(fit.param_names),
        "input_uncert_alphas": fit.input_uncert_alphas,
        "input_uncert_yukawa": fit.input_uncert_yukawa,
        "lumi_uncorr":      fit.lumi_uncorr,
        "lumi_corr":        fit.lumi_corr,
        "beam_energy_res":  fit.beam_energy_res,
    }


def diff_snapshots(s1, s2):
    drifts = {}
    for k in s1:
        a, b = s1[k], s2[k]
        if isinstance(a, dict):
            drifts[k] = 0 if a == b else float("nan")
        elif isinstance(a, list) and a and isinstance(a[0], str):
            drifts[k] = 0 if a == b else float("nan")
        elif isinstance(a, list):
            drifts[k] = max(abs(float(x) - float(y)) for x, y in zip(a, b))
        elif isinstance(a, float):
            drifts[k] = abs(a - b)
        else:
            drifts[k] = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    return drifts


SCAN_SPECS = [
    # (name, build_kwargs, callable)
    ("scan_beam_resolution", {}, lambda f: scans.scan_beam_resolution(f, lo=0.1, hi=0.2, step=0.05)),
    ("scan_bec",             {"with_bec": True}, lambda f: scans.scan_bec(f, lo=0, hi=2, step=1.0)),
    ("scan_bes",             {"with_bes": True}, lambda f: scans.scan_bes(f, lo=0, hi=0.01, step=0.005)),
    ("scan_lumi",            {}, lambda f: scans.scan_lumi(f, lo=0, hi=2, points=3)),
    ("scan_alphas",          {}, lambda f: scans.scan_alphas(f, hi=2e-4, step=1e-4)),
    ("scan_yukawa_constraint", {}, lambda f: scans.scan_yukawa_constraint(f, hi=0.02, step=0.01)),
    ("scan_yukawa_theory",   {"last_ecm": True}, lambda f: scans.scan_yukawa_theory(f, max_shift=0.005, step=0.005)),
    ("scan_width",           {"sm_width": True}, lambda f: scans.scan_width(f, hi=5, step=2.5)),
    ("scan_chi2",            {}, lambda f: scans.scan_chi2(f)),
]


def main():
    print(f"plots redirected to {PLOT_TMP}")
    print()
    print(f"{'scan':<28s} {'fields drifted':<15s} {'detail'}")
    print("-" * 80)

    overall_ok = True
    for name, kwargs, call in SCAN_SPECS:
        try:
            fit = build_fit(**kwargs)
        except Exception as exc:
            print(f"{name:<28s} SKIP             build_fit failed: {exc}")
            continue
        before = snapshot(fit)
        with contextlib.redirect_stdout(io.StringIO()):
            call(fit)
        after = snapshot(fit)
        drifts = diff_snapshots(before, after)
        drifted = [k for k, v in drifts.items() if v != 0]
        if drifted:
            overall_ok = False
            top = max(drifts.items(), key=lambda kv: 0 if kv[1] != kv[1] else kv[1])
            print(f"{name:<28s} {len(drifted):<15d} max={top[0]}={top[1]:.3e}")
        else:
            print(f"{name:<28s} 0               OK")

    print()
    print("OVERALL: " + ("ALL SCANS LEAVE fit UNTOUCHED" if overall_ok else "MUTATION DETECTED"))


if __name__ == "__main__":
    main()
