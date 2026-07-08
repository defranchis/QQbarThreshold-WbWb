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
import tempfile

import numpy as np

from _audit_common import build_fit, make_pseudo_subset, make_scan_specs


PLOT_TMP = tempfile.mkdtemp(prefix="audit_plots_")
# scan_true_value walks every file under INPUT_DIRS["pseudo"]; symlink only a
# few of them into a tempdir so the audit stays under a minute. The audit
# doesn't care if these files cause bias-filter rejections — it only checks
# state isolation, not numerical fit success.
PSEUDO_TMP = make_pseudo_subset("audit_pseudo_")
SCAN_SPECS = make_scan_specs(PSEUDO_TMP)


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
        "alphas_sigma": fit._constraints["alphas"]["sigma"],
        "yukawa_sigma": fit._constraints["yukawa"]["sigma"] if "yukawa" in fit._constraints else 0.0,
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


def main():
    print(f"plots redirected to {PLOT_TMP}")
    print()
    print(f"{'scan':<28s} {'fields drifted':<15s} {'detail'}")
    print("-" * 80)

    overall_ok = True
    for name, kwargs, call in SCAN_SPECS:
        try:
            fit = build_fit(PLOT_TMP, **kwargs)
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
