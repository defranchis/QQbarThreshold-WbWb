#!/usr/bin/env python
"""Parse the WHIZARD LO mnuqq sqrts scan and plot sigma vs sqrts.

Process: e+ e- -> mu- nu_mu_bar u d_bar (single semi-leptonic 4f channel),
full tree-level 4f matrix element, no ISR.  See whizard/work/mnuqq_scan.sin.

Reads the WHIZARD run log, extracts (sqrts, sigma, error) per scan point,
writes a CSV + a PDF/PNG plot, and prints the sigma(240)/sigma(163) ratio.
"""
import re
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARDIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
    "whizard/work/mnuqq_par")
OUTDIR = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
          "WW_threshold/plots/mnuqq_lo_scan")

NUM = r"([-+0-9.Ee]+)"
RE_INT = re.compile(r"integral\(mnuqq\)\s*=\s*" + NUM)
RE_ERR = re.compile(r"error\(mnuqq\)\s*=\s*" + NUM)


def parse(pardir):
    """Read per-point dirs mnuqq_par/p_<sqrts>/run.log.

    sqrts from the dir name; sigma/error from the `show` block.
    Returns arrays (sqrts, sigma, err) in GeV / fb, sorted by sqrts.
    """
    rows = []
    for d in glob.glob(os.path.join(pardir, "p_*")):
        m = re.search(r"p_([0-9.]+)$", d)
        if not m:
            continue
        s = float(m.group(1))
        txt = open(os.path.join(d, "run.log")).read()
        mi, me = RE_INT.search(txt), RE_ERR.search(txt)
        if not (mi and me):
            print("WARN: no result in %s" % d, file=sys.stderr)
            continue
        rows.append((s, float(mi.group(1)), float(me.group(1))))
    rows.sort()
    s = np.array([r[0] for r in rows])
    sig = np.array([r[1] for r in rows])
    err = np.array([r[2] for r in rows])
    return s, sig, err


def main():
    s, sig, err = parse(PARDIR)
    if len(s) == 0:
        sys.exit("No scan points parsed from %s" % PARDIR)

    # CSV
    os.makedirs(OUTDIR, exist_ok=True)
    csv = os.path.join(OUTDIR, "mnuqq_lo_scan.csv")
    with open(csv, "w") as f:
        f.write("# sqrts[GeV]  sigma[fb]  error[fb]  rel[%]\n")
        for a, b, c in zip(s, sig, err):
            f.write("%8.3f  %12.6f  %12.6f  %8.4f\n"
                    % (a, b, c, 100 * c / b))

    print("WHIZARD LO mnuqq (e+e- -> mu- nubar u dbar), no ISR")
    print(" sqrts[GeV]   sigma[fb]      err[fb]    rel[%]")
    for a, b, c in zip(s, sig, err):
        print("  %7.1f   %11.5f   %9.5f   %6.3f" % (a, b, c, 100 * c / b))

    # Ratio 240/163
    def at(x):
        j = np.argmin(np.abs(s - x))
        if abs(s[j] - x) > 1e-6:
            return None, None, None
        return s[j], sig[j], err[j]

    s240, sig240, e240 = at(240.0)
    s163, sig163, e163 = at(163.0)
    print()
    if sig240 is not None and sig163 is not None:
        ratio = sig240 / sig163
        rel = ratio * np.hypot(e240 / sig240, e163 / sig163)
        print("sigma(240) = %.5f +/- %.5f fb" % (sig240, e240))
        print("sigma(163) = %.5f +/- %.5f fb" % (sig163, e163))
        print("RATIO sigma(240)/sigma(163) = %.5f +/- %.5f" % (ratio, rel))
    else:
        print("240 and/or 163 GeV not found among scan points.")
        ratio = None

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(s, sig, yerr=err, fmt="o-", ms=4, lw=1.2,
                color="#1f77b4", capsize=2,
                label=r"$e^+e^-\to\mu^-\bar\nu_\mu u\bar d$ (LO, no ISR)")
    ax.axvline(2 * 80.377, ls=":", color="grey", lw=1,
               label=r"$2m_W=160.75$ GeV")
    for x in (163.0, 240.0):
        sx, sgx, _ = at(x)
        if sgx is not None:
            ax.plot([sx], [sgx], "s", ms=8, mfc="none",
                    mec="crimson", mew=1.6, zorder=5)
            ax.annotate("%g" % x, (sx, sgx), textcoords="offset points",
                        xytext=(6, 6), color="crimson", fontsize=9)
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\sigma$ [fb]")
    ttl = "WHIZARD 3.1.5 LO  $e^+e^-\\to\\mu^-\\bar\\nu_\\mu u\\bar d$"
    if ratio is not None:
        ttl += "    $\\sigma(240)/\\sigma(163)=%.3f$" % ratio
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, "mnuqq_lo_scan." + ext), dpi=140)
    print("\nWrote: %s/mnuqq_lo_scan.{pdf,png,csv}" % OUTDIR)


if __name__ == "__main__":
    main()
