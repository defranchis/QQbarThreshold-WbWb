"""Quantify the correlated-vs-decorrelated seed de-noise gain (lnuqq A/B).

Reads the partonic σ̂_NLO at {nominal, mp10, mm10} × √ŝ for both seed modes and
compares the m_W response  r±(√ŝ) = σ̂(±10)/σ̂(nominal) − 1:

  • antisym  ½[r₊ − r₋]   = the genuine linear m_W slope (signal)
  • even     ½[r₊ + r₋]   = curvature + noise (≈0 if clean; inflated by MC noise)

Correlated seeds (nominal & variation share the MC sequence) should cancel the
even-part noise → small, smooth |even|; decorrelated should leave it large/scattered.

Output: plots/indep_mocanlo/seedtest_response.png + a printed table.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = ("/afs/cern.ch/work/m/mdefranc/private/FCC/"
           "QQbar_threshold/mocanlo/grid_gen/seedtest_results")
ECMS = (159.0, 160.0, 161.0, 162.0, 163.0)
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def sigma_nlo(channel, vp, ecm, mode):
    sfx = "_gf_decorr" if mode == "decorr" else "_gf"
    path = os.path.join(RESULTS, f"{channel}_{vp}_ecm{ecm:.4f}{sfx}.csv")
    if not os.path.exists(path):
        return None, None
    r = list(csv.DictReader(open(path)))[0]
    return float(r["sigma_nlo"]), float(r["err_nlo"])


def responses(mode):
    """Return dict √ŝ → (r_up, r_dn, err_up, err_dn) in %, or None if missing."""
    out = {}
    for e in ECMS:
        s0, e0 = sigma_nlo("lnuqq", "nominal", e, mode)
        sp, ep = sigma_nlo("lnuqq", "mp10", e, mode)
        sm, em = sigma_nlo("lnuqq", "mm10", e, mode)
        if None in (s0, sp, sm):
            continue
        out[e] = (100 * (sp / s0 - 1), 100 * (sm / s0 - 1),
                  100 * ep / sp, 100 * em / sm)
    return out


def main():
    res = {m: responses(m) for m in ("corr", "decorr")}
    print(f"{'mode':7s} {'√ŝ':>6s} {'r(+10)%':>9s} {'r(-10)%':>9s} "
          f"{'antisym%':>9s} {'even%':>9s}")
    summary = {}
    for mode in ("corr", "decorr"):
        evens = []
        for e in sorted(res[mode]):
            rp, rm, _, _ = res[mode][e]
            anti = 0.5 * (rp - rm); even = 0.5 * (rp + rm)
            evens.append(even)
            print(f"{mode:7s} {e:6.1f} {rp:9.3f} {rm:9.3f} {anti:9.3f} {even:9.3f}")
        if evens:
            summary[mode] = np.sqrt(np.mean(np.array(evens) ** 2))
    print()
    for mode in summary:
        print(f"  RMS(even) [{mode:6s}] = {summary[mode]:.3f}%   "
              f"(noise proxy; smaller = cleaner)")
    if len(summary) == 2:
        print(f"  → correlated seeds reduce even-part noise by "
              f"{summary['decorr']/summary['corr']:.1f}×")

    # plot
    outdir = os.path.join(_REPO, "plots", "indep_mocanlo")
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for mode, col in (("decorr", "tab:orange"), ("corr", "tab:blue")):
        es = sorted(res[mode])
        if not es:
            continue
        rp = [res[mode][e][0] for e in es]; rm = [res[mode][e][1] for e in es]
        ax.plot(es, rp, "o-", color=col, label=f"{mode}: r(+10 MeV)")
        ax.plot(es, rm, "s--", color=col, alpha=0.6, label=f"{mode}: r(−10 MeV)")
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"$\sqrt{\hat s}$ [GeV]")
    ax.set_ylabel(r"$\sigmâ(m_W\pm10)/\sigmâ(m_W)-1$ [%]")
    ax.set_title("Correlated vs decorrelated seeds — lnuqq m_W response (50k ev)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "seedtest_response.png"), dpi=140)
    print(f"\nwrote {outdir}/seedtest_response.png")


if __name__ == "__main__":
    main()
