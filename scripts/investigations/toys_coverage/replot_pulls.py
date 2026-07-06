"""Re-render toy-ensemble pull histograms from a saved toys_*.csv.

Same panels as toy_ensemble.py's inline plot, but publication-grade
(PDF + PNG, Gaussian overlay, pull mean/width/coverage in the panel
titles) and without re-running the 1000 fits.

    PYTHONPATH=. python3 scripts/investigations/toys_coverage/replot_pulls.py \
        [plots/toys_coverage/toys_production_full.csv]
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(csv_path: str) -> int:
    rows = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    r = rows[rows[:, 0] > 0.5]
    stem = os.path.splitext(csv_path)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, label, dv, sv in ((axes[0], r"$m_W$", r[:, 1], r[:, 2]),
                              (axes[1], r"$\Gamma_W$", r[:, 3], r[:, 4])):
        pull = dv / sv
        n = len(pull)
        mean, std = float(np.mean(pull)), float(np.std(pull, ddof=1))
        cov = float(np.mean(np.abs(dv) <= sv))
        ax.hist(pull, bins=30, range=(-4, 4), histtype="stepfilled",
                color="#377eb8", alpha=0.35, edgecolor="#377eb8")
        xs = np.linspace(-4, 4, 200)
        binw = 8 / 30
        ax.plot(xs, n * binw * np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi),
                color="#e41a1c", lw=1.4, label="unit Gaussian")
        ax.set_xlabel(f"pull({label})")
        ax.set_ylabel("toys / bin")
        ax.set_title(f"{label}:  mean $= {mean:+.3f}$,  width $= {std:.3f}$,  "
                     f"cov($\\pm1\\sigma$) $= {100*cov:.1f}\\,\\%$", fontsize=10)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(f"Self-consistent toys, production configuration (N={len(r)})",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}_pulls.{ext}", dpi=140)
        print(f"  wrote {stem}_pulls.{ext}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join("plots", "toys_coverage",
                                    "toys_production_full.csv")))
