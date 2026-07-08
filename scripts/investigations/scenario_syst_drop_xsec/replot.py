"""Re-plot the scenario systematics-breakdown bar charts (m_W, Gamma_W) from the
cached ``scenario_compare_syst.csv``, dropping the ``xsec`` source.

The cross-section placeholder is not among the scenario-fit nuisances, so its
breakdown value is ``None`` (empty in the CSV) for every scenario/POI and its
bar would render as a meaningless 0.00.  This matches the (now-general) filter
in ``framework/process/ww/scenario_compare.py`` Figure 3, but regenerates the
two figures from the cached CSV so no fits are re-run.

Style is imported from ``scenario_compare`` (``_SCN_STYLE``, ``_SYST_ROWS``,
``_POI_TAG``) + ``plot_labels`` so the output is identical to the framework's
Figure 3, minus the xsec category.  Overwrites
``plots/scenario_compare/scenario_compare_syst_{mW,gW}.{pdf,png}`` in place.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cards import ww_default as card
from framework.process.ww import plot_labels
from framework.process.ww.scenario_compare import (
    _SCN_STYLE, _SYST_ROWS, _POI_TAG,
)

REPO = os.environ.get(
    "WW_REPO",
    "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold",
)
BASE = os.path.join(REPO, "plots", "scenario_compare", "scenario_compare")
CSV = BASE + "_syst.csv"


def load_csv(path):
    """data[scenario][poi][source] = float | None ; preserve scenario order."""
    data, order = {}, []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            scn, poi, src = row["scenario"], row["poi"], row["source"]
            raw = row["value_displayunit"].strip()
            data.setdefault(scn, {}).setdefault(poi, {})[src] = (
                float(raw) if raw else None)
            if scn not in order:
                order.append(scn)
    return data, order


def main():
    data, order = load_csv(CSV)
    names = [n for n in order if n in _SCN_STYLE]  # canonical scenario order
    pois = list(card.POI_DISPLAY)

    # keep sources with at least one finite value across all scenarios/POIs
    srcs = [s for s in _SYST_ROWS
            if any(data[n].get(poi, {}).get(s) is not None
                   for n in names for poi in pois)]
    dropped = [s for s in _SYST_ROWS if s not in srcs]
    print(f"[replot] sources kept: {srcs}")
    print(f"[replot] sources dropped (all-None): {dropped}")

    badge = (f"{plot_labels.process_label_short(card)} "
             f"{plot_labels.generator_label_short(card)}").strip()
    L_ab = card.SCENARIO["total_lumi"] / 1e6

    x = np.arange(len(srcs))
    w = 0.8 / max(len(names), 1)
    for poi in pois:
        disp = card.POI_DISPLAY[poi]
        tag = _POI_TAG.get(poi, poi)
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        for k, name in enumerate(names):
            st = _SCN_STYLE.get(name, dict(color=f"C{k}"))
            vals = [(data[name].get(poi, {}).get(s) or 0.0) for s in srcs]
            xb = x + (k - (len(names) - 1) / 2.0) * w
            bars = ax.bar(xb, vals, width=w, color=st["color"],
                          edgecolor="k", linewidth=0.4, label=name)
            for rect, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=6, rotation=90,
                            xytext=(0, 1), textcoords="offset points")
        ax.set_ylabel(rf"$\sigma({disp['symbol']})$ [{disp['unit']}]")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels(srcs, rotation=0)
        ax.set_xlabel("uncertainty source")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=len(names))
        ax.set_title(rf"$\sigma({disp['symbol']})$ breakdown across scan "
                     f"scenarios (same {L_ab:.1f} ab$^{{-1}}$)")
        fig.text(0.985, 0.005, badge, ha="right", va="bottom", fontsize=7,
                 color="0.4")
        fig.savefig(BASE + f"_syst_{tag}.pdf", bbox_inches="tight")
        fig.savefig(BASE + f"_syst_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[replot] wrote {BASE}_syst_{tag}.{{pdf,png}}")


if __name__ == "__main__":
    main()
