"""Plotting helpers shared between WbWb and WW fits.

Two scopes:

* :func:`save_figure`, :func:`projection_title`, :func:`process_annotation` —
  shared decoration applied by every scan / plot helper.
* :func:`plot_fit_scenario`, :func:`plot_parameter_variations` — top-level
  diagnostic plots historically baked into the ``fit`` class itself.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)


# ---------------------------------------------------------------------------
# Decoration helpers
# ---------------------------------------------------------------------------
def projection_title(total_lumi, unit="fb", fmt="{:.0f}"):
    """Top-right "Projection (N fb^-1 / ab^-1)" title used throughout."""
    if unit == "fb":
        amount = total_lumi / 1e3
    elif unit == "ab":
        amount = total_lumi / 1e6
    else:
        raise ValueError(f"Unknown lumi unit: {unit}")
    return rf"$\mathit{{Projection}}$ ({fmt.format(amount)} {unit}$^{{-1}}$)"


def process_annotation(card, *, ax=None, x=0.92, y=0.17, ha="right", offset=0.0,
                       include_reference=False, chain_label=None):
    """Stamp the standard process / generator / BES annotation.

    ``chain_label`` overrides the static ``card.GENERATOR_LABEL`` — pass
    the value read from the actual template preamble (via
    ``FitCore.template_metadata()``) so the plot describes the templates
    being fit, not the live card.
    """
    ax = ax or plt.gca()
    lines = [card.PROCESS_LABEL]
    if include_reference:
        gen_label = chain_label if chain_label else card.GENERATOR_LABEL
        if gen_label:
            lines.append(gen_label)
        if card.GENERATOR_REF:
            lines.append(card.GENERATOR_REF)
    if card.BES_LABEL:
        lines.append(card.BES_LABEL)
    for i, line in enumerate(lines):
        ax.text(x, y - i * 0.04 + offset, line, fontsize=23 if i == 0 else 21,
                transform=ax.transAxes, ha=ha)


_ACTIVE_CHAIN_LABEL: str | None = None


def set_active_chain_label(label: str | None) -> None:
    """Set the chain-summary string stamped at the bottom of every figure
    saved via :func:`save_figure`. Pass ``None`` to disable. Used by the
    fit entry points so every fit plot reflects which physics chain
    produced the input templates.
    """
    global _ACTIVE_CHAIN_LABEL
    _ACTIVE_CHAIN_LABEL = label


def save_figure(plot_dir, name, *, also_pdf=True, clf=True):
    """Save ``<plot_dir>/<name>.png`` and (optionally) the matching ``.pdf``.

    If a chain label has been set via :func:`set_active_chain_label`, it
    is stamped at the bottom of the figure before saving.
    """
    if _ACTIVE_CHAIN_LABEL:
        plt.gcf().text(0.5, 0.005, _ACTIVE_CHAIN_LABEL, ha="center", va="bottom",
                       fontsize=7, color="0.35")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{name}.png"))
    if also_pdf:
        plt.savefig(os.path.join(plot_dir, f"{name}.pdf"))
    if clf:
        plt.clf()


# ---------------------------------------------------------------------------
# Top-level diagnostics
# ---------------------------------------------------------------------------
def plot_fit_scenario(fit):
    """Pseudo/asimov data vs. fitted lineshape (and ratio panel)."""
    suffix = "asimov" if fit.asimov else "pseudo"
    xsec_nom = fit.template()
    xsec_pseudo = fit.template(fit.pseudodata_tag)
    xsec_fit = _fit_xsec_with_uncert(fit)

    if not fit.scenario_dict["add_last_ecm"]:
        xsec_nom = xsec_nom[:-1]
        xsec_fit = xsec_fit[:-1]

    # Absolute cross sections ------------------------------------------------
    plt.figure()
    label = "Asimov data" if fit.asimov else "Pseudo data"
    plt.errorbar(fit.xsec_scenario["ecm"], fit.pseudo_data_scenario,
                 yerr=fit.unc_pseudodata_scenario, fmt=".", label=label)
    plt.plot(xsec_nom["ecm"], xsec_fit["xsec"], label="Fitted model", linewidth=2)
    plt.plot(xsec_nom["ecm"], xsec_nom["xsec"], label="Nominal model", linestyle="--", linewidth=2)
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel("Cross section [pb]")
    plt.legend()
    save_figure(fit.plot_dir, f"fit_scenario_{suffix}", also_pdf=False)

    # Ratio to nominal -------------------------------------------------------
    plt.figure()
    nom_at_scenario = fit.slice_to_scenario(xsec_nom)["xsec"]
    plt.errorbar(fit.xsec_scenario["ecm"],
                 fit.pseudo_data_scenario / nom_at_scenario,
                 yerr=fit.unc_pseudodata_scenario / nom_at_scenario,
                 fmt=".", label=label, linewidth=2, markersize=10)
    plt.plot(xsec_nom["ecm"], xsec_fit["xsec"] / xsec_nom["xsec"], label="Fitted lineshape", linewidth=2)
    plt.fill_between(xsec_nom["ecm"],
                     (xsec_fit["xsec"] - xsec_fit["unc"]) / xsec_nom["xsec"],
                     (xsec_fit["xsec"] + xsec_fit["unc"]) / xsec_nom["xsec"],
                     alpha=0.5, label="Post-fit uncertainty")
    pseudo_label = "Pseudodata cross section" if not fit.asimov else "Asimov cross section"
    plt.plot(xsec_pseudo["ecm"], xsec_pseudo["xsec"] / xsec_nom["xsec"],
             label=pseudo_label, linestyle="--", linewidth=2)
    plt.axhline(1, color="grey", linestyle="--", label="Reference cross section", linewidth=2)
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel(f"{fit.card.PROCESS_LABEL.split(' ')[0]} total cross section ratio")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    if not fit.scenario_dict["add_last_ecm"]:
        plt.xlim(339.7, 347.3)
    process_annotation(fit.card, x=0.96, y=0.45, include_reference=True,
                       chain_label=fit.template_metadata().get("chain"))

    save_figure(fit.plot_dir, f"fit_scenario_ratio_{suffix}", also_pdf=not fit.asimov)


def plot_parameter_variations(fit):
    """Per-parameter variation templates, normalised to the nominal lineshape."""
    plt.figure()
    xsec_nom = fit.template()
    plt.plot(xsec_nom["ecm"], np.ones(len(xsec_nom["ecm"])),
             label="Nominal model", linestyle="--", color="C0")
    binned_kinds = set(fit._systematics_meta["binned"])
    for i, name in enumerate(fit.param_names):
        if name in binned_kinds or i in fit._per_bin_meta:
            continue
        xsec_var = fit.template(f"{name}_var")
        factor = 1000 if name == "sw2" else 1
        ratio = (xsec_var["xsec"] / xsec_nom["xsec"] - 1) * factor + 1
        inv_ratio = (xsec_nom["xsec"] / xsec_var["xsec"] - 1) * factor + 1
        label = fit.card.PARAM_LABELS.get(name, name)
        plt.plot(xsec_nom["ecm"], ratio, label=label, color=f"C{i+1}")
        plt.plot(xsec_nom["ecm"], inv_ratio, linestyle="--", color=f"C{i+1}")
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel("Cross section variation")
    plt.legend()
    plt.title("Parameter variations normalized to nominal cross section")
    save_figure(fit.plot_dir, "param_variations", also_pdf=False)


def _fit_xsec_with_uncert(fit):
    """Return the fitted-model cross section with per-point uncertainty.

    Mirrors the original ``getXsecParams``.
    """
    import pandas as pd

    params = fit.fit_params_with_cov()
    th = np.array(fit.template()["xsec"])
    binned_kinds = set(fit._systematics_meta["binned"])
    for i, name in enumerate(fit.param_names):
        if name in binned_kinds or i in fit._per_bin_meta:
            continue
        p = params[fit._idx[name]]
        th = th * (1 + p * np.array(fit.morph_dict[name]["xsec"]))
    return pd.DataFrame({"ecm": fit.l_ecm,
                         "xsec": [t.n for t in th],
                         "unc": [t.s for t in th]})
