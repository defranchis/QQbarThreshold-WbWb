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
def _fit_scenario_caption(fit, *, ax=None, x=0.92, y=0.17):
    """Two-line process + generator + BES annotation, drawn at the same
    lower-right position and fontsizes as the legacy ``process_annotation``.
    Line 1 merges the card's ``PROCESS_LABEL_SHORT`` with
    ``GENERATOR_LABEL_SHORT``; line 2 is ``BES_LABEL``. Card-driven so
    every panel of :func:`plot_fit_scenario` carries an identical caption."""
    ax = ax or plt.gca()
    process_short = getattr(fit.card, "PROCESS_LABEL_SHORT",
                            fit.card.PROCESS_LABEL.split(" at ")[0])
    gen_short = getattr(fit.card, "GENERATOR_LABEL_SHORT", "")
    lines = [f"{process_short} {gen_short}".strip()]
    if fit.card.BES_LABEL:
        lines.append(fit.card.BES_LABEL)
    for i, line in enumerate(lines):
        ax.text(x, y - i * 0.04, line, transform=ax.transAxes,
                ha="right", fontsize=23 if i == 0 else 21)


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
    _fit_scenario_caption(fit)
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
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel(r"$\sigma / \sigma_{\rm nom}$")
    plt.title(projection_title(fit.scenario_dict["total_lumi"]), loc="right", fontsize=20)
    plt.legend(loc="upper left", fontsize=20)
    xlim = _scan_xlim(fit)
    if xlim is not None:
        plt.xlim(*xlim)
    _fit_scenario_caption(fit)

    save_figure(fit.plot_dir, f"fit_scenario_ratio_{suffix}", also_pdf=not fit.asimov)


def plot_parameter_variations(fit):
    """Per-parameter variation templates, normalised to the nominal lineshape.
    Legend annotates each curve with the ±Δ magnitude (from the card's
    ``PARAMETERS[name]["variation"]``) so the reader sees at a glance which
    variation the line corresponds to.
    """
    plt.figure()
    xsec_nom = fit.template()
    plt.plot(xsec_nom["ecm"], np.ones(len(xsec_nom["ecm"])),
             label="Nominal model", linestyle="--", color="C0")
    binned_kinds = set(fit._systematics_meta["binned"])
    poi_display = getattr(fit.card, "POI_DISPLAY", {})
    math_labels = getattr(fit.card, "PARAM_MATH_LABELS", {})
    for i, name in enumerate(fit.param_names):
        if name in binned_kinds or i in fit._per_bin_meta:
            continue
        xsec_var = fit.template(f"{name}_var")
        factor = 1000 if name == "sw2" else 1
        ratio = (xsec_var["xsec"] / xsec_nom["xsec"] - 1) * factor + 1
        inv_ratio = (xsec_nom["xsec"] / xsec_var["xsec"] - 1) * factor + 1
        # Compose "$<sym> \pm <Δ>$ <unit>" using the POI display spec when
        # available (POI's scale/unit), or fall back to the math label and
        # raw card variation.
        spec = poi_display.get(name, {})
        symbol = spec.get("symbol", math_labels.get(name, name))
        delta = float(fit.card.PARAMETERS[name]["variation"])
        scale = float(spec.get("scale", 1.0))
        unit = spec.get("unit", "")
        delta_disp = delta * scale
        delta_str = f"{int(delta_disp)}" if unit == "MeV" else f"{delta_disp:g}"
        unit_str = f" {unit}" if unit else ""
        label = rf"${symbol} \pm {delta_str}${unit_str}"
        plt.plot(xsec_nom["ecm"], ratio, label=label, color=f"C{i+1}")
        plt.plot(xsec_nom["ecm"], inv_ratio, linestyle="--", color=f"C{i+1}")
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel("Cross section variation")
    # Opt-in scan-window clipping (only set on cards that want it — WW does,
    # WbWb keeps its legacy full-template view).
    if getattr(fit.card, "RESTRICT_PARAM_VARIATIONS_PLOT_TO_SCAN", False):
        xlim = _scan_xlim(fit)
        if xlim is not None:
            plt.xlim(*xlim)
    plt.legend(fontsize=20)
    save_figure(fit.plot_dir, "param_variations", also_pdf=False)


# ---------------------------------------------------------------------------
# Fit-input validation plots
# ---------------------------------------------------------------------------
# Read the on-disk templates the fit consumes and reproduce the
# m_W / Γ_W variation visualisations directly from them — no re-evaluation
# of the cross-section chain. The point is to *prove* the fit inputs
# carry the expected behaviour. The ±σ ratio is the ±(POI-variation)/σ_nom
# at the variation magnitude declared in the card; the negative side is
# the linear-morph reflection (2σ_nom − σ_var)/σ_nom, which is exactly the
# template the fit's morphing extrapolates to.
def param_axis_label(card, name: str) -> str:
    """Compose a full axis label ``"$<math>$ [<unit>]"`` for parameter
    ``name`` from the card's ``PARAM_MATH_LABELS`` and ``PARAM_UNITS``
    dicts. Falls back to the bare name if no math symbol is registered."""
    math = getattr(card, "PARAM_MATH_LABELS", {}).get(name, name)
    unit = getattr(card, "PARAM_UNITS", {}).get(name, "")
    return rf"${math}$" + (f" [{unit}]" if unit else "")


def _scan_xlim(fit, pad_factor: float = 0.05):
    """Return ``(lo, hi)`` for the fit's scan window. Card may declare an
    explicit ``SCAN_XLIM = (lo, hi)`` override (used by WbWb to preserve
    legacy plot bounds); otherwise the range is derived from
    ``card.SCENARIO["scan_min" / "scan_max"]`` with a small padding so
    edge points stay clear of the axis. Returns ``None`` when
    ``add_last_ecm`` is set, so the caller can display the full template
    range including the far reference point."""
    if fit.scenario_dict.get("add_last_ecm"):
        return None
    explicit = getattr(fit.card, "SCAN_XLIM", None)
    if explicit is not None:
        return tuple(explicit)
    lo = float(fit.card.SCENARIO["scan_min"])
    hi = float(fit.card.SCENARIO["scan_max"])
    pad = (hi - lo) * pad_factor
    return (lo - pad, hi + pad)


def _fit_input_poi_variations(fit):
    """For each POI return (name, math_label, Δ_display, unit_str). Iterates
    ``fit.tracked_pois()`` (POIs declared in ``card.POI_DISPLAY`` and free
    in the fit — constrained nuisances like α_s are skipped). Reads the
    math symbol from ``card.POI_DISPLAY[name]["symbol"]`` and rescales the
    card's variation by ``POI_DISPLAY[name]["scale"]`` into the declared
    display unit."""
    pd = fit.card.POI_DISPLAY
    out = []
    for name in fit.tracked_pois():
        spec = pd.get(name, {})
        delta = float(fit.card.PARAMETERS[name]["variation"])
        scale = float(spec.get("scale", 1.0))
        unit = spec.get("unit", "")
        symbol = spec.get("symbol", name)
        out.append((name, symbol, delta * scale, unit))
    return out


def plot_fit_input_ratios(fit):
    """Two-panel σ(var)/σ(nom) for ``mass`` and ``width`` from the fit's
    own templates. Equivalent to the diagnostic ``ratios_mW_GammaW`` plot
    but sourced from disk so it reflects what the fit actually consumes.
    """
    xsec_nom = fit.template()
    ecm = np.asarray(xsec_nom["ecm"])
    sig_nom = np.asarray(xsec_nom["xsec"])

    pois = _fit_input_poi_variations(fit)
    if not pois:
        return

    n = len(pois)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 4.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    xlim = _scan_xlim(fit)
    for ax, (name, math_label, delta_disp, unit) in zip(axes, pois):
        sig_p = np.asarray(fit.template(f"{name}_var")["xsec"])
        ratio_p = sig_p / sig_nom
        ratio_m = 2.0 - ratio_p   # linear-morph −Δ counterpart
        d_str = f"{int(delta_disp)}" if unit == "MeV" else f"{delta_disp:g}"
        u = f" {unit}" if unit else ""
        ax.plot(ecm, ratio_p, color="#a50f15", linewidth=1.8,
                label=rf"${math_label} + {d_str}${u} (template)")
        ax.plot(ecm, ratio_m, color="#08519c", linewidth=1.8,
                label=rf"${math_label} - {d_str}${u} (linear morph)")
        ax.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
        ax.set_ylabel(rf"$\sigma({math_label} \pm \delta)/\sigma_{{\rm nom}}$")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.legend(loc="best", fontsize=16, framealpha=0.9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    plt.tight_layout()
    save_figure(fit.plot_dir, "fit_input_ratios")


def plot_fit_input_azzurri_overlay(fit):
    """Single square panel: every POI's variation band overlaid on the
    nominal σ, each linearly inflated by ``card.AZZURRI_OVERLAY_INFLATE``
    so the band is visible (×100 turns a ±10 MeV mass/width template into
    a ±1 GeV-equivalent band — Azzurri 2107.04444 Fig. 1 scale). The
    deviation σ_var − σ_nom is read straight from the fit's templates
    and scaled — purely visual, no extra physics."""
    xsec_nom = fit.template()
    ecm = np.asarray(xsec_nom["ecm"])
    sig_nom = np.asarray(xsec_nom["xsec"])

    pois = _fit_input_poi_variations(fit)
    if not pois:
        return

    palette = [("#9e6bbf", "#54278f"),
               ("#4daf4a", "#1b7837"),
               ("#fdae6b", "#a63603"),
               ("#9ecae1", "#08519c")]

    inflate = int(getattr(fit.card, "AZZURRI_OVERLAY_INFLATE", 100))
    # Optional σ_template → σ_WW unit conversion so the y-axis matches the
    # σ_WW scale of the diagnostic Azzurri overlay. For WW the templates
    # are σ_observed = σ_WW × BR_inclusive(μν qq̄); divide by that BR to
    # recover σ_WW (≈ 0–15 pb). Cards that don't expose W-leptonic /
    # W-hadronic BR primitives (e.g. WbWb) plot the raw template σ.
    br_munu = getattr(fit.card, "BR_W_MUNU", None)
    br_had = getattr(fit.card, "BR_W_HAD", None)
    divisor = 2.0 * br_munu * br_had if (br_munu and br_had) else 1.0
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for (name, math_label, delta_disp, unit), (color_band, color_edge) in zip(pois, palette):
        sig_var = np.asarray(fit.template(f"{name}_var")["xsec"])
        sig_p = (sig_nom + inflate * (sig_var - sig_nom)) / divisor
        sig_m = (sig_nom - inflate * (sig_var - sig_nom)) / divisor
        d_str = f"{int(delta_disp)}" if unit == "MeV" else f"{delta_disp:g}"
        u = f" {unit}" if unit else ""
        ax.fill_between(ecm, sig_m, sig_p, color=color_band, alpha=0.30,
                        label=rf"${math_label} \pm {d_str}${u} $\times {inflate}$")
        ax.plot(ecm, sig_p, color=color_edge, linewidth=1.0, linestyle="--")
        ax.plot(ecm, sig_m, color=color_edge, linewidth=1.0, linestyle=":")

    ax.plot(ecm, sig_nom / divisor, color="black", linewidth=1.8, label="nominal (template)")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\sigma_{WW}$ [pb]" if divisor != 1.0 else r"$\sigma$ [pb]")
    # Match the diagnostic Azzurri overlay's view: full lineshape range,
    # so the rising shoulder + above-threshold plateau are both visible.
    ax.set_xlim(*getattr(fit.card, "AZZURRI_OVERLAY_XLIM", (155, 170)))
    ax.set_ylim(*getattr(fit.card, "AZZURRI_OVERLAY_YLIM", (0.0, 13.0)))
    ax.legend(loc="upper left", fontsize=16, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure(fit.plot_dir, "fit_input_azzurri_overlay")


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
