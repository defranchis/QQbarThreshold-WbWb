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


def _labels_module(card):
    """Resolve the per-process plot_labels module via ``card.PROCESS_ID``."""
    pid = getattr(card, "PROCESS_ID", None)
    if pid == "ww":
        from framework.process.ww import plot_labels as labels
    elif pid == "wbwb":
        from framework.process.wbwb import plot_labels as labels
    else:
        raise ValueError(
            f"card.PROCESS_ID={pid!r} is not registered with a plot_labels module")
    return labels


def _bes_label(card) -> str:
    """``"+ FCC-ee BES"`` when the card configures a non-zero beam-energy
    spread, empty otherwise. Auto-derived from ``BEAM_ENERGY_RES`` so the
    label can't drift out of sync with the actual smearing kernel."""
    return "+ FCC-ee BES" if float(getattr(card, "BEAM_ENERGY_RES", 0.0)) > 0.0 else ""


def process_annotation(card, *, ax=None, x=0.92, y=0.17, ha="right", offset=0.0,
                       include_reference=False, chain_label=None):
    """Stamp the standard process / generator / BES annotation.

    ``chain_label`` overrides the per-process static generator label — pass
    the value read from the actual template preamble (via
    ``FitCore.template_metadata()``) so the plot describes the templates
    being fit, not the live card.
    """
    ax = ax or plt.gca()
    labels = _labels_module(card)
    lines = [labels.process_label(card)]
    if include_reference:
        gen_label = chain_label if chain_label else labels.generator_label(card)
        if gen_label:
            lines.append(gen_label)
        if labels.GENERATOR_REF:
            lines.append(labels.GENERATOR_REF)
    bes = _bes_label(card)
    if bes:
        lines.append(bes)
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
    """Two-line process + generator-short + BES annotation. Line 1 merges
    the per-process short process label with the short generator badge;
    line 2 is the auto-derived BES tag (empty if the card has no BES)."""
    ax = ax or plt.gca()
    labels = _labels_module(fit.card)
    lines = [f"{labels.process_label_short(fit.card)} {labels.generator_label_short(fit.card)}".strip()]
    bes = _bes_label(fit.card)
    if bes:
        lines.append(bes)
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

    save_figure(fit.plot_dir, f"fit_scenario_ratio_{suffix}", also_pdf=True)


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
    for i, name in enumerate(fit.param_names):
        if name in binned_kinds or i in fit._per_bin_meta:
            continue
        xsec_var = fit.template(f"{name}_var")
        # Visible-deviation amplification: scale lifts the lineshape ratio
        # into display units (×1000 for mass/width in GeV→MeV, ×1 for
        # dimensionless POIs) so the analysis-target distortion is visible.
        _, scale, _ = _poi_display_spec(fit.card, name)
        ratio = (xsec_var["xsec"] / xsec_nom["xsec"] - 1) * scale + 1
        inv_ratio = (xsec_nom["xsec"] / xsec_var["xsec"] - 1) * scale + 1
        plt.plot(xsec_nom["ecm"], ratio, label=_poi_pm_label(fit.card, name),
                 color=f"C{i+1}")
        plt.plot(xsec_nom["ecm"], inv_ratio, linestyle="--", color=f"C{i+1}")
    plt.xlabel(r"$\sqrt{s}$ [GeV]")
    plt.ylabel("Cross section variation")
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
    ``name``. Math symbol comes from ``card.POI_DISPLAY[name]["symbol"]``
    (POIs) or ``card.PARAM_MATH_LABELS`` (nuisances), falling back to the
    bare name. Unit is the parameter's *native* unit on the plot axis —
    ``card.PARAM_UNITS`` — distinct from ``POI_DISPLAY["unit"]`` which is
    the syst-table display unit (e.g. MeV vs the GeV the axis carries)."""
    poi_spec = getattr(card, "POI_DISPLAY", {}).get(name)
    if poi_spec is not None:
        math = poi_spec.get("symbol", name)
    else:
        math = getattr(card, "PARAM_MATH_LABELS", {}).get(name, name)
    unit = getattr(card, "PARAM_UNITS", {}).get(name, "")
    return rf"${math}$" + (f" [{unit}]" if unit else "")


def _format_delta(delta_disp: float, unit: str) -> str:
    """Δ magnitude as ``"<value> <unit>"`` — integer for MeV, %g elsewhere."""
    d_str = f"{int(delta_disp)}" if unit == "MeV" else f"{delta_disp:g}"
    return d_str + (f" {unit}" if unit else "")


def _poi_display_spec(card, name: str) -> tuple[str, float, str]:
    """``(math_symbol, display_scale, unit)`` for ``name`` from
    ``card.POI_DISPLAY``. For non-POI nuisances (no POI_DISPLAY entry, e.g. a
    global flat normalization nuisance) the math symbol falls back to
    ``card.PARAM_MATH_LABELS``; failing that, the bare name with no scale/unit."""
    spec = getattr(card, "POI_DISPLAY", {}).get(name, {})
    symbol = (spec.get("symbol")
              or getattr(card, "PARAM_MATH_LABELS", {}).get(name, name))
    return symbol, float(spec.get("scale", 1.0)), spec.get("unit", "")


def _poi_pm_label(card, name: str) -> str:
    """LaTeX ``"<sym> ± <Δ> <unit>"`` for parameter ``name``. Single point
    that resolves the POI display spec + the card's variation magnitude
    + the dimension-aware Δ formatting — used by every plot that draws
    "POI ± Δ" curves so they stay consistent."""
    symbol, scale, unit = _poi_display_spec(card, name)
    params = getattr(card, "PARAMETERS", {})
    if name in params:
        delta = float(params[name]["variation"]) * scale
    else:
        # nuisance without a PARAMETERS entry (e.g. a global flat normalization
        # nuisance): annotate with its morph unit INPUT_VAR[name].
        delta = float(getattr(card, "INPUT_VAR", {}).get(name, 1.0)) * scale
    return rf"${symbol} \pm {_format_delta(delta, unit)}$"


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
    """For each POI return ``(name, math_label, Δ_display, unit_str)``.
    Iterates ``fit.tracked_pois()`` (POIs declared in ``card.POI_DISPLAY``
    and free in the fit — constrained nuisances like α_s are skipped).
    Symbol / scale / unit come from :func:`_poi_display_spec`."""
    out = []
    for name in fit.tracked_pois():
        symbol, scale, unit = _poi_display_spec(fit.card, name)
        delta = float(fit.card.PARAMETERS[name]["variation"])
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
        delta_str = _format_delta(delta_disp, unit)
        ax.plot(ecm, ratio_p, color="#a50f15", linewidth=1.8,
                label=rf"${math_label} + {delta_str}$ (template)")
        ax.plot(ecm, ratio_m, color="#08519c", linewidth=1.8,
                label=rf"${math_label} - {delta_str}$ (linear morph)")
        ax.axhline(1.0, color="grey", alpha=0.4, linewidth=0.7)
        ax.set_ylabel(rf"$\sigma({math_label} \pm \delta)/\sigma_{{\rm nom}}$")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.legend(loc="best", fontsize=16, framealpha=0.9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel(r"$\sqrt{s}$ [GeV]")
    plt.tight_layout()
    save_figure(fit.plot_dir, "fit_input_ratios")


# Display-scale "wide band" per-POI shifts for the Azzurri overlay
# (Azzurri 2107.04444 Fig. 1 scale). ±1 GeV on mass and width to match
# the literature reference. Each value is in the POI's *native* unit (GeV).
# Only tracked POIs are overlaid (`_fit_input_poi_variations` skips the
# constrained α_s/α_em_isr nuisances), so no nuisance entry is needed here.
_AZZURRI_POI_DELTA = {"mass": 1.0, "width": 1.0}


def plot_fit_input_azzurri_overlay(fit):
    """Single square panel: each POI's σ band evaluated at the nominal ±
    a wide display-scale shift (`_AZZURRI_POI_DELTA`; ±1 GeV for mass to
    match Azzurri 2107.04444 Fig. 1), directly through the σ chain via
    `WWGenerator.from_card` — full non-linear response (Born + NLO loops +
    NNLO + ISR + anchor), not a linear extrapolation of the variation
    templates."""
    from framework.process.ww.generator import WWGenerator

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

    # σ_template → σ_WW unit conversion so the y-axis matches the
    # diagnostic σ_WW scale. Templates carry σ_observed = σ_WW × BR;
    # divide by BR to recover σ_WW.
    br_munu = getattr(fit.card, "BR_W_MUNU", None)
    br_had = getattr(fit.card, "BR_W_HAD", None)
    divisor = 2.0 * br_munu * br_had if (br_munu and br_had) else 1.0

    # Build a card-faithful generator once; reuse for every (POI ± Δ) call.
    gen = WWGenerator.from_card(fit.card)
    from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq

    def sigma_at(values):
        """σ_observed at the given POI values (mass, width, alphas overrides)."""
        mW = float(values.get("mass", fit.card.PARAMETERS["mass"]["nominal"]))
        gW = float(values.get("width", fit.card.PARAMETERS["width"]["nominal"]))
        a_s = gen.alpha_s + float(values.get("alphas", 0.0))
        return sigma_observed_munuqq(
            ecm,
            mW=mW, gammaW=gW,
            channel=gen.channel,
            include_coulomb=gen.include_coulomb,
            bfs=gen.bfs,
            include_NLO_hard_decay=gen.include_NLO_hard_decay,
            include_BFS_NNLO=gen.include_BFS_NNLO,
            apply_delta_QCD=gen.apply_delta_QCD,
            alpha_s=a_s, alpha_s_ref=gen.alpha_s,
            br_convention=gen.br_convention,
            apply_whizard_anchor=gen.apply_whizard_anchor,
            whizard_anchor_source=gen.whizard_anchor_source,
            isr_scheme=gen.isr_scheme,
            # Match the production ISR scheme of the on-disk nominal template
            # (generator.do_scan): without these the ±Δ bands default to
            # isr_nll=False (LL+exp) while the nominal curve is NLL — two
            # different schemes on the same axes.
            isr_nll=gen.isr_nll,
            isr_emela_ll=gen.isr_emela_ll,
            isr_emela_pert_order=gen.isr_emela_pert_order,
            isr_emela_fac_scheme=gen.isr_emela_fac_scheme,
            isr_emela_ren_scheme=gen.isr_emela_ren_scheme,
            alpha_em_isr=gen.alpha_em_isr,
            coulomb_kc_safe=gen.coulomb_kc_safe,
            decay_uses_full_born=gen.decay_uses_full_born,
            m_t=gen.m_t, M_H=gen.M_H, MZ=gen.MZ,
        )

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for (name, math_label, _delta_disp_template, unit), (color_band, color_edge) in zip(pois, palette):
        delta = float(_AZZURRI_POI_DELTA.get(name, 0.0))
        if delta == 0.0:
            continue
        nom_val = float(fit.card.PARAMETERS[name]["nominal"])
        sig_p = sigma_at({name: nom_val + delta}) / divisor
        sig_m = sigma_at({name: nom_val - delta}) / divisor
        # Wide-band shifts: label in GeV for mass/width (more readable than
        # "1000 MeV"); raw value for α_s.
        if name in ("mass", "width"):
            delta_str = rf"{delta:g}\,\mathrm{{GeV}}"
        else:
            delta_str = f"{delta:g}"
        ax.fill_between(ecm, sig_m, sig_p, color=color_band, alpha=0.30,
                        label=rf"${math_label} \pm {delta_str}$")
        ax.plot(ecm, sig_p, color=color_edge, linewidth=1.0, linestyle="--")
        ax.plot(ecm, sig_m, color=color_edge, linewidth=1.0, linestyle=":")

    ax.plot(ecm, sig_nom / divisor, color="black", linewidth=1.8, label="nominal (template)")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\sigma_{WW}$ [pb]" if divisor != 1.0 else r"$\sigma$ [pb]")
    # Full lineshape view: rising shoulder + above-threshold plateau.
    ax.set_xlim(155, 170)
    ax.set_ylim(0.0, 13.0)
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
