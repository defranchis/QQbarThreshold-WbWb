"""Extrapolate the *experimental* uncertainty from the reconstructed μνqq̄
channel to inclusive WW.

Motivation
----------
The threshold lineshape σ(√s) has the same shape for every WW decay channel:
in the production ``pdg-constant`` BR convention the branching ratio is a flat
multiplicative factor, so it cancels in the fit's *relative* response to
(m_W, Γ_W) and to every relative systematic. Going from the μνqq̄ channel to
the full inclusive WW sample therefore multiplies the event yield by
``1/B(μνqq̄)`` with ``B(μνqq̄) = 2·BR(W→μν)·BR(W→had) ≈ 0.143`` (a factor
≈ 7), and changes **nothing else** about the experimental fit:

  * the **statistical** term scales as 1/√N → shrinks by √(1/B) ≈ 2.6;
  * the **luminosity** uncertainty (δL/L, from the Bhabha counting, common to
    all channels), the **beam-energy** calibration/spread (BEC/BES, a √s
    effect), and the **parametric** α_s nuisance are all *relative* and map
    through unchanged — their MeV contributions to (m_W, Γ_W) are
    channel-independent.

So the naive "√B scaling of the *total* uncertainty" is wrong: only the stat
part scales. Running the *full* Asimov fit with the rescaled rate is the only
way to map every systematic correctly, which is what this module does.

Implementation
--------------
Since ``N = σ·L``, the inclusive rate boost is applied as an **effective
luminosity** scaling ``L → L/B`` — this reproduces the inclusive yield
*exactly* while leaving the cross-section templates (the lineshape) and every
relative systematic bit-identical, so no template regeneration is needed and
all systematics map by construction. The full experimental breakdown
(:func:`framework.common.systematics.compute_syst_breakdown`: stat + α_s +
BES + BEC + lumi, realistic priors — the ``--systTable`` configuration) is
computed at both the μνqq̄ and the inclusive effective luminosity and compared.

Scope / caveats
---------------
* **Experimental uncertainty only.** THEORY uncertainties are deliberately
  *not* extrapolated: they differ channel-by-channel (different final states,
  different modelling) and are reported separately (the μνqq̄-specific theory
  ladder / ISR-scheme study). This module touches only stat + beam +
  parametric terms.
* **Future parametric nuisance: α_em_isr (ISR input coupling).** Once the
  ISR coupling α(M_Z) is wired as a profiled nuisance (its FCC-ee Tera-Z input
  uncertainty ≈ 2.4×10⁻⁷, ~0.04 MeV — *not* the ISR α-scheme spread, which is
  theory), it is a *parametric/experimental* term that is **channel-common**
  (ISR is identical for every WW final state), so it maps through this
  extrapolation *unchanged*, exactly like luminosity/BES/BEC. No code change is
  needed here: ``_SYST_ROWS`` is derived from ``card.SYST_TABLE_ORDER``, so the
  new source appears automatically once added to the card.
* This is the *idealised* statistical ceiling — it treats the full WW rate as
  if reconstructed with the μνqq̄ per-event sensitivity (no channel-dependent
  background, purity or resolution). A real multi-channel combination sits
  between the μνqq̄ result and this bound; since the budget is luminosity-
  dominated, the gap is small.

Run via ``python3 doFit_ww.py --channelExtrap`` or import
:func:`run_channel_extrapolation`.
"""

from __future__ import annotations

import os

from cards import ww_default as card
from framework.common.systematics import compute_syst_breakdown
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.process.ww.scenario_compare import _SYST_ROWS, _grouped_syst
from framework.process.ww.xsec_calculator.eft_xsec import BR_INCLUSIVE_MUNUQQ

#: Inclusive-rate boost: μνqq̄ yield × 1/B(μνqq̄) → full WW yield.
INCLUSIVE_FACTOR = 1.0 / BR_INCLUSIVE_MUNUQQ


def _build_full_fit(lumi_scale: float) -> WWFit:
    """Full production fit (all POIs + α_s/BES/BEC/lumi nuisances, realistic
    priors — the ``--systTable`` configuration) on the card's baseline 7-point
    scan, with the effective luminosity scaled by ``lumi_scale``."""
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    S = card.SCENARIO
    fit.init_scenario(
        scan_min=S["scan_min"], scan_max=S["scan_max"], scan_step=S["scan_step"],
        total_lumi=S["total_lumi"] * lumi_scale,
        last_lumi=S["last_lumi"] * lumi_scale,
    )
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.fit_parameters()
    return fit


def run_channel_extrapolation(out=None):
    """Compute the experimental systematics breakdown for the μνqq̄ channel and
    its inclusive-WW extrapolation (effective lumi × 1/B), and emit a
    side-by-side comparison table."""
    out = out or os.path.join("plots", "channel_extrap")
    channels = [
        ("munuqq", 1.0),
        ("inclusive WW", INCLUSIVE_FACTOR),
    ]
    results = {}
    for label, scale in channels:
        print(f"\n[channel-extrap] {label}: effective lumi × {scale:.3f} ...")
        fit = _build_full_fit(scale)
        results[label] = compute_syst_breakdown(fit)
    _emit(channels, results, out)
    return results


def _emit(channels, results, out):
    names = [c[0] for c in channels]
    lines = []
    lines.append("WW experimental-uncertainty extrapolation: μνqq̄ → inclusive WW")
    lines.append(f"Inclusive rate boost = 1/B(μνqq̄) = {INCLUSIVE_FACTOR:.3f} "
                 f"(B = 2·BR_μν·BR_had = {BR_INCLUSIVE_MUNUQQ:.4f}).")
    lines.append("Applied as an effective-luminosity scaling (N=σ·L); the")
    lines.append("lineshape σ and every RELATIVE systematic are unchanged, so only")
    lines.append("the statistical term scales (∝1/√N). Full production fit (all POIs")
    lines.append("+ α_s/BES/BEC/lumi nuisances, realistic priors); baseline 7-point")
    lines.append("scan. THEORY uncertainties are channel-specific and NOT extrapolated.")
    lines.append("")
    for poi in card.POI_DISPLAY:
        disp = card.POI_DISPLAY[poi]
        ch = (f"{'source':10s} " + " ".join(f"{n:>16s}" for n in names) +
              f" {'stat-scaling':>13s}")
        lines.append("-" * len(ch))
        lines.append(f"σ({disp['symbol']})  [{disp['unit']}]")
        lines.append(ch)
        lines.append("-" * len(ch))
        for src in _SYST_ROWS + ["total exp"]:
            key = "total" if src == "total exp" else src
            vals = [_grouped_syst(results.get(n), poi, key) for n in names]
            cells = " ".join(f"{v:>16.2f}" if v is not None else f"{'—':>16s}"
                             for v in vals)
            # Annotate the expected √-scaling only for the stat row.
            note = ""
            if src == "stat" and all(v is not None for v in vals) and vals[1] > 0:
                note = f"×{vals[0]/vals[1]:>11.2f}"
            lines.append(f"{src:10s} {cells} {note:>13s}")
        lines.append("-" * len(ch))
        lines.append("")
    lines.append(f"Expected stat improvement √(1/B) = {INCLUSIVE_FACTOR**0.5:.2f} "
                 f"(if a row scales by this, it is stat-like; if it stays flat,")
    lines.append(" it is a systematic that does NOT improve with more channels).")
    text = "\n".join(lines)
    print("\n" + text + "\n")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".txt", "w") as fh:
        fh.write(text + "\n")
    import csv
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "poi", "source", "value_MeV"])
        for name in names:
            for poi in card.POI_DISPLAY:
                for src in _SYST_ROWS + ["total"]:
                    v = _grouped_syst(results.get(name), poi, src)
                    w.writerow([name, poi, src, "" if v is None else f"{v:.4f}"])
    print(f"[channel-extrap] wrote {out}.txt and {out}.csv")
