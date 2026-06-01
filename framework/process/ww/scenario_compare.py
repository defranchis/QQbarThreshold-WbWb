"""Scan-scenario comparison for the WW threshold fit.

Compares three √s data-taking layouts — all using the SAME cross-section
templates (the σ is stored on the fine √s grid; a scenario merely selects which
points the fit consumes), all with the SAME total luminosity (the project's own
12 ab⁻¹), and all fit with the reduced 2-POI stat + correlated-lumi setup of the
theory ladder (so the comparison isolates the effect of the scan geometry):

  * **7pt**         — the project baseline: 157–163 GeV, 1 GeV step (7 points),
                      equal luminosity split.
  * **3pt-FCC**     — a 3-point "FCC baseline": 157, 160, 163 GeV, equal split.
  * **Azzurri-like**— a 2-point optimised layout in the spirit of P. Azzurri
                      (arXiv:2107.04444 §2.4): one point at 157 GeV and one at
                      the dσ/dΓ_W = 0 ("Γ_W-insensitive") crossing — computed
                      from *our own* templates, not Azzurri's number — with the
                      luminosity split 60 % low / 40 % high (Azzurri's f=0.40 at
                      the upper point). We keep our own total luminosity, only
                      Azzurri's split ratio.

For each scenario it reports (a) the production-chain (+δ_QCD/NLL) Asimov
sensitivity σ(m_W), σ(Γ_W) and their correlation ρ, and (b) the full
perturbative theory ladder (LO→+NLO→+NNLO→+δ_QCD). Both fall out of one
``run_theory_ladder`` call per scenario; the (expensive NLL) templates are
generated once and reused from the persistent cache for the other scenarios.

Run via ``python3 doFit_ww.py --compareScenarios`` or import
:func:`run_scenario_comparison`.
"""

from __future__ import annotations

import os

import numpy as np

from cards import ww_default as card
from framework.common.fit_core import ecm_to_str
from framework.common.parameters import Parameters
from framework.common.systematics import compute_syst_breakdown
from framework.process.ww.fit import WWFit
from framework.process.ww.generator import WWGenerator
from framework.process.ww.theory_ladder import (
    HARD_RUNGS, LADDER_CACHE_DIR, run_theory_ladder,
)


# ---------------------------------------------------------------------------
# Γ_W-insensitive crossing from our own production templates
# ---------------------------------------------------------------------------
def gamma_crossing_ecm(*, lo: float = 156.0, hi: float = 166.0) -> float:
    """Return the √s (snapped to the 0.1-GeV template grid) where
    dσ_obs/dΓ_W = 0 in the production templates — i.e. where the width-varied
    template crosses the nominal one. Restricted to the threshold window
    [``lo``, ``hi``] to avoid the appended above-threshold anchor point."""
    gen = WWGenerator.from_card(card)
    params = Parameters(card.PARAMETERS, cross_terms=getattr(card, "CROSS_TERMS", ()))
    scales = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0})
    nd = card.INPUT_DIRS["nominal"]
    kw = dict(mass_scale=scales["mass"], width_scale=scales["width"],
              mass_scheme=getattr(card, "MASS_SCHEME", "OS"), indir=nd)
    f_nom = gen.file_name(params.values("nominal"), **kw)
    f_wid = gen.file_name(params.values("width_var"), **kw)
    nom = np.loadtxt(f_nom, delimiter=",", comments="#")
    wid = np.loadtxt(f_wid, delimiter=",", comments="#")
    ecm = nom[:, 0]
    d = wid[:, 1] - nom[:, 1]                # dσ for +Δ width
    m = (ecm >= lo) & (ecm <= hi)
    ecm, d = ecm[m], d[m]
    sign = np.sign(d)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        raise RuntimeError("no dσ/dΓ_W = 0 crossing found in "
                           f"[{lo}, {hi}] GeV — check the templates")
    i = idx[0]
    xc = ecm[i] - d[i] * (ecm[i + 1] - ecm[i]) / (d[i + 1] - d[i])
    return round(float(xc), 1)               # snap to the 0.1-GeV grid


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
_AZZURRI_F_HIGH = 0.40   # Azzurri arXiv:2107.04444 §2.4: lumi fraction at the
                         # upper (Γ_W-insensitive) point; 1−f at the low point.


def build_scenarios() -> dict:
    """Return ``{name: scenario_dict}`` for the three layouts. All carry the
    same total luminosity (card ``SCENARIO['total_lumi']``); only the √s points
    and (for Azzurri-like) the per-point split differ."""
    L = card.SCENARIO["total_lumi"]
    xc = gamma_crossing_ecm()
    xc_str = ecm_to_str(xc)
    lo_str = ecm_to_str(157.0)

    scan7 = [ecm_to_str(e) for e in np.arange(157.0, 163.0 + 1e-9, 1.0)]
    scan3 = [ecm_to_str(e) for e in (157.0, 160.0, 163.0)]
    azz_lumi = {lo_str: (1.0 - _AZZURRI_F_HIGH) * L, xc_str: _AZZURRI_F_HIGH * L}

    return {
        "7pt baseline": {"scan_list": scan7, "total_lumi": L,
                         "desc": "157–163, 1.0 GeV step (equal split)"},
        "3pt FCC base": {"scan_list": scan3, "total_lumi": L,
                         "desc": "157, 160, 163 (equal split)"},
        "Azzurri-like": {"scan_list": [lo_str, xc_str], "lumi_dict": azz_lumi,
                         "total_lumi": L,
                         "desc": f"157 ({100*(1-_AZZURRI_F_HIGH):.0f}%) + "
                                 f"{xc_str} ({100*_AZZURRI_F_HIGH:.0f}%) "
                                 f"[Γ_W-insensitive crossing]"},
    }


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------
def _row(rows, *, hard, isr, lumi):
    return next((r for r in rows
                 if r["hard"] == hard and r["isr"] == isr and r["lumi"] == lumi), None)


# ---------------------------------------------------------------------------
# Full-production systematics breakdown per scenario
# ---------------------------------------------------------------------------
def _full_syst_breakdown(scn):
    """Run the FULL production fit (all POIs + α_s/BES/BEC/lumi nuisances, real
    priors — the same configuration as ``doFit_ww.py --systTable``) under the
    scenario's scan geometry, and return its per-source systematics breakdown.

    Returns ``(syst, totals, centrals)`` from
    :func:`framework.common.systematics.compute_syst_breakdown`, or ``None`` if
    the fit fails to converge (e.g. the 2-point Azzurri-like layout cannot
    constrain all POIs + nuisances)."""
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=True,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(
        scan_list=scn["scan_list"],
        lumi_dict=scn.get("lumi_dict"),
        total_lumi=scn.get("total_lumi", card.SCENARIO["total_lumi"]),
        last_lumi=scn.get("last_lumi", card.SCENARIO["last_lumi"]),
        add_last_ecm=scn.get("add_last_ecm", False),
    )
    # Match the --systTable configuration: BES + BEC binned nuisances on top of
    # the auto-activated lumi nuisance (nuisance LUMI_MODE).
    fit.add_binned_nuisance("BEC")
    fit.add_binned_nuisance("BES")
    fit.fit_parameters()
    if not fit.minuit.valid:
        print(f"   [syst] nominal fit did not converge — skipping breakdown")
        return None
    try:
        return compute_syst_breakdown(fit)
    except Exception as exc:  # under-constrained scenario → degenerate syst fit
        print(f"   [syst] breakdown failed ({type(exc).__name__}: {exc}) — skipping")
        return None


def run_scenario_comparison(*, workers=48, out=None, scheme_var=False):
    """Run, for each scan scenario (shared cached templates): the theory ladder
    (sensitivity + per-rung bias) AND the full-production systematics breakdown,
    then emit one consolidated comparison table."""
    out = out or os.path.join("plots", "scenario_compare")
    scenarios = build_scenarios()

    results = {}
    syst_results = {}
    for name, scn in scenarios.items():
        print(f"\n{'='*70}\n[scenario] {name}: {scn['desc']}\n{'='*70}")
        safe = name.replace(" ", "_").replace("/", "")
        rows, _scheme = run_theory_ladder(
            isr="both", workers=workers, scheme_var=scheme_var,
            out=os.path.join("plots", f"scenario_{safe}"),
            base=LADDER_CACHE_DIR, scenario=scn,
        )
        results[name] = rows
        print(f"[scenario] {name}: full-production systematics breakdown ...")
        syst_results[name] = _full_syst_breakdown(scn)

    _emit_comparison(scenarios, results, out, syst_results)
    return results, syst_results


# Sources shown in the per-scenario breakdown (matches the baseline syst
# table: stat first, then the configured systematics, then the total).
_SYST_ROWS = ["stat"] + list(card.SYST_TABLE_ORDER)


def _grouped_syst(sb, poi, src):
    """Return one source's contribution to σ(``poi``) [display units] from a
    ``compute_syst_breakdown`` result ``sb=(syst, totals, centrals)``.

    Binned nuisances are stored split as ``<src>_uncorr`` / ``<src>_corr``;
    this quadrature-combines them into the single ``src`` (e.g. ``lumi``) so
    the per-scenario comparison shows one row per physical source. Returns
    ``None`` if ``sb`` is ``None`` or no finite entry is found."""
    if sb is None:
        return None
    syst, totals, _c = sb
    if src == "total":
        v = totals.get(poi)
        return v if (v is not None and np.isfinite(v)) else None
    d = syst.get(poi, {})
    if src == "stat":
        v = d.get("stat")
        return v if (v is not None and np.isfinite(v)) else None
    acc, found = 0.0, False
    for k, v in d.items():
        if (k == src or k.startswith(src + "_")) and v is not None and np.isfinite(v):
            acc += v * v
            found = True
    return acc ** 0.5 if found else None


def _emit_comparison(scenarios, results, out, syst_results=None):
    lines = []
    lines.append("WW scan-scenario comparison — Asimov sensitivity + theory ladder")
    L_ab = card.SCENARIO["total_lumi"] / 1e6
    lines.append(f"Total luminosity = {L_ab:.1f} /ab for ALL scenarios (same lumi).")
    lines.append("Fit: m_W, Γ_W float; stat (Asimov) + correlated-lumi only "
                 "(2-POI ladder setup).")
    lines.append("Sensitivity = production chain (+δ_QCD / NLL ISR), realistic "
                 "FCC-ee corr-lumi prior;")
    lines.append("the 'free-lumi' σ (rate floated) is shown in parentheses for "
                 "reference.")
    lines.append("")

    # --- (a) sensitivity + correlation ------------------------------------
    h = (f"{'scenario':14s} {'pts':>3s}  {'σ_mW':>16s} {'σ_ΓW':>16s} "
         f"{'ρ':>7s}   points")
    lines.append("-" * len(h))
    lines.append("(a) SENSITIVITY  [prior]  (free)")
    lines.append(h)
    lines.append(f"{'':14s} {'':>3s}  {'[MeV]':>16s} {'[MeV]':>16s} {'':>7s}")
    lines.append("-" * len(h))
    for name, scn in scenarios.items():
        rows = results[name]
        rp = _row(rows, hard="+dQCD", isr="NLL", lumi="prior")
        rf = _row(rows, hard="+dQCD", isr="NLL", lumi="free")
        npts = len(scn["scan_list"])
        smw = f"{rp['sig_mW']:6.2f} ({rf['sig_mW']:5.2f})"
        sgw = f"{rp['sig_gW']:6.2f} ({rf['sig_gW']:5.2f})"
        lines.append(f"{name:14s} {npts:>3d}  {smw:>16s} {sgw:>16s} "
                     f"{rp['rho']:+7.2f}   {scn['desc']}")
    lines.append("-" * len(h))
    lines.append("")

    # --- (b) theory ladder per scenario -----------------------------------
    lines.append("(b) THEORY LADDER  Δm_W vs production truth [MeV], realistic "
                 "corr-lumi prior")
    rungs = [k for k, _ in HARD_RUNGS]
    hh = f"{'scenario':14s} {'ISR':4s} " + " ".join(f"{r:>9s}" for r in rungs)
    lines.append("-" * len(hh))
    lines.append(hh)
    lines.append("-" * len(hh))
    for name in scenarios:
        rows = results[name]
        for isr_key in ("LL", "NLL"):
            cells = []
            for rung in rungs:
                r = _row(rows, hard=rung, isr=isr_key, lumi="prior")
                cells.append(f"{r['bias_mW']:+9.3f}" if r else f"{'—':>9s}")
            lines.append(f"{name:14s} {isr_key:4s} " + " ".join(cells))
        lines.append("-" * len(hh))
    lines.append("")

    # --- (c) full-production systematics breakdown per scenario -----------
    if syst_results:
        names = list(scenarios.keys())
        lines.append("(c) SYSTEMATICS BREAKDOWN  full production fit (all POIs +")
        lines.append("    α_s/BES/BEC/lumi nuisances, realistic priors — as --systTable)")
        for poi in card.POI_DISPLAY:
            disp = card.POI_DISPLAY[poi]
            ch = (f"{'source':10s} " +
                  " ".join(f"{n:>14s}" for n in names))
            lines.append("-" * len(ch))
            lines.append(f"σ({disp['symbol']})  [{disp['unit']}]")
            lines.append(ch)
            lines.append("-" * len(ch))
            for src in _SYST_ROWS + ["total exp"]:
                key = "total" if src == "total exp" else src
                cells = []
                for n in names:
                    sb = syst_results.get(n)
                    if sb is None:
                        cells.append(f"{'n/c':>14s}")
                        continue
                    val = _grouped_syst(sb, poi, key)
                    if val is None:
                        cells.append(f"{'—':>14s}")
                    else:
                        cells.append(f"{val:>14.2f}")
                lines.append(f"{src:10s} " + " ".join(cells))
            lines.append("-" * len(ch))
        if any(v is None for v in syst_results.values()):
            lines.append("  n/c = full-syst fit did not converge for this scenario")
            lines.append("        (too few points to constrain all POIs + nuisances).")
        lines.append("")

    # --- (d) uncertainty + correlation stability across the ladder --------
    # σ(m_W), σ(Γ_W) and ρ are set by the lineshape + lumi constraint, NOT by
    # which perturbative pieces are switched on, so they should be ~flat across
    # the ladder rungs. The 'spread' column (max−min over the four rungs) makes
    # that explicit per scenario; the scenario-to-scenario change is the
    # geometry effect. NLL ISR, realistic corr-lumi prior throughout.
    rungs = [k for k, _ in HARD_RUNGS]
    qh = (f"{'scenario':14s} {'quantity':11s} " +
          " ".join(f"{r:>8s}" for r in rungs) + f" {'spread':>8s}")
    lines.append("(d) UNCERTAINTY & CORRELATION STABILITY across the ladder")
    lines.append("    (NLL ISR, realistic corr-lumi prior; σ in MeV)")
    lines.append("-" * len(qh))
    lines.append(qh)
    lines.append("-" * len(qh))
    for name in scenarios:
        rows = results[name]
        rung_rows = [_row(rows, hard=r, isr="NLL", lumi="prior") for r in rungs]
        for qkey, qlabel, fmt in (("sig_mW", "σ_mW [MeV]", "{:8.2f}"),
                                  ("sig_gW", "σ_ΓW [MeV]", "{:8.2f}"),
                                  ("rho",    "ρ",          "{:+8.2f}")):
            vals = [rr[qkey] if rr else None for rr in rung_rows]
            cells = " ".join(fmt.format(v) if v is not None else f"{'—':>8s}"
                             for v in vals)
            fin = [v for v in vals if v is not None]
            spread = (max(fin) - min(fin)) if fin else float("nan")
            sp = f"{spread:8.2f}" if np.isfinite(spread) else f"{'—':>8s}"
            lines.append(f"{name:14s} {qlabel:11s} {cells} {sp}")
        lines.append("-" * len(qh))
    lines.append("")

    lines.append("Notes:")
    lines.append("  • Templates are identical across scenarios (σ on the fine √s")
    lines.append("    grid); only which points enter the fit changes.")
    lines.append("  • The Azzurri-like upper point is OUR dσ/dΓ_W=0 crossing")
    lines.append("    (snapped to the 0.1-GeV grid), not Azzurri's 162.3 GeV.")
    lines.append("  • The Azzurri-like layout reduces ρ (here +0.19 vs +0.51 for")
    lines.append("    the 7-point) — the upper point carries little Γ_W info, partly")
    lines.append("    decorrelating the POIs — but NOT to ρ≈0: the corr-lumi-only")
    lines.append("    constraint leaves a residual (Azzurri's ρ≈0 uses a different")
    lines.append("    setup). It also gives the tightest σ_ΓW.")
    lines.append("  • The 'free-lumi' σ is meaningless for the 2-point Azzurri-like")
    lines.append("    scenario (huge values): 2 points cannot constrain 2 POIs PLUS")
    lines.append("    a floating normalisation — read its 'prior' (realistic) column.")

    text = "\n".join(lines)
    print("\n" + text + "\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".txt", "w") as fh:
        fh.write(text + "\n")

    import csv
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "n_pts", "isr", "rung", "lumi",
                    "bias_mW_MeV", "bias_gW_MeV", "sig_mW_MeV", "sig_gW_MeV", "rho"])
        for name, scn in scenarios.items():
            for r in results[name]:
                w.writerow([name, len(scn["scan_list"]), r["isr"], r["hard"],
                            r["lumi"], f"{r['bias_mW']:.4f}", f"{r['bias_gW']:.4f}",
                            f"{r['sig_mW']:.4f}", f"{r['sig_gW']:.4f}", f"{r['rho']:.4f}"])
    print(f"[scenario] wrote {out}.txt and {out}.csv")

    # Systematics breakdown CSV (one row per scenario × POI × source).
    if syst_results:
        with open(out + "_syst.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "poi", "source", "value_displayunit", "converged"])
            for name in scenarios:
                sb = syst_results.get(name)
                for poi in card.POI_DISPLAY:
                    for src in _SYST_ROWS + ["total"]:
                        if sb is None:
                            w.writerow([name, poi, src, "", 0])
                            continue
                        val = _grouped_syst(sb, poi, src)
                        cell = "" if val is None else f"{val:.4f}"
                        w.writerow([name, poi, src, cell, 1])
        print(f"[scenario] wrote {out}_syst.csv")
