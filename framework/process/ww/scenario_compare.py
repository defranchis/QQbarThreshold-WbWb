"""Scan-scenario comparison for the WW threshold fit.

Compares three √s data-taking layouts — all using the SAME cross-section
templates (the σ is stored on the fine √s grid; a scenario merely selects which
points the fit consumes), all with the SAME total luminosity
(``card.SCENARIO["total_lumi"]``), and all fit with the reduced 2-POI stat +
correlated-lumi setup of the theory ladder (so the comparison isolates the
effect of the scan geometry):

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
def _build_scenario_fit(scn):
    """Build + fit the FULL production fit (all POIs + α_s/aem_isr/BES/BEC/lumi
    nuisances, realistic priors — the ``doFit_ww.py --systTable`` configuration)
    under the scenario's scan geometry. Returns the fitted :class:`WWFit`."""
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
    return fit


def _full_syst_breakdown(scn):
    """Per-source systematics breakdown for a scenario (``doFit_ww.py
    --systTable`` configuration).

    Returns ``(syst, totals, centrals)`` from
    :func:`framework.common.systematics.compute_syst_breakdown`, or ``None`` if
    the fit fails to converge (e.g. the 2-point Azzurri-like layout cannot
    constrain all POIs + nuisances)."""
    fit = _build_scenario_fit(scn)
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
    then emit one consolidated comparison table + the comparison plots."""
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
    _plot_comparison(scenarios, results, out, syst_results)
    run_scenario_syst_scans(out)
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
    rungs = [k for k, _ in HARD_RUNGS]   # ladder rung keys, used by (b) and (d)
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
    lines.append("  • The Azzurri-like layout reduces ρ relative to the 7-point")
    lines.append("    (see panel (a) for the live values) — the upper point carries")
    lines.append("    little Γ_W info, partly decorrelating the POIs — but NOT to")
    lines.append("    ρ≈0: the corr-lumi-only constraint leaves a residual (Azzurri's")
    lines.append("    ρ≈0 uses a different setup). It also gives the tightest σ_ΓW.")
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


# ---------------------------------------------------------------------------
# Plots: scan-point layout on the threshold lineshape, and the (m_W, Γ_W)
# Asimov error ellipses per scenario.
# ---------------------------------------------------------------------------
#: Colour + marker per scenario (keyed by the build_scenarios() names).
_SCN_STYLE = {
    "7pt baseline": dict(color="C0", marker="o"),
    "3pt FCC base": dict(color="C1", marker="s"),
    "Azzurri-like": dict(color="C3", marker="D"),
}
#: √s window for the lineshape panel [GeV] (threshold region only; the
#: production templates extend to 240 GeV via the above-threshold anchor).
_LINE_LO, _LINE_HI = 154.5, 166.0
#: Short filename tag per POI (mass/width separated into their own figures).
_POI_TAG = {"mass": "mW", "width": "gW"}


def _nominal_lineshape():
    """Return ``(ecm, sigma)`` of the nominal production template, restricted to
    the threshold window. Loaded exactly like :func:`gamma_crossing_ecm` so the
    plotted curve is the same σ the fit consumes."""
    gen = WWGenerator.from_card(card)
    params = Parameters(card.PARAMETERS, cross_terms=getattr(card, "CROSS_TERMS", ()))
    scales = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0})
    kw = dict(mass_scale=scales["mass"], width_scale=scales["width"],
              mass_scheme=getattr(card, "MASS_SCHEME", "OS"),
              indir=card.INPUT_DIRS["nominal"])
    a = np.loadtxt(gen.file_name(params.values("nominal"), **kw),
                   delimiter=",", comments="#")
    ecm = a[:, 0]
    m = (ecm >= _LINE_LO) & (ecm <= _LINE_HI)
    return ecm[m], a[m, 1]


def _per_point_lumi(scn):
    """Return ``{ecm_float: lumi}`` for a scenario: its explicit per-point
    ``lumi_dict`` if present, else the total split equally over the scan points
    (matching ``init_scenario``'s equal-split default)."""
    L = scn.get("total_lumi", card.SCENARIO["total_lumi"])
    pts = [float(s) for s in scn["scan_list"]]
    ld = scn.get("lumi_dict")
    if ld:
        return {float(k): v for k, v in ld.items()}
    return {e: L / len(pts) for e in pts}


def _sigma_at(ecm_grid, sigma_grid, e):
    """Linear-interpolated σ at √s = ``e`` [GeV] on the template grid."""
    return float(np.interp(e, ecm_grid, sigma_grid))


def _cov_ellipse(ax, sx, sy, rho, *, n_sigma=1.0, **kw):
    """Draw the ``n_sigma`` covariance ellipse for a 2-D Gaussian with marginal
    widths ``sx``/``sy`` and correlation ``rho``, centred at the origin (Asimov
    truth). Returns the matplotlib patch."""
    from matplotlib.patches import Ellipse
    cov = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
    vals, vecs = np.linalg.eigh(cov)              # ascending eigenvalues
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_sigma * np.sqrt(vals)
    e = Ellipse((0.0, 0.0), width=w, height=h, angle=angle, fill=False, **kw)
    ax.add_patch(e)
    return e


def _final_sensitivity(rows):
    """The production-chain (+δ_QCD, NLL ISR, realistic corr-lumi prior) Asimov
    sensitivity row — same one panel (a) of the table reports."""
    return _row(rows, hard="+dQCD", isr="NLL", lumi="prior")


def _plot_comparison(scenarios, results, out, syst_results=None):
    """Comparison figures: (1) the scan-point layout on the threshold lineshape
    with the per-point luminosity split, (2) the (m_W, Γ_W) Asimov error ellipses
    per scenario, and — when ``syst_results`` is given — (3) the per-source
    systematic breakdown across scenarios (does the dependence change with the
    scan geometry?). Saved as ``<out>_layout.{pdf,png}``,
    ``<out>_ellipses.{pdf,png}`` and ``<out>_syst.{pdf,png}``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from framework.process.ww import plot_labels
    # These compact multi-line comparison figures use plain matplotlib styling;
    # framework.common.plots applies the big-font hep.style.CMS globally on
    # import (leaks in via the scans import), which overflows small panels.
    plt.style.use("default")

    badge = (f"{plot_labels.process_label_short(card)} "
             f"{plot_labels.generator_label_short(card)}").strip()
    L_ab = card.SCENARIO["total_lumi"] / 1e6
    ecm, sig = _nominal_lineshape()

    # --- Figure 1: scan layout on the lineshape ---------------------------
    fig, (ax_t, ax_b) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.0], hspace=0.07))
    ax_t.axvspan(157.0, 163.0, color="0.93", zorder=0)
    ax_b.axvspan(157.0, 163.0, color="0.93", zorder=0)
    ax_t.plot(ecm, sig, "-", color="0.35", lw=1.6, zorder=1,
              label=r"$\sigma_{WW}(\sqrt{s})$ (NLL)")

    names = list(scenarios.keys())
    dodge = np.linspace(-0.13, 0.13, len(names))     # bottom-panel bar dodge
    for k, name in enumerate(names):
        st = _SCN_STYLE.get(name, dict(color=f"C{k}", marker="o"))
        lumi = _per_point_lumi(scenarios[name])
        xs = sorted(lumi)
        ys = [_sigma_at(ecm, sig, e) for e in xs]
        # marker area ∝ luminosity at the point (visual cue; bars give numbers)
        lmax = max(lumi.values())
        sizes = [40 + 150 * (lumi[e] / lmax) for e in xs]
        ax_t.scatter(xs, ys, s=sizes, color=st["color"], marker=st["marker"],
                     edgecolor="k", linewidth=0.6, zorder=3,
                     label=f"{name} ({len(xs)} pt)")
        ax_b.bar([e + dodge[k] for e in xs], [lumi[e] / 1e6 for e in xs],
                 width=0.11, color=st["color"], edgecolor="k", linewidth=0.4,
                 label=name)

    ax_t.set_ylabel(r"$\sigma_{WW}$ [pb]")
    ax_t.set_title("WW threshold scan layouts (same total "
                   f"{L_ab:.1f} ab$^{{-1}}$)")
    ax_t.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_b.set_ylabel(r"$\mathcal{L}$/point [ab$^{-1}$]")
    ax_b.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax_b.set_xlim(_LINE_LO, _LINE_HI)
    ax_b.grid(axis="y", color="0.9", lw=0.6)
    fig.text(0.985, 0.01, badge, ha="right", va="bottom", fontsize=7,
             color="0.4")
    fig.savefig(out + "_layout.pdf", bbox_inches="tight")
    fig.savefig(out + "_layout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: (m_W, Γ_W) error ellipses ------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lim = 0.0
    for k, name in enumerate(names):
        r = _final_sensitivity(results[name])
        if r is None:
            continue
        st = _SCN_STYLE.get(name, dict(color=f"C{k}", marker="o"))
        sx, sy, rho = r["sig_mW"], r["sig_gW"], r["rho"]
        _cov_ellipse(ax, sx, sy, rho, n_sigma=1.0, edgecolor=st["color"],
                     lw=2.0, zorder=3,
                     label=(f"{name}: "
                            rf"$\sigma_{{m_W}}={sx:.2f}$, "
                            rf"$\sigma_{{\Gamma_W}}={sy:.2f}$ MeV, "
                            rf"$\rho={rho:+.2f}$"))
        _cov_ellipse(ax, sx, sy, rho, n_sigma=2.0, edgecolor=st["color"],
                     lw=1.0, ls="--", alpha=0.6, zorder=2)
        lim = max(lim, 2.2 * max(sx, sy))
    ax.axhline(0, color="0.7", lw=0.7, zorder=0)
    ax.axvline(0, color="0.7", lw=0.7, zorder=0)
    ax.plot(0, 0, "+", color="k", ms=9, mew=1.4, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta m_W$ [MeV]")
    ax.set_ylabel(r"$\Delta \Gamma_W$ [MeV]")
    ax.set_title("Asimov 1$\\sigma$ / 2$\\sigma$ contours "
                 "(stat + corr-lumi, NLL)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.text(0.985, 0.01, badge, ha="right", va="bottom", fontsize=7,
             color="0.4")
    fig.savefig(out + "_ellipses.pdf", bbox_inches="tight")
    fig.savefig(out + "_ellipses.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: per-source systematic breakdown across scenarios -------
    # Does each uncertainty source depend on the scan geometry? Grouped bars by
    # source, coloured by scenario; m_W and Γ_W on SEPARATE figures to avoid
    # clutter. A source whose bars are equal-height is geometry-independent; a
    # source that changes (e.g. lumi on Γ_W, or stat) carries the scenario
    # dependence. Same numbers as panel (c) of the text table.
    if syst_results and any(v is not None for v in syst_results.values()):
        names = [n for n in scenarios if syst_results.get(n) is not None]
        srcs = _SYST_ROWS                       # stat + configured systematics
        x = np.arange(len(srcs))
        w = 0.8 / max(len(names), 1)
        for poi in card.POI_DISPLAY:
            disp = card.POI_DISPLAY[poi]
            tag = _POI_TAG.get(poi, poi)
            fig, ax = plt.subplots(figsize=(8.0, 4.4))
            for k, name in enumerate(names):
                st = _SCN_STYLE.get(name, dict(color=f"C{k}"))
                vals = [(_grouped_syst(syst_results[name], poi, s) or 0.0)
                        for s in srcs]
                xb = x + (k - (len(names) - 1) / 2.0) * w
                bars = ax.bar(xb, vals, width=w, color=st["color"],
                              edgecolor="k", linewidth=0.4, label=name)
                for rect, v in zip(bars, vals):       # annotate every bar
                    ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2,
                                v), ha="center", va="bottom", fontsize=6,
                                rotation=90, xytext=(0, 1),
                                textcoords="offset points")
            ax.set_ylabel(rf"$\sigma({disp['symbol']})$ [{disp['unit']}]")
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            ax.margins(y=0.18)
            ax.set_xticks(x)
            ax.set_xticklabels(srcs, rotation=0)
            ax.set_xlabel("uncertainty source")
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
                      ncol=len(names))
            ax.set_title(rf"$\sigma({disp['symbol']})$ breakdown across scan "
                         f"scenarios (same {L_ab:.1f} ab$^{{-1}}$)")
            fig.text(0.985, 0.005, badge, ha="right", va="bottom", fontsize=7,
                     color="0.4")
            fig.savefig(out + f"_syst_{tag}.pdf", bbox_inches="tight")
            fig.savefig(out + f"_syst_{tag}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
        print(f"[scenario] wrote {out}_layout / _ellipses / "
              f"_syst_{{mW,gW}} .{{pdf,png}}")
    else:
        print(f"[scenario] wrote {out}_layout.{{pdf,png}} and "
              f"{out}_ellipses.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Per-scenario systematic-prior sweeps overlaid: the single-scenario
# ``uncert_mass_width_vs_<syst>`` impact curves (framework.common.scans), drawn
# for all three layouts on one figure so the systematic's *dependence* on the
# scan geometry is visible directly.
# ---------------------------------------------------------------------------
# x-axis presentation per systematic for the sweep plots: display unit, axis
# label, and grid upper edge (absolute ``hi`` or ``hi_factor`` × the production
# prior). The SET of systematics scanned and each one's kind (binned vs
# constraint) are taken from card.SYST_TABLE_ORDER / SYSTEMATICS / PRIORS, so
# adding or removing a systematic in the card updates the scans automatically;
# only the presentation lives here. A systematic with no entry is skipped (and
# logged) rather than silently dropped. Ranges match the single-scenario scans
# in framework.common.scans (scan_bes/scan_bec/scan_lumi/scan_alphas).
_SCAN_AXIS = {
    "alphas":  dict(axis_unit=1e3, hi=3e-4,
                    axis_label=r"$\alpha_s(m_Z)$ uncertainty [$\times 10^{3}$]"),
    "aem_isr": dict(axis_unit=1e8, hi_factor=3.0,
                    axis_label=r"$\alpha(M_Z)_{\rm ISR}$ uncertainty [$\times 10^{-8}$]"),
    "BES":     dict(axis_unit=100.0, hi=0.03, axis_label="BES uncertainty [%]"),
    "BEC":     dict(axis_unit=1.0,   hi=10.0,
                    axis_label=r"$\sqrt{s}$ calibration uncertainty [MeV]"),
    "lumi":    dict(axis_unit=100.0, hi_factor=3.0,
                    axis_label="Luminosity uncertainty [%]"),
}
_SCAN_NPTS = 31


def _syst_kind(key):
    """Classify a SYST_TABLE_ORDER entry as 'binned' or 'constraint' from the
    card: explicit ``SYSTEMATICS[key]['type']`` if declared, else inferred from
    PRIORS (dict uncorr/corr ⇒ binned, e.g. the LUMI_MODE-injected ``lumi``;
    scalar ⇒ constraint)."""
    t = card.SYSTEMATICS.get(key, {}).get("type")
    if t in ("binned", "constraint"):
        return t
    return "binned" if isinstance(card.PRIORS.get(key), dict) else "constraint"


def _sweep_specs():
    """Card-driven systematic-prior sweep definitions: iterate
    card.SYST_TABLE_ORDER, take each systematic's kind + baseline prior from the
    card and its x-axis from ``_SCAN_AXIS``. So the scenario sweeps cover exactly
    the systematics in the card's table — no hand-maintained list."""
    specs = []
    for key in card.SYST_TABLE_ORDER:
        ax = _SCAN_AXIS.get(key)
        if ax is None:
            print(f"[scan] no x-axis metadata for {key!r} in _SCAN_AXIS — "
                  f"skipping its scenario sweep")
            continue
        kind = _syst_kind(key)
        if kind == "binned":
            base, directions = card.PRIORS[key]["uncorr"], ("uncorr", "corr")
        else:
            base, directions = card.PRIORS[key], ("",)
        hi = ax.get("hi", ax.get("hi_factor", 3.0) * base)
        specs.append(dict(
            key=key, kind=kind, directions=directions,
            grid=np.linspace(0.0, hi, _SCAN_NPTS), axis_unit=ax["axis_unit"],
            axis_label=ax["axis_label"], baseline=base))
    return specs


def _collect_binned(fit, kind, grid, directions):
    """Sweep a binned nuisance's ``uncorr``/``corr`` prior over ``grid`` and
    return ``{direction: {poi: impact_array}}`` (quadrature-subtracted impact in
    display units), mirroring ``framework.common.scans._scan_nuisance``."""
    from framework.common.scans import run_local_migrad, impact
    saved = dict(fit._nuisance_priors.get(kind, {}))
    start = np.zeros(len(fit.param_names))
    pois = ("mass", "width")
    raw = {d: {p: [] for p in pois} for d in directions}
    try:
        for d in directions:
            for v in grid:
                vv = max(float(v), 1e-6)
                fit.set_binned_nuisance_priors(
                    kind, uncorr=(vv if d == "uncorr" else 1e-6),
                    corr=(vv if d == "corr" else 1e-6))
                fr = fit.results_from_minuit(run_local_migrad(fit, start))
                for p in pois:
                    raw[d][p].append(fr[fit._idx[p]].s * card.POI_DISPLAY[p]["scale"])
    finally:
        if saved:
            fit._nuisance_priors[kind] = saved
    return {d: {p: impact(np.array(raw[d][p])) for p in pois} for d in directions}


def _collect_constraint(fit, name, grid):
    """Sweep a 1-D Gaussian-constraint width over ``grid`` and return
    ``{"": {poi: impact_array}}`` (quadrature-subtracted POI impact, display units).

    Per-point re-minimisation makes the quadrature-subtracted impact ``√(σ²−σ₀²)``
    extremely noisy when the impact is ≪ σ_stat (α_s, α_em,ISR): tiny migrad/Hesse
    jitter in σ swamps the signal. But the constraint enters the χ² only as
    ``((p−c)·step/σ)²`` (see ``fit_core``), so changing its width σ shifts a SINGLE
    diagonal element of the Hessian, ``H[θ,θ] = H_data[θ,θ] + k·(step/σ)²``, and
    leaves everything else untouched. We calibrate ``(k, H_data[θ,θ])`` from a few
    well-conditioned migrad+Hesse fits, then compute the covariance for the whole
    grid analytically (Sherman–Morrison rank-1) by overwriting just that element —
    exact for the Gaussian fit, perfectly smooth, and far cheaper (a handful of
    fits instead of one per grid point)."""
    from framework.common.scans import _local_minuit, impact
    start = list(fit.minuit.values)
    saved = fit._constraints[name]["sigma"]
    th = fit._idx[name]
    pois = ("mass", "width")
    poi_idx = {p: fit._idx[p] for p in pois}
    poi_disp = {p: fit.parameters.step(p) * card.POI_DISPLAY[p]["scale"] for p in pois}
    step = fit.parameters.step(name)

    def _cov_inv_at(sigma_phys):
        fit._constraints[name]["sigma"] = float(sigma_phys)
        m = _local_minuit(fit, start)
        m.strategy = 2
        m.migrad()
        m.hesse()
        return np.linalg.inv(np.asarray(m.covariance))

    try:
        # Calibrate H[θ,θ] = H_data + k·(step/σ)² over a few conditioned widths
        # (a least-squares line through several fits averages out per-fit jitter,
        # so the analytic curve is smooth at EVERY point, not just de-spiked).
        cal = [0.5 * saved, saved, 2.0 * saved, 4.0 * saved]
        xs = np.array([(step / s) ** 2 for s in cal])
        Href = None
        ys = []
        for s in cal:
            H = _cov_inv_at(s)
            if Href is None:
                Href = H.copy()
            ys.append(H[th, th])
        k, h_data = np.linalg.lstsq(np.vstack([xs, np.ones_like(xs)]).T,
                                    np.array(ys), rcond=None)[0]
    finally:
        fit._constraints[name]["sigma"] = saved

    # σ→0 ⇒ θ pinned (≡ fixed) ⇒ stat-only baseline impact() subtracts.
    raw = {p: [] for p in pois}
    for u in grid:
        H = Href.copy()
        H[th, th] = h_data + k * (step / max(float(u), float(np.max(grid)) * 1e-9)) ** 2
        C = np.linalg.inv(H)
        for p in pois:
            raw[p].append(np.sqrt(max(C[poi_idx[p], poi_idx[p]], 0.0)) * poi_disp[p])
    return {"": {p: impact(np.array(raw[p])) for p in pois}}


def run_scenario_syst_scans(out=None):
    """For each scan scenario, sweep every systematic prior (BES, BEC, lumi,
    α_s) and overlay the per-POI impact curves across scenarios — one figure per
    systematic (``<out>_scan_<syst>.{pdf,png}``). Answers "does the dependence on
    this systematic differ between scan layouts?". Fits run on the existing
    templates (no MC); each scenario gets a fresh full-production fit."""
    out = out or os.path.join("plots", "scenario_compare")
    scenarios = build_scenarios()
    specs = _sweep_specs()
    data = {s["key"]: dict(axis_label=s["axis_label"],
                           x=np.asarray(s["grid"], float) * s["axis_unit"],
                           baseline_x=s["baseline"] * s["axis_unit"],
                           directions=s["directions"], by_scn={}) for s in specs}
    for name, scn in scenarios.items():
        print(f"[scan] {name}: building fit + sweeping systematics ...")
        fit = _build_scenario_fit(scn)
        if not fit.minuit.valid:
            print(f"[scan] {name}: baseline fit invalid — skipping syst scans")
            continue
        for s in specs:
            try:
                res = (_collect_binned(fit, s["key"], s["grid"], s["directions"])
                       if s["kind"] == "binned"
                       else _collect_constraint(fit, s["key"], s["grid"]))
            except Exception as exc:
                print(f"[scan] {name}/{s['key']} failed "
                      f"({type(exc).__name__}: {exc}) — skipping")
                continue
            data[s["key"]]["by_scn"][name] = res
    _plot_syst_scans(data, out)
    return data


def _plot_syst_scans(data, out):
    """One figure per (systematic × POI) — m_W and Γ_W are kept on separate
    figures to avoid clutter. Each shows that POI's impact vs the systematic's
    input size, one colour per scenario (solid = uncorrelated component, dashed =
    correlated). Dotted vertical line = production prior.
    Files: ``<out>_scan_<syst>_<mW|gW>.{pdf,png}``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from framework.process.ww import plot_labels
    plt.style.use("default")   # override the global hep.style.CMS (see _plot_comparison)

    badge = (f"{plot_labels.process_label_short(card)} "
             f"{plot_labels.generator_label_short(card)}").strip()
    pois = list(card.POI_DISPLAY)
    written = []
    for key, d in data.items():
        if not d["by_scn"]:
            continue
        for poi in pois:
            disp = card.POI_DISPLAY[poi]
            tag = _POI_TAG.get(poi, poi)
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            for name, res in d["by_scn"].items():
                col = _SCN_STYLE.get(name, {}).get("color", "C0")
                for direction in d["directions"]:
                    arr = res.get(direction, {}).get(poi)
                    if arr is None:
                        continue
                    if direction == "corr":
                        ls, suffix = "--", " (corr)"
                    elif direction == "uncorr":
                        ls, suffix = "-", " (uncorr)"
                    else:
                        ls, suffix = "-", ""
                    ax.plot(d["x"], arr, color=col, linestyle=ls, lw=1.8,
                            label=name + suffix)
            ax.axvline(d["baseline_x"], color="0.5", ls=":", lw=1.0)
            ax.annotate("production\nprior", (d["baseline_x"], 0),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="0.4", va="bottom")
            ax.set_ylabel(rf"$\Delta\sigma({disp['symbol']})$ [{disp['unit']}]")
            ax.set_xlabel(d["axis_label"])
            ax.grid(color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0.0)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
                      ncol=2 if len(d["directions"]) > 1 else 1)
            ax.set_title(rf"$\sigma({disp['symbol']})$ impact vs {key} input "
                         "across scan scenarios")
            fig.text(0.985, 0.005, badge, ha="right", va="bottom", fontsize=7,
                     color="0.4")
            stem = out + f"_scan_{key}_{tag}"
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            fig.savefig(stem + ".png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            written.append(f"{key}_{tag}")
    if written:
        print(f"[scenario] wrote {out}_scan_{{{','.join(written)}}}.{{pdf,png}}")
