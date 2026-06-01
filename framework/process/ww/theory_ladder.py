"""Incremental-piece theory-uncertainty ladder for the WW threshold fit.

Turns the BFS perturbative pieces on one at a time —

    LO  →  +NLO loops  →  +NNLO  →  +δ_QCD

each riding on a *fixed* WHIZARD-Born anchor (the anchor is always on, so
the ladder is the pure BFS perturbative series on a common Born), and runs
the whole ladder twice: once with LL+exp ISR and once with NLL ISR.

For every rung an Asimov fit of (m_W, Γ_W) is performed against the full
production chain (``+δ_QCD`` with NLL ISR) used as the injected truth. The
best-fit shift of (m_W, Γ_W) away from the injected truth is the residual
theory bias of stopping the calculation at that rung. The adjacent-rung
differences give the per-piece pulls; the top NLL rung fits itself and
must close to ≈0 (machinery / morph sanity check).

Fit configuration for this test (deliberately minimal — see the user
guideline):

  * POIs: m_W, Γ_W only. α_s / BEC / BES / m_t / M_H / M_Z are fixed at
    their central values (no nuisances, no constraints).
  * Uncertainties: Poisson-stat (Asimov yields) + correlated luminosity
    only. The uncorrelated-lumi term is switched off. Each rung is fit
    twice:
       - ``free``  — the correlated-lumi prior is opened up (≈100%), so any
         flat normalisation difference between rungs is fully absorbed and
         the bias is a *pure shape* effect;
       - ``prior`` — the realistic FCC-ee correlated-lumi prior
         (``card.PRIORS['lumi']['corr']``), so the residual flat component
         that leaks into m_W/Γ_W under the real constraint is included.

What this ladder does **not** capture: pieces entirely absent from the
BFS chain — NLO electroweak (YFSWW3-class non-factorisable + initial-final
interference) and higher-order Coulomb (the BFS unstable-W Green function
G_C vs. the dropped FKM K_C). Those are genuine theory uncertainties that
live outside the ladder and need a separate, non-ladder estimate.

Run via ``python3 doFit_ww.py --theoryLadder`` (see doFit_ww.py for flags),
or import :func:`run_theory_ladder`.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit

# ---------------------------------------------------------------------------
# Ladder definition
# ---------------------------------------------------------------------------
# Cumulative hard-side rungs. ``apply_whizard_anchor`` is True throughout
# (the anchor is not a perturbative order — it is the common Born the whole
# series rides on). Pieces NOT toggled here (include_coulomb=False,
# decay_uses_full_born=True, br_convention=pdg-constant, ...) keep their
# production values, inherited from WWGenerator.from_card(card).
HARD_RUNGS = [
    ("LO",     dict(include_NLO_hard_decay=False, include_BFS_NNLO=False, apply_delta_QCD=False)),
    ("+NLO",   dict(include_NLO_hard_decay=True,  include_BFS_NNLO=False, apply_delta_QCD=False)),
    ("+NNLO",  dict(include_NLO_hard_decay=True,  include_BFS_NNLO=True,  apply_delta_QCD=False)),
    ("+dQCD",  dict(include_NLO_hard_decay=True,  include_BFS_NNLO=True,  apply_delta_QCD=True)),
]

# ISR axis. LL+exp = the analytic β³-truncated single-convolution radiator;
# NLL = the eMELA 2-leg radiator (forces isr_scheme='2leg' downstream).
ISR_LEGS = {
    "LL":  dict(isr_nll=False, isr_emela_ll=False, isr_scheme="single_conv"),
    "NLL": dict(isr_nll=True),
}

# Informational order tag baked into the template filename per rung.
ORDER_OF = {"LO": 0, "+NLO": 1, "+NNLO": 2, "+dQCD": 2}

# The injected-truth rung: full production hard side, NLL ISR.
TRUTH_HARD = "+dQCD"
TRUTH_ISR = "NLL"

# Tags whose σ we actually compute. ``pseudodata`` is intentionally absent —
# the fit always overrides the pseudodata with the injected truth, so its
# file content is irrelevant; we satisfy FitCore's unconditional read by
# copying the nominal template onto the pseudodata filename.
_GEN_TAGS = ["nominal", "mass_var", "width_var", "cross_mass_width"]

# Lumi prior used in 'free' mode: huge correlated prior ⇒ overall
# normalisation effectively unconstrained.
_LUMI_CORR_FREE = 1.0


def _ladder_parameters() -> Parameters:
    """POI-only parameter set: m_W and Γ_W (drop α_s), cross-term kept."""
    reduced = {"mass": card.PARAMETERS["mass"], "width": card.PARAMETERS["width"]}
    return Parameters(reduced, cross_terms=[("mass", "width")])


def _make_generator(hard_over: dict, isr_over: dict, order: int) -> WWGenerator:
    """Production generator (from_card) with the rung's chain toggles applied."""
    gen = WWGenerator.from_card(card)
    for k, v in {**hard_over, **isr_over}.items():
        setattr(gen, k, v)
    gen.order = order
    return gen


def _scales():
    s = getattr(card, "RENORM_SCALES", {"mass": 80.0, "width": 80.0, "vars": []})
    return s["mass"], s["width"], getattr(card, "MASS_SCHEME", "OS")


# ---------------------------------------------------------------------------
# Template generation (parallel over rung × tag)
# ---------------------------------------------------------------------------
def _gen_one(spec):
    """Worker: generate one (rung, tag) template into its rung dir.

    ``spec`` is fully picklable (strings + plain dicts) so the generator is
    reconstructed inside the worker — the steering card is a module and does
    not survive pickling.
    """
    hard_key, hard_over, isr_key, isr_over, tag, outdir = spec
    gen = _make_generator(hard_over, isr_over, ORDER_OF[hard_key])
    params = _ladder_parameters()
    mass_scale, width_scale, mass_scheme = _scales()
    path = gen.do_scan(
        params.values(tag),
        mass_scale=mass_scale, width_scale=width_scale, mass_scheme=mass_scheme,
        outdir=outdir,
    )
    return f"{hard_key:7s} {isr_key:3s} {tag:18s} → {os.path.basename(path)}"


def _rung_dir(base: str, hard_key: str, isr_key: str) -> str:
    safe = hard_key.replace("+", "p")
    return os.path.join(base, f"{safe}_{isr_key}")


def _generate_all_templates(base: str, isr_keys, max_workers: int, verbose=True):
    """Generate every (rung, isr, tag) template under ``base``; parallel."""
    specs = []
    for hard_key, hard_over in HARD_RUNGS:
        for isr_key in isr_keys:
            isr_over = ISR_LEGS[isr_key]
            outdir = _rung_dir(base, hard_key, isr_key)
            for tag in _GEN_TAGS:
                specs.append((hard_key, hard_over, isr_key, isr_over, tag, outdir))
    if verbose:
        print(f"[ladder] generating {len(specs)} templates "
              f"({len(HARD_RUNGS)} rungs × {len(isr_keys)} ISR × {len(_GEN_TAGS)} tags) "
              f"on {max_workers} workers ...")
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for line in ex.map(_gen_one, specs):
                if verbose:
                    print("   " + line)
    else:
        for spec in specs:
            line = _gen_one(spec)
            if verbose:
                print("   " + line)

    # Satisfy FitCore's unconditional read of the 'pseudodata' tag by copying
    # each rung's nominal template onto the pseudodata filename (content is
    # discarded — the fit always injects the production truth as pseudodata).
    params = _ladder_parameters()
    mass_scale, width_scale, mass_scheme = _scales()
    for hard_key, _ in HARD_RUNGS:
        for isr_key in isr_keys:
            gen = _make_generator(HARD_RUNGS[0][1], ISR_LEGS[isr_key], ORDER_OF[hard_key])
            outdir = _rung_dir(base, hard_key, isr_key)
            nom = gen.file_name(params.values("nominal"), mass_scale=mass_scale,
                                width_scale=width_scale, mass_scheme=mass_scheme, indir=outdir)
            pse = gen.file_name(params.values("pseudodata"), mass_scale=mass_scale,
                                width_scale=width_scale, mass_scheme=mass_scheme, indir=outdir)
            shutil.copyfile(nom, pse)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _ladder_card():
    """SimpleNamespace clone of the WW card, reduced to a 2-POI cov-lumi fit:
    no nuisances/constraints, lumi handled in the covariance."""
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"], "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def _build_fit(card_ov, hard_key, isr_key, base):
    gen = _make_generator(dict(HARD_RUNGS_BY_KEY[hard_key]), ISR_LEGS[isr_key], ORDER_OF[hard_key])
    fit = WWFit(card_ov, gen, input_dir=_rung_dir(base, hard_key, isr_key), asimov=True)
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])
    return fit


HARD_RUNGS_BY_KEY = dict(HARD_RUNGS)


def _fit_rung(card_ov, hard_key, isr_key, base, truth_nom, lumi_corr):
    """Fit one rung against the injected truth; return a result row dict."""
    fit = _build_fit(card_ov, hard_key, isr_key, base)
    fit.lumi_uncorr = 0.0
    fit.lumi_corr = lumi_corr
    fit.create_scenario(pseudodata=truth_nom)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)  # [mass, width] ufloat, GeV
    mass, width = res[0], res[1]
    truth_mass = fit.d_params["nominal"]["mass"]
    truth_width = fit.d_params["nominal"]["width"]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    return {
        "hard": hard_key, "isr": isr_key,
        "bias_mW": (mass.n - truth_mass) * 1e3,
        "bias_gW": (width.n - truth_width) * 1e3,
        "sig_mW": mass.s * 1e3,
        "sig_gW": width.s * 1e3,
        "rho": rho,
        "valid": bool(fit.minuit.valid),
    }


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------
def run_theory_ladder(*, isr="both", workers=8, out=None, keep=False, base=None):
    """Run the full ladder and emit the residual-bias table.

    Parameters
    ----------
    isr : "both" | "LL" | "NLL"
    workers : process-pool size for template generation
    out : path stem for the output table (``.csv`` + ``.md`` written); default
          ``plots/theory_ladder``
    keep : keep the temporary template directory (else removed at the end)
    base : explicit template base dir (else a fresh tempdir)
    """
    # Legs reported in the table vs. legs whose templates we must build. The
    # truth rung (NLL) is always generated so it can be injected as the
    # Asimov truth, even when only the LL leg is reported.
    report_keys = ["LL", "NLL"] if isr == "both" else [isr]
    gen_keys = list(dict.fromkeys(report_keys + [TRUTH_ISR]))
    out = out or os.path.join("plots", "theory_ladder")

    own_base = base is None
    base = base or tempfile.mkdtemp(prefix="ww_theory_ladder_")
    print(f"[ladder] template base: {base}")
    try:
        _generate_all_templates(base, gen_keys, max_workers=workers)

        # Injected truth = production hard side + NLL ISR, smeared nominal.
        card_ov = _ladder_card()
        truth_fit = _build_fit(card_ov, TRUTH_HARD, TRUTH_ISR, base)
        truth_nom = truth_fit.template("nominal")

        rows = []
        for lumi_mode, lumi_corr in (("free", _LUMI_CORR_FREE),
                                     ("prior", card.PRIORS["lumi"]["corr"])):
            for isr_key in [k for k in ("LL", "NLL") if k in report_keys]:
                for hard_key, _ in HARD_RUNGS:
                    row = _fit_rung(card_ov, hard_key, isr_key, base, truth_nom, lumi_corr)
                    row["lumi"] = lumi_mode
                    rows.append(row)
    finally:
        if own_base and not keep:
            shutil.rmtree(base, ignore_errors=True)

    _emit_table(rows, out, report_keys)
    return rows


def _emit_table(rows, out, isr_keys):
    lines = []
    lines.append("WW theory-uncertainty ladder — Asimov (m_W, Γ_W) residual bias")
    lines.append(f"Injected truth: {TRUTH_HARD} hard side + {TRUTH_ISR} ISR (full production).")
    lines.append("Fit: m_W, Γ_W float; stat (Asimov) + correlated lumi only; "
                 "all other parameters fixed.")
    lines.append("'free' = corr-lumi opened up (pure shape bias); "
                 "'prior' = realistic FCC-ee corr-lumi prior.")
    lines.append("")
    header = (f"{'lumi':5s} {'ISR':4s} {'rung':7s} "
              f"{'Δm_W':>9s} {'δ(prev)':>9s} {'ΔΓ_W':>9s} "
              f"{'σ_mW':>7s} {'σ_ΓW':>7s} {'ρ':>6s}  conv")
    units = (f"{'':5s} {'':4s} {'':7s} "
             f"{'[MeV]':>9s} {'[MeV]':>9s} {'[MeV]':>9s} "
             f"{'[MeV]':>7s} {'[MeV]':>7s} {'':>6s}")
    for lumi_mode in ("free", "prior"):
        sub = [r for r in rows if r["lumi"] == lumi_mode]
        if not sub:
            continue
        lines.append("=" * len(header))
        lines.append(header)
        lines.append(units)
        lines.append("-" * len(header))
        for isr_key in [k for k in ("LL", "NLL") if k in isr_keys]:
            prev = None
            for hard_key, _ in HARD_RUNGS:
                r = next((x for x in sub if x["isr"] == isr_key and x["hard"] == hard_key), None)
                if r is None:
                    continue
                dprev = "" if prev is None else f"{r['bias_mW'] - prev:+9.3f}"
                lines.append(
                    f"{lumi_mode:5s} {isr_key:4s} {hard_key:7s} "
                    f"{r['bias_mW']:+9.3f} {dprev:>9s} {r['bias_gW']:+9.3f} "
                    f"{r['sig_mW']:7.3f} {r['sig_gW']:7.3f} {r['rho']:+6.2f}  "
                    f"{'ok' if r['valid'] else 'BAD'}")
                prev = r["bias_mW"]
            lines.append("-" * len(header))
    lines.append("")
    lines.append("NOT captured by this ladder (separate estimate needed):")
    lines.append("  • NLO electroweak (YFSWW3-class non-factorisable + initial-final).")
    lines.append("  • Higher-order Coulomb (BFS G_C vs. dropped FKM K_C).")
    text = "\n".join(lines)
    print("\n" + text + "\n")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".txt", "w") as fh:
        fh.write(text + "\n")
    # Machine-readable CSV.
    import csv
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lumi", "isr", "rung", "bias_mW_MeV", "bias_gW_MeV",
                    "sig_mW_MeV", "sig_gW_MeV", "rho", "minuit_valid"])
        for r in rows:
            w.writerow([r["lumi"], r["isr"], r["hard"],
                        f"{r['bias_mW']:.4f}", f"{r['bias_gW']:.4f}",
                        f"{r['sig_mW']:.4f}", f"{r['sig_gW']:.4f}",
                        f"{r['rho']:.4f}", int(r["valid"])])
    print(f"[ladder] wrote {out}.txt and {out}.csv")
