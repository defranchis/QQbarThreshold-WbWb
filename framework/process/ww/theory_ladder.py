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

On top of the perturbative ladder, an **ISR scheme-variation** block
probes the ISR theory uncertainty. After the 2026-06-16 DELTA-NLL
resolution (report sec:val-isr-cross; scripts/investigations/nll_isr/
scheme_alpha_scan.py + mu_f_truncation.py), the ISR theory systematic is
reported as SEPARATE components — **NOT combined in quadrature** — and is
kept distinct from renorm-scheme / scale STABILITY diagnostics:

  * **NNLL truncation** ('trunc' row) — the size of the highest included
    order: the α-FIXED order step eMELA-LL ↔ eMELA-NLL (same production
    α(M_Z)=1/128.943 on both sides, so the leading-log α-value piece
    cancels and only the genuine NLL kernel remains). This is the dominant
    ISR theory-systematic component (≈1 MeV; cf. indep-chain 0.98 MeV).
  * **residual factorisation scheme** (DELTA↔MSBAR) — RESOLVED 2026-06-16
    (indep-chain scheme campaign: scripts/investigations/nll_isr/{oalpha_
    matching_test, msbar_rematched_crossfit, ren_scheme_crossfit}.py). The
    O(α) DELTA⊗DELTA matching is exact (Task 0: eMELA=DELTA; σ̂_NLO carries no
    IS-collinear log; the ΔC₁ finite term cancels in the matched observable),
    and the MS̄ resummed ePDF is endpoint-pathological (Task 2: D_MS̄/D_Δ→1.44
    as x→1, MS̄⊗MS̄≈2.2×Δ⊗Δ — an all-orders blow-up no O(α) counterterm can
    cancel). So DELTA is the UNIQUE valid NLL factorisation scheme and the
    residual is O(α²)/sub-MeV; the coupling-renorm scheme at FIXED α (Task 3,
    ALPMZ↔MSBAR) is +0.36 MeV shape (negligible). The ≈0.14 MeV proxy is
    RETIRED (it was an α-VALUE swing, not a scheme effect; no valid alternative
    scheme remains to bracket).
  * **α(M_Z) input** — the measurement uncertainty on the ISR coupling,
    ≈0.005 MeV, carried as the profiled ``aem_isr`` nuisance (not a row
    here; it lives in the scenario-systematics table).

DIAGNOSTICS — stability checks, NOT systematic components:
  * **α renormalisation scheme** ('ren' rows) — ALPMZ (production) vs ALGMU
    (α_Gμ≈1/132.17) vs FIXED/α(0) (Thomson). The spread (≈36 MeV
    ALPMZ↔ALGMU) is DOMINATED by the non-linear LL α-VALUE sensitivity (the
    fit absorbs Δα into m_W; step 1), NOT the NLL kernel. The production α
    is fixed by the EW input scheme to α(M_Z), so this is a renorm-scheme
    stability diagnostic whose physical residual is the ``aem_isr`` input
    nuisance above — NOT the headline systematic (superseded 2026-06-16).
  * **factorisation scale ξ** ('scale' rows, Q=ξ√s, ξ∈{0.5,2}) — DGLAP
    evolution-stability only: σ̂ has no μ_F counterterm (pdf_set=none) so
    ξ-variation is uncompensated (shape ≈1.4 MeV ≈ the truncation; cov-lumi
    ≈31 MeV is a normalisation artefact), NOT a truncation uncertainty.

What this ladder does **not** capture: pieces entirely absent from the
BFS chain — NLO electroweak (YFSWW3-class non-factorisable + initial-final
interference) and higher-order Coulomb (the BFS unstable-W Green function
G_C vs. the dropped FKM K_C). Those are genuine theory uncertainties that
live outside the ladder and need a separate, non-ladder estimate. The
independent MoCaNLO+Recola chain DOES carry full NLO-EW: its hard-EW
renormalisation-scheme spread (gf↔alphaz ≈ −12 MeV shape — the now-largest
indep-chain theory systematic; ew_scheme_crossfit.py) quantifies part of
this NLO-EW uncertainty. A G_F-consistency test (gf with G_F tuned so the
derived α_Gμ equals the alphaz α(M_Z), gf_consistency_crossfit.py) separates
the genuine NNLO-EW renormalisation-prescription residual from the (spurious)
non-SM-consistent input-α-value difference; see report sec:indep-ew.

Run via ``python3 doFit_ww.py --theoryLadder`` (see doFit_ww.py for flags),
or import :func:`run_theory_ladder`.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import types
from concurrent.futures import ProcessPoolExecutor

import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.generator import WWGenerator
from framework.process.ww.fit import WWFit
from framework.process.ww.xsec_calculator.eft_xsec import (
    ALPHA_EM_0, ALPHA_MZ_PDG, alpha_Gmu, M_W_BFS_REF,
)

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
HARD_RUNGS_BY_KEY = dict(HARD_RUNGS)

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
# normalisation effectively unconstrained. Public so the production
# ``doFit_ww.py --shapeOnly`` fit reuses the SAME free-lumi value (single
# source of truth — a flat 100% correlated normalisation variance).
LUMI_CORR_FREE = 1.0


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
# Template generation
# ---------------------------------------------------------------------------
# All templates — perturbative-ladder rungs (LL/NLL) AND the ISR-scheme
# variants — are generated in a SINGLE flat pool so every (slow) NLL template
# competes for the worker pool at once, in one ~10-min wave, rather than two
# sequential waves that each leave most of the pool idle. Each template pins
# exactly ONE core: ``run_theory_ladder`` forces ``WW_ISR_NJOBS=1`` so the
# inner per-√s eMELA pool never nests inside this outer pool and oversubscribes
# (workers × WW_ISR_NJOBS). The outer pool IS the parallelism.
def _gen_one(spec):
    """Worker: generate one template, reusing an up-to-date cached file if the
    persistent base already holds one with a matching fingerprint
    (``ensure_scan``). ``spec`` is fully picklable (strings + plain dicts) so
    the generator is rebuilt inside the worker — the steering card is a module
    and does not survive pickling."""
    label, hard_over, isr_over, order, tag, outdir = spec
    gen = _make_generator(hard_over, isr_over, order)
    params = _ladder_parameters()
    mass_scale, width_scale, mass_scheme = _scales()
    path, regen = gen.ensure_scan(
        params.values(tag),
        mass_scale=mass_scale, width_scale=width_scale, mass_scheme=mass_scheme,
        outdir=outdir,
    )
    return f"{label:16s} {tag:18s} [{'gen  ' if regen else 'reuse'}] → {os.path.basename(path)}"


def _rung_dir(base: str, hard_key: str, isr_key: str) -> str:
    safe = hard_key.replace("+", "p")
    return os.path.join(base, f"{safe}_{isr_key}")


def _ladder_specs(base, isr_keys):
    """Unified specs for every perturbative rung × ISR leg × tag."""
    for hard_key, hard_over in HARD_RUNGS:
        for isr_key in isr_keys:
            outdir = _rung_dir(base, hard_key, isr_key)
            for tag in _GEN_TAGS:
                yield (f"{hard_key} {isr_key}", hard_over, ISR_LEGS[isr_key],
                       ORDER_OF[hard_key], tag, outdir)


def _variant_specs(base, variants):
    """Unified specs for the ISR-scheme variants (hard side fixed at the truth
    rung; only the eMELA α-scheme / ξ overrides change)."""
    hard_over = dict(HARD_RUNGS_BY_KEY[TRUTH_HARD])
    for label, _kind, overrides in variants:
        outdir = _isr_var_dir(base, label)
        for tag in _GEN_TAGS:
            yield (f"isr:{label}", hard_over, overrides,
                   ORDER_OF[TRUTH_HARD], tag, outdir)


def _generate_templates(base, isr_keys, variants, max_workers, verbose=True):
    """Generate ALL templates (ladder rungs + ISR-scheme variants) in one pool,
    then drop the pseudodata-tag copies FitCore unconditionally reads."""
    specs = list(_ladder_specs(base, isr_keys)) + list(_variant_specs(base, variants))
    if verbose:
        print(f"[ladder] generating {len(specs)} templates "
              f"({len(HARD_RUNGS)}×{len(isr_keys)} ladder + {len(variants)} ISR-scheme, "
              f"× {len(_GEN_TAGS)} tags) on {max_workers} workers ...")
    if max_workers > 1 and len(specs) > 1:
        # Cap the pool to the number of templates (no point forking more workers
        # than there is work — ``--ladderWorkers`` defaults to 48 while a typical
        # ladder has ~32–52 specs), and use the ``forkserver`` start method.
        #
        # forkserver forks each worker from a *clean* server process rather than
        # from the threaded main interpreter (matplotlib/Agg + eMELA pull in
        # background threads at import). The default ``fork`` method copies those
        # threads' locks in arbitrary states, which trips the CPython
        # fork-with-threads teardown race (cpython#90622): on an all-cache-hit
        # ladder every ``_gen_one`` returns in ~ms, so the workers finish almost
        # simultaneously — exactly the timing that left the ProcessPoolExecutor
        # manager thread unable to join them. The idle (state-S) workers were
        # then never reaped, the ``with`` block's shutdown blocked forever, and
        # because the children are non-daemonic the foreground ssh never
        # returned (the run looked "still running" long after the table printed).
        # forkserver's sentinel-based teardown is not subject to that race.
        #
        # ``_gen_one`` + its specs are fully picklable by design (strings + plain
        # dicts; the generator is rebuilt inside the worker), so forkserver — which
        # re-imports this module per worker instead of inheriting parent memory —
        # needs no code change. Workers inherit ``os.environ`` (incl. the
        # ``WW_ISR_NJOBS=1`` guard set by ``run_theory_ladder``); the morph grid is
        # pre-warmed into the AFS client cache by the launch scripts, so the
        # per-worker re-read on a cache miss does not re-introduce grid contention.
        n_workers = min(max_workers, len(specs))
        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            for line in ex.map(_gen_one, specs):
                if verbose:
                    print("   " + line)
    else:
        for spec in specs:
            line = _gen_one(spec)
            if verbose:
                print("   " + line)

    # Satisfy FitCore's unconditional read of the 'pseudodata' tag by copying
    # each output dir's nominal template onto the pseudodata filename (content
    # discarded — the fit always injects the production truth as pseudodata).
    # Only ``order`` enters the filename, so a throwaway LO-override generator
    # at the dir's order suffices.
    params = _ladder_parameters()
    mass_scale, width_scale, mass_scheme = _scales()
    dir_order = (
        [(_rung_dir(base, hk, ik), ORDER_OF[hk]) for hk, _ in HARD_RUNGS for ik in isr_keys]
        + [(_isr_var_dir(base, lbl), ORDER_OF[TRUTH_HARD]) for lbl, _, _ in variants]
    )
    for outdir, order in dir_order:
        gen = _make_generator(HARD_RUNGS[0][1], ISR_LEGS[isr_keys[0]], order)
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


def _init_scenario(fit, scenario):
    """Apply a scan scenario to ``fit``. ``scenario=None`` → the card's default
    uniform 7-point grid (157–163, step 1). Otherwise ``scenario`` is a dict
    with an explicit ``scan_list`` (and optional per-point ``lumi_dict``,
    ``total_lumi``, ``last_lumi``, ``add_last_ecm``) — used by
    ``--compareScenarios`` to fit the SAME templates over different √s sets."""
    if scenario is None:
        S = card.SCENARIO
        fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                          scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                          last_lumi=S["last_lumi"])
    else:
        fit.init_scenario(
            scan_list=scenario["scan_list"],
            lumi_dict=scenario.get("lumi_dict"),
            total_lumi=scenario.get("total_lumi", card.SCENARIO["total_lumi"]),
            last_lumi=scenario.get("last_lumi", card.SCENARIO["last_lumi"]),
            add_last_ecm=scenario.get("add_last_ecm", False),
        )


def _build_fit(card_ov, hard_key, isr_key, base, scenario=None):
    gen = _make_generator(dict(HARD_RUNGS_BY_KEY[hard_key]), ISR_LEGS[isr_key], ORDER_OF[hard_key])
    fit = WWFit(card_ov, gen, input_dir=_rung_dir(base, hard_key, isr_key), asimov=True)
    _init_scenario(fit, scenario)
    return fit


# ---------------------------------------------------------------------------
# ISR scheme variation (α-fixed NNLL truncation + renorm-scheme/ξ stability)
# ---------------------------------------------------------------------------
# All variants ride on the full production hard side (+dQCD). Each is fit
# against the production truth (NLL, ALPMZ, ξ=1), so its best-fit shift is the
# scheme-/order-induced m_W/Γ_W bias. After the 2026-06-16 DELTA-NLL
# resolution the kinds split into ONE systematic component and two diagnostics
# (see module docstring); components are reported SEPARATELY, not in quadrature.
#
# 'trunc' row  → NNLL TRUNCATION (the ISR theory-systematic component): the
#                α-fixed order step eMELA-LL ↔ eMELA-NLL (same production α on
#                both sides → leading-log α-value cancels, genuine NLL kernel).
# 'ren'   rows → renorm-scheme STABILITY diagnostic (ALPMZ↔ALGMU↔α(0)); the
#                ≈36 MeV spread is the non-linear LL α-VALUE sensitivity, NOT
#                the kernel. α is fixed by the EW input scheme; the physical
#                residual is the aem_isr nuisance (≈0.005 MeV). NOT the headline.
# 'scale' rows → ξ DGLAP-stability diagnostic, NOT a truncation uncertainty in
#                the DELTA scheme (σ̂ has no μ_F counterterm; see docstring).
_ALPHA_MZ_FALLBACK = ALPHA_MZ_PDG    # ALPMZ; framework single source (eft_xsec),
                                     # = card PARAM_INPUTS["alpha_em_isr"] default


def _isr_scheme_variants():
    """Return [(label, kind, isr_overrides), ...] for the ISR scheme scan.

    The production ISR α (ALPMZ) is read from the live card so the baseline
    variant is bit-identical to the injected truth (closure sanity row)."""
    prod = WWGenerator.from_card(card)
    a_mz = prod.alpha_em_isr if prod.alpha_em_isr is not None else _ALPHA_MZ_FALLBACK
    a_gmu = alpha_Gmu(M_W_BFS_REF)   # ALGMU, BFS prescription (≈1/132.17)
    a_0 = ALPHA_EM_0                 # FIXED/Thomson α(0) (≈1/137.036)
    nll = {"isr_nll": True}          # +dQCD hard already production; NLL ISR
    return [
        # NNLL-truncation component: eMELA DGLAP LL at the SAME production α →
        # the bias vs the eMELA-NLL truth is the α-fixed NLL kernel.
        ("eMELA-LL",    "trunc", {"isr_nll": False, "isr_emela_ll": True,
                                   "isr_emela_ren_scheme": "ALPMZ",
                                   "alpha_em_isr": a_mz,  "isr_scale_factor": 1.0}),
        ("ALPMZ(prod)", "ren",   {**nll, "isr_emela_ren_scheme": "ALPMZ",
                                   "alpha_em_isr": a_mz,  "isr_scale_factor": 1.0}),
        ("ALGMU",       "ren",   {**nll, "isr_emela_ren_scheme": "ALGMU",
                                   "alpha_em_isr": a_gmu, "isr_scale_factor": 1.0}),
        ("alpha(0)",    "ren",   {**nll, "isr_emela_ren_scheme": "FIXED",
                                   "alpha_em_isr": a_0,   "isr_scale_factor": 1.0}),
        ("xi=0.5",      "scale", {**nll, "isr_emela_ren_scheme": "ALPMZ",
                                   "alpha_em_isr": a_mz,  "isr_scale_factor": 0.5}),
        ("xi=2.0",      "scale", {**nll, "isr_emela_ren_scheme": "ALPMZ",
                                   "alpha_em_isr": a_mz,  "isr_scale_factor": 2.0}),
    ]


def _isr_var_dir(base: str, label: str) -> str:
    safe = (label.replace("(", "").replace(")", "").replace("=", "")
            .replace(".", "p").replace("/", "").replace(" ", ""))
    return os.path.join(base, f"isrvar_{safe}")


def _fit_and_row(fit, truth_nom, lumi_corr, extra):
    """Run the lumi-corr-only Asimov fit (uncorr lumi off, correlated lumi at
    ``lumi_corr``) on ``fit`` against the injected ``truth_nom`` and return a
    result row: the caller's identifier keys (``extra``) merged with the common
    bias/sig/rho/valid fields."""
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
        **extra,
        "bias_mW": (mass.n - truth_mass) * 1e3,
        "bias_gW": (width.n - truth_width) * 1e3,
        "sig_mW": mass.s * 1e3,
        "sig_gW": width.s * 1e3,
        "rho": rho,
        "valid": bool(fit.minuit.valid),
    }


def _fit_isr_variant(card_ov, label, overrides, base, truth_nom, lumi_corr,
                     scenario=None):
    """Fit one ISR-scheme variant against the production truth; return a row."""
    gen = _make_generator(dict(HARD_RUNGS_BY_KEY[TRUTH_HARD]), overrides,
                          ORDER_OF[TRUTH_HARD])
    fit = WWFit(card_ov, gen, input_dir=_isr_var_dir(base, label), asimov=True)
    _init_scenario(fit, scenario)
    return _fit_and_row(fit, truth_nom, lumi_corr, {"variant": label})


def _fit_rung(card_ov, hard_key, isr_key, base, truth_nom, lumi_corr, scenario=None):
    """Fit one rung against the injected truth; return a result row dict."""
    fit = _build_fit(card_ov, hard_key, isr_key, base, scenario=scenario)
    return _fit_and_row(fit, truth_nom, lumi_corr, {"hard": hard_key, "isr": isr_key})


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------
#: Persistent on-disk cache for the ladder/scheme-variation templates. Lives
#: under the (git-ignored) output area so a re-run reuses every unchanged
#: template (``ensure_scan`` fingerprint check) instead of rebuilding the
#: expensive NLL set. Wiped only by deleting the directory by hand.
LADDER_CACHE_DIR = os.path.join("output_xsec", "ww", "theory_ladder")


def run_theory_ladder(*, isr="both", workers=48, out=None, keep=True, base=None,
                       scheme_var=True, scenario=None):
    """Run the full ladder and emit the residual-bias table.

    Parameters
    ----------
    isr : "both" | "LL" | "NLL"
    workers : process-pool size for template generation
    out : path stem for the output table (``.csv`` + ``.md`` written); default
          ``plots/theory_ladder``
    keep : keep the template directory after the run. Default ``True``: the
        templates persist in :data:`LADDER_CACHE_DIR` so the next run reuses
        the unchanged ones. Set ``False`` only with an explicit throwaway
        ``base`` to have it removed at the end.
    base : template directory (default :data:`LADDER_CACHE_DIR`, persistent).
    scheme_var : also run the ISR scheme-variation block (α-renormalisation
        scheme + ξ stability) on top of the perturbative ladder. Always NLL
        (eMELA); fit against the production truth. Default True.
    scenario : optional scan-scenario dict (``scan_list`` + optional
        ``lumi_dict`` / ``total_lumi`` / ``last_lumi`` / ``add_last_ecm``). The
        templates are scenario-independent (fine √s grid), so only the FIT step
        changes — this is what ``--compareScenarios`` uses to fit one cached
        template set over different √s sets. ``None`` → card's 7-point grid.
    """
    # Legs reported in the table vs. legs whose templates we must build. The
    # truth rung (NLL) is always generated so it can be injected as the
    # Asimov truth, even when only the LL leg is reported.
    report_keys = ["LL", "NLL"] if isr == "both" else [isr]
    gen_keys = list(dict.fromkeys(report_keys + [TRUTH_ISR]))
    out = out or os.path.join("plots", "theory_ladder", "theory_ladder")

    # Persistent cache by default (templates saved on disk, reused next run);
    # an explicit ``base`` can opt into a throwaway dir + ``keep=False``.
    own_base = base is not None
    base = base or LADDER_CACHE_DIR
    os.makedirs(base, exist_ok=True)
    print(f"[ladder] template cache: {base} "
          f"(reused where the fingerprint matches; pass keep=False + an "
          f"explicit base to discard)")
    variants = _isr_scheme_variants() if scheme_var else []
    # Force the inner per-√s eMELA pool to serial for the duration of template
    # generation, so it does NOT nest inside the outer template pool and
    # oversubscribe (workers × WW_ISR_NJOBS processes). The outer pool is the
    # parallelism — one core per template. Saved/restored around the run.
    _prev_njobs = os.environ.get("WW_ISR_NJOBS")
    os.environ["WW_ISR_NJOBS"] = "1"
    try:
        _generate_templates(base, gen_keys, variants, max_workers=workers)

        # Injected truth = production hard side + NLL ISR, smeared nominal.
        card_ov = _ladder_card()
        truth_fit = _build_fit(card_ov, TRUTH_HARD, TRUTH_ISR, base, scenario=scenario)
        truth_nom = truth_fit.template("nominal")

        rows = []
        scheme_rows = []
        for lumi_mode, lumi_corr in (("free", LUMI_CORR_FREE),
                                     ("prior", card.PRIORS["lumi"]["corr"])):
            for isr_key in [k for k in ("LL", "NLL") if k in report_keys]:
                for hard_key, _ in HARD_RUNGS:
                    row = _fit_rung(card_ov, hard_key, isr_key, base, truth_nom,
                                    lumi_corr, scenario=scenario)
                    row["lumi"] = lumi_mode
                    rows.append(row)
            for label, kind, overrides in variants:
                srow = _fit_isr_variant(card_ov, label, overrides, base,
                                        truth_nom, lumi_corr, scenario=scenario)
                srow["lumi"] = lumi_mode
                srow["kind"] = kind
                scheme_rows.append(srow)
    finally:
        # Only discard a *throwaway* base (explicit base + keep=False); the
        # default persistent cache (own_base False) is always kept.
        if own_base and not keep:
            shutil.rmtree(base, ignore_errors=True)
        if _prev_njobs is None:
            os.environ.pop("WW_ISR_NJOBS", None)
        else:
            os.environ["WW_ISR_NJOBS"] = _prev_njobs

    _emit_table(rows, out, report_keys, scheme_rows)
    return rows, scheme_rows


def _emit_table(rows, out, isr_keys, scheme_rows=None):
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

    # --- ISR scheme-variation block ----------------------------------------
    if scheme_rows:
        s_header = (f"{'lumi':5s} {'variant':12s} {'kind':6s} "
                    f"{'Δm_W':>9s} {'ΔΓ_W':>9s} "
                    f"{'σ_mW':>7s} {'σ_ΓW':>7s} {'ρ':>6s}  conv")
        s_units = (f"{'':5s} {'':12s} {'':6s} "
                   f"{'[MeV]':>9s} {'[MeV]':>9s} "
                   f"{'[MeV]':>7s} {'[MeV]':>7s} {'':>6s}")
        lines.append("")
        lines.append("#" * len(s_header))
        lines.append("ISR scheme variation — hard side = +dQCD; truth = NLL/ALPMZ/ξ=1 (production).")
        lines.append("Components reported SEPARATELY, not in quadrature (2026-06-16 DELTA-NLL resolution):")
        lines.append("'trunc' = NNLL TRUNCATION (the ISR theory-systematic component): α-fixed")
        lines.append("          order step eMELA-LL ↔ eMELA-NLL (same α → only the NLL kernel).")
        lines.append("'ren'   = α-renorm-scheme STABILITY diagnostic (ALPMZ↔ALGMU↔α(0)); the")
        lines.append("          spread is the LL α-VALUE sensitivity, NOT the kernel. α is fixed by")
        lines.append("          the EW input scheme; physical residual = aem_isr nuisance (~0.005).")
        lines.append("'scale' = ξ (Q=ξ√s) DGLAP-stability diagnostic ONLY — NOT a truncation")
        lines.append("          uncertainty in the DELTA scheme (σ̂ has no μ_F counterterm).")
        lines.append("#" * len(s_header))
        for lumi_mode in ("free", "prior"):
            sub = [r for r in scheme_rows if r["lumi"] == lumi_mode]
            if not sub:
                continue
            lines.append(s_header)
            lines.append(s_units)
            lines.append("-" * len(s_header))
            for r in sub:
                lines.append(
                    f"{lumi_mode:5s} {r['variant']:12s} {r['kind']:6s} "
                    f"{r['bias_mW']:+9.3f} {r['bias_gW']:+9.3f} "
                    f"{r['sig_mW']:7.3f} {r['sig_gW']:7.3f} {r['rho']:+6.2f}  "
                    f"{'ok' if r['valid'] else 'BAD'}")
            lines.append("-" * len(s_header))

    # --- ISR theory systematic: separate components (NOT in quadrature) -----
    if scheme_rows:
        def _trunc(lm):
            r = next((x for x in scheme_rows if x.get("kind") == "trunc"
                      and x["lumi"] == lm), None)
            return None if r is None else r["bias_mW"]
        lines.append("")
        lines.append("ISR theory systematic — SEPARATE components (NOT added in quadrature):")
        for lm in ("free", "prior"):
            t = _trunc(lm)
            if t is not None:
                lines.append(f"  [{lm:5s}] NNLL truncation = {abs(t):6.2f} MeV  "
                             f"(eMELA-LL↔eMELA-NLL, α fixed)")
        lines.append("         residual DELTA scheme =   0.14 MeV  "
                     "(kernel α-swing; step 5, indep chain)")
        lines.append("         α(M_Z) input          =   0.005 MeV "
                     "(aem_isr profiled nuisance)")
        lines.append("         ('ren'/'scale' rows above are STABILITY diagnostics, "
                     "not components.)")
        # Quote the live free (shape-only) vs prior (cov-lumi) truncation rather
        # than frozen literals, so the NOTE can never drift away from the numbers
        # emitted a few lines above when the fit/grid changes.
        _t_free, _t_prior = _trunc("free"), _trunc("prior")
        _shape_str = f"~{abs(_t_free):.1f}" if _t_free is not None else "~0.4"
        _cov_str = f"~{abs(_t_prior):.0f}" if _t_prior is not None else "~12"
        lines.append(f"  NOTE: the truncation is {_shape_str} MeV shape-only but "
                     f"{_cov_str} MeV cov-lumi —")
        lines.append("  LL->NLL is almost pure NORMALISATION (~+0.3% ISR flux, little shape);")
        lines.append("  under the tight lumi prior that flux change leaks into m_W via the rate")
        lines.append("  handle (same pattern as 'scale'/'ren'). Whether the cov-lumi leakage is")
        lines.append("  a separate ISR-norm component or absorbed into the lumi + cross-section-")
        lines.append("  normalisation budget is an OPEN accounting choice (both shown above).")

    lines.append("")
    lines.append("NOT captured by this ladder (separate estimate needed):")
    lines.append("  • NLO electroweak (YFSWW3-class non-factorisable + initial-final).")
    lines.append("  • Higher-order Coulomb (BFS G_C vs. dropped FKM K_C).")
    lines.append("  • Factorisation-scheme (DELTA↔MSBAR) ISR uncertainty — proper variation")
    lines.append("    needs the +∫K(x)σ_Born collinear counterterm in σ̂ (deferred); bounded")
    lines.append("    ≈0.14 MeV by the NLL-kernel α-swing (step 5, listed as a component above).")
    text = "\n".join(lines)
    print("\n" + text + "\n")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".txt", "w") as fh:
        fh.write(text + "\n")
    # Machine-readable CSV.
    import csv
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "lumi", "isr", "rung", "bias_mW_MeV", "bias_gW_MeV",
                    "sig_mW_MeV", "sig_gW_MeV", "rho", "minuit_valid"])
        for r in rows:
            w.writerow(["ladder", r["lumi"], r["isr"], r["hard"],
                        f"{r['bias_mW']:.4f}", f"{r['bias_gW']:.4f}",
                        f"{r['sig_mW']:.4f}", f"{r['sig_gW']:.4f}",
                        f"{r['rho']:.4f}", int(r["valid"])])
        for r in (scheme_rows or []):
            w.writerow([f"isr_scheme:{r['kind']}", r["lumi"], "NLL", r["variant"],
                        f"{r['bias_mW']:.4f}", f"{r['bias_gW']:.4f}",
                        f"{r['sig_mW']:.4f}", f"{r['sig_gW']:.4f}",
                        f"{r['rho']:.4f}", int(r["valid"])])
    print(f"[ladder] wrote {out}.txt and {out}.csv")
