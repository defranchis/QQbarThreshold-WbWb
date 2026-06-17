# HANDOFF — scheme-variation campaign, continuation (2026-06-16, end of day)

Picks up the program in `HANDOFF_scheme_variations.md`. Findings are also in memory
`project_scheme_variations_2026-06-16.md`. This file = the **live state + open items**
so another agent can resume cold. User had to leave; resume later.

## What is DONE (committed findings — see the memory file for physics detail)

- **Task 0** — O(α) matching test (`oalpha_matching_test.py`): Δ⊗Δ matching PROVEN; eMELA = Δ;
  σ̂_NLO inclusive/Δ; idip resolved. ΔC₁ large at σ̂ level, cancels in matched obs → O(α²).
- **Task 1** — hard-EW-coupling scheme of σ̂ (`ew_scheme_crossfit.py`): GENUINE ~12 MeV shape.
  gf↔alphaz = **−12.3 shape / −9.2 cov-lumi**; gf↔alpha0 = **+11.3 / +23.6**; gf↔gf = 0.00.
  NOW the LARGEST theory systematic. Param/coupling mismatch RULED OUT (e/(g·sinθ_W)=1 all
  schemes; only α changes; channel-indep flat Born). Coupling-consistency check (a) done.
- **Task 2** — DELTA↔MS̄ (`msbar_rematched_crossfit.py`): MS̄ ePDF endpoint-pathological
  → Δ is the UNIQUE valid NLL fac scheme; fac-scheme residual O(α²). 0.14 proxy RETIRED.
- **Task 3** — ISR ren-scheme ALPMZ↔MSBAR at fixed α (`ren_scheme_crossfit.py`):
  **+0.36 shape / +15.2 cov-lumi** (shape negligible).
- **Quoting decision** (user): quote INDIVIDUAL variations, NO envelope; α(0) is a poor
  scheme for W physics → gf↔α(M_Z) is the representative bracket, flag α(0) caveat.
- **Plot** — `report/figs/ww_scheme_variations.pdf` generated (`plot_scheme_lineshapes.py`):
  2×2, EW(gf/alphaz/alpha0) left | ISR(LL+exp/NLL-Δ-ALPMZ/NLL-Δ-MSBARren) right;
  top = σ(√s) absolute, bottom = line-shape ratio (norm removed) = the m_W lever.

## OPEN ITEMS (priority order)

### 1. ISR-ratio STEPS pathology  ← user flagged "pathological, investigate tomorrow"
- **Symptom:** bottom-RIGHT panel of `ww_scheme_variations.pdf` (ISR line-shape ratio,
  normalisation removed) shows step-like structure at the ~0.05–0.1 % level.
- **Localised THIS session** (fine √s scan, 2nd-difference of the BETA/prod shape ratio):
  curvature spikes are spread across the steep turn-on **158–160 GeV at ~0.25 GeV spacing
  = the MoCaNLO σ̂ grid-node spacing**, NOT at 161.0 → **NOT the corruption** (item 2).
- **Root-cause hypothesis (strong):** the σ̂(√ŝ) interpolator is a scipy `UnivariateSpline`
  with weights=1/err, k=order, s=len(points) (`partonic_grid.py:63-68`). Its knot-level
  non-smoothness (kinks) in the steep turn-on, convolved with TWO different radiators
  (BETA analytic vs prod cubic-spline) and ratio'd (norm removed), does NOT cancel → steps
  at √s≈σ̂-node. NOT the eMELA grid (16 Q-knots over [75,350], cubic → ~no knots in
  [157,163], can't make 0.25 GeV steps; `isr_emela_grid.py` EmelaGrid RectBivariateSpline).
- **Actions tomorrow:**
  1. Regenerate the plot AFTER restoration (item 2) completes — rule out any residual 161
     contribution. Steps should persist (they're in the turn-on, not at 161).
  2. If persistent: smoother σ̂ interpolant — try larger `s`, monotone PCHIP, or denser ecm
     grid; OR accept as sub-0.1 % numerical artefact (≪ 12 MeV EW syst) and note in caption.
  3. **Check it does NOT bias the cross-fit Δm_W**: truth & morph share the same σ̂ spline so
     it likely cancels in the fit, but verify (the shape-only numbers feed the report).
- Relevant memory: `feedback_always_rerun_diagnostic_plots`, `feedback_verify_plot_features_numerically`.

### 2. Restore 3 corrupted production EOS points  ← NOW ON CONDOR
- **Cluster 12886569** (3 jobs, submitted 2026-06-16 19:07, JobFlavour "tomorrow", ~2h each).
  Files: `restore161.sub` + `points_restore161.txt` (500k ev, 1.0 % prec, base-seed 1000 =
  varpoint-correlated, default --outdir = prod EOS). Local jobs were KILLED.
- **Corrupted files:** `/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results/`
  `lnuqq_nominal_ecm161.0000_{gf,alphaz,alpha0}.csv` — overwritten with 200-event noise
  (σ̂_Born ~134–174 ± ~10, wall 16 s) by my throwaway Recola-banner run_point runs (default
  outdir = prod EOS). Healthy neighbour: σ̂_Born ~170 ± 0.35, wall ~6700 s.
- **VERIFY when done:** `condor_q 12886569`; then each CSV nominal-161 row should show
  err_born ~0.3, σ̂_Born ~170, wall_s ~thousands.
- **Cross-fits / plots that READ these points must wait for restoration** (ew_scheme_crossfit,
  the line-shape plot). Re-run them after to get clean numbers.

### 3. Report update (PENDING) — §sec:indep-ew + §sec:indep-isr + figure
- Embed `report/figs/ww_scheme_variations.pdf` (already generated): `\begin{figure}` + `\ref`
  from both subsections.
- **§sec:indep-ew (report lines ~3055–3086): STALE NUMBERS — reconcile before writing.**
  Report currently says α(M_Z): +1.8 cov-lumi / +19.9 shape; α(0): −3.3 / −11.5. The NEW
  campaign (`ew_scheme_crossfit.py`, prod grid + identical prod NLL ISR across all 3 schemes)
  gives gf↔alphaz −12.3 shape / −9.2 cov-lumi; gf↔alpha0 +11.3 / +23.6. **Signs AND
  magnitudes differ** — the old values came from a DIFFERENT script (`crossfit_scheme.py`,
  different ISR config / cross-fit direction / pre-fix). DO NOT paste from memory: **re-run
  `ew_scheme_crossfit.py` AFTER restoration** for clean current numbers, and work out WHY
  they differ from the report's stale values (truth↔morph direction? grid? C₁ fix?). The
  campaign numbers are the authoritative ones to report.
- Add coupling-consistency discussion: Recola derives e=√(4πα), g=e/sinθ_W, sin²θ_W=
  1−M_W²/M_Z²=0.2230 FIXED all schemes ⇒ e/(g·sinθ_W)=1; ONLY α changes (gf 1/132.18,
  alphaz 1/128.94, alpha0 1/137.04); Born∝α⁴ flat-norm spread is lumi-absorbed; the √ŝ-tilt
  is the m_W lever. Source: `get_alpha_rcl` + `input_rcl.f90:135-137`. NOT a param mismatch.
- Quoting: individual variations, NO envelope; α(0) poor-scheme caveat; gf↔α(M_Z) the bracket.
- Framing: hard-EW-scheme σ̂ spread (~12 MeV) is now the LARGEST theory syst of the indep
  chain (ISR sub-MeV after Tasks 0/2/3). Scheme-vs-parametric: realistic budget FIXES G_μ
  (FCC-ee EW-fit baseline) + propagates α(M_Z) PARAMETRIC unc, NOT the full scheme span.
- **§sec:indep-isr:** add Task 0 (Δ⊗Δ proven), Task 2 (MS̄ pathological→Δ unique, 0.14 retired,
  O(α²)), Task 3 (ren-scheme +0.36 shape / +15.2 cov-lumi). Reference the ISR panel.
- Then republish via `report/publish.sh` + send live PDF URL. **ASK before commit**
  (`feedback_ask_before_committing`, `feedback_keep_report_in_sync`, `feedback_report_publish_link`).

### 4. G_F-consistency test (the user's "last item")
- **User's concern:** the 3 EW schemes use today's PDG central α values, which are NOT
  mutually SM-consistent ⇒ part of the ~12 MeV may be spurious input-inconsistency, not
  genuine NNLO-EW truncation.
- **User's prescription (I confirmed it is CORRECT):** adjust an INPUT so the PHYSICAL
  α(M_Z) is the same across schemes; the residual is then the pure renormalisation-
  prescription effect (input-value difference removed). NOT "set the LO couplings equal".
- **Cleanest implementation:** adjust G_F in the gf scheme so α_Gμ = √2 G_F M_W² sw²/π =
  α(M_Z) = 1/128.936 (Recola alZ default). G_F_adj ≈ 1.1663787e-5 × (132.18/128.936) ≈
  **1.1957e-5** (+2.5 %; sw²=1−M_W²/M_Z²≈0.2230). Then gf(G_F_adj) and alphaz share the
  same LO coupling → cross-fit gf(adj)↔alphaz = PURE renorm-prescription residual.
  - collapses to ~0 → original ~12 MeV was largely input-α-value (spurious) → demote.
  - stays O(several MeV) → genuine renorm-scheme (NNLO-EW trunc) → keep.
- **CODE NEEDED (not currently supported):** `run_point.py` has NO --fermi-constant / --alphaz
  flag (only --scheme-alpha {gf,alphaz,alpha0,alphamsbar}, --m-t/--m-h/--m-z). The card DOES
  emit `<fermi_constant>` (gf reads it) → adjust `SMInputs.fermi_constant`. Add a
  `--fermi-constant` flag to run_point.py threading to SMInputs (small additive change).
  [Alt: add --alphaz + SMInputs.alphaz + `<alphaz>` card emission to pin the alphaz input.]
- **Then:** verify G_F_adj with Recola `get_alpha_rcl` (1-point run printing α = 1/128.936),
  THEN regenerate the gf grid (nominal varpoint over 33 ecm for shape; full morph for the
  cross-fit) with G_F_adj to a **SCRATCH outdir (NOT prod EOS!)**, **ON CONDOR** (~hours).
  Cross-fit vs the existing alphaz grid.

### 5. Task 4 — ISR α-running order LL↔NLL
- Needs `emela_c_wrapper.cpp` to expose `SetPerturbativeOrderAlpha` (`eMELA.hh:120`) +
  recompile `libeMELApy.so`. `build_grid` already has a `pert_order` param. Light after rebuild.

### 6. Task 5 — compensated μ_F
- Handoff said "needs MS̄ σ̂" but MS̄ is pathological; DOABLE in Δ via an analytic μ_F-shift
  counterterm on σ̂ (grid Q∈[75,350] ⊇ ξ√s already covers it). Reassess.

### 7. Fold into `theory_ladder.py` rows + republish; commit (ASK first).

## KEY FILES / NUMBERS
- Scripts: `scripts/investigations/nll_isr/{ew_scheme_crossfit, oalpha_matching_test,
  msbar_rematched_crossfit, ren_scheme_crossfit, plot_scheme_lineshapes}.py`
- Grids: prod `framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz`;
  MSBAR-ren `/tmp/ww_nll_scheme_scan/grids/emela_nll_delta_msbarren_alpmz.npz`
  (**/tmp is VOLATILE — rebuild via `ren_scheme_crossfit.py` if gone**).
- Figure: `report/figs/ww_scheme_variations.pdf`.
- σ̂ interpolator: `partonic_grid.py:63-68` (UnivariateSpline, w=1/err, k=order, s=len(pts)).
- eMELA grid: `isr_emela_grid.py` EmelaGrid (RectBivariateSpline cubic over (ln omx, ln Q),
  16 Q-knots, Q∈[75,350]).
- Coupling source: Recola `get_alpha_rcl` (`input_rcl.f90:2336-2370`); defaults
  al0=1/137.036, alZ=1/128.936, alMS=1/127.930 (`input_rcl.f90:135-137`).

## GOTCHAS
- **NEVER** run `run_point.py` without `--outdir <scratch>` for throwaway runs — the default
  is the production EOS dir (this is exactly how the 3 points got corrupted).
- **Long jobs → CONDOR** (user directive: "always submit to condor stuff that runs forever").
  Local Bash IS an ironic node for interactive/short work (`feedback_no_ssh_ironic02`), but
  anything multi-hour goes to condor.
- `/tmp` grids are volatile across sessions/nodes.
- Quote syst components SEPARATELY, never in quadrature (`feedback_systematics_components_separate`).
