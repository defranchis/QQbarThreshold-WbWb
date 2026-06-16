# HANDOFF — ISR scheme-dependence & coupling-consistency studies (WW threshold m_W)

Prepared 2026-06-16. Self-contained brief for an agent picking up the ISR scheme work
on the FCC-ee WW-threshold m_W measurement. You have repo access but not the originating
conversation — everything you need is here or in the pointers at the end.

---

## 0. Objective

Replace the current **proxy** estimate of the ISR scheme/coupling dependence with
**measured, genuine** variations, and resolve the "is gf-σ̂ ⊗ alpmz-ISR a mismatch?" concern.
Today the residual ISR factorisation-scheme systematic is quoted as a **0.14 MeV proxy**
(an α-value swing). At the FCC-ee target σ(m_W)=0.25 MeV that is NOT negligible, and it is
a coarse stand-in, not a real measurement. Produce the real numbers.

**All tasks are read-only/additive — do NOT change the production chain or the headline result.**
**Ask the user before any git commit (standing rule).**

---

## 1. Background (what you must understand first)

The independent WW chain computes σ_obs(√s) = ∫∫ dx₁dx₂ D(x₁)D(x₂) σ̂_NLO(x₁x₂s):
- **σ̂_NLO**: e+e-→4f NLO-EW hard cross section from MoCaNLO+Recola1, run with `pdf_set=none`
  (bare electrons), Catani-Seymour dipoles, EW coupling scheme `scheme_alpha=gf` (G_μ).
  Because `pdf_set=none` there is NO IS mass factorisation → **σ̂ is in the DELTA (DIS-like)
  factorisation scheme** (σ̂ ≈ σ_Born inclusive; verified: σ̂_NLO/σ̂_Born=+8.3%, no −30%
  radiative tail).
- **D**: eMELA electron ePDF (BCFS arXiv:1911.12040), NLL, factorisation scheme **DELTA**,
  renormalisation scheme **ALPMZ**, α(M_Z)=**1/128.943** (PDG).
- Production = **DELTA σ̂ ⊗ DELTA ePDF = matched** (correct).

**Two distinct α roles (NOT a mismatch — do not "fix" by making them equal):**
- *Hard EW coupling* — `scheme_alpha=gf` → α_Gμ≈1/132. Renormalises the WW production/decay
  EW couplings. G_μ is the standard scheme for W physics (absorbs Δr / universal corrections).
  Luminosity-degenerate (little direct m_W bias).
- *ISR QED coupling* — eMELA `ALPMZ` → α(M_Z)≈1/128.9. Drives the collinear ISR logs
  ln(s/m_e²). Shape-relevant (biases m_W). Fixed by the Tera-Z α(M_Z) measurement.
These enter different factors (σ̂ vs D) and are renormalised independently. What MUST match is
the **collinear factorisation scheme** (DELTA/MS̄) — and it does. The coupling residual where
they overlap (the O(α) IS-QED piece carries α_Gμ in σ̂ vs α(M_Z) in D) is genuinely O(α²);
the job is to MEASURE it, not assume it.

**Status of the matching argument:** DELTA is established 3 ways (empirical α-stability scan,
literature, numerical closure) but the *mechanism* ("pdf_set=none ⇒ σ̂ is DELTA") was
*inferred*, not proven by O(α) expansion. There is also an internal wording tension:
`isr_beta.py` says the `idip` run "applies the IS collinear counterterm" (sounds MS̄), while
the conclusion is σ̂=DELTA. Resolution: `idip` removes the collinear *divergence/log*; the
DELTA↔MS̄ *finite* term (few-%) is the residual. **Task 0 settles this directly.**

---

## 2. Environment & where things live

```bash
cd /afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
source setup.sh                 # sources cvmfs LCG_106 (py3.11 + numpy/scipy/iminuit/
                                # uncertainties) + QQBAR_INSTALL libs for eMELA dlopen
export PYTHONPATH=$PWD:$PYTHONPATH
# Verify: python3 -c "import scipy,iminuit,uncertainties; from framework.process.ww.indep
#   import isr_beta; from framework.process.ww.xsec_calculator import emela_wrapper"
```
Do NOT rely on a bare `/usr/bin/python3` (3.9 user-site is flaky / lacks the stack).
Heavy MC (only Task 1's `alphamsbar`, if pursued) runs on `fcc-ironic-02/-03`, never lxplus
(see CLAUDE.md). The eMELA-grid / cross-fit tasks below are light and run locally on LCG_106.

**Code map (file:line where stable):**
- `framework/process/ww/indep/mocanlo_cards.py` — MoCaNLO card writer; `scheme_alpha` (line 36/113),
  `pdf_set=none` (134), dipoles (141-146), scales (114-116), run types born/virt/real/idip (75).
- `framework/process/ww/indep/generator_mocanlo.py` — `WWGeneratorMoCaNLO(scheme_alpha=…, isr_cfg=…)`;
  `_isr_cfg` builds the ISRConfig; `_varpoint_lineshape` calls the convolution.
- `framework/process/ww/indep/partonic_grid.py` — loads EOS σ̂ grids by suffix `_{scheme_alpha}`;
  `ChannelVarGrid` keeps **born + nlo** (drops virt/real/idip). EOS dir = `DEFAULT_RESULTS_DIR`
  (`/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results`).
- `framework/process/ww/indep/isr_beta.py` — ISRConfig; `convolve_2leg`; `_per_leg_emela_nll`
  (direct) / `_per_leg_grid_nll` (grid); **`oalpha_isr_subtraction` (≈609-662) = the LL β C₁,
  diagnostic only** (needed for Task 0).
- `framework/process/ww/indep/isr_emela_grid.py` — `build_and_write(npz, fac_scheme, ren_scheme,
  alpha, pert_order, omx_knots, q_knots)`; grids bake fac/ren/alpha into meta (a guard checks them).
- `framework/process/ww/xsec_calculator/emela_wrapper.py` — `initialize(pert_order, fac_scheme,
  ren_scheme, alpha)`, `code_pdf(x,omx,Q)`, `ll_pdf(idx,…)`.
- `framework/process/ww/theory_ladder.py` — ISR scheme-var block ('trunc'/'ren'/'scale').
- **Cross-fit harness (reuse these):** `scripts/investigations/nll_isr/scheme_alpha_scan.py`
  (helpers `_card_2poi`, `_gen_templates`, `_fresh_fit`, `_crossfit`, `nll_cfg`, `ll_cfg`,
  `grid_path`, `ALPHAS`), and `mu_f_truncation.py` (μ_F study), `confirm_alpha_decomp.py`.

**eMELA capability (verified from `install/include/eMELA/eMELA.hh`):**
- order: LL/NLL (`SetPerturbativeOrder`) — **no NNLL**.
- α-running order: LL/NLL (`SetPerturbativeOrderAlpha`).
- fac scheme: DELTA, MSBAR. ren scheme: MSBAR, FIXED, ALPMZ, ALGMU. (+`SetDELTAGMU`.)

**MoCaNLO capability (verified):** `scheme_alpha` ∈ {gf, alpha0, alphaz, alphamsbar}
(`standard_model.F90:7610-7622`; CLI `run_point.py --scheme-alpha`). `pdf_scheme` (MSbar/beta/…)
exists but needs a beam PDF whose e+e- soft-endpoint sampling is **broken** (NaN/σ→0) → MoCaNLO
**cannot give a usable MS̄ σ̂**; use the analytic route (Task 2). Install:
`/afs/.../FCC/QQbar_threshold/mocanlo/src/{mocanlo-1.0.0,recola1-1.5.1,COLLIER-1.2.9}`.

---

## 3. Tasks — the genuine variations (ranked by value/cost)

Reporting convention for ALL: quote components **separately, NEVER summed in quadrature**;
give **both shape-only (lumi free) and cov-lumi (realistic prior)**; flag the cov-lumi
normalisation leakage as lumi-degenerate (accounting left open). See
`[[feedback-systematics-components-separate]]`.

### Task 0 — O(α)-expansion matching test  [KEYSTONE, light, do FIRST]
**Goal:** prove σ̂ is DELTA and that σ̂↔ISR are matched at O(α) — directly, not via an α-slope.
This controls the 0.14 proxy (expectation, to verify: true residual ≪ 0.14) and resolves the
idip/DELTA wording tension.
**How:** expand the eMELA-NLL two-leg convolution to O(α) and compare to σ̂_NLO. The O(α) piece
of the radiator applied to σ̂_Born is exactly `oalpha_isr_subtraction` (isr_beta.py, the LL β C₁).
Check: `[D_NLL ⊗ σ̂_Born]|_{O(α)}` reproduces the IS-collinear content of `σ̂_NLO − σ̂_Born`
(per √s on the 157–163 grid). Reuse `convolve_2leg`, the σ̂ grid, and `oalpha_isr_subtraction`;
also see `scripts/investigations/nll_isr/validate_emela_isr.py` (closest existing check).
**Acceptance:** a per-√s table of [O(α) expansion] vs [σ̂_NLO finite IS piece]; agreement at the
expected few-% (DELTA) confirms matching; a large discrepancy would indicate MS̄-like σ̂.
**Caution:** derive the O(α) expansion of the *eMELA* NLL ePDF carefully; verify the C₁ formula
against source, do not paste from memory (`[[feedback-verify-paper-formulae]]`).

### Task 1 — Hard EW-coupling scheme: gf ↔ alphaz ↔ alpha0  [CHEAP — grids exist]
**Goal:** measure the hard-EW-renormalisation-scheme dependence of σ̂ → directly answers the
gf/alpmz concern (alphaz puts the HARD coupling at α(M_Z), the SAME value as the ISR).
**Key fact:** `*_alphaz.csv` and `*_alpha0.csv` σ̂ grids ALREADY EXIST on EOS (≈1980 inclusive
points each, alongside gf). **No MC needed** — just refit. (`alphamsbar` would need an MC grid
on ironic; optional.)
**How:** with the `scheme_alpha_scan.py` harness, build NLL templates for
`WWGeneratorMoCaNLO(scheme_alpha="alphaz", isr_cfg=<DELTA/ALPMZ NLL>)` and cross-fit against the
`scheme_alpha="gf"` templates (truth = one scheme, morph = the other), shape-only and cov-lumi.
Same for alpha0.
**Acceptance:** Δm_W(gf↔alphaz), Δm_W(gf↔alpha0), shape + cov-lumi. Small shape spread ⇒ the
gf/alpmz coupling choice is harmless (concern resolved); large ⇒ a real hard-EW-scheme systematic
to carry. **First confirm the alphaz/alpha0 grids are input-synced with gf** (same m_t/M_H/M_Z,
varpoints, √s); if not, regenerate or restrict to matched points.

### Task 2 — IS factorisation DELTA ↔ MS̄, RE-MATCHED  [replaces the 0.14 proxy]
**Goal:** the genuine *matched-to-matched* factorisation-scheme dependence.
**How:** build an MS̄ σ̂ = DELTA σ̂ + ΔC₁ analytically, then convolve with an eMELA **MSBAR** grid,
and compare m_W to the DELTA⊗DELTA production. ΔC₁ comes from the DELTA→MS̄ finite term
D_MS̄−D_Δ = −(α/2π)[(1+x²)/(1−x)(2ln(1−x)+1)]₊ ⊗ σ̂_Born (Frixione arXiv:2105.06688 eq.33).
Code pattern for an additive σ̂: `match_bfs.py`; a `delta_c1` exists in
`scripts/investigations/bfs_match/c1_delta_vs_beta_fixedorder.py`. σ̂_Born is available
(`ChannelVarGrid` keeps born). Build the MSBAR eMELA grid with
`isr_emela_grid.build_and_write(..., fac_scheme="MSBAR", ren_scheme="ALPMZ", alpha=1/128.943)`.
**Acceptance:** Δm_W(DELTA⊗DELTA vs MS̄⊗MS̄), shape + cov-lumi → the real residual-scheme number
(should be ≪ the −384 MeV *mismatched* DELTA⊗MS̄ diagnostic, and bracket the 0.14 proxy).
**CAUTION:** ΔC₁ is the FINITE K_ee^Δ scheme term, NOT the LL collinear log; derive sign/form
from the Frixione source and unit-test against eq.33 before trusting m_W.

### Task 3 — ISR coupling-renorm scheme at FIXED α value: ALPMZ ↔ MSBAR  [light]
**Goal:** isolate the QED-coupling *scheme* (not value). Rebuild an eMELA grid with
`ren_scheme="MSBAR"` at the **same** α(M_Z)=1/128.943, cross-fit vs ALPMZ.
**Do NOT** use ALGMU/α(0) here — those change the α *value* and are the demoted 'ren' diagnostic
(the large numbers are the non-linear value sensitivity, not a genuine scheme effect; don't
double-count).
**Acceptance:** Δm_W(ALPMZ↔MSBAR) at fixed α, shape + cov-lumi.

### Task 4 — ISR α-running order: LL ↔ NLL  [light]
**Goal:** a true higher-order handle. eMELA `SetPerturbativeOrderAlpha` (0=LL,1=NLL); rebuild
grids at fixed fac/ren/α, cross-fit. **Acceptance:** Δm_W, shape + cov-lumi.

### Task 5 — Compensated μ_F scale variation  [needs Task 2 first]
**Goal:** a genuine scale-variation truncation (the current ξ is uncompensated — σ̂ has no μ_F
counterterm — so it only probes DGLAP numerics). With the MS̄ σ̂(μ_F) from Task 2 (whose
counterterm carries μ_F), vary μ_F=ξ√s, ξ∈{½,2} in σ̂ AND D together. **Acceptance:** the
compensated Δm_W envelope, shape + cov-lumi.

---

## 4. Do NOT do (non-genuine — never quote as a scheme systematic)
- **DELTA σ̂ ⊗ MS̄ ePDF** (gives −384 MeV cov-lumi): deliberately MISMATCHED diagnostic only.
- **Uncompensated ξ with pdf_set=none** (31 MeV cov-lumi): DGLAP-numerics, not truncation.
- **ISR α-VALUE swings** ALGMU/α(0)/the 0.14 (1/128.232↔1/128.943): value sensitivity, not a
  scheme variation. The physical α(M_Z)-input uncertainty is the `aem_isr` nuisance ≈ 0.005 MeV.
- **Forcing the couplings equal** (running σ̂ in alphaz "to match" the ISR): wrong physics —
  alphaz is a worse HARD scheme. gf(hard)+α(M_Z)(ISR) is correct (Task 1 *measures* the spread).

## 5. Hard limits
- **eMELA has no NNLL** (LL/NLL only) → a direct NNLL−NLL truncation is impossible; the LL→NLL
  order-step proxy is the ceiling.
- **MoCaNLO can't give a usable MS̄ σ̂** (broken lepton-PDF sampling) → use the Task-2 analytic route.

## 6. Deliverables
- Per-task: shape + cov-lumi Δm_W, components kept separate (no quadrature).
- Fold results into `theory_ladder.py` rows where natural and the report
  `report/ww_bfs_implementation.tex` §sec:val-isr-cross; keep report in sync and republish via
  `report/publish.sh` (needs `pdflatex` on PATH:
  `/cvmfs/sft.cern.ch/lcg/external/texlive/latest/bin/x86_64-linux`; builds 3× for refs).
- Investigation scripts go under `scripts/investigations/nll_isr/`; never commit generated PDFs.
- **Ask before committing.** End any report republish with the live URL
  (https://mdefranc.web.cern.ch/WW_threshold/report/ww_bfs_implementation.pdf).

## 7. Pointers
- `memory/research/SETUP_CAPABILITIES_2026-06-16.md` — full verified capability/plan (this brief
  is its actionable form).
- `memory/research/SCHEME_MATCHING_SCOPING.md` — the DELTA-vs-MS̄ resolution (RESOLVED header).
- `memory/project_nll_promotion_plan_2026-06-15.md` — overall NLL-promotion roadmap (step 6 =
  wire the grid into production; the scheme menu lives here too).
- Existing results to reproduce/extend: `scheme_alpha_scan.py` (DELTA kernel −0.98, swing 0.14;
  MSBAR mismatch −384), `mu_f_truncation.py` (μ_F shape 1.4 / cov-lumi 31), theory_ladder 'trunc'
  row (0.42 shape / 11.6 cov-lumi).
