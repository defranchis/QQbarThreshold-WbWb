# Synthesis: BFS-on-MoCaNLO matching layer — code + physics review

Scope: commits 080492c / 74993f0 / 48a453b. Construction = graft BFS NNLO (relative K-factor
δ_NNLO = (σ_NNLO−σ_NLO)/σ_Born) + per-channel δ_QCD onto the independent MoCaNLO+Recola1 NLO-EW
line shape, then one ISR convolution with an O(α) matching subtraction (C₁) keyed to MoCaNLO's Born.
Review = 7 dimensions (double-count, isr-matching, dqcd, nll-port, assembly-morph, code-quality,
report-accuracy) × adversarial verification (20 agents, workflow wf_b0de4149-ce8). This document
consolidates the lead reviewer's hand-checkpoint (REVIEW_bfs_moca_matching.md) with the per-agent
StructuredOutput findings and their final verdicts. No physics was re-run here; verdicts are final.

---

## 1. Bottom line

The matching construction is **physically sound and correctly built**, and it is **safe as the
current default-OFF, WIP-bannered exploratory code** — no defect was found that biases the published
headline shape-only pull (δ_NNLO +1.6 MeV, NLL −3.7 MeV). Double-counting protection against
MoCaNLO's own NLO is **structural** (δ_NNLO cancels the entire BFS NLO, incl. the O(α) Coulomb eq.62,
to ~1e-12) rather than numerical, every NNLO piece is genuinely α-suppressed, the O(α) ISR matching
cancels exactly (matched−Born scales as α² not α), the per-channel δ_QCD bookkeeping was traced to
MoCaNLO source and confirmed double-count-free, and the morph/units/channel-assembly chain is unbiased
(self-pull = 0.00 MeV). The single most important caveat is operational, not physical: **before the
WIP banner is dropped, two MAJOR latent footguns must be closed** — (a) the generator cache key omits
all matching flags, so reusing an instance across a flag toggle silently serves a stale/unmatched line
shape, and (b) there is no automated test exercising the production NLL path. Neither bites today's
scripts (each builds a fresh generator per variant), but both would silently bias m_W if the usage
pattern changed.

---

## 2. Confirmed issues, ranked by severity

All findings below have verdict **confirmed** or **partial**. Severities are the adversarially-corrected
values. None biases the currently-published numbers; the impact column states the latent/conditional risk.

### MAJOR

**M1 — Generator `_cache` key omits all matching flags; non-frozen dataclass → silent stale hit on a reused instance** (CQ-1, confirmed)
- *What's wrong:* `WWGeneratorMoCaNLO` is `@dataclass` with `frozen=False`. `_varpoint_lineshape`
  caches on `ck=(varpoint, _grid_key(sqrt_s))` and `_fit_morph` on `("coeffs", _grid_key)` — neither
  includes `match_bfs / match_bfs_nnlo / match_bfs_dqcd / isr_nll / alpha_s / sm`. The cache check
  runs *before* `cfg = self._isr_cfg()` is built, so the flags structurally cannot reach the key.
  The verifier **reproduced a silent stale hit**: on a used instance, flipping `match_bfs=True`
  returned the OLD unmatched array (`allclose(before,after)=True`, `allclose(after,fresh)=False`,
  ~0.12% line-shape error); same for `alpha_s` (0.1199→0.130), `isr_nll` (False→True, served LL),
  and the coeffs cache at off-nominal m_W.
- *Impact:* If an instance is ever reused across a flag toggle, the full neglected piece is dropped →
  m_W bias up to ~1.6 MeV (δ_NNLO) or ~3.7 MeV (NLL). **Not triggered today** — `matched_pull.py:120-132`
  and `plot_match_lineshape.py:29` build a fresh `_gen()`/instance per variant.
- *Fix:* fold a generator-state fingerprint (`match_bfs, _nnlo, _dqcd, isr_nll, alpha_s, sm.mt/mH/mZ,
  br_convention, scheme_alpha, lepton_cut, _cfg_fingerprint(self._isr_cfg())`) into both cache keys, or
  set `frozen=True` (forces a fresh instance on any change). Add a toggle self-check.
- *Cite:* `framework/process/ww/indep/generator_mocanlo.py:77` (`@dataclass`, no frozen), `:155`, `:202`.

**M2 — No automated test exercises the NLL eMELA path** (nll-port-1, confirmed)
- *What's wrong:* `isr_beta.py:18-19` docstring claims a cross-check vs `isr.sigma_ISR_2leg_convolution`
  is "performed in tests/__main__", but `isr_beta.py` has **no `__main__`** and the only validation
  script `scripts/investigations/indep_mocanlo/test_isr_beta.py` runs **LL only** (`:69` `nll=False`,
  zero `assert`s — print/plot diagnostics, not regression guards). `_per_leg_emela_nll` is invoked by
  no test or script. The 6e-10 closure figure is real (verifier independently reproduced it, in fact
  to ~1e-15 with matched configs) but only ever ad-hoc.
- *Impact:* `isr_nll` is the production NLL knob that sets the m_W central value when on. A future change
  to `_per_leg_emela_nll`, `emela_wrapper`, or `LAMBDA1_NF0` would silently break the NLL line shape with
  no test failing. No bias today.
- *Fix:* add an `nll=True` assertion to `test_isr_beta.py` (compare `convolve_2leg` vs
  `bfs_isr.sigma_ISR_2leg_convolution` on identical σ̂, tol ~1e-8, skip if libeMELApy unavailable);
  fix the docstring (no `__main__`).
- *Cite:* `framework/process/ww/indep/isr_beta.py:18-19`, `:316/:371`; `scripts/investigations/indep_mocanlo/test_isr_beta.py:69`.

**M3 — δ_NNLO@163 GeV documented as +0.21% (text/table) vs +0.12% (figure) for the same symbol** (RA-1, confirmed)
- *What's wrong:* two different physical objects carry the same label. +0.21% (report `:2997`, Table `:3012`)
  is the **partonic** K-factor δ_NNLO(√ŝ) (eq:match-dnnlo); +0.12% (figure caption `:3110`, §match-pull `:3097`)
  is the **ISR-convolved observed** line-shape ratio, diluted by the radiator. Verifier confirmed both:
  partonic +0.214%, observed +0.124%. The text at `:2996-2998` even mixes the partonic +0.21% with the
  observed NLL +0.26–0.59% in one sentence.
- *Impact:* **No m_W/Γ_W effect** (descriptive sizing, not a fit input). But a reader cannot reconcile
  table vs figure and would conclude one is wrong.
- *Fix:* label one "partonic δ_NNLO K-factor" and the other "ISR-convolved observed ratio", or quote
  both consistently in both places.
- *Cite:* `report/ww_bfs_implementation.tex:2997, 3012, 3110, 3097`.

### MINOR

**m4 — Soft-limit branch (omx<1e-15) is heavily exercised; substituting it shifts NLL norm ~0.39% with a ~0.035% energy slope** (nll-port-2, **partial**; severity corrected major→minor)
- *What's wrong:* the hypothesis that the `omx<1e-15` branch is dead code is **disproven** — with
  β_e≈0.058 (1/β_e≈17) about 31/128 GL nodes fall below the cutoff, carrying ~14% of the per-leg weight.
  There the code substitutes the flat analytic `norm_nll` instead of eMELA's `code_pdf`, which keeps
  growing (~logarithmically) toward the endpoint → a downward discontinuity at the cutoff. **Corrected
  numbers:** the σ_obs shift from evaluating `code_pdf` everywhere is **+0.39%** (the finding's first
  draft said +0.51%) with a **~0.035% energy slope** across the scan (draft said 0.082%); the
  `norm_nll≈1.0434` quoted in the draft was wrong (true ≈1.0421). The branch is **numerically identical
  between the port and the BFS-side isr.py** (rel diff 2.8e-11), so it does NOT break the 6e-10
  port-vs-reference closure — it is a genuine NLL soft-limit systematic on **both** chains.
- *Impact:* the ~0.39% piece is a near-flat normalization that cancels in a line-shape fit; only the
  ~0.035% energy slope is m_W-relevant, projecting to **sub-MeV / low-MeV** once Γ_W floats (NOT the
  "tens of MeV" of the first draft).
- *Fix:* push the cutoff far lower (e.g. 1e-300 with x floored so `code_pdf` stays callable) and let
  eMELA's own asymptotics run, OR document the ~0.4% norm / ~0.035% slope as an ISR-soft-limit
  uncertainty in §sec:match. Correct the misleading "those nodes contribute negligibly" claim at the
  **single** location `isr.py:441` (the draft's dual citation and `isr_beta.py:323` cite were wrong).
- *Cite:* `framework/process/ww/indep/isr_beta.py` soft branch; reference `isr.py:441`.

**m5 — LL `isr_cfg.alpha` defaults to α_Gμ and is never coupled to the MoCaNLO `scheme_alpha` grid** (ISR-5, confirmed)
- *What's wrong:* `scheme_alpha` selects which MoCaNLO σ̂ grid (hence the α in σ̂_NLO's intrinsic O(α) ISR
  collinear log) via `load_grids` globbing `*_{scheme_alpha}.csv` and the card's `<scheme_alpha>`. But the
  LL `ISRConfig.alpha` defaults to α_Gμ and `_isr_cfg()` only swaps α inside the `isr_nll` branch — the LL
  branch returns `self.isr_cfg` unchanged, with no reference to `scheme_alpha`. Verified at runtime:
  `WWGeneratorMoCaNLO(scheme_alpha='alphaz')._isr_cfg().resolved_alpha()` = α_Gμ, not α(M_Z).
- *Impact:* **None on the default gf run** (α_Gμ consistent everywhere). The moment the EW-scheme-variation
  grids (`alphaz`/`alpha0`) land and are fitted with default LL ISR, the radiator+C₁ (α_Gμ) silently
  mismatch σ̂_NLO's ISR-log α at O(α²) — an uncontrolled, unflagged contamination of the ALPMZ↔ALGMU
  ~36 MeV variation that is the dominant theory systematic. The double-count still cancels (radiator and
  C₁ agree); only the second-order overlap with σ̂_NLO's log is inconsistent. **Currently latent: only
  `*_gf.csv` grids exist** (555 in the results dir, zero alphaz/alpha0).
- *Fix:* derive `isr_cfg.alpha` from `scheme_alpha` (`alpha_for_scheme`) when at default, or assert/warn if
  `resolved_alpha()` differs from the grid's scheme α, before those grids are used. Document the rule in
  `cards/README`.
- *Cite:* `framework/process/ww/indep/generator_mocanlo.py:81, 83, 140-151`; `mocanlo_cards.py:113`.

**m6 — Incomplete σ̂ grid → cryptic `LinAlgError` instead of a clear "incomplete grid" guard** (AM-8, confirmed)
- *What's wrong:* the default `results` dir holds lnuqq+mutau but **no qqqq** (verified: 553 CSVs, 0 qqqq).
  Pure-WW needs {lnuqq, qqqq, mutau}; the all-or-nothing varpoint filter drops all 20 varpoints → empty
  design matrix → `np.linalg.lstsq` raises `LinAlgError: 1-dimensional array given. Array must be
  two-dimensional` rather than a clear message. Fails loudly, no bias (headline pull used the complete
  `results_mt174p2` grid). Note: `dofit_indep.py` consequently **can no longer run on the default grid**.
- *Fix:* in `_fit_morph`, if `len(vps) < 6` (or any required `(ch,vp)` missing), raise a clear error
  naming the missing channel/varpoints and the rank-6 requirement before calling `lstsq`.
- *Cite:* `framework/process/ww/indep/generator_mocanlo.py:207-208, 227-229`.

**m7 — `delta_nnlo_relative` docstring overstates exact convention-independence** (AM-4, confirmed)
- *What's wrong:* the docstring asserts anchor/BR/multiplicity all "cancel" so δ_NNLO is convention-
  independent. Multiplicity (the 27/4 factor) does cancel exactly, but the **WHIZARD anchor and BR
  correction do NOT** — the anchor multiplies only the Born base, not the additive NLO/NNLO blocks, so
  leaving them on shifts the ratio by ~0.9–1.2% (predicted 1/f−1 matches observed anchor-only shift
  exactly). The code's choice to **strip** them is the cleaner option; this is a doc inaccuracy, not a bug.
- *Impact:* ≈0.016 MeV (1% of the +1.6 MeV NNLO pull) — well below the 0.1 MeV threshold; the production
  fit carries zero bias because the stripped value is used.
- *Fix:* soften to "anchor/BR are stripped to remove a ~1% √ŝ-dependent contamination; the stripped ratio
  is the convention-clean definition."
- *Cite:* `framework/process/ww/indep/match_bfs.py:115-126` (docstring); mechanism at `xsec_calculator/bfs_eft.py:753-768`.

**m8 — 6e-10 NLL closure floor is the m_e constant mismatch, not an NLL truncation/soft-branch residual** (nll-port-4, confirmed)
- *What's wrong:* the report/memory present the 6e-10 port-vs-reference floor as a closure residual. It is
  **entirely** the electron-mass constant difference `isr_beta.M_E=0.51099895069e-3` vs
  `eft_xsec.M_E=0.5109989461e-3` (8th sig fig, rel 9e-9 → 7.4e-10 in β → ~6.7e-10 in σ). Forcing a common
  m_e collapses closure to machine precision (~1e-16).
- *Impact:* sub-keV on m_W; provenance/doc hygiene only.
- *Fix:* attribute the 6e-10 to the m_e constant in §sec:match and memory; optionally align the two
  constants' PDG digits (kept separate is fine for the no-shared-code independence goal).
- *Cite:* `isr_beta.py:83` vs `eft_xsec.py:50`.

**m9 — Report wording: δ_QCD "channel-weights ≈⟨n_had⟩×4%", tab:indep-br cross-ref, NLL-shift range** (RA-2 / RA-3 / RA-4, all confirmed)
- *RA-2:* "+5.43% summed over the pure-WW channel weights (≈⟨n_had⟩×4.0%)" is misleading — the literal
  multiplicities (12/4/9) give only +3.24% (⟨n_had⟩=0.80); the +5.43% requires **σ-weighting**
  (colour-N_c-boosted hadronic shares → σ-weighted ⟨n_had⟩≈1.33). The code σ-weights correctly; only the
  explanatory text is wrong about *why*. Reword to make the σ-weighting / colour boost explicit.
  (`report:2950-2951`.)
- *RA-3:* §match-pull (`:3038-3040`) cites tab:indep-br sensitivities as σ(m_W)=0.44 pdg / 0.90 off-shell,
  but the table actually prints **0.42 / 0.83**. The 0.44/0.90 are the match-pull values on the interim
  m_t=174.2 grid, not the table's (different grid; `dofit_indep.py` no longer runs on the default grid —
  see m6). Reconcile by labeling the grid, or regenerate tab:indep-br on the 174.2 grid.
- *RA-4:* the NLL ISR shape is quoted two ways: "+0.26%→+0.59%" (text `:2998`, raw) vs "~+0.3–0.45%"
  (figure `:3111` / §match-pull `:3098`, savgol-smoothed). Pick one convention; note the 157 GeV dip to ~0
  if quoting the smoothed curve.
- *Impact:* none on m_W/Γ_W; presentation/consistency only.

---

## 3. Refuted / corrected claims (checked, do not act on)

- **"C₁ LL-vs-NLL treatment → ~2 MeV of the −3.7 NLL pull" — REFUTED** (ISR-4, refuted; downgraded to
  minor). The first-draft alarm projected an α-toggle proxy that **overstates the genuine ambiguity by
  ~20×** (it rescales the entire O(α) subtraction by 3.2%, whereas the real missing piece is the NLL part
  of the O(α) coefficient ~(α/π)·λ₁/4 ≈ 0.165% of the O(α) ISR, nearly flat) **and** fixed Γ_W in the
  projection. A direct Γ_W-floated cross-fit gives the tilt absorbed by Γ_W; the defensible m_W impact is
  **~0.05–0.1 MeV**, consistent with the report's existing ≲0.1–0.3% estimate. Report TODO 2 (O(α)-exact
  NLL C₁) is a real but **<0.1 MeV** refinement, not a few-MeV effect. The −3.7 MeV NLL pull is a genuine
  radiator effect, not a subtraction artefact.
- **δ_QCD double-count worry — RESOLVED in favour of the code** (dqcd-1, the one substantive open physics
  item). The dqcd reviewer read MoCaNLO source: Γ_W=2.085 GeV is an **external physical (QCD-corrected)**
  input — `standard_model.F90:7571-7586` only does the OS→pole conversion and **does not** recompute the
  total width from partials — while W→qq̄ runs at `loop_qcd_order=0` (EW-only partial). So MoCaNLO's
  effective BR_had = Γ_had^EW/Γ_W^phys is too small by exactly δ_QCD, and ×δ_QCD^n_had restores it with
  **no double-count**. The +5.43% σ-weighted ⟨δ_QCD^n_had⟩ was reproduced from the actual grids. Closes
  report TODO 3.
- **"factor-2 in the eMELA soft-limit norm_nll vs isr.py" (P3) — DISPROVEN** (nll-port-3). isr.py's β is
  the combined 2-leg β with `kappa=β/2`; isr_beta's `be` is the per-leg β = β_total/2. They are
  numerically equal (ratio 1−7e-10), so the NLL exponents and LL norm prefactors are algebraically
  identical — a per-leg-vs-combined convention, not a factor of 2.
- **"α mismatch in C₁ when isr_nll=True" (P1) — RETRACTED** (ISR-3). C₁ must use the **radiator's** α
  (α(M_Z) in the NLL path), not MoCaNLO's α_Gμ: C₁ cancels the resummation's own O(α) double-count, and
  `_isr_cfg()` feeds one cfg to both `convolve_2leg` and `oalpha_isr_subtraction`. MoCaNLO's genuine α_Gμ
  fixed-order NLO survives untouched. The α_Gμ-vs-α(M_Z) split is an O(α²) scheme feature, not a bug.

---

## 4. Confirmed-correct strong points

- **Double-count protection is structural (1e-12).** δ_NNLO = (σ_NNLO−σ_NLO)/σ_Born cancels ALL NLO incl.
  the full O(α) Coulomb eq.62 (term1+term2), because `_add_nnlo_to_LR` only adds the 5 NNLO pieces on top
  of the unchanged NLO loops. Independent of the chain's `include_coulomb=False` default. (DC-1, DC-5.)
- **Every NNLO piece is genuinely α-suppressed** vs the NLO single-Coulomb MoCaNLO already has: NLO-C is
  4.103·α·C1 (coeff 0.031), C×res is width-suppressed (Γ_W/M_W), none is a hidden bare O(α) Coulomb. (DC-2/3/6.)
- **Relative-K transplant error ≤2.5e-6** — MoCaNLO-vs-BFS Born shape mismatch ≤0.6%, well inside the
  report's 6e-3% slope bound. (DC-4.)
- **O(α) ISR matching is exact:** matched−Born scales as α² not α; endpoint substitution u=(1−x)^β_e is
  exact; the finite-cutoff (1−x_min)^β_e artefact cancels between convolve_2leg and C₁; `_radiator_norm`
  O(α)=1+3β_s/4 with γ_E cancellation matches C₁'s 0.75·β_s. (ISR-1/2.)
- **δ_QCD placement is order-independent and factorizes correctly:** flat per-channel factor commutes with
  the linear ISR convolution and C₁ (verified to 3e-15); also multiplies the δ_NNLO·Born term correctly
  (δ_NNLO is EW-only, all 3 BFS calls pass `apply_delta_QCD=False`). Needed identically in both BR
  conventions. (dqcd-3/4.)
- **Assembly/morph/units are unbiased:** PURE_WW and 6-block weights both → 81 sum-rule = (2N_c+3)²;
  quad+bilinear morph (basis full-rank, cond≈1820) captures the K-factor m_W slope to ~7e-9; self-pull =
  0.00 MeV; `_br_factor`=(gW/GW0)²(MW0/mW)⁶ is the exact tree BR² strip (≡1 off-shell, the pull's mode);
  fb→pb clean; sub-threshold negative-Born neutralised; 240 GeV last_ecm zero never enters the likelihood.
  (AM-1/2/3/6/7.)
- **Code hygiene solid:** `_grid_key` content-hash retires the old id() cache bug; `nlo_eff` default-arg
  binding correct (no late-binding-loop bug); fork-pool morph bit-identical to serial; spline denoising
  has no m_W-slope bias (0.18%); numerical guards never silently produce NaN/0. (CQ-2..6.)
- **NLL scheme + port correct:** α(M_Z)/ALPMZ/DELTA matches BFS production; `_cfg_fingerprint` captures
  every radiator knob; the port reproduces BFS eMELA-NLL to machine precision (the 6e-10 floor is the m_e
  constant). (nll-port-5/6, RA-5.) Every tab:match-pull entry reproduces to ~0.05 MeV; equations match
  code; WIP/m_t=174.2 caveats present — no over-claim of production readiness. (RA-6.)

---

## 5. Prioritized next actions

1. **Fix M1 (cache key) + add M2 (NLL test).** Both cheap, both MAJOR-latent. Fold a generator-state
   fingerprint into the cache keys (or freeze the dataclass) + a toggle self-check; add an `nll=True`
   assertion to `test_isr_beta.py` and fix the `isr_beta.py` `__main__` docstring. Optionally commit the
   port-validation snippet so the 6e-10 claim is reproducible in-repo (RA-5).
2. **Couple LL `isr_cfg.alpha` to `scheme_alpha` (m5 / ISR-5) before the EW-scheme grids land** — guards
   the dominant ~36 MeV theory systematic from silent α mixing the moment alphaz/alpha0 grids are fitted.
3. **Decide on m4 (soft-endpoint cutoff):** either lower the cutoff so eMELA's asymptotics run, or document
   the ~0.4% norm / ~0.035% slope as an ISR-soft-limit systematic in §sec:match (and fix the single
   "negligible" claim at `isr.py:441`). Sub-MeV/low-MeV, shared with the BFS-side isr.py.
4. **Add the AM-8 grid guard (m6)** and fix the report polish items: M3/RA-1 (label partonic vs observed
   δ_NNLO), RA-2/3/4 wording (σ-weighting, tab:indep-br cross-ref, NLL-shift range), AM-4 docstring (m7),
   and the 6e-10 m_e-floor attribution (m8). Close report TODO 3 with the dqcd-1 finding; note report
   TODO 2 (O(α)-exact C₁) is a **<0.1 MeV** refinement, not few-MeV (ISR-4 refutation).
5. **Then redo the headline pull on the m_t=172.5 correlated-seed grid (report TODO 1), with δ_QCD in the
   baseline templates, and drop the WIP banner** once items 1–4 are closed.
