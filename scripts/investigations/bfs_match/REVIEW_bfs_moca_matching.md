# Review: BFS-on-MoCaNLO matching layer (incl. ISR) — code + physics

Status: LIVING DOCUMENT (checkpoint while review workflow runs). Reviewer: Claude (Opus 4.8).
Scope: commits 080492c / 74993f0 / 48a453b. Files:
- `framework/process/ww/indep/match_bfs.py`  (δ_NNLO relative K-factor + δ_QCD per-channel)
- `framework/process/ww/indep/generator_mocanlo.py` (wiring: σ̂_comb, morph, BR factor)
- `framework/process/ww/indep/isr_beta.py` (LL+exp radiator, O(α) matching subtraction, eMELA NLL port)
- supporting: channels.py, partonic_grid.py, varpoints.py, grid.py
- BFS source: xsec_calculator/bfs_eft.py (sigma_BFS_LO_total_WW_pb, delta_QCD_factor, _add_nnlo_to_LR, _add_nlo_loops_to_LR)
- scripts: matched_pull.py, plot_match_lineshape.py
- report: ww_bfs_implementation.tex §sec:match (2846–3132)

## VERDICT (preliminary): construction is physically sound; double-counting protection is structural and correct. A handful of concrete items to verify/fix, none fatal. The layer is correctly DEFAULT-OFF and labelled exploratory/WIP.

---

## Confirmed-correct (physics)
1. **δ_NNLO = (σ_NNLO − σ_NLO)/σ_Born is purely the NNLO block.** Both σ_NLO and σ_NNLO in the
   numerator carry the SAME full BFS NLO (incl. the O(α) Coulomb, which `_add_nlo_loops_to_LR`
   adds with `subleading_only=coulomb_kc_safe`, default False → full NLO Coulomb). The NLO loops
   cancel in the difference, so δ_NNLO contains NO O(α) piece. Protection vs double-counting
   MoCaNLO's NLO is STRUCTURAL, not numerical. ✓
2. **`_add_nnlo_to_LR` terms are all ≥O(α²):** C×[S+H], NLO-C (=NLO correction to Coulomb
   potential, O(α²/v)), C×decay, C×res, C3. None re-introduces an O(α) Coulomb that MoCaNLO
   already has. (VERIFY each term's order — see workflow.)
3. **Additive matching σ̂_comb = σ̂_NLO + δ_NNLO·σ̂_Born is the standard top-threshold prescription**
   (complete fixed-order + higher-order threshold remainder with lower-order subtracted; here the
   subtraction is automatic because δ_NNLO's numerator cancels the NLO).
4. **C₁ asymmetry is correct:** the O(α) ISR matching subtraction is keyed to MoCaNLO's Born ONLY
   (σ̂_NLO carries the explicit O(α) ISR log from `pdf_set=none`; the BFS δ_NNLO·Born term is
   ISR-naked, so it must ride the resummation unsubtracted). Implemented via `nlo_eff = nlo + δ·born`
   convolved, minus `C₁[born]`. ✓
5. **Endpoint substitution u=(1−x)^β_e is exact:** jac dx/du = u^(1/β_e−1)/β_e = jac_NS; the singular
   β_e(1−x)^(β_e−1) maps to a UNIFORM measure in u (= norm N), so it's added directly (not ×jac_NS).
   Verified algebraically. ✓
6. **O(α) subtraction self-consistency:** `oalpha_isr_subtraction` includes the
   `β_e·σ̂(s)·ln(1−x_min)` endpoint-norm term that exactly cancels convolve_2leg's
   (1−x_min)^β_e = 1+β_e ln(1−x_min) finite-cutoff artifact. The γ_E/Γ(1+β_e) terms in `_radiator_norm`
   are O(α²) (cancel at O(α): exp(−γ_E β_e)/Γ(1+β_e) ≈ 1), so the O(α) norm is just 1+3β_s/4,
   matching the subtraction's `0.75·β_s·σ̂(s)`. Same x_min/n_quad used on both sides → cancellation
   holds. ✓
7. **δ_QCD placement (flat factor, post-convolution) is order-independent:** a √ŝ-independent factor
   commutes with the linear ISR convolution and the C₁ subtraction, so pre/post placement is identical.
   Per-channel δ_QCD^n_had (n_had∈{0,1,2}) is MORE correct than BFS's single global factor. ✓
8. **Relative-K-factor transplant error is bounded:** δ_NNLO defined vs BFS Born, applied to MoCaNLO
   Born. Error ~ (shape diff)×δ_NNLO ~ few% × 0.2% ~ 4e-3 %. Report's "NLO premise" checks
   (<0.1% shape @160.5-162; −3.2% slope) bound it. ✓
9. **Channel sum-rule** (channels.check_sum_rule): Σ weight·colour = 81 = (2N_c+3)². ✓
10. **pdg-constant `_br_factor` = (gW/GW0)²(MW0/mW)⁶** matches BR² ∝ m_W⁶/Γ_W² (Γ_partial ∝ m_W³).
    Freezes Γ_W to line-shape-only WITHOUT discarding the m_W rate handle. (VERIFY the m_W⁶ scaling
    numerically vs MoCaNLO's actual off-shell BR² dependence.)
11. **Closure / sanity:** self-fit pull = 0.00 MeV (validates machinery); δ_NNLO shape pull +1.6 MeV
    lands on the independent theory-ladder NNLO step ~2.3 MeV; NLL −3.7 MeV consistent w/ ladder
    α-scheme scale. NLL port reproduces BFS-side eMELA-NLL to 6e-10 on identical σ̂.

## ITEMS TO VERIFY / POSSIBLE ISSUES (for the workflow)

### P1 — α mismatch in C₁ when isr_nll=True  [MEDIUM, concrete]
When `isr_nll=True`, `_isr_cfg()` rebuilds the WHOLE cfg with `alpha=ALPHA_MZ` (α(M_Z)). This cfg is
also used by `oalpha_isr_subtraction` (C₁). But C₁ must remove exactly the O(α) ISR that MoCaNLO
ALREADY put into σ̂_NLO — and the MoCaNLO grids were generated with `scheme_alpha="gf"` → α_Gμ.
So C₁ uses α(M_Z)=1/128.94 while MoCaNLO used α_Gμ=1/132.4 → ~3.2% of the O(α) ISR subtraction is
mismatched → a residual ~0.1% line-shape effect that may contaminate the −3.7 MeV NLL pull.
The report acknowledges a generic "residual O(α) NLL/DELTA-scheme constant" but does NOT name this
specific α_Gμ-vs-α(M_Z) subtraction-α inconsistency. → fix (use MoCaNLO's α in C₁) or document explicitly.

### P2 — δ_QCD vs MoCaNLO width normalization (double-count?)  [MEDIUM, = report TODO 3]
δ_QCD^n_had ASSUMES MoCaNLO's hadronic partial width (numerator of BR) is QCD-order-0 (EW only) while
the input Γ_W=2.085 (denominator) is the PHYSICAL width (QCD-corrected). Then ×δ_QCD restores the
missing QCD in the numerator → correct, NOT double-count. BUT if MoCaNLO's input Γ_W is EW-only, or if
it internally re-derives Γ_had with QCD, ×δ_QCD over-corrects. Must trace MoCaNLO's actual width/BR
handling (mocanlo_cards.py + the run cards). Same in BOTH BR conventions because `_br_factor` only
rescales the m_W/Γ_W SCALING, not the absolute (still-EW-only) BR magnitude.

### P3 — eMELA soft-limit norm_nll: factor-2 vs isr.py + possibly dead code  [LOW-MED, concrete]
`_per_leg_emela_nll`: `norm_nll = norm·exp(be·(α/π)·λ₁/4)`. isr.py line 372 uses `kappa·(α/π)·λ₁/4`
with `kappa=β/2`. If isr.py's β is the same per-leg β_e, isr_beta's exponent is 2× isr.py's. This
branch only fires when `omx<1e-15` (GL nodes never land exactly at the endpoint) → likely UNREACHED,
which would explain why the 6e-10 validation passes regardless. If so: (a) confirm consistency with
isr.py, (b) note the validation does NOT exercise this branch (latent risk if it ever becomes reachable).

### P4 — mutable dataclass + per-instance cache staleness  [LOW, latent footgun]
`WWGeneratorMoCaNLO` is a non-frozen dataclass; `_cache` keyed by (varpoint, grid_key) does NOT include
the matching flags (match_bfs/isr_nll/alpha_s/sm). Safe in current scripts (fresh _gen() per variant),
but mutating a flag on a used instance → stale cache. Either freeze, or fold the flags into the cache key.

### P5 — 240 GeV last_ecm point is always 0 in the indep generator  [LOW / pre-existing]
`_build_fine_grid` appends 240; `inside=(ecm>=156)&(<=164)` zeroes it. If any scenario uses last_ecm=240
as a normalization anchor, the indep template is 0 there. Pre-existing indep-calc design; confirm the
matched-pull scenario doesn't rely on it.

### P6 — δ_QCD also multiplies the δ_NNLO term  [INFO, believed correct]
Code: obs = convolve(nlo + δ_NNLO·born) − C₁; then obs ×= δ_QCD. So δ_QCD multiplies the δ_NNLO
contribution too. Physically correct (production Coulomb × hadronic-decay QCD factorize), and δ_NNLO is
EW-only so no QCD double-count. Confirm.

### P7 — Report number cross-checks  [INFO]
- δ_QCD per channel 1.000/1.040/1.082 ✓ (delta_QCD_factor(0.1199)=1.0402, ^2=1.082).
- "+5.4%" flat requires σ×multiplicity weighting (qqqq has largest σ, gets δ_QCD²);
  ⟨δ_QCD^n_had⟩ with BR_had≈0.674 → +5.5%. Report wording "summed over channel weights" understates
  that σ-weighting is needed; the figure script computes the true σ-weighted ratio → +5.4% OK.
- Fig caption says δ_NNLO 0→+0.12% while text/Table say 0→+0.21% @163 — CHECK consistency (157-vs-163
  range and which curve). Possible minor inconsistency.

---

## INTERIM CONCLUSIONS (2026-06-03, token-budget cutoff; verification workflow wf_b0de4149-ce8 still running)

**P1 RETRACTED — was reviewer error.** On re-derivation: C₁ MUST use the *radiator's* α (α_MZ when
isr_nll=True), NOT MoCaNLO's α_Gμ. C₁'s job is to cancel the *resummation's own* O(α) double-count so
that MoCaNLO's exact fixed-order NLO (which legitimately uses α_Gμ throughout) survives untouched:
  σ_obs = ∫∫D D σ̂_NLO − C₁[σ̂_Born] = σ̂_NLO + [resummation − its own O(α)].
The O(α) piece of the resummed convolution is in the radiator's α, so C₁ must match that. The
α_Gμ-vs-α_MZ difference surfaces only at O(α²) and is exactly the scheme ambiguity the report already
flags as the "residual O(α) NLL/DELTA-scheme constant". The ONLY genuine approximation is that C₁ keeps
its analytic *LL* functional form rather than eMELA's NLL O(α) — already acknowledged (report TODO 2).

**P3 REFUTED.** isr.py's β is the 2-leg total; kappa=β/2 = isr_beta's per-leg β_e (cf. memory note
"isr_beta β_e = isr.py β/2"), so the soft-limit exponents agree — no factor-2. Remaining sub-point is
latent only: the omx<1e-15 endpoint branch is essentially never hit by the GL nodes, so the 6e-10 port
validation does not exercise norm_nll (fine today; flag if that branch ever becomes reachable).

**Net standing issues after self-verification:**
- P2 [MED, ~75% conf] — δ_QCD vs MoCaNLO width/BR bookkeeping (= report TODO 3). NOT yet settled; needs
  a read of mocanlo_cards.py + the run-card width handling. This is the one physics item that could, in
  principle, mean a real ~%-level rate mis-normalization (→ large via the rate handle, cf. the −106 MeV
  cov-lumi pull). Highest-value next check.
- P4 [MINOR] — non-frozen dataclass + match-flags absent from the generator `_cache` key. Footgun only.
- P7 [MINOR/doc] — δ_NNLO "0→+0.12%" (fig caption) vs "0→+0.21%" (text/table). Reconcile.
- P5/P6 — info only (240 GeV zero point pre-existing; δ_QCD×δ_NNLO is correct EW×QCD factorization).

**Verdict:** the BFS-on-MoCaNLO matching is a correct, well-constructed, conservatively-scoped
(default-OFF, WIP-bannered) cross-combination. No correctness defect found that would bias the headline
shape-only pull (δ_NNLO +1.6 / NLL −3.7 MeV). The single substantive open physics question is the δ_QCD
width/BR trace; everything else is doc polish or latent-only.

### Recovery paths for the in-flight verification workflow (wf_b0de4149-ce8)
If this session ends before the workflow finishes, its per-agent outputs persist on disk and can be
recovered or re-run from a NEW session:
- Script (re-runnable as a fresh workflow): `…/9923c2fd-…/workflows/scripts/review-bfs-moca-matching-wf_b0de4149-ce8.js`
- Transcript dir (per-agent jsonl, written as each agent completes):
  `…/9923c2fd-…/subagents/workflows/wf_b0de4149-ce8/`
  (both under ~/.claude/projects/-afs-…-WW-threshold/)
- Same-session resume: Workflow({scriptPath, resumeFromRunId:"wf_b0de4149-ce8"}) — cached agents return instantly.
- New-session: just re-invoke the script (Workflow scriptPath) — it re-runs from scratch, deterministic.
The analyst review ABOVE is complete and standalone; the workflow only adds independent cross-checks.
## FINAL — workflow wf_b0de4149-ce8 complete (7 dims × adversarial verify, 20 agents, 1.18M tok).
(Synthesis agent died on the 7pm session limit; consolidated by hand from the per-agent jsonl.)

### Biggest results
- **P2 (δ_QCD double-count) RESOLVED — CORRECT.** dqcd reviewer read MoCaNLO source: Γ_W=2.085 is an
  EXTERNAL physical (QCD-corrected) input (standard_model.F90:7571-7586 only does OS→pole, does NOT
  recompute the total width), while W→qq̄ runs at loop_qcd_order=0 (EW-only partial). So MoCaNLO's BR =
  Γ_had^EW/Γ_W^phys is too small by exactly δ_QCD, and ×δ_QCD^n_had restores it with NO double-count.
  +5.43% σ-weighted ⟨δ_QCD^n_had⟩ reproduced from the actual grids. This was the one substantive open
  physics item → settled in favour of the implementation.
- **Double-counting protection verified to 1e-12** (δ_NNLO cancels ALL NLO incl. the O(α) Coulomb eq.62);
  every NNLO piece is α-suppressed vs the NLO single-Coulomb (NLO-C conversion coeff 0.031, NOT hidden
  O(α)); relative-K transplant error ≤2.5e-6 (< the report's 6e-3% bound).
- **O(α) ISR matching cancellation verified exact** (matched−Born scales as α², not α). My P1/P3 retractions
  both independently confirmed (C₁@α_MZ is correct; no factor-2).

### CONFIRMED ISSUES (ranked; fix before dropping the WIP banner / going to production)
1. **[MAJOR] CQ-1 (=my P4) generator `_cache` omits all matching flags + non-frozen dataclass.** Reviewer
   REPRODUCED a silent stale hit: mutating `match_bfs` on a used instance returns the OLD unmatched line
   shape. Keys are `(varpoint,_grid_key)` / `("coeffs",_grid_key)` — missing match_bfs/_nnlo/_dqcd/isr_nll/
   alpha_s/sm/br_convention/scheme_alpha/lepton_cut. Safe in current scripts (fresh _gen() per variant).
   FIX: fold a generator-state fingerprint into the cache keys (or freeze the dataclass).
2. **[MAJOR] nll-port-1 no automated test exercises the NLL path.** isr_beta.py:18-19 docstring claims a
   cross-check "in tests/__main__" but there is NO `__main__` and test_isr_beta.py only tests LL. The
   6e-10 figure is real but only ever reproduced ad-hoc. isr_nll IS the production NLL knob → latent: a
   future change to _per_leg_emela_nll/emela_wrapper/LAMBDA1_NF0 breaks NLL silently. FIX: add an nll=True
   assertion (convolve_2leg vs bfs_isr.sigma_ISR_2leg_convolution on identical σ̂) to test_isr_beta.py.
3. **[MAJOR/partial] nll-port-2 the soft-endpoint cutoff omx<1e-15 is NOT dead code.** With β_e≈0.058
   (1/β_e≈17), ~30 of 128 GL nodes fall below 1e-15 and carry ~13% of the per-leg weight; there the code
   substitutes the flat analytic norm_nll instead of eMELA's code_pdf → ~0.5% normalisation shift with a
   ~0.08% ENERGY SLOPE (the m_W-relevant part; flat part cancels in a line-shape fit) → low-MeV potential.
   SHARED with the BFS-side isr.py reference (so it does NOT break the 6e-10 port-vs-reference closure, but
   it is a genuine NLL systematic on BOTH chains). FIX: push the cutoff far lower (e.g. 1e-300 with x
   floored so code_pdf stays callable) and let eMELA's own asymptotics run, or document it as an NLL syst.
   [This was my P3's "unreached branch" hypothesis — DISPROVEN: it is heavily reached.]
4. **[MAJOR/doc] RA-1 (=my P7) δ_NNLO@163 = +0.21% (text/table) vs +0.12% (figure).** Same symbol, two
   different objects: +0.21% is the PARTONIC K-factor (eq:match-dnnlo); +0.12% is the ISR-CONVOLVED
   observed line-shape ratio (diluted by the radiator). No m_W impact; misleads the reader. FIX: label them.
5. **[MINOR] ISR-5 latent α-scheme footgun.** The LL `isr_cfg.alpha` defaults to α_Gμ and is NOT coupled
   to the generator's `scheme_alpha`. Running an α(M_Z)/α(0) MoCaNLO grid (the EW-scheme-variation grids
   expected to land) with the default LL ISR would silently mix α between σ̂_NLO's ISR log and C₁. Default
   gf run is fully consistent. FIX: derive isr_cfg.alpha from scheme_alpha (or assert they match) before
   those grids are used.
6. **[MINOR] AM-8** incomplete σ̂ grid → cryptic `LinAlgError` (the DEFAULT_RESULTS_DIR has lnuqq+mutau but
   no qqqq). Add a clear "<6 varpoints / missing channel" guard in _fit_morph.
7. **[MINOR] AM-4** delta_nnlo_relative docstring overstates EXACT convention-independence: anchor+BR are
   √ŝ-dependent, leave a ~1% residual in the ratio (impact ≈0.016 MeV). Soften the wording.
8. **[MINOR] nll-port-4** the 6e-10 closure floor is the m_e constant mismatch (isr_beta.M_E
   0.51099895069e-3 vs eft_xsec.M_E 0.5109989461e-3, 8th sig fig), NOT an intrinsic NLL residual —
   attribute it correctly in report/memory. (Sub-keV on m_W.)
9. **[MINOR] RA-2/3/4** report wording: "+5.43% summed over channel weights ≈⟨n_had⟩×4%" needs σ-weighting
   language (colour-boosted ⟨n_had⟩≈1.33, not the multiplicity 0.8); §match-pull cites tab:indep-br as
   0.44/0.90 but the table prints 0.42/0.83 (interim-grid difference — reconcile); NLL shift quoted as both
   +0.26→0.59% and +0.3–0.45% (pick one).

### REFUTED / corrected by adversarial verification
- **ISR-4 REFUTED:** the reviewer's alarming "C₁ LL-vs-NLL treatment → ~2 MeV of the −3.7 NLL pull" was
  checked by toggling C₁'s α at fixed NLL conv → actual impact **~0.05–0.1 MeV**. So report TODO 2
  (O(α)-exact NLL) is a real but <0.1 MeV refinement, NOT a few-MeV effect. Headline −3.7 MeV NLL pull is
  a genuine radiator effect, not a subtraction artefact.
- My **P1** (α mismatch in C₁) and **P3** (factor-2) — both refuted (recorded above).

### Confirmed-correct strong points (survived scrutiny)
Double-count protection structural (1e-12); δ_QCD bookkeeping correct; O(α) matching exact (α² residual);
morph captures the K-factor m_W slope (self-pull 0.00, basis full-rank cond≈1820); _br_factor m_W⁶/Γ_W²
exact (off-shell≡1, the pull's mode); units fb→pb clean; weights→81 sum-rule; sub-threshold negative-Born
neutralised; NLL scheme (α(MZ)/ALPMZ/DELTA) + cache fingerprint complete; spline denoising no m_W-slope
bias (0.18%); fork-pool bit-identical to serial; every tab:match-pull entry reproduces to ~0.05 MeV.

### Top next actions
(a) Fix CQ-1 cache key + add the nll-port-1 NLL test (both cheap, both MAJOR-latent).
(b) Quantify/clip nll-port-2 soft-endpoint cutoff — the only finding with low-MeV physics potential, and it
    touches the BFS-side isr.py too. (c) Fix the ISR-5 α coupling before the EW-scheme grids are used.
(d) Report polish RA-1..4 + AM-4 docstring. (e) Then redo the pull on the 172.5 grid + drop the WIP banner.

---
## FIXES APPLIED (2026-06-03, same session, verified)
Scope: ONLY framework/process/ww/indep/ + report + test (production BFS chain untouched; default path byte-identical).
- **CQ-1** generator_mocanlo.py: added `_state_key()` (match flags + alpha_s + sm + scheme_alpha + lepton_cut +
  smooth + isr cfg fingerprint) folded into both cache keys → reused-instance toggle now recomputes
  (verified: False→True gives +5.6%, was 0/stale; fresh-matched==reused-toggled; default byte-identical).
- **ISR-5** generator_mocanlo.py `_isr_cfg()`: LL ISR α now couples to scheme_alpha (gf→α_Gμ, alphaz→α_MZ,
  alpha0→α_0; explicit non-default α respected; unknown scheme left untouched). Verified.
- **AM-8** generator_mocanlo.py `_fit_morph`: clear ValueError naming missing channels when <6 varpoints
  (verified vs the lnuqq+mutau-only default dir).
- **nll-port-1** test_isr_beta.py: added check [5] (eMELA NLL convolve_2leg vs BFS isr nll=True on identical
  σ̂) + assertions on [2]/[5]; both pass (LL 4.6e-15, NLL 3.4e-15 with common m_e → confirms nll-port-4:
  the 6e-10 is purely the m_e-constant 8th-sig-fig difference).
- **nll-port-2** isr_beta.py `_per_leg_emela_nll`: documented the soft-endpoint (omx<1e-15) systematic as a
  known low-MeV NLL uncertainty (NOT changed — shared with production isr.py; physics decision). + report
  sec:match-todo new item.
- **nll-port-5 / AM-4** clarifying docstrings (jac is the universal dx/du; δ_NNLO anchor/BR stripped not
  exactly cancelling, ~0.02 MeV).
- **Report** RA-1 (δ_NNLO +0.21% partonic K-factor vs +0.12% observed ISR-convolved — verified: 0.214% /
  0.124% @163), RA-2 (σ-weighting language), RA-3 (0.44/0.90 interim vs 0.42/0.83 production cross-ref),
  RA-4 (NLL range), nll-port-4 (6e-10 = m_e const), δ_QCD bookkeeping marked RESOLVED, O(α)-NLL bounded
  <0.1 MeV. Compiles 67pp clean; **republished to EOS**.
DEFERRED (needs user/physics decision, documented not changed): nll-port-2 numeric cutoff change (touches
production isr.py); the headline pull regen on the 172.5 grid (grids still landing).

---
## FOLLOW-UP (2026-06-03): nll-port-2 quantified + eMELA bottleneck + disk cache
- **nll-port-2 RESOLVED.** Soft-endpoint cutoff (omx<1e-15, ~13% of per-leg weight): code_pdf vs the
  exact analytic soft limit norm_nll — agree ~0.2% at omx≈1e-7..1e-9 (eMELA's valid regime) then code_pdf
  DIVERGES to +6% at omx≈1e-66 (where x underflows to 1.0). So norm_nll is the trusted endpoint value and
  the cutoff SHIELDS the result from eMELA's x→1 artifact. Lowering it imports +0.53% on the line shape →
  shape-only m_W bias only −0.11 MeV (lumi-wtd) and WRONG sign. Verdict: **cutoff is correct, do NOT lower
  it**; the ~0.1 MeV is an error AVOIDED, not a systematic. Report sec:match-todo + isr_beta docstring updated.
- **eMELA bottleneck (Q3) MEASURED.** initialize = 14ms once (idempotent no-op after). code_pdf = **~22 ms/call**
  (re-evolves DGLAP per query; same cost fixed/varying Q → no internal caching). A morph build ≈ 19k calls
  ≈ 400 s — that is essentially the whole NLL build time.
- **Speed (Q2).** analytic NLL ≈ µs/eval (×10³–10⁴ faster than raw eMELA), but ≈ a precomputed-grid eMELA.
  Speed alone doesn't favor analytic; analytic's wins are endpoint-exactness + no eMELA dependency.
- **SHIPPED: disk-persistent radiator cache** (isr_beta.py). per-leg setup (σ̂-independent) persisted by
  (√s-grid, cfg) under $WW_ISR_RADIATOR_CACHE (default ~/.cache/ww_isr_radiator); NLL-only, atomic writes
  (fork/condor-safe), versioned (_RADIATOR_DISK_VERSION). Verified: cold 7.0s → warm (fresh process) 0.001s,
  byte-identical (same sha1); LL path writes nothing (production default untouched); test_isr_beta passes.
  → the ~400s NLL build is now once-ever; future fits/cross-fits/172.5-pull load in ms.
- Analytic-NLL literature research (Q1): deep-research workflow wf_cd6722d4-04e running (verdict pending).
