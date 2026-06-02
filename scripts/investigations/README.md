# Investigations

Ad-hoc diagnostic scripts kept around because the conclusions they support
are non-trivial. Each script is self-contained — `PYTHONPATH=<repo-root>
python3 scripts/investigations/<topic>/<script>.py`.

Production validation lives in `scripts/validate_bfs_nlo.py`; these are
the **off-tree** explorations that informed the project memories.

## `gammaw_crossing/` — does dσ_WW/dΓ_W cross zero in the scan window?

Triggered by [project_followup_gammaw_crossing.md]. Verdict: yes,
Whizard 4f exact (BFS Tables 1+2) confirms a crossing between 161 and
164 GeV, consistent with Azzurri 2107.04444 Fig. 1. Our BFS-EFT
N^(3/2)LO Born **misses it** — wrong sign at √s ≥ 164 GeV.

| script | what it does |
|---|---|
| `audit_gw.py` | dσ/dΓ_W under every chain toggle (anchor / NLO loops / δ_QCD / K_C / BR convention) |
| `audit_gw_2.py` | absolute σ_WW and dσ/dΓ_W partonic + observed (with ISR) at Azzurri energies |
| `audit_gw_3.py` | dσ/dΓ_W by successive BFS Born order (LO → N^(3/2)LO) — isolates which component carries Γ_W |
| `apples.py` | apples-to-apples: BORN-only chain vs Whizard 4f Born from BFS Tables 1+2 |
| `check_anchor.py` | verify anchor reproduces Whizard at the two reference Γ_W points |
| `check_dsigmadGW.py` | dσ/dΓ_W fine-grid scan around the 168-GeV bump |
| `scan_window_gw.py` | restrict to [157, 163] GeV — confirms no crossing in the FCC-ee scan |
| `single_res_test.py` | does dropping σ^(1/2) single-resonant content shift the crossing? (no) |
| `whizard_ref_gw.py` | extract Whizard 4f dσ/dΓ_W from BFS Tables 1+2 directly |

## `born_bias/` — Born-side m_W bias from BR-correction-induced residuals

Triggered by [project_followup_born_side_mW_bias.md]. Verdict:
sub-percent BFS-internal bookkeeping subtlety in how (Γ_W^(0)/Γ_W)²
interacts with σ^(1/2); ~0.3 MeV (FCC weights) / ~1 MeV (conservative
2-point) bias on m_W. Five BR-convention / propagator-width
hypotheses tested, none uniformly closes the residual.

| script | what it does |
|---|---|
| `decompose.py` | per-component decomposition of the Scenario B residual |
| `back_out.py` | back out the implied BR factor on σ^(1/2) at each √s |
| `back_out_both.py` | back out (r_sq, r_lin) simultaneously |
| `propagator_test.py` | test propagator Γ_W^(0) vs Γ_W^phys hypotheses |
| `quantify_v2.py` | rigorous m_W bias at BFS reference points |

## `whizard_grid_highstats/` — morphing scheme validation on the highstats grid

Triggered by the highstats-grid campaign (9 m_W × 7 Γ_W × 37 √s @ ~0.016% MC).
Verdict: per-√s quadratic morphs in (m_W, Γ_W) + a single bilinear
cross-term coefficient β(√s) + cubic-spline of all 8 morph quantities
along √s — on a grid first denoised in √s — reproduces σ to MC-bar
(≲ 0.1% interior), sub-MeV m_W-bias-safe. Dissected + documented in
report Appendix B. NOT yet wired into the fit (the anchor still uses
trilinear interpolation; see `project_followup_morphing_scheme.md`).

| script | what it does |
|---|---|
| `morph.py` | shared primitives: denoise_grid, fit_morph_at_sqrts, build_splines, sigma_morph, build_morph_from_grid |
| `plot_grid_overview.py` | input grid: (m_W, Γ_W) plane + per-point MC precision vs √s |
| `plot_axis_fits.py` | per-√s quadratic R_m/R_Γ fits dissected — families, residuals, linear/quad/cubic model order |
| `plot_axes_loo.py` | 1D LOO on m_W and Γ_W axes (input grid quality test) |
| `plot_cross_term.py` | bilinear β dissected — (m_W, Γ_W) residual heatmaps, β linearity, β(√s) |
| `plot_sqrts_quantities.py` | morph quantities vs √s, absolute values (raw nodes + denoised curve) |
| `plot_sqrts_interpolation.py` | morph quantities vs √s, deviation of the raw per-√s fit from the denoised curve |
| `plot_sqrts_validation.py` | √s-interpolation blind test on the 0.25-GeV-offset densify points |
| `plot_morphing_scheme.py` | per-√s naive vs bilinear morphing residual (scheme buildup) |
| `plot_morph_predictions.py` | smooth σ(√s), Azzurri σ_obs, 4D LOO-on-√s validation |
| `plot_validate_substep.py` | held-out 1-MeV (m_W, Γ_W) plane vs morph (depends on grid_validate/) |
| `validate_bfs_tables.py` | morph vs BFS arXiv:0707.0773 Tables 1/2 (external closure) |

## `isr_audit/` — 2-leg ISR vs single-conv, anchor checks, β audits

From the 2-leg LL+exp ISR implementation work and earlier audits.
Verdict (see [project_followup_isr_scheme.md] and
[project_followup_isr_residual.md]): single-conv and 2-leg agree to
<0.1 % at LL+exp; the residual ~0.8 % to BFS Table 3 is NLL beyond
LL+exp, not scheme choice.

| script | what it does |
|---|---|
| `test_anchor.py` | anchor on/off effect on σ vs BFS Table 3 |
| `test_beta.py` | β value subtleties (Q² scale, NLL α correction) |
| `test_bump.py` | anchor spline behavior near 168 GeV |
| `test_mixed.py` | BETA vs MIXED scheme comparison |
| `test_nquad.py` | numerical convergence with n_quad |
| `test_zmin.py` | z_min cutoff variation |

## `whizard_isr_verification/` — WHIZARD ISR vs LL+exp, BFS Born(ISR) reproduction

Triggered by the BFS Table 4 closure-budget campaign (see report §5.5
and [project_followup_isr_residual.md]). Runs WHIZARD~3.1.5 at the BFS
Table 4 inputs in four configurations to decompose the
LL+exp-vs-WHIZARD-ISR residual into version drift, kernel difference, and
α-choice contributions. Verdict: our WHIZARD Born (no ISR) matches BFS
to MC stat (0.01-0.15 %); our WHIZARD with ISR-on no-cut is +0.8-1.2 %
above BFS — clean WHIZARD-3.1.5-vs-WHIZARD-1.x version drift.
The LL+exp BETA vs WHIZARD-multiplicative kernel adds 0.3-2 % more
(worst at 158 GeV).

| script | what it does |
|---|---|
| `job_born.sin` | WHIZARD process, no beam structure → reproduces BFS Born column |
| `job_isr.sin.tmpl` | WHIZARD process + `beams=e1,E1=>isr`, with `sqrts_hat>155 GeV` cut |
| `job_isr_nocut.sin.tmpl` | same but no partonic cut — BFS's actual Born(ISR) recipe |
| `run_born.sh`, `run_one.sh`, `run_one_nocut.sh` | drivers (sed-substitute α, source WHIZARD env, run on /tmp) |
| `work_born/`, `work_alpha0/`, `work_alphaGmu/`, `work_alpha0_nocut/` | results per configuration |

## `c1fin_analytic/` — analytic BFS appendix-B c^(1,fin) + closure-budget tests

Triggered by the c^(1,fin) analytic implementation
([project_c1fin_analytic_2026-05-26.md], commit 7d87038) and the
closure-budget probe. Verdict: the standard-chain residual at
[161, 170] GeV is dominated by BFS's σ̂_LR^(0)→σ̂_Born substitution in
the decay correction (line 2660 of arXiv:0707.0773); the option-5
hybrid + decay swap closes BFS NLO to MC stat (+0.02-0.13 %).

| script | what it does |
|---|---|
| `c1fin.py` | standalone analytic c_p,LR^(1,fin), c_d,l^(1,fin), c_d,h^(1,fin) — reference closure |
| `debug_cp_bare.py`, `debug_cp_ct.py` | per-term debug of bare + counterterm assembly (used to find a factor-of-2 typo in the 1-mass triangle derivation) |
| `option5_postisr_anchor_test.py` | hybrid: BFS-quoted Born(ISR) + our analytic NLO × LL+exp |
| `option5_with_decay_swap.py` | option-5 + BFS's σ̂_LR^(0)→σ̂_Born decay substitution → closure to +0.02-0.35 % vs BFS Table 4 NLO column |
| `chain_decay_swap_validation.py` | chain-level closure after the substitution lands in production (`decay_uses_full_born=True`); confirms knob ON reproduces option-5+swap |
| `chain_decay_swap_slopes.py` | σ_obs + dσ/dm_W + dσ/dΓ_W shifts at scan-window √s under knob ON vs OFF |
| `decay_swap_asimov_AB.py` | Asimov A/B with two template sets (knob OFF + knob ON) — cross-fits give the m_W central-value shift attributable to the substitution (|Δm_W| = 5.5 MeV) |

## `bes_uncorr_scaling/` — BES uncorr prior shares lumi's per-point counting rescale

Triggered by the 2026-06-02 generalization of `LUMI_UNCORR_SCALES` into a
counting-measurement mechanism (commit 6d0a738, `UNCORR_COUNTING_KINDS`).
Verdict: the beam-energy spread is read off the di-muon sample, so its
uncorrelated component is a counting measurement whose per-point precision
scales as `√(L_ref/L_i)` — exactly like the luminosity — and now uses the same
`fit._uncorr_perbin_scale`. BES contributes ≈0 to m_W regardless, so this is a
self-consistency refinement, not a headline shift. BEC stays fixed per point
(resonant-depolarisation calibration reproducibility, not a counting statistic —
and the term that actually drives m_W). See `cards/README.md`.

| script | what it does |
|---|---|
| `verify_bes_scaling.py` | builds the production-config Asimov fit at the 7-point baseline and a 3-point FCC layout; asserts BES is in `_counting_uncorr_kinds`, that `_uncorr_perbin_scale = √(L_ref/L_i)` aligns with the BES per-bin count, and that the 3-point layout tightens the BES uncorr prior (more lumi/point) while m_W impact stays ≈0 |

## `syst_table_dedup/` — byte-identity gate for the systematics-table emitters

Triggered by the NEED-4 dedup (commit 1c5084f): `_print_table` and
`_write_latex` each re-implemented row iteration / NaN handling / `THEORY_UNC`
gating, so the text and LaTeX tables could structurally drift. Verdict: both
emitters were refactored onto a shared `_emit_rows()` generator and proven
byte-identical to the pre-refactor output on a synthetic golden hitting every
formatting trap (NaN diagonal → dash, WbWb theory row, relative-yukawa divide).

| file | what it does |
|---|---|
| `check_byte_identical.py` | renders `_print_table` + `_write_latex` on mock WW-like (2 POI, no theory, NaN diagonal) and WbWb-like (3 POI, theory, relative-yukawa) cards; `write` mode stores the golden, `check` mode diffs against it |
| `golden.txt` | captured pre-refactor output — the byte-identity reference |

## `anchor_ecm_pattern/` — ground-truth WHIZARD grid m_W/Γ_W dependence

Triggered by the anchor refactor / Γ_W-crossing discussion (see
[project_anchor_ecm_pattern_resolved.md] and
[project_gammaw_crossing_resolved.md]). Plots σ_Whiz(√s, m_W, Γ_W) read
straight from the 1295-pt grid — no EFT chain, no anchor, no ISR — as the
apples-to-apples reference for what WHIZARD actually predicts at the nodes.
Verdict (resolved): the anchor strips WHIZARD's BR² squeeze under the
pdg-constant chain, the dσ/dΓ_W=0 crossing near 162 GeV is preserved, and
ρ(m_W,Γ_W)=+0.64.

| script | what it does |
|---|---|
| `plot_whizard_grid_dependencies.py` | 4-panel σ vs √s and σ-ratio vs √s at several m_W / Γ_W slices, raw grid only |

## `bfs_nnlo/` — closed-form NNLO closure (production validation)

**Not an off-tree exploration** — these two are production-validation scripts
cross-referenced from `report/README.md` §3.5–3.6 (they mirror Tables 1 and 2
of arXiv:0807.0102). Kept here because they began as the NNLO audit
([project_bfs_nnlo_audit_2026-05-26.md], [reference_bfs_nnlo.md]). Verdict:
the closed-form pieces close Table 1 to |Δ|≤0.0006 fb; the ISR-improved
Δσ̂^(3/2) round-trips Table 2.

| script | what it does |
|---|---|
| `check_closed_form_pieces.py` | each NNLO piece (C×[S+H], NLO-C, decay, res, C3) + sum vs Table 1 |
| `check_isr_table2.py` | Δσ̂^(3/2) shift after ISR convolution, round-trip vs Table 2 (pins BFS paper EW inputs) |

## `bilinear_morph/` — (m_W × Γ_W) cross-term morph closure

Triggered by the bilinear cross-term morph ([project_bilinear_morph_2026-05-29.md],
commit 68a3874). Verdict: at the worst-case (+10, +10) MeV corner the linear-only
chain carries a residual bilinear bias that the production `CROSS_TERMS=[("mass","width")]`
chain removes — m_W bias 40 keV → ~1 keV (sub-eV MIGRAD closure).

| script | what it does |
|---|---|
| `closure_corner.py` | Asimov closure at the corner template, bilinear vs linear-only chain; reports the bilinear bias on each POI |

## `coulomb_double_count/` — K_C vs BFS eq. 62 overlap → dropping K_C

Triggered by the Coulomb double-count audit ([project_followup_kc_dropped_2026-05-26.md],
[project_followup_deep_audit_2026-05-20.md]). Verdict: the FKM K_C factor and BFS
eq. 62 term 1 encode the same leading-α/v Coulomb physics (~5–7% overlap at
threshold), so they double-count — which is why `include_coulomb=False` became
the production default. Two figures from here are embedded in the report
(`coul_compare_kc_safe_chain.pdf`, `coul_kc_vs_term1_overlap.pdf`).

| script | what it does |
|---|---|
| `validate_kc_safe_decomposition.py` | 4 cross-checks pinning the `coulomb_kc_safe` knob: route-A/B identity, K_C↔eq.62 overlap, asymptotic limit, FKM 1995 5.21% near-threshold |
| `check_chain_over_whizard_ratio.py` | σ_chain/σ_WHIZARD_4f_Born for the three Coulomb configurations |
| `compare_anchor_before_after.py` | σ-shape impact of the K_C-safe fix |
| `plot_kc_vs_term1_overlap.py` | visual of the K_C−1 vs Δσ_Coul_NLO term1/σ_LR^(0) same-physics overlap |

## `isr_scheme/` — ISR α-renormalisation scheme dependence is genuine

Triggered by the theory-ladder ISR scheme-variation finding
([project_theory_ladder_2026-06-01.md], [project_followup_isr_alpha_mz_scale_var.md]).
Verdict: the surprisingly large ALPMZ→ALGMU / α(0) shift of the fitted m_W
(+3.5 MeV pure-shape, +36 MeV under a realistic lumi prior) is a **genuine**
σ-level effect, not a plumbing/quadrature bug — it is purely ISR (partonic σ̂
identical across schemes), quadrature-robust, and linear in α. It is the
dominant theory systematic.

| script | what it does |
|---|---|
| `verify_isr_alpha_scheme.py` | σ-level scheme shift decomposed into normalisation vs shape; confirms purely-ISR + plumbing + quadrature + β_e linearity |
| `quad_meanR_check.py` | follow-up confirming the ALGMU/ALPMZ mean-R normalisation |

## `nll_isr/` — NLL eMELA ISR landing campaign

The full work behind the NLL ISR landing ([project_nll_isr_state_2026-05-27.md],
[project_followup_nll_isr_plan.md], [reference_emela_nll_isr.md]): eMELA C-wrapper
build, NLL validation, quadrature convergence, ISR-scenario cross-fits, the
α_em_isr A/B, and post-ISR morph closure. Verdict: NLL ISR is essentially
mandatory at FCC-ee precision — the LL+exp→NLL cross-fit bias on m_W is
±22.4 MeV (±17.0 LL-truncation + ±5.4 pure-NLL); morph closure holds to ~ppm
post-ISR. Carries committed `.log` files (the kept run outputs).

| script | what it does |
|---|---|
| `build_emela_wrapper.sh`, `emela_c_wrapper.cpp` | build the eMELA C wrapper |
| `validate_emela_isr.py` | NLL ISR σ validation |
| `convergence_study.py` | n_quad / z_min convergence |
| `compare_isr_scenarios.py`, `cross_fit_isr_scenarios.py` | LL+exp vs NLL σ comparison + Asimov cross-fit bias |
| `asimov_alpha_em_isr_AB.py` | α_em_isr (α(M_Z) input) Asimov A/B |
| `morph_closure_post_isr.py` | morph closure after the ISR convolution |
| `scale_decouple_diagnostic.py`, `smoke_test_scale_alphaMZ.py` | α-scale vs α-value separation diagnostics |
| `plot_isr_comparison.py` | the `isr_comparison.pdf` report figure |

## `old_new_AB/` — chain-routing cross-fit decomposition

Triggered by the open-issues chain-routing items ([project_chain_routing_cross_fit_2026-05-29.md],
[project_followup_2107_04444_comparison.md]). Builds matched OLD/NEW template
sets differing only in one targeted knob and Asimov-cross-fits them. Verdict:
K_C-drop alone ±140 MeV, decay substitution ±5 MeV, δ_QCD routing ±85 MeV, full
old→new chain ±220 MeV on m_W — the K_C drop was the dominant correction. The
2107.04444 follow-up reaches Δm_W ≲ 0.6 MeV against the current chain.

| script | what it does |
|---|---|
| `build_templates.py` | builds `prod/`, `legacy_dqcd/`, `legacy_combined/` template sets (one targeted knob each) |
| `run_AB.py` | item 3 (δ_QCD routing) + item 4 (full old chain) cross-fits |
| `run_2107_followup.py` | 2107.04444 (Azzurri) Asimov cross-check vs the current production chain |

## `lumi_nuisance/` — LUMI_MODE nuisance ≡ cov equivalence

Validates that representing luminosity as a floating per-bin+correlated Gaussian
nuisance (`LUMI_MODE='nuisance'`) is physically equivalent at LO to the
covariance-matrix mode (`'cov'`). Verdict: identical POI uncertainties up to
Minuit precision; the small BES_corr/BEC_corr "outliers" are quadrature-subtraction
noise (same-mode jitter ≈ cross-mode delta), not a real disagreement; nuisance
mode is only mildly slower (N+1 extra fit params).

| script | what it does |
|---|---|
| `validate.py` | side-by-side nominal / syst-table / scan_lumi comparison of the two modes |
| `detailed_compare.py` | consolidated diff table with a noise-floor-aware OK/OK*/FAIL verdict |
| `syst_table_jitter.py` | same-mode syst-table jitter — proves the outlier rows are noise |
| `smoke_scans.py` | full scan suite runs clean under nuisance mode (no NaN POI unc) |
| `timing.py` | wall-time cov vs nuisance for nominal / syst-table / lumi sweep |

## `lumi_scaling_review/` — per-point lumi `√(L_ref/L_i)` scaling review

Focused review of the per-point uncorrelated-lumi counting rescale in `fit_core`
(the mechanism since generalized to BES via `UNCORR_COUNTING_KINDS` — see
`bes_uncorr_scaling/` and `cards/README.md`). Verdict: direction (more L_i →
tighter prior), bin-order alignment, and uneven-split behaviour all check out;
and the `add_last_ecm × LUMI_UNCORR_SCALES` last-bin tightening is **not**
double-counted (the cov path is gated mutually-exclusive with the per-point
scale).

| script | what it does |
|---|---|
| `check_lumi_perbin.py` | direction / bin-order / uneven-split / same_evts / reinit-to-stat across 6 layouts, cov vs nuisance |
| `check_double_count.py` | confirms the last-bin tightening isn't applied twice when add_last_ecm and LUMI_UNCORR_SCALES are both on |

## `parametric_nuisance_variations/` — sizing the profiled parametric nuisances

Triggered by the parametric-prior campaign ([project_followup_param_unc_propagation.md],
[project_followup_alpha_em_isr_nuisance.md]). Sizes the m_t / M_H / α_em_isr
template variations and validates their fit-side wiring. Verdict: a variation
giving ~0.5 % σ-response (≈25× the 0.02 % morph noise floor) keeps the response
linear; the aem_isr profiled nuisance, per-point lumi scaling, and channel
extrapolation all wire through correctly (validated on fast LL templates so
production templates are untouched).

| script | what it does |
|---|---|
| `measure_response.py` | relative σ-response per natural unit of m_t / M_H / α_em_isr + linearity check |
| `validate_wiring.py` | fast LL-template check that aem_isr nuisance + per-point lumi + channel extrap all wire through |
