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
