# Handoff — efficient + smooth + production-faithful ISR convolution

**Date:** 2026-06-17  •  **Status:** EXPLORATORY, not wired into production  •  **Owner:** open

## One-line goal
Produce a 1D ("luminosity") reformulation of the two-leg ISR convolution that is
**(a) production-faithful** (reproduces the 2D `convolve_2leg` numerics to
<0.05 % / sub-MeV in the m_W fit), **(b) smooth** (no GL ripple), and **(c) fast**
(≪ the O(n_quad²) cost of the 2D), then wire it as an **opt-in** in `isr_beta`.
If <0.05 % faithfulness turns out not to be reachable, document precisely *why*
and recommend the fallback (keep the 2D, raise `n_quad`). **Do not** change
production defaults; **do not** commit (ASK-before-commit standing rule).

## Why this exists (context)
The NLL line shape has a visible ripple in the 2 m_W turn-on at the production
`n_quad=128`. This was chased down and is **not** physics and **not** a grid
problem — it is a **Gauss–Legendre quadrature artefact**: σ̂'s steep turn-on
("kink" at the σ̂ grid edge √ŝ=156 GeV) is straddled by GL nodes that shift with
√s, so the convolution error scales **∝1/n_quad** (max|2nd-diff| 4.5e-3 @128 →
1.2e-3 @512 → 7.1e-4 @1024, halving per doubling — textbook). See memory
`project_scheme_variations_2026-06-16.md` (CONTINUATION 2026-06-17) and
`isr_steps_investigation.py` / `isr_steps_ladder_check.py`.

**Production fits are robust and need no change** — proven: same-radiator Asimov
cancels exactly; EW σ̂-denoise spread 0.64 MeV (Part C); LL↔NLL worst-case
cross-fit 0.16 MeV across denoising. The motivation for a 1D form is (i) clean
figures without a 16× cost, and (ii) future-proofing if σ̂ ever becomes expensive
(it is currently a cheap spline, which is exactly why the 2D's vectorisation
already wins on wall-clock — see "honest caveat" below).

## What's already built (all in `scripts/investigations/nll_isr/`)
- `convolve_1d_prototype.py` — iterated single-leg C∘C, kink-split both legs.
  Reaches truth-smoothness at ~7× fewer σ̂-evals than 2D-512 **but wall-clock
  SLOWER** (Python loop; per-leg grid rebuilt per energy because the kink-split
  nodes are E-dependent). Kept for reference; not the recommended path.
- `luminosity_prototype.py` — **the candidate.** σ_obs(s)=∫L(z)σ̂(√z·√s)dz with the
  luminosity L = radiator self-convolution in V=−ln z. Double-soft endpoint
  V^{2β_e−1} handled exactly by Gauss–Jacobi; L̃(V) (all the radiator work)
  precomputed once; single Gauss–Jacobi panel [0,V_top] (NO interior kink split —
  σ̂ is off-shell-smooth through 2m_W, so a split injects a spurious panel-transition
  kink at √s=2m_W; the σ̂ grid-edge step is the integration *limit* V_top, never
  interior). Smoother than the 2D and ~1000× fewer σ̂-evals.
- `lumi_grid.py` — production-grade form of the above: `L̃(V; μ_F)` on a (V, μ_F)
  grid (so μ_F=√s drift across the scan is interpolated, not approximated), an
  `npz` with provenance meta mirroring `isr_emela_grid`, `LumiGrid.load/ltilde`,
  `sigma_obs_grid`, `line_shape_grid`. Build 0.5 s / 21 kB once; eval Δσ_core
  0.042 % (n_out=16), max|2nd-diff| 3.4e-4 (smoother than 2D-1024), <1 s.
- `plot_lumi_validation.py`, `plot_1d_validation.py` — publish validation plots to
  the EOS web area (`…/WW_threshold/scheme_variations/{lumi,conv1d}_validation.png`).

## The crux — why it's "not liked yet" (the real open problem)
The luminosity L̃ is a **separate re-derivation** of the radiator self-convolution.
It is NOT yet *guaranteed identical* to what the production `convolve_2leg` actually
integrates. Concretely, the residual (~0.04–0.1 % core) is a mix of:
1. the 2D truth's **own non-convergence** at finite n_quad (the kink kills GL
   convergence — even n_quad=1024 is not the s→∞ limit), and
2. an **endpoint-convention mismatch**: `convolve_2leg` applies a specific
   soft+virtual normalisation (`norm_nll` / `beta_s`,`beta_h` split, the
   `(1−x)^β_e` endpoint substitution, `x_min`), and the prototype's L̃ rebuilds the
   per-leg density from `xfxQ` *without* threading those exact pieces through.

Until L̃ is **derived from the identical per-leg radiator objects the 2D uses**, so
that `lumi` and `convolve_2leg` agree to machine precision *as n_quad→∞*, the 1D
form is not a trustworthy drop-in — it's "close" but not provably the same physics.
That is almost certainly the source of the lingering dissatisfaction.

## Concrete plan (recommended)
1. **Read** `framework/process/ww/indep/isr_beta.py` end-to-end, in particular
   `convolve_2leg` (~lines 602–622), `_radiator_setup`, `_per_leg_grid_nll`,
   `_endpoint_grid`, `_radiator_norm`, `beta_components`, and the `ISRConfig`
   fields (`n_quad`, `x_min`, `emela_*`). Write down *exactly* what per-leg density
   ρ(x) the 2D integrates, including the soft+virtual `norm_nll` factor and the
   `(1−x)^β_e` substitution.
2. **Re-derive L̃ from THAT ρ** (not a fresh `xfxQ` call): the luminosity must be the
   self-convolution of the *same* ρ the 2D uses, so the only difference between the
   1D and 2D results is the quadrature, not the integrand.
3. **Convergence test (the acceptance gate):** with the corrected L̃, show
   `lumi(n_out) → convolve_2leg(n_quad)` agree to **machine precision as
   n_quad→∞** (run the 2D at n_quad=2048 as the reference), and that the lumi form
   hits <0.05 % at small n_out. If they do NOT converge to each other, the L̃
   derivation is still wrong — iterate on step 2.
4. **Cross-fit neutrality:** inject a truth line shape via the new conv and fit with
   a morph via the new conv — Δm_W must be 0 ± <0.05 MeV vs the 2D path
   (reuse the cross-fit machinery in `gf_consistency_crossfit.py` /
   `ew_scheme_crossfit.py`).
5. **Wire as opt-in** in `isr_beta` (e.g. `ISRConfig(use_lumi_grid=…/lumi_grid=…)`),
   provenance-checked exactly like the eMELA grid, with an end-to-end
   `doFit_ww.py`/`dofit_indep.py` smoke test. Production default stays the 2D.
6. If step 3 cannot reach <0.05 % for a *defensible* reason (e.g. the soft+virtual
   structure is genuinely not a pure z=x₁x₂ luminosity), **stop and document it** —
   the robust fallback is to keep the 2D and raise `n_quad` (256–512) for figures
   only; production fits are already robust.

## Acceptance criteria (all four)
- **Faithful:** lumi vs `convolve_2leg(n_quad=2048)` < 0.05 % over √s∈[157.5,162.5];
  the two *converge to each other* as n_quad→∞.
- **Smooth:** max|2nd-diff| ≤ the 2D-1024 truth.
- **Fast:** σ̂-evals ≪ 2D-128 (the lumi grid is already ~5000× fewer).
- **Cross-fit neutral:** Δm_W = 0 ± <0.05 MeV truth-vs-morph through the new conv.

## Environment / ground rules
- `source setup.sh && PYTHONPATH=$PWD:$PYTHONPATH python3 …`  — **PREPEND** PYTHONPATH
  (a bare `PYTHONPATH=$PWD` wipes the LCG site-packages → `ModuleNotFoundError:
  uncertainties`). No trailing `| grep` (subshell). See `feedback_python_env_fallback`.
- Heavy runs are fine in the local Bash env (it *is* an ironic node — run directly,
  no ssh; cap ≈48 threads). Long jobs → CONDOR.
- Scratch/scheme grids live in `/tmp/ww_nll_scheme_scan/grids/`; prod NLL grid is
  `framework/process/ww/indep/grids/emela_nll_delta_alpmz.npz`.
- **Do not** touch production defaults; **do not** commit (ASK first). Leave findings
  in a results file + update memory `project_scheme_variations_2026-06-16.md`.
