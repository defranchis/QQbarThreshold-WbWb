# Cards — physics rationale

This file documents the non-obvious choices in `ww_default.py`.
For parameter provenance and BFS equation cross-references see the BFS paper
arXiv:0707.0773 (NLO) and arXiv:0807.0102 (NNLO).

---

## PARAM_INPUTS / PARAM_UNC

### PARAM_INPUTS — production theory inputs

`alpha_s_MW`, `m_t`, `M_H`, `M_Z` enter the NLO hard-matching coefficient
`c_p,LR^(1,fin)` and the BFS EW couplings ξ(s), χ(s), sin²θ_W.  Production
defaults: PDG 2024 / FCC-ee FSR Vol. 1 Table 2 (arXiv:2505.00272).
`m_t=172.5` GeV is the FSR central value, distinct from the BFS reference
(see below).

The BFS reference set used to produce Tables 1–4 is different:
`m_t=174.2, M_H=115 (pre-Higgs-discovery), M_Z=91.188`.  These are kept as
`M_*_BFS_REF` constants in `process/ww/xsec_calculator/eft_xsec.py`; all closure
validation scripts pass them explicitly.  Note the production
default `m_t=172.5` is now ≠ `M_T_BFS_REF=174.2`, so BFS Tables 1–4
closure must use `M_T_BFS_REF` (already pinned in validation scripts).

`alpha_em` and `alpha_em_isr` are α_em overrides.

- **σ chain (`alpha_em`)** — `None` (default) → derived
  `α_Gμ(m_W, M_Z) = √2 G_F m_W² sin²θ_W / π` (tree-level G_μ scheme;
  `G_F = 1.1663787e-5` in `eft_xsec.py`).  Applies to Born + NLO + NNLO
  via `alpha_Gmu(mW, MZ, override=alpha_em)`.
- **ISR (`alpha_em_isr`)** — production card default is the explicit
  float `1.0/128.943` (PDG α(M_Z); matches `isr_emela_ren_scheme="ALPMZ"`
  and the FCC-ee target via A_FB^μμ off-peak ~ Riembau arXiv:2501.05508).
  `None` → legacy fallback `α_Gμ(M_W_BFS_REF) ≈ 1/132.1`, fixed across
  the fit (BFS prescription, arXiv:0707.0773 line 2514); kept for paper-
  closure tests.

Only the ISR side carries a fit nuisance (`PARAM_UNC.alpha_em_isr` →
`aem_isr`); the σ-chain `alpha_em` (Gμ scheme) is luminosity-degenerate for
m_W and is left unpropagated (`PARAM_UNC.alpha_em = None`). See PARAM_UNC.

### PARAM_UNC — parametric uncertainties (FCC-ee projections)

`PARAM_UNC` holds the FCC-ee parametric-input uncertainties. A value of
`None` means the input is **not** propagated as a fit nuisance (it has no
usable handle on the threshold lineshape, or it is degenerate with another
nuisance). Only `alpha_s` and `alpha_em_isr` are profiled.

| Key            | Value   | Profiled? | Basis / why |
|----------------|---------|-----------|-------------|
| `alpha_s`      | 1.0e-4  | yes (`alphas`)   | FCC-ee FSR Vol.1 Tera-Z; enters δ_QCD routing |
| `alpha_em_isr` | 4.7e-8  | yes (`aem_isr`)  | Riembau arXiv:2501.05508 (see below) |
| `m_t`          | None    | no        | negligible σ-response (see below) |
| `M_H`          | None    | no        | negligible σ-response (see below) |
| `M_Z`          | None    | no        | sub-0.1 MeV on m_W; not propagated |
| `alpha_em`     | None    | no        | Gμ-scheme α: luminosity-degenerate for m_W |

**`alpha_em_isr` = 4.7×10⁻⁸** — δα(M_Z) absolute = 0.6×10⁻⁵ (relative) × α(M_Z),
the Riembau (arXiv:2501.05508) **combined on-peak** projection (A_FB^μμ + e/μ + e/e
ratios, coverage to cosθ=0.99). This supersedes the earlier 2.4×10⁻⁷, which was
Riembau's more conservative off-peak A_FB^μμ-only figure (≈3×10⁻⁵ rel). Profiled
as the `aem_isr` offset nuisance (template variation 1×10⁻⁴, prior 4.7×10⁻⁸; see
PARAMETERS). Asimov impact ~0.008 MeV — the *input* uncertainty, distinct from the
ISR renormalisation-scheme ambiguity (ALPMZ↔ALGMU, ~36 MeV under prior), which is a
missing-higher-order **theory** uncertainty reported separately (theory ladder).

**`m_t`/`M_H` are negligible by measurement, not by omission.** The σ-response of
the threshold lineshape to these inputs (measured in
`scripts/investigations/parametric_nuisance_variations/measure_response.py`) is
+0.0016 %/GeV (m_t) and −0.0058 %/GeV (M_H) — so over the *entire* FCC-ee input
uncertainty the σ shifts by ~10⁻⁵ %, a thousand-fold below the ~0.02 % morph noise
floor, i.e. <1 keV on m_W. Physically, a *direct* threshold-lineshape m_W reads the
mass off the kinematic turn-on; unlike an indirect EW fit it does **not** inherit the
m_t/M_H uncertainty through the Δr(m_t, M_H) relation (and with `pdg-constant` BR +
α_Gμ, m_t/M_H never enter the normalisation either). There is no clean morph template
to build for them, so they are documented as negligible rather than profiled.

**`alpha_em` (Gμ-scheme, hard σ̂)** is luminosity-degenerate for m_W (it rescales the
overall normalisation, absorbed by the lumi nuisance), so profiling it buys nothing
for the mass; left `None`. Only the ISR-side α (`alpha_em_isr`, a shape effect via
β_e) matters.

BR_W_* uncertainties are deliberately not in `PARAM_UNC`: in `pdg-constant` BR mode
the BR is a measured constant, an analysis-level systematic rather than a theory-input
nuisance.

---

## PARAMETERS — fit parameters & template variations

`mass` and `width` are the POIs (absolute values). `alphas` and `aem_isr` are
**offsets** from their nominal inputs (α_s(M_W) and the ISR coupling α(M_Z)),
profiled as Gaussian-constraint nuisances (`SYSTEMATICS[...] = constraint`).

The `variation` field is the **template finite-difference step** used to build the
morph slope — a purely numerical choice, decoupled from the physical prior in
`PRIORS`. It is sized to lift the σ-response well above the ~0.02 % morph noise
floor while staying in the linear regime, exactly as the `mass` POI uses a 10 MeV
variation yet the fit reports sub-MeV. For `aem_isr`, variation `1×10⁻⁴` gives a
~0.46 % σ-response (≈23× noise, clean & linear); the prior `4.7×10⁻⁸` is ~2000×
smaller and is the physical FCC-ee input width (the constraint is applied as
`σ_fit = PRIORS[name]/variation`, so the two never need to match). `aem_isr` is
applied in `generator.do_scan` as `α_em_isr → α_em_isr + offset` (ISR β_e only).

## Priors / nuisances — beam energy, spread, luminosity

`PRIORS` gives the Gaussian-constraint widths for the beam/luminosity nuisances
(`uncorr` = point-to-point between scan points, `corr` = fully correlated). For
**BES and luminosity** the correlated component is half the uncorrelated
(`corr = ½·uncorr`, following the WbWb threshold paper arXiv:2503.18713); **BEC**
instead has a *larger* fully-correlated resonant-depolarisation component
(`corr 0.3 > uncorr 0.1` MeV — the absolute √s scale is the dominant common term).

- **BEC (beam energy calibration)** — MeV on √s. `corr 0.3 / uncorr 0.1`. FCC-ee
  resonant depolarisation: Blondel & Janot arXiv:1909.12245 quote a 300 keV √s
  determination at the WW threshold (di-muon Z-return, calibrated to similar
  accuracy by RDP) with ~100 keV point-to-point. (The LEP-era ±10 MeV is *not* the
  FCC-ee projection.)

- **BES (beam energy spread)** — relative, on the 0.105 % spread. `uncorr 0.01 /
  corr 0.005` (1 % / 0.5 %). Blondel-Janot: σ_√s monitored continuously from
  di-muon events to ~0.6 % at the Z (and "virtually infinite precision"), with 1 %
  used as their benchmark variation. BES contributes ≈0 to m_W regardless.

- **Luminosity** — relative, per scan point. `uncorr 2×10⁻⁴ / corr 1×10⁻⁴`.
  Estimated from a **central di-photon (e⁺e⁻→γγ) counting** measurement, the method
  used in the WbWb paper (arXiv:2503.18713, 0.1 %/point at 41 fb⁻¹). Using the
  fiducial σ_γγ = 13.2 pb at √s=160 GeV (Carloni Calame et al. arXiv:1906.08056,
  central acceptance 20°<θ_γ<160°, E_γ>0.25√s) and L = 2.74 ab⁻¹/point
  (19.2 ab⁻¹ over 7 points): δℒ/ℒ = 1/√(σ·L) = **1.7×10⁻⁴**, which we conservatively
  inflate to **2×10⁻⁴**. (σ_γγ ∝ 1/s exactly within fixed acceptance — confirmed by
  1906.08056 to <0.5 % — so the 4.6× higher γγ rate at WW than at the tt̄ threshold
  would give ~5.7×10⁻⁵; we do not bank that gain pending a fiducial efficiency study.
  Small-angle Bhabha, ~20× the rate, would do better — di-photon is the conservative
  fallback.)

### LUMI_UNCORR_SCALES — per-point luminosity scaling

The luminosity is a **counting** measurement, so its uncorrelated (point-to-point)
prior scales as `1/√(L_point)`: a scan with fewer points puts more luminosity per
point → smaller per-point uncorr lumi. With `LUMI_UNCORR_SCALES = True`, the fit
rescales the prior per scan point as `uncorr_i = uncorr_ref·√(L_ref/L_i)`, where
`(uncorr_ref, L_ref) = (PRIORS["lumi"]["uncorr"], LUMI_UNCORR_CALIB_LUMI)` is
calibrated at the baseline 7-point per-point luminosity. So the 7-point baseline
keeps 2×10⁻⁴, the 3-point FCC scenario gets 2×10⁻⁴·√(3/7) ≈ 1.3×10⁻⁴, etc. (applied
in `fit_core._nuisance_prior`/`_build_cov`). The correlated (common-normalisation)
component does **not** scale. The channel extrapolation applies its inclusive yield
boost as a stat-only rescale at the real machine luminosity, so this prior is
unaffected by the (fictitious) yield boost.

### PEAK_ECM — BES-smearing reference

`PEAK_ECM = 160.0` is used **only** as the reference √s that converts the relative
beam-energy spread (BES, % per beam) into the absolute Gaussian smearing-kernel
width (`σ_√s = PEAK_ECM·BES%/100/√2`, in `common/smearing.py`). It does not enter
the cross-section calculation, scan grid, or morph. 160 GeV is the WW operating
point (E_beam = 80 GeV, where the 0.105 % spread is quoted), replacing the earlier
162.5 scan-centre placeholder (a 1.5 % narrower kernel; sub-MeV on the fit).

---

## SCENARIO & INPUT_VAR

`SCENARIO` defines the baseline scan and luminosity:

- `scan_min/max/step = 157/163/1.0` → the 7-point threshold scan.
- `total_lumi = 19.2e6` /pb — FCC-ee FSR Vol.1 (arXiv:2505.00272) `tab:seqbaseline`
  WW baseline: 20×10³⁴/cm²s per IP × 9.6 ab⁻¹/yr × 2 yr × 4 IP = 19.2 ab⁻¹, split
  equally across the 7 points (2.74 ab⁻¹/point).
- `last_lumi = 10.8e6` /pb — ZH 240 GeV baseline (FSR `tab:seqbaseline`); only used
  with `--lastecm`, dormant otherwise.
- `stat_inflation = 1.0` — global stat multiplier (WbWb uses 1.2 for background
  contamination; WW reco study pending). `coarse_scan` is a 3-point fallback
  geometry; `true_value_pivot = "mass"` selects which POI the Asimov truth pivots on.

`INPUT_VAR` gives the **fit-parameter units** of the binned nuisances — the step a
nuisance value of 1.0 corresponds to: `BEC = 10` MeV (matches the
`output_xsec/ww/BEC/scan_{p,m}10/` template shift), `BES = 0.1` (10 % of the spread),
`lumi = 0.01` (1 %). These are numerical units; the physical constraint widths are in
`PRIORS` (the fit applies `σ_fit = PRIORS/INPUT_VAR`).

**Scan-scenario comparison.** `doFit_ww.py --compareScenarios` compares three √s
layouts (7-point baseline, 3-point FCC, 2-point Azzurri-like) at the same total
luminosity, writing `plots/scenario_compare*`: the 2-POI sensitivity, the
per-scenario full-systematics breakdown and theory ladder (text/CSV), and plots —
`_layout` (scan points on the lineshape + per-point lumi), `_ellipses`
((m_W,Γ_W) error contours), `_syst_{mW,gW}` (per-source budget bars), and
`_scan_<syst>_{mW,gW}` (the `uncert_mass_width_vs_<syst>` impact sweeps overlaid
across layouts, one figure per systematic × POI). The systematics scanned are
derived from `SYST_TABLE_ORDER` (no hand-maintained list). The constraint-nuisance
sweeps (`alphas`, `aem_isr`) use an **exact rank-1 Gaussian covariance update** —
one calibrated Hesse per nuisance, then an analytic curve — instead of a per-point
re-fit, which removes the quadrature-subtraction noise that otherwise swamps their
sub-0.1 MeV impacts. All of this is documented in the report's scan-scenario
comparison section.

---

## NLO_CONFIG

### `br_convention`

- `"pdg-constant"` (default): multiply σ_WW by the PDG-measured BR, treated as a
  fixed constant.  δ_QCD is routed through BR via
  `BR(α_s) = BR_PDG × δ_QCD(α_s)/δ_QCD(α_s_ref)` so the α_s differential is
  preserved without double-counting.
- `"bfs-eft"`: δ_QCD multiplies σ per BFS §6.1; BR is computed from BFS EFT partials.
  Use for BFS paper round-trips only.

### `include_coulomb` / `coulomb_kc_safe`

`include_coulomb` controls the **FKM (Fadin-Khoze-Martin 1993) multiplicative K_C
factor**, a 1993-vintage LEP-era engineering shortcut.  It is **off by default**
because:

1. BFS treats Coulomb via an α-expansion of the Coulomb Green function (additive
   insertions, eq. 61–62 of arXiv:0707.0773), not a multiplicative K_C.
2. With `include_NLO_hard_decay=True` the BFS NLO Coulomb (eq. 62) is already in the
   chain via `delta_sigma_Coulomb_NLO_specific_pb`.  Adding K_C on top double-counts
   the leading α/v piece at the ~5% level at threshold.
3. BFS explicitly justify not resumming the two-photon piece (lines 1722–1725:
   "a few permille").
4. Scenario F (BFS Table 4 closure) and `plot_bfs_table4` both use
   `include_coulomb=False`; the production chain matches that comparison.

`coulomb_kc_safe=True` (with `include_coulomb=True`) adds only the subleading NLO
two-photon piece (~0.2% at threshold) while dropping the leading piece that K_C and
BFS eq. 62 share.  Diagnostic only.

A full Grade-B refactor replacing K_C with the BFS unstable-W Coulomb Green function
G_C is planned but gated on other work; see memory `project_grade_b_gc_refactor_plan`.

### `decay_uses_full_born`

BFS §6.2 (line 2255) states that in the NLO decay correction (eq. 84) one should
replace σ̂^(0) by the **full Born** σ_Born (LO + 1/2 + NLO Coulomb potential + 3/2,a,
with the Whizard anchor applied).  Setting `True` (default) implements this recipe
exactly and closes BFS Table 4 to ±0.5% at [161, 170] GeV.  Setting `False` reverts
to Δσ_decay = δ_decay × σ̂^(0) and mis-closes Table 4 by up to 0.9%.

### `include_BFS_NNLO`

Five closed-form pieces from arXiv:0807.0102 eq. 49:

| Term     | Equation | Description |
|----------|----------|-------------|
| C×[S+H]  | eq. 34   | Coulomb × (soft+hard), ISR-subtracted |
| NLO-C    | eq. 39   | Coulomb potential running + α(M_Z)→G_μ |
| C×decay  | eq. 40   | Coulomb × EW partial-width factor |
| C×res    | eq. 48   | Residue × single-Coulomb |
| C3       | eq. 11   | Triple-Coulomb with ζ(3) |

Combined ~+1 fb at the WW peak; ~3 MeV impact on m_W per BFS §6.4.
Validated to 4 digits vs Table 1 of arXiv:0807.0102.

### `apply_delta_QCD`

QCD correction δ_QCD = 1 + α_s/π + 1.409(α_s/π)² (BFS §6.1).
In `"pdg-constant"` BR mode this is routed through the hadronic BR rather than
multiplying σ directly, to avoid double-counting the PDG-measured BR.

### `whizard_anchor_source`

Three sources, all implementing the BFS §6.2 prescription:

- **`"morph"`** (default): per-√s quadratic morphing predictor trained on
  `whizard/work/grid_fine/grid.csv` (6363 pts, 101 √s × 9 m_W × 7 Γ_W, ~0.008% MC).
  Validated sub-MeV-safe (max 0.020% on held-out `grid_validate_fine`).
  Implementation: `process/ww/xsec_calculator/grid_morph.py`.
- **`"grid"`**: trilinear interpolation of the 1295-point WHIZARD scan
  (5 m_W × 7 Γ_W × 37 √s); ~0.05–0.2% MC noise per point.
- **`"spline"`**: cubic spline in δ=√s−2m_W over BFS Tables 1+2, linear in Γ_W
  between two anchor points.  Smooth but limited Γ_W coverage.

### `isr_scheme`

Both schemes implement LL+exp BETA radiator per LEP2 YR Beenakker (hep-ph/9602351)
eq. 67 and BFS eq. 71.  They agree to <0.1%.

- `"single_conv"` (default): LEP2 YR α→2α 1D shortcut; ~10× faster.
- `"2leg"`: full per-leg double-convolution.  Auto-selected when `isr_nll=True`
  or `isr_emela_ll=True` (eMELA's ePDF is a per-leg `D(x, Q)`).

### `isr_nll` / `isr_emela_ll` / `isr_emela_{pert_order,fac_scheme,ren_scheme}`

NLL ISR via eMELA (BCFS arXiv:1911.12040) is the **production default
since 2026-05-29** (`isr_nll=True`).  Motivation: the LL+exp→NLL
cross-fit bias is ±22.4 MeV on m_W (see report §val-isr-cross),
~90× the FCC-ee σ(m_W)=0.25 MeV target.  The mutually-exclusive
`isr_emela_ll` (DGLAP-evolved BETA-scheme LL diagnostic) and the
LL+exp BETA baseline (`isr_nll=False`) remain available for paper-
closure tests that match BFS's own structure-function radiator
(e.g. BFS Tables 1-4).  When flipping `isr_nll` off→on, regenerate
the canonical templates via `condor/ww_templates/submit.py` +
`condor_submit` (15 jobs × 4 cores, ~5 min wall) or rerun
`compute_xsec_ww.py` locally.

The eMELA scheme knobs default to the production NLL central
DELTA + ALPMZ + α(M_Z) = 1/128.943 (see `alpha_em_isr` above); ALGMU
is the BFS-prescription diagnostic.  The renormalisation-scheme
variation ALPMZ ↔ ALGMU ↔ α(0) is the genuine NLL theory uncertainty
channel and is propagated via the `alpha_em_isr` PARAM_UNC nuisance.

An `isr_scale_factor` (ξ ∈ [0.5, 2] on Q = ξ √s and the LL log) is
implemented in `isr.py` for legacy investigation but intentionally not
exposed at card level: in DELTA + collinear-finite c^(1,fin) the σ̂ has
no μ_F dependence to cancel against D's evolution, so the variation
measures DGLAP-evolution stability rather than NLL truncation
(`scripts/investigations/nll_isr/scale_decouple_diagnostic.py`).

### `alpha_em_isr`

See the `alpha_em_isr` paragraph in PARAM_INPUTS above for the
production default; the PARAM_UNC entry holds the FCC-ee Δα(M_Z) = 4.7×10⁻⁸
(Riembau combined projection), profiled as the `aem_isr` nuisance.

### `diagnostic_bfs_coulomb_nlo`

Adds BFS NLO Coulomb via the standalone `BFSCorrections.delta_NLO` path.
**Double-counts** when `include_NLO_hard_decay=True`.  Only useful together with
`include_NLO_hard_decay=False` to reproduce BFS paper plots showing this piece in
isolation.

---

## Open work for sub-MeV precision

1. **RACOONWW grid above 170 GeV** — current calibration grid
   disagrees with BFS-era Born; replace before publishing plots
   outside [157, 165] GeV.

## CROSS_TERMS

`CROSS_TERMS = [("mass", "width")]` in `ww_default.py` declares the
POI pairs whose bilinear (cross-term) corner template is materialised.
The fit applies the residual non-multiplicative factor
`(1 + m_corner) / ((1 + m_a)(1 + m_b)) - 1` so the corner closure is
exact (linear-morph residual ~40 keV on m_W reduced to ~1 keV;
`scripts/investigations/bilinear_morph/closure_corner.py`).  Opt-in:
omit the attribute (WbWb) to recover pure linear morphing.
