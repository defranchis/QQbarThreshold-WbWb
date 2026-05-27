# Cards — physics rationale

This file documents the non-obvious choices in `ww_default.py`.
For parameter provenance and BFS equation cross-references see the BFS paper
arXiv:0707.0773 (NLO) and arXiv:0807.0102 (NNLO).

---

## PARAM_INPUTS / PARAM_UNC

### PARAM_INPUTS — production theory inputs

`alpha_s_MW`, `m_t`, `M_H`, `M_Z` enter the NLO hard-matching coefficient
`c_p,LR^(1,fin)` and the BFS EW couplings ξ(s), χ(s), sin²θ_W.  Production
defaults are PDG 2024 values.

The BFS reference set used to produce Tables 1–4 is different:
`m_t=174.2, M_H=115 (pre-Higgs-discovery), M_Z=91.188`.  These are kept as
`M_*_BFS_REF` constants in `process/ww/xsec_calculator/eft_xsec.py`; all closure
validation scripts pass them explicitly.

`alpha_em` and `alpha_em_isr` are α_em overrides.  `None` (default) means:

- **σ chain (`alpha_em`)** → derived `α_Gμ(m_W, M_Z) = √2 G_F m_W² sin²θ_W / π`
  (tree-level G_μ scheme; `G_F = 1.1663787e-5` in `eft_xsec.py`).  Applies
  to Born + NLO + NNLO via `alpha_Gmu(mW, MZ, override=alpha_em)`.
- **ISR (`alpha_em_isr`)** → `α_Gμ(M_W_BFS_REF) ≈ 1/132.1`, fixed across the
  fit (BFS prescription, arXiv:0707.0773 line 2514 — avoids fictitious m_W
  dependence in the LL β_e kernel).

Set to a float to inject a user-supplied value (used for the
`PARAM_UNC.{alpha_em, alpha_em_isr}` nuisances).

### PARAM_UNC — parametric uncertainties (FCC-ee projections)

Values are the FCC-ee Feasibility Study Report Vol. 1 projections
(arXiv:2505.00272, Table 2 of §1.4) — stat ⊕ syst, where available.
**Not yet wired** into the fit; no SYSTEMATICS entry reads them.
Provisional values (all absolute):

| Key            | Value     | FCC-ee projection (FSR Vol. 1)              | Source            |
|----------------|-----------|----------------------------------------------|-------------------|
| `m_t`          | 7 MeV     | stat 4.2 ⊕ syst 4.9 MeV (FSR ≈ 6.5; rounded) | tt̄ threshold scan |
| `M_H`          | 5 MeV     | ZH recoil ≈ 4 MeV (FSR §4.3; rounded)        | Higgs run         |
| `M_Z`          | 0.1 MeV   | stat 4 keV ⊕ syst 100 keV                    | Z line-shape scan |
| `alpha_s`      | 1.0e-4    | (stat 0.1 ⊕ syst 1.0) × 10⁻⁴                | combined Z fit    |
| `alpha_em`     | 2.4e-7    | δα abs from δα⁻¹ ≈ 4×10⁻³ at M_Z²            | A_FB^μμ off-peak  |
| `alpha_em_isr` | 1.0e-7    | scheme/scale variation (not direct FCC-ee obs) | held conservative |

Notes:

- α_em is split into two independent nuisances so the ISR-side scheme
  uncertainty decorrelates from the rest of the σ chain. The σ-chain
  value tracks the FSR Table 2 conservative projection from A_FB^μμ
  off-peak. The more aggressive Riembau-method projection (forward
  dilepton ratios e⁻/μ⁻ + e⁻/e⁺, arXiv:2501.05508) is ~0.6×10⁻⁵
  relative → δα ≈ 5×10⁻⁸ absolute, ~5× tighter than the FSR value
  used here.
- `alpha_em_isr` reflects an *ISR-scheme* variation (α(0) ↔ α_Gμ(M_W) ↔
  α(M_Z) — see "PARAM_INPUTS"). FCC-ee does not directly measure this,
  so the value is a held-conservative scheme uncertainty rather than a
  parametric one.
- BR_W_* uncertainties are deliberately not in PARAM_UNC: in
  `pdg-constant` BR mode the BR is a measured constant, so its
  uncertainty is an analysis-level systematic rather than a
  theory-input nuisance. The W BR is also measured at FCC-ee but
  enters σ via the BR normalisation, not via the BFS partonic chain.

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
- `"2leg"`: full per-leg double-convolution; natural starting point for the NLL ISR
  upgrade.

### `alpha_em_isr`

`None` → α_Gμ(M_W_BFS_REF) ≈ 1/132.1, fixed across the fit (BFS prescription —
avoids fictitious m_W dependence in the ISR kernel).

Override with a float for a scheme-variation systematic.  A running-α option
`alpha_em_running(Q)` is planned as part of the NLL ISR upgrade.

### `diagnostic_bfs_coulomb_nlo`

Adds BFS NLO Coulomb via the standalone `BFSCorrections.delta_NLO` path.
**Double-counts** when `include_NLO_hard_decay=True`.  Only useful together with
`include_NLO_hard_decay=False` to reproduce BFS paper plots showing this piece in
isolation.

---

## Open work for sub-MeV precision

1. **NLL ISR** — analytic Skrzypek-Jadach or BCFS (arXiv:1911.12040).  Closes the
   ±0.5% ISR residual to BFS Table 3.  Largest remaining theory systematic.
2. **RACOONWW grid above 170 GeV** — current calibration grid disagrees with BFS-era
   Born; replace before publishing plots outside [157, 165] GeV.
