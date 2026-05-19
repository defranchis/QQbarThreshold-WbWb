# FCC-ee threshold-scan fit framework

A modular chi2/Minuit fit of the cross-section lineshape near production
thresholds at FCC-ee. Currently configured for **WbWb** (top-quark threshold,
~343 GeV) and scaffolded for **WW** (W-pair threshold, ~161 GeV).

The framework grew out of the monolithic `work/doFit.py` (see commit
history in this repository for the legacy reference); the goal of the
refactor was to split fit machinery from process-specific glue so that
different theory generators and different threshold processes can share
the same chi2/scans/systematic-table pipeline.

## Layout

```
WW_threshold/
├── cards/                Steering cards (plain Python modules)
│   ├── wbwb_default.py     - every magic number for the WbWb fit
│   └── ww_default.py       - WW threshold scan
├── framework/             All importable framework code (common + process)
│   ├── common/               Process-agnostic fit machinery
│   │   ├── parameters.py       - parameter bookkeeping (nominal / pseudo /
│   │                              variation tags per fit parameter)
│   │   ├── smearing.py         - Gaussian convolution with the beam-energy
│   │                              spectrum
│   │   ├── fit_core.py         - FitCore: read templates, smear, morph, build
│   │                              scenarios, chi2, init_minuit, nuisances
│   │   ├── plots.py            - shared plot decoration + the two top-level
│   │                              diagnostic plots (fit_scenario,
│   │                              parameter_variations)
│   │   ├── scans.py            - process-agnostic scans: LS / BEC / BES / lumi /
│   │                              alphaS / scale / shift / chi2 / true-value
│   │                              (one impact line per POI)
│   │   ├── systematics.py      - print_syst_table (text + LaTeX)
│   │   └── parallel.py         - fork-based scan dispatcher
│   └── process/
│       ├── wbwb/                 - WbWb process code (fit-side + calculation)
│       │   ├── generator.py        - thin wrapper around xsec_calculator.xsec_calc
│       │   │                         (QQbar_threshold N3LO+ISR tt)
│       │   ├── fit.py              - WbWbFit subclass: SM-width hook,
│       │   │                         constrain_yukawa property, scenario validator,
│       │   │                         pseudodata-tag / pull / print-extra / scannable
│       │   │                         POI overrides
│       │   ├── scans.py            - WbWb-only scans: width, yukawa-constraint,
│       │   │                         yukawa-theory, lumi-yukawa-ratio,
│       │   │                         scale-vars-yukawa
│       │   └── xsec_calculator/    - pybind11 wrappers around the C++ QQbar_threshold
│       │                             library (tt N3LO+ISR template producer +
│       │                             PS↔MS mass scheme conversion)
│       └── ww/                   - WW process code (fit-side + calculation)
│           ├── generator.py        - WWGenerator + .from_card factory + chain-
│           │                         kwargs helpers (single source of truth)
│           ├── fit.py              - WWFit (placeholder; inherits FitCore)
│           └── xsec_calculator/    - WW BFS-EFT cross-section calculation (Python)
│               ├── bfs_eft.py        - BFS-EFT N^(3/2)LO Born (eq. 17+33+37+39 of
│               │                       arXiv:0707.0773) + h4-h7 single-resonant +
│               │                       NLO loops (HSC + Coulomb_NLO + EW-decay)
│               │                       + Whizard 4f Born anchor (sec. 6.2) + δ_QCD
│               ├── eft_xsec.py       - σ partonic entry points + Fadin-Khoze-Martin
│               │                       Coulomb K-factor + RACOONWW spline above 170 GeV
│               └── isr.py            - LL+exp BETA-scheme ISR (LEP2 YR eq. 67),
│                                       single-conv default + 2-leg per BFS eq. 71
├── output_xsec/            - pre-computed σ-template input files for the
│   ├── wbwb/                 fit (gitignored; one subdir per process, each
│   └── ww/                   with {nominal,scale_vars,BEC,sw2,pseudo,…})
├── scripts/
│   ├── _audit_common.py    - shared build_fit + SCAN_SPECS for the harness
│   ├── audit_scans.py      - state-isolation regression check
│   └── scan_dump.py        - per-scan plot-data dumper (numerical regression check)
├── legacy_code/            - pre-refactor monolithic doFit.py + its driver
│                             scripts; archived for reference, not on the
│                             import path (see legacy_code/README.md)
├── allFits_wbwb.sh         - full WbWb diagnostic suite (every scan + syst table)
├── allFits_ww.sh           - same for WW
├── compute_xsec_ww.py      - WW template-generation driver (nominal + BEC vars)
├── setup.sh                - prepend the QQbar_threshold .so directory to
│                             LIBRARY_PATH / LD_LIBRARY_PATH / CPLUS_INCLUDE_PATH
├── doFit_wbwb.py           - WbWb entry script
└── doFit_ww.py             - WW entry script
```

## Quick start

```bash
source setup.sh                   # WbWb only; WW doesn't need it
python3 doFit_wbwb.py --systTable
```

Or `./allFits_wbwb.sh` — runs the full diagnostic suite and sources `setup.sh` itself.

## Running scans

Multiple scans can be requested in one invocation:

```bash
python doFit_wbwb.py --LSscan --BECscans --BESscans --lumiscans \
                     --alphaSscan --chi2scans --systTable
```

By default the scans run in parallel (up to 6 worker processes); pass
`--parallel 1` to force sequential. The systematic table always runs
sequentially after the scans complete.

### Scan flags

| Flag | Scan | Notes |
|---|---|---|
| `--LSscan` | beam-energy resolution → stat uncertainty | |
| `--BECscans` | BEC prior strength → impact | |
| `--BESscans` | BES prior strength → impact | |
| `--lumiscans` | luminosity uncertainty → impact | |
| `--alphaSscan` | αₛ prior strength → impact | also runs the Yukawa-constraint scan unless `--fitYukawa` |
| `--yukawaThScan` | above-threshold xsec shift → fitted yt | requires `--lastecm` |
| `--widthscan` | SM-width theory uncertainty → mass | requires `--SMwidth` |
| `--scaleVarsScan` | renormalisation scale variation | requires `--scaleVars` |
| `--chi2scans` | 1D + 2D chi2 profile scans | runs 51×51 profile fits per pair (~7 s) |
| `--truevaluescan` | iterate over pseudo-data templates | |
| `--shiftScan` | uniform shift of the scan ecm grid | needs templates outside the original ecm range |

### Common flags

| Flag | Effect |
|---|---|
| `--pseudo` | use a toy pseudo-experiment (default: Asimov) |
| `--SMwidth` | constrain Γₜ to the SM/QCD prediction |
| `--fitYukawa` | float Yukawa (default: constrained) |
| `--addSw2` | add sw2 nuisance |
| `--lastecm` | include the 365 GeV point for the Yukawa lever |
| `--sameNevts` | distribute lumi to equalise N_events per point |
| `--scaleVars` | read scale-variation templates |
| `--BECnuisances` / `--BESnuisances` | add the BEC / BES nuisance bins explicitly |
| `--oneS` | use the 1S mass scheme |
| `--twopoints` | coarse two-point scan |
| `--inputDir` | override the nominal input directory |
| `--noPlots` | skip the diagnostic plots (does not skip scan plots) |
| `--parallel N` | up to N worker processes (default 6) |

## Steering cards

Each card is a plain Python module exposing top-level dicts/scalars.
**Numbers** live in `PRIORS`; **metadata** (how each number is consumed)
lives in `SYSTEMATICS`. The two are physically separate so the same
value never lives in two places.

```python
# Physics parameters of interest — derived from framework.process.wbwb.xsec_calculator.parameter_def
# so the card and the C++ template generator share one source of truth.
PARAMETERS = {"mass": {...}, "width": {...}, "yukawa": {...}, "alphas": {...}}
PARAMETERS_1S = {...}                                # alternate mass scheme
ORDER         = 3                                    # 0=LO ... 3=N3LO
RENORM_SCALES = {"mass": 80.0, "width": 350.0, ...}

# Scan grid + integrated luminosity. ``coarse_scan`` is the comparison
# grid used by scan_true_value; ``true_value_pivot`` names the POI whose
# true value is parsed from the pseudodata filename.
SCENARIO          = {"scan_min": 340.0, "scan_max": 344.5, "scan_step": 0.5,
                     "total_lumi": 0.41e6, "last_lumi": 2.65e6,
                     "stat_inflation": 1.2,
                     "coarse_scan": {"scan_min": 340.5, "scan_max": 345.0,
                                     "scan_step": 1.0, "lumi_factor": 1/2**0.5},
                     "true_value_pivot": "mass"}
BEAM_ENERGY_RES   = 0.186                            # %; 0 disables smearing
PEAK_ECM, LAST_ECM = 345.0, 365.0

# Template-generation-frozen variation magnitudes (must match the C++ scan).
INPUT_VAR = {"BEC": 10.0, "BES": 0.1, "sw2": 2.5e-6}

# ALL prior magnitudes — single source of truth, no duplication elsewhere.
PRIORS = {
    "alphas":   1.0e-4,                              # constraint sigma
    "yukawa":   0.03,                                # constraint sigma
    "BEC":      {"uncorr": 5.0,    "corr": 2.5},     # binned nuisance (MeV)
    "BES":      {"uncorr": 0.01,   "corr": 5.0e-3},  # binned nuisance
    "sw2":      2.5e-6,                              # global nuisance prior
    "lumi":     {"uncorr": 1.0e-3, "corr": 5.0e-4},  # cov-matrix entry
    "SM_width": 5.0,                                 # WbWb-specific theory band on Γ_t (MeV)
}

# Metadata for each systematic in PRIORS that goes through chi2.
# type ∈ {"constraint", "binned", "global"}.
# Lumi (cov-matrix) and SM_width (WbWb physical_fit_params hook) have
# no SYSTEMATICS entry — FitCore reads them directly from PRIORS.
SYSTEMATICS = {
    "alphas": {"type": "constraint", "always_on": True},
    "yukawa": {"type": "constraint", "always_on": False},
    "BEC":    {"type": "binned",
               "source": {"kind": "template_dir",
                          "var_subdir": True, "snap_to_grid": True}},
    "BES":    {"type": "binned", "source": {"kind": "smear_shift"}},
    "sw2":    {"type": "global",
               "source": {"kind": "template_dir"}},
}

# Row order in the printed / LaTeX systematic table. "BEC"/"BES"/"lumi"
# are shorthand for both ${name}_uncorr and ${name}_corr in that order.
SYST_TABLE_ORDER = ["alphas", "yukawa", "sw2", "BES", "BEC", "lumi"]

# Parameters of interest displayed by the syst-table machinery.
# Column order = dict insertion order. `relative=True` divides the raw
# uncertainty by the fitted central value (fractional uncertainty); the
# default is absolute scaling.
POI_DISPLAY = {
    "mass":   {"symbol": r"m_t",      "unit": "MeV", "scale": 1000},
    "width":  {"symbol": r"\Gamma_t", "unit": "MeV", "scale": 1000},
    "yukawa": {"symbol": r"y_t",      "unit": "%",   "scale": 100, "relative": True},
}

INPUT_DIRS = {"nominal": "output_xsec/wbwb/nominal", "BEC": "output_xsec/wbwb/BEC", ...}
THEORY_UNC = {"mass": 35.0, "width": 25.0, "yukawa": 10.0}
```

To run a variant fit, copy `cards/wbwb_default.py` to e.g.
`cards/wbwb_high_lumi.py`, edit the relevant fields, and import it from a
new entry script.

### Extending to a new process (or adding a new systematic)

Both reduce to **card-only edits**; no FitCore changes.

**Adding a new 1-D Gaussian constraint** (e.g. sin²θ_W for WW):

```python
# in cards/ww_default.py
PRIORS["sin2thetaW"] = 1.0e-5                        # sigma
SYSTEMATICS["sin2thetaW"] = {"type": "constraint", "always_on": True}
```

The parameter name must also appear in `PARAMETERS` so it's a known fit
parameter. `FitCore.chi2` picks it up automatically; the syst table
includes it (alphabetically appended if not in `SYST_TABLE_ORDER`).

**Adding a new binned nuisance** (per-ECM uncorrelated + shared
correlated, à la BEC/BES):

```python
PRIORS["new_binned"] = {"uncorr": 1.0, "corr": 0.5}
SYSTEMATICS["new_binned"] = {
    "type": "binned",
    "source": {"kind": "template_dir",
               "var_subdir": True, "snap_to_grid": True},
}
INPUT_DIRS["new_binned"] = "new_binned_variations"   # template dir
INPUT_VAR["new_binned"] = 10.0                       # template-frozen
```

Plus call `fit.add_binned_nuisance("new_binned")` in the entry script
(or wire it into a CLI flag). The chi² loop, `_morph_one` dispatcher,
nuisance-prior helper, and syst-table machinery all pick it up.
`template_dir` sources read their path from `INPUT_DIRS[name]` — one
mapping for all on-disk template directories, including the nominal /
scale-vars / pseudo ones.

**Adding a new global nuisance** is analogous with `"type": "global"`
and a scalar `PRIORS["x"] = ...`.

**Adding a new template source mechanism** (something other than
`template_dir` or `smear_shift`): add a new branch in
`FitCore._morph_one` keyed off `source["kind"]`. That's the only place
that needs to know about the new kind.

**Adding (or dropping) a parameter of interest** (e.g. WW's `mW`/`ΓW`,
or a hypothetical fourth POI on top of the WbWb default):

```python
POI_DISPLAY["my_poi"] = {"symbol": r"\mu", "unit": "MeV", "scale": 1000}
THEORY_UNC["my_poi"]  = 5.0   # MeV, optional
```

Both the syst-table (`print_syst_table`) and the scan-impact plot
helpers in `common/scans.py` pick it up automatically:

* `print_syst_table` adds one column per POI in the printed table and
  LaTeX file.
* `scan_alphas` / `scan_bec` / `scan_bes` / `scan_lumi` /
  `scan_beam_resolution` / `scan_scale_vars` / `scan_true_value` /
  `scan_shift` / `scan_chi2` iterate `fit.tracked_pois()` and emit one
  line per POI per panel. POIs sharing a unit go on the same panel
  (e.g. `mass + width` in MeV); separate units get their own panels
  (e.g. `yukawa` in %). Filenames follow the convention
  `uncert_<pois>_vs_<scan-axis>` — recovers the legacy
  `uncert_mass_width_vs_alphas` on WbWb-default.

POIs filtered out at runtime: those not in `PARAMETERS` (so not
actually fit) and those currently treated as constrained nuisances
(e.g. `yukawa` under `--fitYukawa` not set). Use `"relative": True`
for POIs whose uncertainty is naturally quoted as a fraction of the
central value (divide-by-central applied at display time).

WbWb-specific scans (yukawa-constraint sweep, yukawa-theory shift,
the legacy yukawa-vs-lumi-ratio panel, the yukawa-shift-vs-scale
panel) live in `process/wbwb/scans.py`. They consume the
`fit.constrain_yukawa` property + the `_constraints["yukawa"]` store —
both defined on `WbWbFit`, not `FitCore`. Adding analogous scans for
a third process means a new `process/X/scans.py` and entry-script
imports; `common/` stays untouched.

**Process subclass hooks** (in `common/fit_core.py`): for cross-POI
relations or process-specific output, subclass `FitCore` and override
any of:

* `physical_fit_params(params)` — resolve cross-parameter relations
  inside chi² (WbWb uses this for the SM-width hook).
* `_validate_scenario(add_last_ecm)` — raise if the requested
  scenario combination doesn't make physical sense.
* `_select_pseudodata_tag()` — which template tag to use as the
  pseudodata reference. Default `"pseudodata"`; WbWb returns
  `"mass_var"` under SM_width.
* `_print_param_extras(name, val)` — print annotation lines after the
  `Fitted <name>` line (WbWb uses this for SM-width theory parameter
  + Yukawa-constraint info).
* `_pull_for(name, val)` — custom pull formula for parameters whose
  semantics differ from the standard `val - pseudodata` (WbWb uses
  it for the SM-width theory knob).
* `is_scannable_poi(name)` — whether a POI should be included in scan
  plots. Default `True`; WbWb returns `False` for `"width"` under
  SM_width (it's a constrained theory knob, not a free POI to scan).
* `stat_breakdown_default()` — whether `print_syst_table` should
  compute the per-POI stat breakdown. Default `True`; WbWb returns
  `not self.constrain_yukawa`.

All hooks have sensible defaults in `FitCore`, so a no-op subclass
(`class WWFit(FitCore): pass`) Just Works.

## Cross-section templates

The fit reads pre-computed cross-section templates from on-disk text files
of the form `N3LO_scan_PS_ISR_massX.XX_widthY.YY_yukawaZ.ZZ_asVarA.AAAA_scaleM…_scaleW….txt`,
one (ecm, xsec) per line. Templates are generated by the C++ extension
`framework/process/wbwb/xsec_calculator/xsec_calc` (a pybind11 wrapper around Andreas Maier's
`QQbar_threshold` library). To re-generate:

```bash
cd framework/process/wbwb/xsec_calculator
bash compile_calc.sh        # builds xsec_calc.cpython-*.so
python ../../../../compute_xsec_wbwb.py --ncores 8 --outdir ../../../../output_xsec/wbwb/nominal
```

The fit itself does not call do_scan; the entry scripts read existing
text files only.

## Physics conventions

- **Chi2 form**: `(d − f)ᵀ C⁻¹ (d − f)` with `f = xsec_nom · Π_i (1 + p_i · morph_i)`
  (multiplicative morphing — valid in the small-variation regime, which
  this fit lives in: |p_i| ~ O(1), |morph_i| ~ O(few %)). Gaussian
  constraints declared in `card.SYSTEMATICS` (today: αₛ unconditionally,
  yukawa when `WbWbFit.constrain_yukawa=True`) are added in quadrature.
- **Constraint centring**: by default the αₛ / Yukawa priors are centred
  on the pseudodata "true" value (Asimov-self-consistent: data, fit, and
  constraint all sit at the pseudo point, no bias at the minimum).
  Override per-entry via `card.SYSTEMATICS["alphas"]["center"]` /
  `["yukawa"]["center"]` for SM-centred or bias-study runs.
- **Covariance** = stat (diagonal √N inflated by `stat_inflation`) +
  uncorrelated lumi (diagonal) + correlated lumi (rank-1 outer-product).
  Cholesky-factored once per `init_minuit` for fast per-chi²-call solves.
- **BEC / BES binned nuisances**: per-bin uncorrelated nuisance (one per
  scan ecm) + one fully-correlated nuisance, both with Gaussian priors
  of width set via `card.PRIORS["BEC"]` / `card.PRIORS["BES"]`.
- **BEC template loader**: BEC variations live in subdirectories named
  `scan_p{var}` / `scan_m{var}` (one per ±MeV magnitude); the loader
  snaps the shifted ECMs back to the nominal 0.1-GeV grid. Constraint:
  variation magnitude ≤ 40 MeV — beyond that banker's rounding at .05
  would alias onto the next ECM bin. Encoded in
  `SYSTEMATICS["BEC"]["source"]` via `var_subdir` / `snap_to_grid` flags.
- **SM-width constraint** (WbWb only, `--SMwidth`): rewrites the width
  fit parameter from the mass via `Γₜ = Γ_ref + 0.027·(mₜ − mₜ_ref) +
  k·δ_th`, with `k` a floating "theory knob" given a Gaussian prior of
  width `card.PRIORS["SM_width"]` MeV. See
  `process/wbwb/fit.py:_width_n3lo_local_linearisation`.
- **Pseudo-data RNG**: `--pseudo` draws noise from a local
  `numpy.random.default_rng(42)` seeded once at `FitCore.__init__` and
  advanced per `create_scenario` call. The legacy global-`np.random.seed(42)`
  behaviour (same noise on every call, including across scan iterations)
  is available behind `--legacyPseudoRng` for byte-reproducing old
  --pseudo runs.

## Diagnostics

Two regression harnesses live under `scripts/`:

- `python scripts/audit_scans.py` — for each scan helper, builds a fresh
  fit, snapshots ~20 inspected fields of `fit`, runs the scan, and
  diffs the snapshot against itself. Catches state-isolation regressions
  (a scan accidentally mutating `fit`'s state). Use after touching any
  scan code.
- `python scripts/scan_dump.py` — monkey-patches `plt.plot` so every
  `(x, y)` pair fed into matplotlib is captured. Run on both sides of a
  change (`git stash` between dumps) and `diff` the captures to check
  numerical equivalence of every scan + `print_syst_table`. The
  expected baseline is **zero-line diff** after any code reorganisation
  that doesn't mean to change physics output.

The chained `--systTable` pipeline is the most end-to-end check; the
output `systematics_table.tex` should be byte-identical (modulo
intentional label / casing changes) across refactors.

## WW status

The WW threshold fit is wired end-to-end and uses the **full BFS-EFT
chain** as the default partonic cross section. Quick start:

```bash
python3 compute_xsec_ww.py            # generate nominal + BEC-variation templates
python3 doFit_ww.py --systTable       # canonical fit with full systematics table
python3 -m scripts.validate_bfs_nlo   # validate against BFS Tables 1+2+3+4
python3 -m scripts.plot_ww_diagnostics  # diagnostic plot set
```

Cross-section pipeline (in `process/ww/`):

* `bfs_eft.py` — BFS-EFT N^(3/2)LO Born expansion from arXiv:0707.0773:
  eq. (17) σ_LR^(0), eq. (33) σ^(1)_pot, eq. (37+38+appendix A)
  σ^(1/2) including h1–h3 *and* h4–h7 single-resonant, eq. (39+40)
  σ^(3/2),a. Plus NLO loops (HSC eq. 54-56, NLO Coulomb eq. 62, EW
  decay eq. 60), δ_QCD multiplier (eq. delta_qcd), and the
  Whizard 4f Born anchor (BFS sec. 6.2 prescription). All knobs
  card-driven and on by default; see `cards/ww_default.py` NLO_CONFIG.
* `eft_xsec.py` — partonic entry points `sigma_partonic_munuqq` and
  `sigma_WW_partonic`. Coulomb K-factor (Fadin-Khoze-Martin +
  Bardin-Riemann α²). Switches to a RACOONWW CC03 spline above
  √s = 170 GeV (only the `--lastecm` 240-GeV point uses this region).
* `isr.py` — LL+exp BETA-scheme ISR per LEP2 YR Beenakker eq. (67).
  Two formally-equivalent implementations: `single_conv` (LEP2 YR α→2α
  1D form, default) and `2leg` (BFS eq. 71 double convolution); they
  agree to <0.1 %. NLL upgrade (analytic Skrzypek-Jadach or eMELA) is
  the next step — removes BFS's own ~31 MeV ISR systematic on m_W.
* `generator.py` — `WWGenerator.from_card(card)` factory + `.do_scan`
  template writer + chain-kwargs helpers (`partonic_kwargs_from_card`,
  `observed_kwargs_from_card`) as the single source of truth for
  card-driven configuration.

Validation (`scripts/validate_bfs_nlo.py`, 10 scenarios A–J):
- Scenario A: BFS Table 1 (LO width, no BR corr) closes to **4-5 digits**.
- Scenario B: BFS Table 2 (NLO+QCD width, BR corr) closes to 0.1 %
  in the scan window, 0.8 % at 155 GeV (EFT-validity edge).
- Scenario F: full NLO chain vs BFS Table 4 σ_obs — 0.4-1.0 % deficit,
  driven by BFS using Whizard's own ISR for the Born vs my LEP2 YR
  LL+exp (the same ~31 MeV NLL-ISR systematic BFS itself quotes).
- Scenario I: Whizard-anchor closure to 4-5 digits at Table 1
  reference points.

What's NOT yet in the calculation (priority order for sub-MeV m_W):

1. **BFS dominant NNLO** ([arXiv:0807.0102](https://arxiv.org/abs/0807.0102))
   — σ^(2) Born + NNLO Coulomb beyond eq. 62. Expected ~0.2-0.5 % on σ.
2. **NLL ISR** — analytic Skrzypek-Jadach or eMELA
   (Bertone-Cacciari-Frixione-Stagnitto). Removes the residual 0.7-1.0 %
   ISR deficit to BFS and the ~31 MeV ISR systematic on m_W.
3. **Finer Whizard anchor grid** — currently uses only the 6 √s × 2 Γ_W
   reference points from BFS Tables 1+2; a denser grid (run Whizard
   ourselves) would remove the 168-GeV dσ/dΓ_W bump and shrink the
   Born-side ~0.3 MeV systematic.

A parallel implementation based entirely on established generators
(WHIZARD / Recola / MoCaNLO) is planned as a second `*Generator`
class slotting into the same `do_scan` / `file_name` contract — needed
for publication-level cross-validation.
