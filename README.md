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
│   └── ww_default.py       - placeholder for WW
├── common/               Process-agnostic fit machinery
│   ├── parameters.py       - parameter bookkeeping (nominal / pseudo /
│                              variation tags per fit parameter)
│   ├── smearing.py         - Gaussian convolution with the beam-energy
│                              spectrum
│   ├── fit_core.py         - FitCore: read templates, smear, morph, build
│                              scenarios, chi2, init_minuit, nuisances
│   ├── plots.py            - shared plot decoration + the two top-level
│                              diagnostic plots (fit_scenario,
│                              parameter_variations)
│   ├── scans.py            - LS / BEC / BES / lumi / alphaS / Yukawa /
│                              yukawa-theory / width / scale / shift /
│                              chi2 / true-value scans
│   ├── systematics.py      - print_syst_table (text + LaTeX)
│   └── parallel.py         - fork-based scan dispatcher
├── process/
│   ├── wbwb/
│   │   ├── generator.py    - thin wrapper around xsec_calculator.xsec_calc
│   │   │                     (QQbar_threshold N3LO+ISR tt)
│   │   └── fit.py          - WbWbFit subclass with the SM-width hook
│   └── ww/                 - placeholders; do_scan raises NotImplementedError
├── scripts/
│   └── audit_scans.py      - state-isolation regression check
├── xsec_calculator/        - pybind11 wrappers around the C++ QQbar_threshold
│                             library (tt N3LO+ISR template producer)
├── utils_convert/          - pybind11 wrapper for PS↔MS mass conversion
├── doFit_wbwb.py           - WbWb entry script
└── doFit_ww.py             - WW entry script (placeholder)
```

## Quick start

```bash
source setup.sh           # sets LD_LIBRARY_PATH for the QQbar_threshold .so
python doFit_wbwb.py --systTable
```

This runs the default-scenario WbWb fit and prints the systematic table.

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

Each card is a plain Python module exposing top-level dicts/scalars:

```python
PARAMETERS = {
    "mass":   {"nominal": 171.5, "pseudo": 0.01, "variation": 0.03, "round_dec": 2},
    ...
}
PRIORS = {"alphas": {"default": 1e-4}, "lumi": {"uncorr": 1e-3, "corr": 5e-4, ...}, ...}
SCENARIO = {"scan_min": 340.0, "scan_max": 344.5, "scan_step": 0.5,
            "total_lumi": 0.41e6, "last_lumi": 2.65e6, "stat_inflation": 1.2}
INPUT_DIRS = {"nominal": "output_full", "BEC": "BEC_variations", ...}
THEORY_UNC = {"mass": 35.0, "width": 25.0, "yukawa": 10.0}
```

To run a variant fit, copy `cards/wbwb_default.py` to e.g.
`cards/wbwb_high_lumi.py`, edit the relevant fields, and import it from a
new entry script.

## Cross-section templates

The fit reads pre-computed cross-section templates from on-disk text files
of the form `N3LO_scan_PS_ISR_massX.XX_widthY.YY_yukawaZ.ZZ_asVarA.AAAA_scaleM…_scaleW….txt`,
one (ecm, xsec) per line. Templates are generated by the C++ extension
`xsec_calculator/xsec_calc` (a pybind11 wrapper around Andreas Maier's
`QQbar_threshold` library). To re-generate:

```bash
cd xsec_calculator
bash compile_calc.sh        # builds xsec_calc.cpython-*.so
python ../compute_xsec_parallel.py --ncores 8 --outdir ../output_full
```

The fit itself does not call do_scan; the entry scripts read existing
text files only.

## Physics conventions

- **Chi2 form**: `(d − f)ᵀ C⁻¹ (d − f)` with `f = xsec_nom · Π_i (1 + p_i · morph_i)`
  (multiplicative morphing — valid in the small-variation regime, which
  this fit lives in: |p_i| ~ O(1), |morph_i| ~ O(few %)). External
  constraints on αₛ (always) and yt (when `constrain_yukawa=True`) are
  added in quadrature.
- **Covariance** = stat (diagonal sqrt(N) inflated by `stat_inflation`) +
  uncorrelated lumi (diagonal) + correlated lumi (rank-1 outer-product).
  Cholesky-factored once per `init_minuit` for fast per-chi2-call solves.
- **Asimov data**: `pseudo_data_scenario` is the smeared lineshape sampled
  at the scan-list ecms with the pseudo-tag parameter values. The Yukawa /
  αₛ priors are centred on the pseudo values, so the projected uncertainty
  answers "what would we report if the true value were the pseudo value?".
- **BEC / BES nuisances**: per-bin uncorrelated nuisance (one per scan
  ecm) + one fully-correlated nuisance, both with Gaussian priors of
  width set via `card.PRIORS[*]`.
- **SM-width constraint** (WbWb only, `--SMwidth`): rewrites the width
  fit parameter from the mass via `Γₜ = Γ_ref + 0.027·(mₜ − mₜ_ref) +
  k·δ_th`, with `k` a floating "theory knob" given a Gaussian prior of
  width `input_uncert_SM_width` MeV. See `process/wbwb/fit.py`.

## Diagnostics

- `python scripts/audit_scans.py` — runs every scan helper in turn and
  confirms that it leaves `fit`'s state bit-identical. Use this after any
  change to scan code.
- `--systTable` is the most sensitive end-to-end check; the reproducibility
  baseline (work/doFit.py) is the same set of 11 rows.

## Validation status

The framework reproduces the legacy `work/doFit.py` bit-for-bit on:
- the systematic table (every row),
- `fit_results` stdout (fitted values, pulls, correlation matrix),
- `param_variations.png` (template-only plot).

Scan output values agree at 10⁻¹⁰–10⁻⁵ relative; the residual is migrad
hesse-convergence noise from the new path (`cho_factor` + vectorised
morph + scalar-attr-mutated rather than deep-copied scans). Plot PNGs
differ at the sub-pixel level along anti-aliased line edges. All
differences are well below physics resolution.

See commit history: every commit message ending in "BIT-IDENTICAL" or
"validation report" records the comparison numbers for that step.

## WW status

`doFit_ww.py`, `cards/ww_default.py`, `process/ww/{generator,fit}.py` are
placeholders. The generator stub raises `NotImplementedError` on
`do_scan`; the card has tentative numerical values that need tuning to
the actual WW scenario. To get the WW fit running:

1. Plug a real WW cross-section calculator into `process/ww/generator.WWGenerator.do_scan`
   (Whizard, RACOONWW, or a theory parameterisation — anything that
   writes the expected file format to disk).
2. Produce template files at the parameter grid implied by
   `cards/ww_default.py` (nominal + pseudo + per-parameter variation).
3. Tune the placeholder values in `cards/ww_default.py` (`BEAM_ENERGY_RES`,
   `PEAK_ECM`, scan grid, `PRIORS`, `THEORY_UNC`).

The chi2/Minuit/scan/syst-table machinery in `common/` works as-is —
nothing in there assumes WbWb.
