"""Default steering card for the WbWb threshold fit.

Every numerical knob that used to live as a module-level constant in the
original `doFit.py` (or as a hardcoded literal inside the `fit` class /
`main()` / `parameters` class) is collected here, grouped by topic.

Cards are plain Python modules: import this module from the entry script,
copy / override individual fields as needed (the entry script may also
expose argparse overrides for the most common ones).

The physics-parameters block (PARAMETERS / PARAMETERS_1S / ORDER /
RENORM_SCALES) is derived from `process.wbwb.xsec_calculator.parameter_def.parameters`
— that class is the single source of truth for the values the C++
template generator (`compute_xsec_wbwb.py`) consumes. Card-only knobs
(`round_dec`, the alt-scale split in `RENORM_SCALES`) are added on top.
"""

from framework.process.wbwb.xsec_calculator.parameter_def import parameters as _ParameterDef

# ---------------------------------------------------------------------------
# Physics: parameters of interest (single source of truth: parameter_def)
# ---------------------------------------------------------------------------
# For each parameter:
#   nominal   = central value used to compute the reference template
#   pseudo    = offset applied to build the "true" pseudodata point
#   variation = offset of the variation template used to morph the lineshape
#   round_dec = decimals to round to in file-name tags (card-only knob)
_p_ps  = _ParameterDef()                         # PS-scheme nominal
_p_1S  = _ParameterDef(oneS_mass=True)           # 1S-scheme nominal
_p_alt = _ParameterDef(do_scale_vars=True)       # alt mass/width scales + scale_vars list

_ROUND_DEC = {"mass": 2, "width": 2, "yukawa": 2, "alphas": 4}


def _from_pd(p):
    return {
        name: {"nominal":   getattr(p, name),
               "pseudo":    getattr(p, f"{name}_pseudo"),
               "variation": getattr(p, f"{name}_var"),
               "round_dec": _ROUND_DEC[name]}
        for name in p.params
    }


PARAMETERS    = _from_pd(_p_ps)
PARAMETERS_1S = _from_pd(_p_1S)

# ---------------------------------------------------------------------------
# Theory / generator settings (also from parameter_def)
# ---------------------------------------------------------------------------
ORDER = _p_ps.order                              # 0=LO ... 3=N3LO
MASS_SCHEME = "PS"                               # "PS" or "1S" — card-only knob
RENORM_SCALES = {
    "mass":  float(_p_ps.mass_scale),
    "width": float(_p_ps.width_scale),
    "alt":   {"mass": float(_p_alt.mass_scale),  # used when scale-variation files exist
              "width": float(_p_alt.width_scale)},
    "vars":  [float(s) for s in _p_alt.scale_vars],
}

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
BEAM_ENERGY_RES = 0.186     # % per beam — set to 0 to disable smearing entirely
PEAK_ECM = 345.0            # GeV, peak used in Gaussian-smearing kernel width
LAST_ECM = 365.0            # GeV, optional point above threshold (Yukawa lever)

SCENARIO = {
    "scan_min":   340.0,
    "scan_max":   344.5,
    "scan_step":  0.5,
    "total_lumi":     0.41e6,   # /pb, full integrated lumi for the scan
    "last_lumi":      2.65e6,   # /pb, lumi at the LAST_ECM point
    # Inflate the per-point stat uncertainty (so the stat cov by 1.44x) to
    # account for selection-efficiency / b-tag / single-top-contamination
    # effects measured in dedicated detector-level studies for the WbWb
    # final state.
    "stat_inflation": 1.2,
    # Coarse-comparison ECM grid used by scan_true_value (one half the
    # number of points, ~lumi_factor of the original per-point lumi).
    "coarse_scan": {
        "scan_min": 340.5, "scan_max": 345.0, "scan_step": 1.0,
        "lumi_factor": 1.0 / 2 ** 0.5,
    },
    # POI whose true value is parsed from each pseudodata filename (e.g.
    # ``..._mass171.50_...``) and used as the x-axis of scan_true_value.
    "true_value_pivot": "mass",
}

# Alternative coarse two-point scan (selected by --twopoints in the entry script)
SCENARIO_TWOPOINTS = {
    "scan_min":   342.0,
    "scan_max":   344.0,
    "scan_step":  1.5,
    "total_lumi": SCENARIO["total_lumi"] / 100,
    "last_lumi":  SCENARIO["last_lumi"],
    "stat_inflation": SCENARIO["stat_inflation"],
}

# ---------------------------------------------------------------------------
# Template-variation sizes used to build the morphing templates
# (must match what the C++ scan was actually run with)
# ---------------------------------------------------------------------------
INPUT_VAR = {
    "BEC":  10.0,    # MeV   - beam energy calibration shift baked into output_xsec/wbwb/BEC/
    "BES":  0.1,     # 10%  - BES variation baked into the BES morph template
    "sw2":  2.5e-6,  #        sw2 shift baked into output_xsec/wbwb/sw2/
    "lumi": 0.01,    # 1%   - fit-param unit for the lumi nuisance (LUMI_MODE="nuisance")
}

# ---------------------------------------------------------------------------
# Lumi-uncertainty treatment.
#   "cov":      σ_lumi enters the data covariance matrix (Σ_ij ∝ σ_th σ_th).
#   "nuisance": σ_lumi is floated as a binned nuisance (one per-ECM-bin
#               parameter + one fully-correlated parameter, exactly like
#               BES/BEC). FitCore auto-activates the nuisance in
#               ``init_scenario``; ``card.PRIORS["lumi"]`` provides the
#               Gaussian priors in either mode. Equivalent to leading order
#               in the prior; rebuilds analytically match cov mode for the
#               threshold-only scan and after per-bin morph scaling when
#               ``add_last_ecm=True``.
# ---------------------------------------------------------------------------
LUMI_MODE = "nuisance"   # "nuisance" | "cov"

# ---------------------------------------------------------------------------
# Priors / systematics schema (numbers in PRIORS, metadata in SYSTEMATICS)
# ---------------------------------------------------------------------------
# PRIORS entry shapes:
#   * scalar          → 1-D Gaussian-constraint sigma (alphas, yukawa,
#                       sw2, SM_width).
#   * {uncorr, corr}  → binned nuisance with per-bin + fully-correlated
#                       components (BEC, BES) or lumi cov-matrix inputs.
# lumi feeds FitCore._build_cov; SM_width feeds WbWbFit.physical_fit_params
# under --SMwidth — neither gets a SYSTEMATICS entry.
PRIORS = {
    "alphas":   1.0e-4,
    "yukawa":   0.03,
    "BEC":      {"uncorr": 5.0,    "corr": 2.5},        # MeV
    "BES":      {"uncorr": 0.01,   "corr": 5.0e-3},
    "sw2":      2.5e-6,
    "lumi":     {"uncorr": 1.0e-3, "corr": 5.0e-4},
    "SM_width": 5.0,                                    # MeV, arXiv:2309.01937
}

# SYSTEMATICS schema:
#   type:      "constraint" | "binned" | "global"
#   always_on: constraints only — False means an entry-script flag
#              decides (today: --fitYukawa gates Yukawa).
#   center:    constraints, optional — defaults to pseudodata "true" value.
#   source:    binned + global — { kind: "template_dir" | "smear_shift", ... }.
#              template_dir loads INPUT_DIRS[name]; var_subdir + snap_to_grid
#              are BEC-specific quirks of the C++ scan output.
#              smear_shift re-smears nominal with bes*(1 + INPUT_VAR[name]).
SYSTEMATICS = {
    "alphas": {"type": "constraint", "always_on": True},
    "yukawa": {"type": "constraint", "always_on": False},
    "BEC":    {"type": "binned",
               "source": {"kind": "template_dir",
                          "var_subdir": True, "snap_to_grid": True}},
    "BES":    {"type": "binned",
               "source": {"kind": "smear_shift"}},
    "sw2":    {"type": "global",
               "source": {"kind": "template_dir"}},
}

# Canonical row order in the systematics table — looked up by
# ``systematics.systematic_list`` to keep the printed / LaTeX-written
# layout stable. Binned-nuisance / lumi entries use shorthand: ``"BEC"``
# expands to ``"BEC_uncorr"`` + ``"BEC_corr"`` (in that order); same for
# ``"BES"``, ``"lumi"``. Any active systematic not listed here is
# appended alphabetically at the end.
SYST_TABLE_ORDER = ["alphas", "yukawa", "sw2", "BES", "BEC", "lumi"]

# ---------------------------------------------------------------------------
# Parameters of interest displayed by the syst-table machinery.
# ---------------------------------------------------------------------------
# Each entry: ``scale`` rescales the raw uncertainty into ``unit``;
# ``relative=True`` additionally divides by the fitted central value
# (used for fractional uncertainties like Yukawa %). Iteration order
# fixes column order. Constraint-active POIs (e.g. yukawa with
# --fitYukawa not set) are filtered out at runtime — they are nuisances,
# not POIs to track.
POI_DISPLAY = {
    "mass":   {"symbol": r"m_t",      "unit": "MeV", "scale": 1000},
    "width":  {"symbol": r"\Gamma_t", "unit": "MeV", "scale": 1000},
    "yukawa": {"symbol": r"y_t",      "unit": "%",   "scale": 100, "relative": True},
}

# ---------------------------------------------------------------------------
# Theory-uncertainty quotes for the systematic-table row "theory"
# (in the same display units as ``POI_DISPLAY``).
# ---------------------------------------------------------------------------
THEORY_UNC = {
    "mass":   35.0,   # MeV
    "width":  25.0,   # MeV
    "yukawa": 10.0,   # %
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
# Paths are interpreted relative to the working directory of the entry script.
INPUT_DIRS = {
    "nominal":    "output_xsec/wbwb/nominal",
    "scale_vars": "output_xsec/wbwb/scale_vars",
    "BEC":        "output_xsec/wbwb/BEC",
    "sw2":        "output_xsec/wbwb/sw2",
    "pseudo":     "output_xsec/wbwb/pseudo",
    "nominal_1S": "output_xsec/wbwb/nominal_1S",
    "scale_1S":   "output_xsec/wbwb/scale_1S",
}

PLOT_DIR = "fit_output/wbwb/plots"
PLOT_DIR_1S = "fit_output/wbwb/plots_1S"
SYST_TABLE_PATH = "fit_output/wbwb/systematics_table.tex"

# ---------------------------------------------------------------------------
# Plot decoration: process branding + nuisance labels
# ---------------------------------------------------------------------------
# Process identifier — dispatches into framework.process.wbwb.plot_labels
# for process / generator / BES strings. No LaTeX branding lives in the card.
PROCESS_ID = "wbwb"

# LaTeX symbols for *nuisance* parameters not in POI_DISPLAY (alphas /
# sw2 are constrained nuisances in WbWb). ``param_axis_label`` falls back
# to PARAM_MATH_LABELS when POI_DISPLAY has no entry for the parameter.
PARAM_MATH_LABELS = {
    "alphas": r"\alpha_s",
    "sw2":    r"\sin^2\theta_W",
}
# Native units on parameter-axis plots — POIs and nuisances together.
# Distinct from POI_DISPLAY's syst-table display unit.
PARAM_UNITS = {
    "mass":   "GeV",
    "width":  "GeV",
    "yukawa": "",
    "alphas": "",
    "sw2":    "",
}

# Linear-inflation factor for the Azzurri-style overlay (see ww_default).
AZZURRI_OVERLAY_INFLATE = 100

# Legacy hardcoded scan-window x-axis for fit-output plots. WbWb has
# historically displayed 339.7-347.3 GeV irrespective of the scan range
# configured in SCENARIO; keep that behaviour by overriding here. WW
# omits this constant and falls through to SCENARIO-derived bounds.
SCAN_XLIM = (339.7, 347.3)
