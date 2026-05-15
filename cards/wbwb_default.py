"""Default steering card for the WbWb threshold fit.

Every numerical knob that used to live as a module-level constant in the
original `doFit.py` (or as a hardcoded literal inside the `fit` class /
`main()` / `parameters` class) is collected here, grouped by topic.

Cards are plain Python modules: import this module from the entry script,
copy / override individual fields as needed (the entry script may also
expose argparse overrides for the most common ones).
"""

# ---------------------------------------------------------------------------
# Physics: parameters of interest
# ---------------------------------------------------------------------------
# For each parameter:
#   nominal   = central value used to compute the reference template
#   pseudo    = offset applied to build the "true" pseudodata point
#   variation = offset of the variation template used to morph the lineshape
#   round_dec = decimals to round to in file-name tags
PARAMETERS = {
    "mass":   {"nominal": 171.5, "pseudo":  0.01,    "variation": 0.03,   "round_dec": 2},
    "width":  {"nominal": 1.33,  "pseudo": -0.02,    "variation": 0.05,   "round_dec": 2},
    "yukawa": {"nominal": 1.0,   "pseudo":  0.05,    "variation": 0.1,    "round_dec": 2},
    "alphas": {"nominal": 0.0,   "pseudo": -0.0001,  "variation": 0.0003, "round_dec": 4},
}
PARAMETERS_1S = dict(PARAMETERS)
PARAMETERS_1S["mass"] = {**PARAMETERS["mass"], "nominal": 171.9}

# ---------------------------------------------------------------------------
# Theory / generator settings
# ---------------------------------------------------------------------------
ORDER = 3                  # 0=LO ... 3=N3LO
MASS_SCHEME = "PS"         # "PS" or "1S"
RENORM_SCALES = {
    "mass":  80.0,
    "width": 350.0,
    "alt":   {"mass": 170.0, "width": 170.0},   # used when scale-variation files exist
    "vars":  [round(s, 1) for s in range(50, 351, 10)],
}

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
BEAM_ENERGY_RES = 0.186     # % per beam
SMEAR_XSEC = True
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
    # final state. WbWb-specific — the WW card uses 1.0.
    "stat_inflation": 1.2,
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
    "BEC":  10.0,    # MeV   - beam energy calibration shift baked into BEC_variations/
    "BES":  0.1,     # 10%  - BES variation baked into the BES morph template
    "sw2":  2.5e-6,  #        sw2 shift baked into output_sw2/
}

# ---------------------------------------------------------------------------
# Prior uncertainties for the nuisance parameters
# ---------------------------------------------------------------------------
# Notes:
#   * `lumi.scale_uncorr` mirrors the old `scale_lumi_uncorr` flag — when True,
#     the per-point uncorrelated lumi uncertainty is scaled by sqrt(N_points).
#   * BES/BEC/lumi have correlated and uncorrelated components.
PRIORS = {
    "yukawa":    {"default": 0.03},          # used only when Yukawa is constrained
    "alphas":    {"default": 1.0e-4},
    "SM_width":  {"default": 5.0},           # MeV, theory unc. on SM width prediction
    "lumi":      {"uncorr": 1.0e-3, "corr": 5.0e-4, "scale_uncorr": False},
    "BES":       {"uncorr": 0.01,   "corr": 5.0e-3},
    "BEC":       {"uncorr": 5.0,    "corr": 2.5},     # MeV
    "sw2":       {"default": 2.5e-6},
}

# ---------------------------------------------------------------------------
# Theory-uncertainty quotes for the systematic-table row "theory"
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
    "nominal":    "output_full",
    "scale_vars": "output_alternative",
    "BEC":        "BEC_variations",
    "sw2":        "output_sw2",
    "pseudo":     "output_pseudo",
    "nominal_1S": "output_1S",
    "scale_1S":   "output_scale_1S",
}

PLOT_DIR = "plots/fit"
PLOT_DIR_1S = "plots/fit_1S"

# ---------------------------------------------------------------------------
# Plot decoration
# ---------------------------------------------------------------------------
PROCESS_LABEL = r"WbWb at $N^{3}LO$+ISR"
GENERATOR_LABEL = r"QQbar_Threshold $N^{3}LO$+ISR"
GENERATOR_REF = r"[JHEP 02 (2018) 125]"
BES_LABEL = r"+ FCC-ee BES"

PARAM_LABELS = {
    "mass":   r"$m_t$ [GeV]",
    "width":  r"$\Gamma_t$ [GeV]",
    "yukawa": r"y_t",
    "alphas": r"\alpha_S",
    "sw2":    r"$\sin^2(\theta_W)$",
}
