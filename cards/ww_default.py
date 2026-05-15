"""Default steering card for the WW threshold fit.

PLACEHOLDER — most values are tentative defaults; tune them to match the
actual WW analysis scenario before drawing physics conclusions.

Only the fields that are genuinely common with the WbWb card (`BES`, `BEC`,
`lumi`, plot decoration helpers) are intended to be production-quality from
day one; everything else is a stub.
"""

# ---------------------------------------------------------------------------
# Physics: parameters of interest
# ---------------------------------------------------------------------------
# Yukawa is not part of the WW fit; alpha_s is the same convention as for
# WbWb (offset wrt. PDG/world average, not an absolute value).
PARAMETERS = {
    "mass":   {"nominal": 80.379, "pseudo":  0.005,   "variation": 0.030,  "round_dec": 3},
    "width":  {"nominal": 2.085,  "pseudo": -0.010,   "variation": 0.030,  "round_dec": 3},
    "alphas": {"nominal": 0.0,    "pseudo": -0.0001,  "variation": 0.0003, "round_dec": 4},
}

# ---------------------------------------------------------------------------
# Theory / generator settings  (PLACEHOLDER: depend on chosen WW generator)
# ---------------------------------------------------------------------------
ORDER = 2                # placeholder
MASS_SCHEME = "OS"       # on-shell — placeholder
RENORM_SCALES = {
    "mass":  80.0,
    "width": 80.0,
    "vars":  [],
}

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
# BES at WW threshold is smaller than at tt threshold; placeholder value.
BEAM_ENERGY_RES = 0.13   # % per beam  (PLACEHOLDER)
SMEAR_XSEC = True
PEAK_ECM = 162.5         # placeholder for smearing kernel width
LAST_ECM = 240.0         # placeholder above-threshold point

SCENARIO = {
    "scan_min":   157.0,
    "scan_max":   163.0,
    "scan_step":  1.0,
    "total_lumi":     12.0e6,   # /pb — FCC-ee WW threshold lumi (PLACEHOLDER)
    "last_lumi":      5.0e6,
    "stat_inflation": 1.0,
}

# ---------------------------------------------------------------------------
# Template-variation sizes used to build the morphing templates
# ---------------------------------------------------------------------------
INPUT_VAR = {
    "BEC":  10.0,    # MeV   - reuse WbWb convention; can be tuned later
    "BES":  0.1,
}

# ---------------------------------------------------------------------------
# Prior uncertainties for the nuisance parameters
# ---------------------------------------------------------------------------
# BES/BEC/lumi structure is shared with WbWb; default numbers differ.
PRIORS = {
    "alphas":   {"default": 1.0e-4},
    "lumi":     {"uncorr": 1.0e-3, "corr": 5.0e-4, "scale_uncorr": False},
    "BES":      {"uncorr": 0.01,   "corr": 5.0e-3},
    "BEC":      {"uncorr": 2.0,    "corr": 1.0},     # MeV — PLACEHOLDER (smaller than WbWb)
}

# ---------------------------------------------------------------------------
# Theory-uncertainty quotes for the systematic-table row "theory"
# ---------------------------------------------------------------------------
THEORY_UNC = {
    "mass":  3.0,    # MeV — PLACEHOLDER
    "width": 3.0,    # MeV — PLACEHOLDER
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
INPUT_DIRS = {
    "nominal":    "output_WW",
    "scale_vars": "output_WW_scale",
    "BEC":        "BEC_variations_WW",
    "pseudo":     "output_WW_pseudo",
}

PLOT_DIR = "plots/fit_WW"

# ---------------------------------------------------------------------------
# Plot decoration
# ---------------------------------------------------------------------------
PROCESS_LABEL = r"WW threshold"            # PLACEHOLDER label
GENERATOR_LABEL = r"<WW generator TBD>"    # PLACEHOLDER
GENERATOR_REF = r""
BES_LABEL = r"+ FCC-ee BES"

PARAM_LABELS = {
    "mass":   r"$m_W$ [GeV]",
    "width":  r"$\Gamma_W$ [GeV]",
    "alphas": r"\alpha_S",
}
