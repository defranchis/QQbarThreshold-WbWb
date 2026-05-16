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
    # No inflation by default; revisit once a WW detector-level study yields
    # a measured factor.
    "stat_inflation": 1.0,
}

# ---------------------------------------------------------------------------
# Template-variation sizes used to build the morphing templates
# ---------------------------------------------------------------------------
INPUT_VAR = {
    "BEC":  10.0,    # MeV   - reuse WbWb convention; can be tuned later
    "BES":  0.1,
    "sw2":  2.5e-6,  # PLACEHOLDER — must match the WW sw2-variation template scan
}

# ---------------------------------------------------------------------------
# Prior uncertainties for the nuisance parameters
# ---------------------------------------------------------------------------
# BES/BEC/lumi structure is shared with WbWb; default numbers differ.
# See `cards/wbwb_default.py` for the meaning of `lumi.uncorr` / `lumi.corr`.
PRIORS = {
    "alphas":   {"default": 1.0e-4},
    "lumi":     {"uncorr": 1.0e-3, "corr": 5.0e-4},
    "BES":      {"uncorr": 0.01,   "corr": 5.0e-3},
    "BEC":      {"uncorr": 2.0,    "corr": 1.0},     # MeV — PLACEHOLDER (smaller than WbWb)
}

# ---------------------------------------------------------------------------
# Refactored systematics schema — see cards/wbwb_default.py for the meaning
# of each section. The WW card carries placeholders until the WW-specific
# generator + nuisance studies pin down the real values.
# ---------------------------------------------------------------------------
CONSTRAINTS = {
    "alphas": {"sigma": 1.0e-4, "always_on": True},
    # TODO: sin2thetaW / alphaEM here once their priors are pinned down.
}

BINNED_NUISANCES = {
    "BEC": {"source": {"kind": "template_dir", "path": "BEC_variations",
                       "var_subdir": True, "snap_to_grid": True},
            "priors": {"uncorr": 2.0, "corr": 1.0}},     # PLACEHOLDER
    "BES": {"source": {"kind": "smear_shift"},
            "priors": {"uncorr": 0.01, "corr": 5e-3}},
}

GLOBAL_NUISANCES = {
    "sw2": {"source": {"kind": "template_dir", "path": "output_sw2"},
            "prior": 2.5e-6},                            # PLACEHOLDER
}

LUMI_PRIORS = {"uncorr": 1.0e-3, "corr": 5.0e-4}

# Canonical row order in the systematics table (see cards/wbwb_default.py
# for the convention). No yukawa rows for WW; future WW-specific
# additions (sin2thetaW, alphaEM, ...) get appended here as their priors
# are pinned down.
SYST_TABLE_ORDER = ["alphas", "sw2", "BES", "BEC", "lumi"]

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
