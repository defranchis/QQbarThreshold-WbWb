"""Default steering card for the WbWb threshold fit.

Every numerical knob that used to live as a module-level constant in the
original `doFit.py` (or as a hardcoded literal inside the `fit` class /
`main()` / `parameters` class) is collected here, grouped by topic.

Cards are plain Python modules: import this module from the entry script,
copy / override individual fields as needed (the entry script may also
expose argparse overrides for the most common ones).

The physics-parameters block (PARAMETERS / PARAMETERS_1S / ORDER /
RENORM_SCALES) is derived from `xsec_calculator.parameter_def.parameters`
— that class is the single source of truth for the values the C++
template generator (`compute_xsec_parallel.py`) consumes. Card-only knobs
(`round_dec`, the alt-scale split in `RENORM_SCALES`) are added on top.
"""

from xsec_calculator.parameter_def import parameters as _ParameterDef

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
    # final state.
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
#   * `lumi.uncorr`: per-ECM-point uncorrelated luminosity uncertainty
#     (fractional). `lumi.corr`: fully-correlated luminosity uncertainty
#     (fractional), applied identically at every ECM point. The above-
#     threshold point (when `add_last_ecm=True`) gets its uncorr entry
#     divided by sqrt(L_above / L_threshold) automatically.
#   * BES/BEC also have correlated and uncorrelated components.
PRIORS = {
    "yukawa":    {"default": 0.03},          # used only when Yukawa is constrained
    "alphas":    {"default": 1.0e-4},
    "SM_width":  {"default": 5.0},           # MeV, theory unc. on SM width prediction
    "lumi":      {"uncorr": 1.0e-3, "corr": 5.0e-4},
    "BES":       {"uncorr": 0.01,   "corr": 5.0e-3},
    "BEC":       {"uncorr": 5.0,    "corr": 2.5},     # MeV
    "sw2":       {"default": 2.5e-6},
}

# ---------------------------------------------------------------------------
# Refactored systematics schema — replacing the legacy PRIORS / INPUT_VAR /
# `add_*_nuisances` plumbing across commits 2-8. While the refactor lands,
# both layouts coexist; ``FitCore`` keeps reading the legacy fields. See
# "Extending to a new process" in README.md (added in commit 8).
# ---------------------------------------------------------------------------

# 1-D Gaussian penalty terms on a single fit parameter, centred at
# ``center`` (defaults to the pseudodata "true" value) with width
# ``sigma``. ``always_on=True`` means the constraint is applied
# unconditionally; ``False`` means an entry-script flag decides (today:
# ``--fitYukawa`` toggles ``constrain_yukawa``, which gates the Yukawa
# constraint).
CONSTRAINTS = {
    "alphas": {"sigma": 1.0e-4, "always_on": True},
    "yukawa": {"sigma": 0.03,   "always_on": False},
    # "center": <value>  # optional per-entry — falls back to the pseudodata
    #                    # "true" value when absent.
}

# Nuisances that perturb the cross-section template per ECM. The
# ``uncorr`` prior constrains the per-ECM-bin parameters; ``corr``
# constrains a single fully-correlated parameter that perturbs every ECM
# identically. ``source`` is a ``{"kind": ..., ...}`` discriminator
# consumed by the morph dispatcher in commit 4 (``template_dir`` loads
# variations from a pre-computed directory; ``smear_shift`` re-smears
# the nominal with ``bes * (1 + INPUT_VAR[kind])``). The variation
# magnitude itself is read from ``INPUT_VAR[<kind>]`` — that value is
# template-generation-frozen (must match the C++ scan) and not
# duplicated here.
BINNED_NUISANCES = {
    # BEC's source carries two BEC-specific quirks generically:
    # `var_subdir` says the actual templates live in a subdir named
    # `scan_p{var}` / `scan_m{var}` (the C++ scan generates a directory
    # per ±variation magnitude); `snap_to_grid` says the loaded ECMs are
    # shifted by ±INPUT_VAR["BEC"] MeV and must be rounded back to the
    # nominal 0.1-GeV grid (see the 40-MeV limit on INPUT_VAR["BEC"]).
    "BEC": {"source": {"kind": "template_dir", "path": "BEC_variations",
                       "var_subdir": True, "snap_to_grid": True},
            "priors": {"uncorr": 5.0, "corr": 2.5}},     # MeV
    "BES": {"source": {"kind": "smear_shift"},
            "priors": {"uncorr": 0.01, "corr": 5e-3}},
}

# Global (non-binned) nuisances — single fit parameter, ``source`` is a
# fully pre-computed variation template (similar in shape to BEC's
# loader, no per-bin expansion). ``INPUT_VAR[<kind>]`` again carries the
# template-generation-frozen variation magnitude.
GLOBAL_NUISANCES = {
    "sw2": {"source": {"kind": "template_dir", "path": "output_sw2"},
            "prior": 2.5e-6},
}

# Luminosity priors stay separate from CONSTRAINTS / BINNED_NUISANCES —
# they feed the cov matrix in ``FitCore._build_cov``, not the chi²
# constraint terms or any nuisance morph row.
LUMI_PRIORS = {"uncorr": 1.0e-3, "corr": 5.0e-4}

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
