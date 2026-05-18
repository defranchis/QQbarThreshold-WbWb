"""Default steering card for the WW threshold fit.

The generator is :class:`process.ww.generator.WWGenerator` — LO Born CC03
(RACOONWW-calibrated 161.33–500 GeV; power-law BW-tail 156–161 GeV) +
Fadin-Khoze-Martin Coulomb + LL YFS-exponentiated ISR. Channel:
inclusive μν qq̄ (BR = 2 × BR(W→μν) × BR(W→had)).

For MeV-level m_W extraction, the BFS NLO+NNLO hooks need to be filled
(see ``process.ww.eft_xsec.BFSCorrections`` and arXiv:0707.0773 / 0807.0102)
and the 156–161 GeV BW-tail replaced with proper RACOONWW / MoCaNLO Born
values. Until then this card configures the infrastructure end-to-end but
is not yet a production physics result.
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
BEAM_ENERGY_RES = 0.13   # % per beam (PLACEHOLDER) — 0 disables smearing entirely
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
    "coarse_scan": {                                       # PLACEHOLDER
        "scan_min": 158.0, "scan_max": 162.0, "scan_step": 2.0,
        "lumi_factor": 1.0 / 2 ** 0.5,
    },
    "true_value_pivot": "mass",
}

# ---------------------------------------------------------------------------
# Template-variation sizes used to build the morphing templates
# ---------------------------------------------------------------------------
INPUT_VAR = {
    "BEC":  10.0,    # MeV — matches BEC_variations_WW/scan_{p,m}10/
    "BES":  0.1,
    # sw2 nuisance is not implemented at the current order (sin²θ_W is
    # derived from m_W via the OS scheme, so its variation is absorbed
    # into the m_W variation). Re-add when running α(s) / NLO_EW are wired
    # in.
}

# ---------------------------------------------------------------------------
# Priors / systematics schema — see cards/wbwb_default.py for the layout
# convention. WW values are mostly PLACEHOLDERS until the generator side
# and nuisance studies pin down the real numbers.
# ---------------------------------------------------------------------------
PRIORS = {
    "alphas": 1.0e-4,
    "BEC":    {"uncorr": 2.0,    "corr": 1.0},          # PLACEHOLDER (smaller than WbWb)
    "BES":    {"uncorr": 0.01,   "corr": 5.0e-3},
    "lumi":   {"uncorr": 1.0e-3, "corr": 5.0e-4},
}

SYSTEMATICS = {
    "alphas": {"type": "constraint", "always_on": True},
    "BEC":    {"type": "binned",
               "source": {"kind": "template_dir",
                          "var_subdir": True, "snap_to_grid": True}},
    "BES":    {"type": "binned",
               "source": {"kind": "smear_shift"}},
}

# Canonical row order in the systematics table.
SYST_TABLE_ORDER = ["alphas", "BES", "BEC", "lumi"]

# ---------------------------------------------------------------------------
# Parameters of interest displayed by the syst-table machinery
# (see cards/wbwb_default.py for the schema).
# ---------------------------------------------------------------------------
POI_DISPLAY = {
    "mass":  {"symbol": r"m_W",      "unit": "MeV", "scale": 1000},
    "width": {"symbol": r"\Gamma_W", "unit": "MeV", "scale": 1000},
}

# ---------------------------------------------------------------------------
# Theory-uncertainty quotes for the systematic-table row "theory"
# (in the same display units as ``POI_DISPLAY``).
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
PROCESS_LABEL = r"$e^+e^-\rightarrow\mu\nu q\bar q$ at WW threshold"
GENERATOR_LABEL = r"LO(CC03,RACOONWW-cal.)+Coulomb+LL ISR"
GENERATOR_REF = r"BFS NLO/NNLO + BW-tail TBD"
BES_LABEL = r"+ FCC-ee BES"

PARAM_LABELS = {
    "mass":   r"$m_W$ [GeV]",
    "width":  r"$\Gamma_W$ [GeV]",
    "alphas": r"\alpha_S",
}
