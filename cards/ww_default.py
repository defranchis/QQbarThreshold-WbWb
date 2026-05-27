"""Default steering card for the WW threshold fit.

Full BFS-EFT N^(3/2)LO chain: Born + NLO loops + NNLO + δ_QCD + Whizard anchor + LL+exp ISR.
Channel: inclusive μνqq̄. Physics rationale for each knob → cards/README.md.
"""

# ---------------------------------------------------------------------------
# Parameters of interest
# ---------------------------------------------------------------------------
PARAMETERS = {
    "mass":   {"nominal": 80.379, "pseudo":  0.005,   "variation": 0.010,  "round_dec": 3},
    "width":  {"nominal": 2.085,  "pseudo": -0.010,   "variation": 0.010,  "round_dec": 3},
    "alphas": {"nominal": 0.0,    "pseudo": -0.0001,  "variation": 0.0003, "round_dec": 4},
}

# ---------------------------------------------------------------------------
# Theory inputs
# ---------------------------------------------------------------------------
# Production defaults are PDG 2024 values.  BFS reference set used for
# Tables 1–4 closure: m_t=174.2, M_H=115, M_Z=91.188 (constants M_*_BFS_REF
# in process/ww/xsec_calculator/eft_xsec.py).
THEORY_INPUTS = {
    "alpha_s_MW": 0.1199,   # α_s(M_W) MS-bar; PDG α_s(M_Z)=0.118 → M_W ≈ 0.1199
    "m_t":        174.2,    # GeV, OS pole mass (PDG ≈ BFS reference)
    "M_H":        125.25,   # GeV PDG 2024 (BFS Tables used 115, pre-discovery)
    "M_Z":        91.1876,  # GeV PDG (BFS Tables used 91.188)
}

# ---------------------------------------------------------------------------
# BFS NLO configuration
# ---------------------------------------------------------------------------
NLO_CONFIG = {
    # --- Channel / BR -------------------------------------------------------
    "channel":                "inclusive",    # 'inclusive' | 'munuud'
    "br_convention":          "pdg-constant", # 'pdg-constant' | 'bfs-eft' (see README)

    # --- Coulomb ------------------------------------------------------------
    # FKM (1993) multiplicative K_C — OFF; BFS Coulomb lives in NLO loops below.
    "include_coulomb":        False,
    # Diagnostic: subleading-only NLO Coulomb (eq. 62 two-photon piece).
    # Only flip True together with include_coulomb=True; see README.
    "coulomb_kc_safe":        False,

    # --- NLO loops ----------------------------------------------------------
    # BFS §4: HSC + Coulomb_NLO + EW-decay correction.
    "include_NLO_hard_decay": True,
    # BFS §6.2 prescription: replace σ̂^(0) by σ_Born in decay correction.
    "decay_uses_full_born":   True,

    # --- NNLO + QCD ---------------------------------------------------------
    # arXiv:0807.0102 eq. 49: C×[S+H] + NLO-C + C×decay + C×res + C3.
    "include_BFS_NNLO":       True,
    # δ_QCD = 1 + α_s/π + 1.409(α_s/π)²; routed through BR in pdg-constant mode.
    "apply_delta_QCD":        True,

    # --- Whizard anchor -----------------------------------------------------
    # Replace BFS-EFT Born by WHIZARD 4f Born (BFS §6.2 prescription).
    "apply_whizard_anchor":   True,
    # 'morph'  — grid_fine quadratic morph, validated sub-MeV (max 0.020%). DEFAULT.
    # 'grid'   — 1295-pt trilinear interpolation, ~0.05–0.2% MC noise.
    # 'spline' — BFS Tables 1+2 cubic spline, two Γ_W points only.
    "whizard_anchor_source":  "morph",

    # --- ISR ----------------------------------------------------------------
    # 'single_conv' — LEP2 YR α→2α 1D form (default, ~10× faster than 2leg).
    # '2leg'        — per-leg double-convolution per BFS eq. 71.
    "isr_scheme":             "single_conv",
    # None → α_Gμ(M_W_BFS_REF) fixed (BFS prescription).  Float overrides.
    "alpha_em_isr":           None,

    # --- Diagnostics --------------------------------------------------------
    # Standalone BFS NLO Coulomb via delta_NLO path.
    # DOUBLE-COUNTS when include_NLO_hard_decay=True — diagnostic only.
    "diagnostic_bfs_coulomb_nlo": False,
}

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
# FCC-ee BES at WW point: σ_δ = 0.105% per beam (arXiv:2505.00272 Table 14, BS).
BEAM_ENERGY_RES = 0.105   # % per beam
PEAK_ECM = 162.5
LAST_ECM = 240.0

SCENARIO = {
    "scan_min":   157.0,
    "scan_max":   163.0,
    "scan_step":  1.0,
    "total_lumi":     12.0e6,   # /pb (PLACEHOLDER)
    "last_lumi":      5.0e6,
    "stat_inflation": 1.0,
    "coarse_scan": {
        "scan_min": 158.0, "scan_max": 162.0, "scan_step": 2.0,
        "lumi_factor": 1.0 / 2**0.5,
    },
    "true_value_pivot": "mass",
}

# ---------------------------------------------------------------------------
# Template-variation sizes
# ---------------------------------------------------------------------------
INPUT_VAR = {
    "BEC":  10.0,   # MeV — matches output_xsec/ww/BEC/scan_{p,m}10/
    "BES":  0.1,
    "lumi": 0.01,   # 1% nuisance unit
}

# ---------------------------------------------------------------------------
# Lumi treatment
# ---------------------------------------------------------------------------
LUMI_MODE = "nuisance"

# ---------------------------------------------------------------------------
# Priors / systematics
# ---------------------------------------------------------------------------
PRIORS = {
    "alphas": 1.0e-4,
    "BEC":    {"uncorr": 2.0,    "corr": 1.0},          # PLACEHOLDER
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

SYST_TABLE_ORDER = ["alphas", "BES", "BEC", "lumi"]

# ---------------------------------------------------------------------------
# POI display + axis labels
# ---------------------------------------------------------------------------
POI_DISPLAY = {
    "mass":  {"symbol": r"m_W",      "unit": "MeV", "scale": 1000},
    "width": {"symbol": r"\Gamma_W", "unit": "MeV", "scale": 1000},
}

PROCESS_ID = "ww"

PARAM_MATH_LABELS = {"alphas": r"\alpha_s"}
PARAM_UNITS = {"mass": "GeV", "width": "GeV", "alphas": ""}

# ---------------------------------------------------------------------------
# Parametric uncertainties on external inputs (PLACEHOLDERS)
# ---------------------------------------------------------------------------
PARAM_UNC = {
    "m_t":       0.30,    # GeV PDG 2024
    "M_H":       0.11,    # GeV PDG 2024
    "M_Z":       0.0021,  # GeV PDG
    "alpha_s":   0.0009,  # PDG α_s(M_Z)
    "BR_W_MUNU": 0.0006,  # PDG
    "BR_W_HAD":  0.0011,  # PDG
}

# ---------------------------------------------------------------------------
# PDG branching ratios
# ---------------------------------------------------------------------------
BR_W_MUNU = 0.1063   # BR(W → μν)
BR_W_HAD  = 0.6741   # BR(W → hadrons)
BR_W_UD   = 0.3358   # BR(W → ud̄ + cs̄)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
INPUT_DIRS = {
    "nominal":    "output_xsec/ww/nominal",
    "scale_vars": "output_xsec/ww/scale_vars",
    "BEC":        "output_xsec/ww/BEC",
    "pseudo":     "output_xsec/ww/pseudo",
}

PLOT_DIR = "fit_output/ww/plots"
SYST_TABLE_PATH = "fit_output/ww/systematics_table.tex"
