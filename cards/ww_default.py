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
# External parameter inputs and uncertainties
# ---------------------------------------------------------------------------
# Production defaults are PDG 2024. α_em=None → derived α_Gμ (σ chain);
# α_em_isr=None → α_Gμ(M_W_BFS_REF) per BFS prescription. See cards/README.md.
PARAM_INPUTS = {
    "m_t":          172.5,    # GeV, OS pole — FCC-ee FSR Table 2 (was 174.2 = BFS ref)
    "M_H":          125.25,   # GeV
    "M_Z":          91.1876,  # GeV
    "alpha_s_MW":   0.1199,   # α_s(M_W) MS-bar
    "alpha_em":     None,     # σ chain override; None → α_Gμ(m_W, M_Z)
    "alpha_em_isr": None,     # ISR β_e override; None → α_Gμ(M_W_BFS_REF)
}

# FCC-ee parametric uncertainties (FSR Vol. 1 arXiv:2505.00272 Table 2).
# Wired via fit-side propagation: TODO (see followup memory).
PARAM_UNC = {
    "m_t":          0.007,   # GeV (FSR Table 2: 4.2 ⊕ 4.9 ≈ 6.5; round to 7)
    "M_H":          0.005,   # GeV (FSR §4.3 ZH recoil ≈ 4; round to 5)
    "M_Z":          1.0e-4,  # GeV (Z line-shape stat 4 ⊕ syst 100 keV)
    "alpha_s":      1.0e-4,  # FSR (Z combined: stat 0.1 ⊕ syst 1.0)×10⁻⁴
    "alpha_em":     2.4e-7,  # δα abs; FSR A_FB^μμ off-peak (~3×10⁻⁵ rel)
    "alpha_em_isr": 1.0e-7,  # ISR scheme/scale variation (not direct FCC-ee obs)
}

# ---------------------------------------------------------------------------
# BFS NLO configuration  (defined in ww_nlo_config.py)
# ---------------------------------------------------------------------------
from cards.ww_nlo_config import NLO_CONFIG  # noqa: E402

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
