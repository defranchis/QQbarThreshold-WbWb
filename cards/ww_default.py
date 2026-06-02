"""Default steering card for the WW threshold fit.

Full BFS-EFT N^(3/2)LO chain: Born + NLO loops + NNLO + δ_QCD + Whizard anchor + LL+exp ISR.
Channel: inclusive μνqq̄. Physics rationale for each knob → cards/README.md.
"""

# ---------------------------------------------------------------------------
# Parameters of interest
# ---------------------------------------------------------------------------
# "alphas"/"aem_isr": input offsets, profiled as constraint nuisances; "variation"
# is a numerical morph step (≠ the PRIORS constraint width). See cards/README.md.
PARAMETERS = {
    "mass":    {"nominal": 80.379, "pseudo":  0.005,   "variation": 0.010,  "round_dec": 3},
    "width":   {"nominal": 2.085,  "pseudo": -0.010,   "variation": 0.010,  "round_dec": 3},
    "alphas":  {"nominal": 0.0,    "pseudo": -0.0001,  "variation": 0.0003, "round_dec": 4},
    "aem_isr": {"nominal": 0.0,    "pseudo":  0.0,     "variation": 1.0e-4, "round_dec": 4},
}

# Bilinear (m_W × Γ_W) cross-term morph — one corner template. WW-only,
# opt-in. Rationale in cards/README.md.
CROSS_TERMS = [("mass", "width")]

# ---------------------------------------------------------------------------
# External parameter inputs and uncertainties
# ---------------------------------------------------------------------------
# Production defaults PDG 2024. α_em=None → derived α_Gμ (σ chain); α_em_isr
# = α(M_Z) paired with NLO_CONFIG["isr_emela_ren_scheme"]="ALPMZ". cards/README.md.
PARAM_INPUTS = {
    "m_t":          172.5,        # GeV, OS pole
    "M_H":          125.25,       # GeV
    "M_Z":          91.1876,      # GeV
    "alpha_s_MW":   0.1199,       # α_s(M_W) MS-bar
    "alpha_em":     None,         # None → α_Gμ(m_W, M_Z)
    "alpha_em_isr": 1.0/128.943,  # α(M_Z), PDG
}

# Parametric input uncertainties; None = not propagated as a nuisance (per-key
# reasons in cards/README.md). Only alpha_s and alpha_em_isr are profiled.
PARAM_UNC = {
    "m_t":          None,
    "M_H":          None,
    "M_Z":          None,
    "alpha_s":      1.0e-4,  # profiled as "alphas"
    "alpha_em":     None,
    "alpha_em_isr": 4.7e-8,  # Riembau 2501.05508 combined; profiled as "aem_isr"
}

# ---------------------------------------------------------------------------
# BFS NLO configuration  (defined in ww_nlo_config.py)
# ---------------------------------------------------------------------------
from cards.ww_nlo_config import NLO_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
# BES = 0.105% per beam (FSR Vol.1 tab:IR_parameters, E_beam=80 GeV). PEAK_ECM
# is the BES-smearing reference √s only (E_beam=80 GeV → √s=160). cards/README.md.
BEAM_ENERGY_RES = 0.105   # % per beam (SR+beamstrahlung)
PEAK_ECM = 160.0
LAST_ECM = 240.0

SCENARIO = {
    "scan_min":   157.0,
    "scan_max":   163.0,
    "scan_step":  1.0,
    "total_lumi":     19.2e6,   # /pb — FCC-ee FSR WW baseline (see README)
    "last_lumi":      10.8e6,   # /pb — ZH 240 GeV, --lastecm only (see README)
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

# Lumi is a counting measurement → uncorr prior scales per point as
# uncorr_i = uncorr_ref·√(L_ref/L_i). See cards/README.md (LUMI_UNCORR_SCALES).
LUMI_UNCORR_SCALES = True
# L_ref = baseline per-point lumi (total ÷ N points), derived from SCENARIO.
_N_BASELINE_POINTS = round((SCENARIO["scan_max"] - SCENARIO["scan_min"]) / SCENARIO["scan_step"]) + 1
LUMI_UNCORR_CALIB_LUMI = SCENARIO["total_lumi"] / _N_BASELINE_POINTS

# Other binned nuisances that share lumi's per-point √(L_ref/L_i) counting
# rescale: BES is di-muon-monitored → precision ∝ 1/√L_point. cards/README.md.
UNCORR_COUNTING_KINDS = ("BES",)

# ---------------------------------------------------------------------------
# Priors / systematics  (rationale + sources in cards/README.md)
# ---------------------------------------------------------------------------
# Constraint widths (uncorr/corr). BEC in MeV on √s; BES rel. on the spread;
# lumi rel. per scan point. Sources + derivations in cards/README.md (Priors).
PRIORS = {
    "alphas":  1.0e-4,
    "aem_isr": 4.7e-8,
    "BEC":     {"uncorr": 0.1,    "corr": 0.3},
    "BES":     {"uncorr": 0.01,   "corr": 5.0e-3},
    "lumi":    {"uncorr": 2.0e-4, "corr": 1.0e-4},
}

SYSTEMATICS = {
    "alphas":  {"type": "constraint", "always_on": True},
    "aem_isr": {"type": "constraint", "always_on": True},
    "BEC":     {"type": "binned",
                "source": {"kind": "template_dir",
                           "var_subdir": True, "snap_to_grid": True}},
    "BES":     {"type": "binned",
                "source": {"kind": "smear_shift"}},
}

SYST_TABLE_ORDER = ["alphas", "aem_isr", "BES", "BEC", "lumi"]

# ---------------------------------------------------------------------------
# POI display + axis labels
# ---------------------------------------------------------------------------
POI_DISPLAY = {
    "mass":  {"symbol": r"m_W",      "unit": "MeV", "scale": 1000},
    "width": {"symbol": r"\Gamma_W", "unit": "MeV", "scale": 1000},
}

PROCESS_ID = "ww"

PARAM_MATH_LABELS = {"alphas": r"\alpha_s", "aem_isr": r"\alpha(M_Z)_\mathrm{ISR}"}
PARAM_UNITS = {"mass": "GeV", "width": "GeV", "alphas": "", "aem_isr": ""}

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
    "BEC":        "output_xsec/ww/BEC",
    "pseudo":     "output_xsec/ww/pseudo",
}

PLOT_DIR = "fit_output/ww/plots"
SYST_TABLE_PATH = "fit_output/ww/systematics_table.tex"
