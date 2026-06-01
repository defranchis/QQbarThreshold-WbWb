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

# Bilinear cross-term morph: one extra corner template per pair
# (both POIs shifted by their variation simultaneously). Captures the
# residual non-multiplicative (m_W, Γ_W) curvature that the per-axis
# linear morph misses. At σ_mW≈2 MeV the residual quadratic correction
# is ~4% of the linear slope → potentially ~0.4 MeV bias if the cross
# term is comparable; the corner template eliminates the leading-order
# bias by closing exactly at (m_W+δm, Γ_W+δΓ).
#
# WW-only — WbWb card does not define CROSS_TERMS (fit-side uses
# `getattr(card, "CROSS_TERMS", ())` so the feature stays opt-in).
CROSS_TERMS = [("mass", "width")]

# ---------------------------------------------------------------------------
# External parameter inputs and uncertainties
# ---------------------------------------------------------------------------
# Production defaults are PDG 2024. α_em=None → derived α_Gμ (σ chain);
# α_em_isr now set to α(M_Z) = 1/128.943 (PDG; Riembau/FCC-ee A_FB^μμ
# off-peak is the projected uncertainty, see PARAM_UNC). Pair with
# NLO_CONFIG["isr_emela_ren_scheme"]="ALPMZ" so eMELA evolves DGLAP
# from the same input. See cards/README.md.
PARAM_INPUTS = {
    "m_t":          172.5,        # GeV, OS pole — FCC-ee FSR Table 2 (was 174.2 = BFS ref)
    "M_H":          125.25,       # GeV
    "M_Z":          91.1876,      # GeV
    "alpha_s_MW":   0.1199,       # α_s(M_W) MS-bar
    "alpha_em":     None,         # σ chain override; None → α_Gμ(m_W, M_Z) (Gμ scheme, unchanged)
    "alpha_em_isr": 1.0/128.943,  # α(M_Z), PDG; paired with isr_emela_ren_scheme=ALPMZ
}

# FCC-ee parametric uncertainties (FSR Vol. 1 arXiv:2505.00272 Table 2).
# Wired via fit-side propagation: TODO (see followup memory).
PARAM_UNC = {
    "m_t":          0.007,   # GeV (FSR Table 2: 4.2 ⊕ 4.9 ≈ 6.5; round to 7)
    "M_H":          0.005,   # GeV (FSR §4.3 ZH recoil ≈ 4; round to 5)
    "M_Z":          1.0e-4,  # GeV (Z line-shape stat 4 ⊕ syst 100 keV)
    "alpha_s":      1.0e-4,  # FSR (Z combined: stat 0.1 ⊕ syst 1.0)×10⁻⁴
    "alpha_em":     2.4e-7,  # δα abs; FSR A_FB^μμ off-peak (~3×10⁻⁵ rel)
    "alpha_em_isr": 2.4e-7,  # δα(M_Z) abs; FCC-ee A_FB^μμ off-peak (Riembau).
                             # Independent of alpha_em above (Option B): hard σ̂
                             # uses α_Gμ via G_F; ISR uses α(M_Z) → distinct
                             # physical inputs, uncorrelated nuisances.
}

# ---------------------------------------------------------------------------
# BFS NLO configuration  (defined in ww_nlo_config.py)
# ---------------------------------------------------------------------------
from cards.ww_nlo_config import NLO_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
# FCC-ee BES at WW point: σ_δ = 0.105% per beam — FSR Vol.1 arXiv:2505.00272
# tab:IR_parameters (E_beam = 80 GeV, SR/BS = 0.069/0.105; BS = SR+beamstrahlung).
BEAM_ENERGY_RES = 0.105   # % per beam (SR+beamstrahlung)
PEAK_ECM = 162.5
LAST_ECM = 240.0

SCENARIO = {
    "scan_min":   157.0,
    "scan_max":   163.0,
    "scan_step":  1.0,
    "total_lumi":     19.2e6,   # /pb — FSR 2505.00272 tab:seqbaseline WW baseline
                                #   (20e34/IP, 9.6/ab/yr × 2 yr × 4 IP = 19.2/ab)
    "last_lumi":      10.8e6,   # /pb — ZH 240 GeV baseline, FSR tab:seqbaseline
                                #   (only used with --lastecm; dormant otherwise)
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
    # MeV on √s. FSR 2505.00272 §EnergyCalibration: absolute (fully correlated)
    # ≈300 keV, point-to-point ≈100 keV at the WW threshold (→ m_W 150/50 keV,
    # tab:DS-Ecal-errors-final). Was 1/2 MeV placeholder.
    "BEC":    {"uncorr": 0.1,    "corr": 0.3},
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
