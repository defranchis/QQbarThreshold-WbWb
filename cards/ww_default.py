"""Default steering card for the WW threshold fit.

The generator is :class:`process.ww.generator.WWGenerator` — full BFS-EFT
N^(3/2)LO chain: Born + NLO loops (HSC + Coulomb_NLO + EW-decay) + δ_QCD
+ Whizard 4f Born anchor + LL+exp ISR (BETA scheme). RACOONWW CC03
calibration spline kicks in only above √s = 170 GeV. Channel: inclusive
μν qq̄ (BR = 2 × BR(W→μν) × BR(W→had), PDG-constant convention).

Remaining open work for sub-MeV m_W: BFS dominant NNLO (arXiv:0807.0102)
and NLL ISR (eMELA / Skrzypek-Jadach). Both deferred.
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
# Theory inputs (BFS NLO calculation parameters)
# ---------------------------------------------------------------------------
# All theory parameters that enter the BFS NLO cross section are configured
# here so the scale/value used in the calculation is explicit and overridable
# from the card. RGE evolution to/from the PDG / FCC-projection scale lives
# downstream — see [[project-followup-parameter-scale-rge]].
THEORY_INPUTS = {
    # α_s(M_W) in MS-bar — enters BFS eq. delta_qcd as the multiplicative
    # QCD correction δ_QCD = 1 + α_s/π + 1.409(α_s/π)². BFS reference: 0.1199
    # (consistent with α_s(M_Z) = 0.118 evolved to M_W). PDG world average
    # α_s(M_Z) = 0.1180 ± 0.0009 → α_s(M_W) ≈ 0.1199.
    "alpha_s_MW": 0.1199,
    # m_t, M_H enter the BFS NLO hard-matching coefficient c_p,LR^(1,fin).
    # The chain currently uses c_fin = -10.076 (Re) evaluated at the BFS
    # reference m_t = 174.2 / M_H = 115 GeV. The m_t/M_H propagation into
    # a varying c_fin is sub-0.001 % on σ_NLO (Scenario H of
    # validate_bfs_nlo.py) and is deferred — these values are recorded
    # here so a future full-PV-C0 implementation can pick them up.
    "m_t":      174.2,   # GeV (pole), BFS Table 4 input
    "M_H":      115.0,   # GeV, BFS Table 4 input (pre-Higgs-discovery value)
}

# ---------------------------------------------------------------------------
# BFS NLO loop corrections — toggle the new physics on/off
# ---------------------------------------------------------------------------
# Default: full BFS NLO loops enabled. ``br_convention = "pdg-constant"``
# uses the PDG-measured BR (≈ 0.143) as a fixed scale factor on σ_WW;
# ``apply_delta_QCD = True`` then multiplies σ by δ_QCD(α_s) = 1 + α_s/π +
# 1.409 (α_s/π)² so that α_s is a *physically active* fit parameter — its
# value shifts σ via the multiplicative QCD enhancement of the hadronic
# content. The slight ~4 % double-count with the QCD content already folded
# into the PDG BR is acceptable: it's a constant rescaling of the cross
# section that the fit absorbs via the luminosity nuisance, while the
# α_s differential (∂σ/∂α_s) is correctly captured.
NLO_CONFIG = {
    "include_NLO_hard_decay": True,    # HSC + EW decay + Coulomb_NLO additive
    "apply_delta_QCD":        True,    # multiplicative δ_QCD(α_s) on σ_partonic
    "br_convention":          "pdg-constant",   # 'pdg-constant' | 'bfs-eft'
    # Whizard-anchor correction f(δ, Γ_W) (BFS section 6.2 prescription).
    # Replaces the BFS-EFT N^(3/2)LO Born by the exact Whizard 4f Born via
    # a smooth multiplicative correction built from BFS Tables 1+2. Closes
    # the residual ~2 % absolute Born deficit; preserves analytic m_W/Γ_W
    # dependence (m_W via δ = √s − 2m_W, Γ_W via linear interp between the
    # two BFS reference Γ_W = {2.045, 2.092}).
    "apply_whizard_anchor":   True,
    # ISR scheme — LL+exp BETA per LEP2 YR Beenakker hep-ph/9602351 eq. (67).
    #   "single_conv": LEP2 YR α→2α 1D convolution shortcut (default, fastest).
    #   "2leg":        full per-leg double-convolution per BFS eq. 71. Agrees
    #                  with single_conv to <0.1 % — formal LL+exp equivalence.
    # See project_followup_isr_scheme.md for the verification details.
    "isr_scheme":             "single_conv",
    # ISR α: None = α_Gμ(m_W=80.377) per BFS prescription (constant across
    # the fit to avoid fictitious m_W dependence in the ISR kernel). Override
    # with a float for a scheme-variation systematic.
    "alpha_em_isr":           None,
}

# ---------------------------------------------------------------------------
# Beam-energy spectrum & scan grid
# ---------------------------------------------------------------------------
# FCC-ee BES at W+W- operating point (E_beam = 80 GeV), with beamstrahlung,
# from FCC FSR Vol. 1 (arXiv:2505.00272) Table 14: σ_δ = 0.105 % per beam.
# SR-only value is 0.069 %; the 0.105 % BS value is the relevant one for
# physics at the IP. At the WW peak (√s ≈ 162.5 GeV) this gives a CM-energy
# spread σ_√s = √s · σ_δ / √2 ≈ 120 MeV.
BEAM_ENERGY_RES = 0.105   # % per beam (FCC FSR Vol 1 Table 14, BS, W+W-)
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
GENERATOR_LABEL = (r"BFS N$^{3/2}$LO + NLO loops + $\delta_{\rm QCD}$ + "
                   r"Whizard anchor + K$_{\rm C}$ + LL+exp ISR")
GENERATOR_REF = r"arXiv:0707.0773 (Beneke-Falgari-Schwinn et al.)"
BES_LABEL = r"+ FCC-ee BES"

PARAM_LABELS = {
    "mass":   r"$m_W$ [GeV]",
    "width":  r"$\Gamma_W$ [GeV]",
    "alphas": r"\alpha_S",
}
