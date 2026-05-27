"""Default steering card for the WW threshold fit.

The generator is :class:`process.ww.generator.WWGenerator` — full BFS-EFT
N^(3/2)LO chain: Born + NLO loops (HSC + Coulomb_NLO + EW-decay) + BFS
dominant NNLO (arXiv:0807.0102 eq. 49) + δ_QCD + Whizard 4f Born anchor
+ LL+exp ISR (BETA scheme). RACOONWW CC03 calibration spline kicks in
only above √s = 170 GeV. Channel: inclusive μν qq̄ (BR = 2 × BR(W→μν) ×
BR(W→had), PDG-constant convention).

Remaining open work for sub-MeV m_W: NLL ISR (eMELA / Skrzypek-Jadach).
"""

# ---------------------------------------------------------------------------
# Physics: parameters of interest
# ---------------------------------------------------------------------------
# Yukawa is not part of the WW fit; alpha_s is the same convention as for
# WbWb (offset wrt. PDG/world average, not an absolute value).
PARAMETERS = {
    "mass":   {"nominal": 80.379, "pseudo":  0.005,   "variation": 0.010,  "round_dec": 3},
    "width":  {"nominal": 2.085,  "pseudo": -0.010,   "variation": 0.010,  "round_dec": 3},
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
# uses the PDG-measured BR (≈ 0.143) as the decay weight on σ_WW. With
# ``apply_delta_QCD = True`` the QCD correction enters the BR (where it
# physically belongs — δ_QCD multiplies Γ_had inside BR(W→qq̄)) via the
# α_s-aware factor
#       BR(α_s) = BR_PDG × δ_QCD(α_s) / δ_QCD(α_s_ref)
# with ``α_s_ref = THEORY_INPUTS["alpha_s_MW"]`` the card-declared nominal.
# At α_s = α_s_ref the ratio is 1 and BR_PDG is recovered exactly (no
# double-count at the reference, contra the older σ × δ_QCD × BR_PDG
# wiring which over-counted by δ_QCD(α_s_PDG) ≈ 1.040 = +4 %). The α_s
# differential ∂σ/∂α_s is preserved (BFS reference dδ_QCD/dα_s = +0.351).
#
# For ``br_convention = "bfs-eft"`` δ_QCD multiplies σ per BFS §6.1
# (theory-LO partials don't contain it), as before.
NLO_CONFIG = {
    # ---- Physics knobs -----------------------------------------------------
    # Channel: inclusive μν qq̄ (both W charges × ud̄+cs̄) or μ⁻ν̄_μ ud̄ specific.
    "channel":                "inclusive",     # 'inclusive' | 'munuud'
    # Branching-ratio convention.
    "br_convention":          "pdg-constant",  # 'pdg-constant' | 'bfs-eft'
    # Fadin-Khoze-Martin Coulomb K-factor (Phys. Lett. B 311 (1993) 311 +
    # hep-ph/9507422). LEP-era closed-form Coulomb resummation factor. NOT
    # used by BFS, who treat the Coulomb correction via an α-expansion of
    # the all-order Coulomb Green function (arXiv:0707.0773 eq. 61-62 and
    # arXiv:0807.0102 eq. 9) and explicitly justify NOT resumming the
    # two-photon piece (BFS lines 1722-1725: "a few permille").
    #
    # Default flipped 2026-05-26 from True → False after a literature audit
    # (see project_followup_kc_dropped_2026-05-26 in memory). Reasons:
    #   (1) K_C overlaps with BFS eq.(62) term 1 at the ~5% level on σ at
    #       threshold (the leading-α/v Coulomb piece is the same physics
    #       in two formulations); multiplying both double-counts.
    #   (2) The Beneke / Hoang school never uses a multiplicative K_C × σ
    #       structure — they use the Coulomb Green function with additive
    #       insertions. Our framework's K_C is a 1993-vintage engineering
    #       shortcut that was bolted on before the BFS chain was assembled.
    #   (3) Scenario F of validate_bfs_nlo.py (BFS Table 4 closure) and the
    #       plot_bfs_table4 figure both call this with include_coulomb=False
    #       — the production chain now matches that apples-to-paper choice.
    #
    # Trade-off: the BFS-validated K_C-off chain loses the ~0.2% two-photon
    # piece that K_C would have added. BFS themselves accept this — see
    # their argument above. Grade B (a full G_C-based refactor) would
    # restore the resummation rigorously; planning lives in memory
    # ([[project-grade-b-gc-refactor-plan]]).
    "include_coulomb":        False,
    # BFS NLO loop chain (HSC + Coulomb_NLO + EW decay correction).
    "include_NLO_hard_decay": True,
    # BFS arXiv:0707.0773 §6.2 line 2255 prescription: in the NLO decay
    # correction (eq. 84), replace σ^(0) by the full Born σ_Born. This is
    # the recipe BFS actually used to produce their published Table 4
    # NLO column. With ``True`` (default), Δσ_decay = δ_decay × σ_LR_Born
    # where σ_LR_Born is the per-helicity LR Born accumulated up to
    # N(3/2)LO (LO + 1/2 + NLO Coulomb potential + 3/2,a) with the
    # Whizard anchor already applied — exactly the "σ_Born" of BFS eq. (84).
    # Setting ``False`` reverts to the historical Δσ_decay = δ_decay × σ^(0)
    # (matches σ̂^(1) of BFS eq. 60 in isolation; mis-closes Table 4 by
    # 0.0-0.9 % at [161,170] GeV, see report §5.5 closure-budget memo).
    # Production default flipped to True on 2026-05-26 to match the BFS
    # recipe and close Table 4 to MC stat.
    "decay_uses_full_born":   True,
    # K_C-safe Coulomb: when True (and include_coulomb=True), the BFS NLO
    # Coulomb (eq. 62 of arXiv:0707.0773) is added with the
    # ``subleading_only=True`` flag (NLO two-photon ~0.2% piece only),
    # dropping the N^(1/2)LO one-photon piece (~5% at threshold) that
    # otherwise double-counts the leading α/v of K_C. Documented in
    # report Appendix C. Vestigial under the K_C-off default — flip
    # include_coulomb=True together with this flag to recover the
    # K_C-safe hybrid combination for diagnostic comparison.
    "coulomb_kc_safe":        False,
    # BFS dominant NNLO (arXiv:0807.0102 eq. 49) — C×[S+H] + NLO-C +
    # C×decay + C×res + C3. All five pieces are closed-form analytic;
    # validated to 4 digits vs Table 1 of the paper. Combined NNLO is
    # ~+1 fb at the WW peak; m_W impact ~3 MeV (5 MeV pre-ISR) per BFS
    # sec. 6.4. See [[reference-bfs-nnlo]] for the equation map.
    "include_BFS_NNLO":       True,
    # Multiplicative δ_QCD(α_s) = 1 + α_s/π + 1.409(α_s/π)² on σ_partonic.
    "apply_delta_QCD":        True,
    # Whizard 4f Born anchor (BFS sec. 6.2 prescription). Closes the
    # residual ~2 % absolute Born deficit.
    "apply_whizard_anchor":   True,
    # Anchor source.  Three implementations in ``bfs_eft.whizard_anchor_factor``:
    #   "morph"  — grid_fine morphing predictor (6363 pts, 0.1-GeV √s step,
    #              9 m_W × 7 Γ_W, ~0.008 % MC): quadratic R_m × R_Γ × bilinear
    #              cross-term, cubic-spline-in-√s, denoised. Validated sub-MeV
    #              (max 0.020 % on held-out grid_validate_fine). PRODUCTION DEFAULT.
    #   "grid"   — 1295-point WHIZARD 3.1.5 scan (5 m_W × 7 Γ_W × 37 √s),
    #              trilinear interp. Full coverage; ~0.05-0.2 % per-point MC noise.
    #   "spline" — BFS arXiv:0707.0773 Tables 1+2: cubic spline in
    #              δ = √s − 2m_W, linear interp in Γ_W between
    #              {2.04483, 2.09201}. Smooth; only two Γ_W anchor points.
    "whizard_anchor_source":  "morph",
    # DIAGNOSTIC ONLY — adds the BFS NLO Coulomb subleading piece (eq. 62 of
    # arXiv:0707.0773) via the standalone BFSCorrections.delta_NLO path.
    # When include_NLO_hard_decay=True (the production default), the FULL
    # eq. 62 is ALREADY added in the chain via
    # delta_sigma_Coulomb_NLO_specific_pb — so enabling this DOUBLE-COUNTS.
    # Only flip True together with include_NLO_hard_decay=False to reproduce
    # BFS paper plots that show this piece in isolation.
    "diagnostic_bfs_coulomb_nlo": False,
    # ---- ISR -------------------------------------------------------------
    # ISR scheme — LL+exp BETA per LEP2 YR Beenakker hep-ph/9602351 eq. (67).
    # The two schemes are algebraically equivalent at LL+exp; the only
    # difference is the quadrature density per dimension.
    #   "single_conv": LEP2 YR α→2α 1D shortcut, n_quad=200. Default.
    #                  ~10× faster than 2-leg for the same residual noise
    #                  floor (see [[project-followup-isr-scheme]]). Revisit
    #                  when NLL ISR lands — at that point the 2-leg
    #                  textbook form will be the natural starting point.
    #   "2leg":        full per-leg double-convolution per BFS eq. 71. The
    #                  canonical "textbook" form; available as an opt-in
    #                  for cross-checks / future NLL upgrades.
    "isr_scheme":             "single_conv",
    # ISR α: None = α_Gμ(M_W_BFS_REF) per BFS prescription (constant across
    # the fit to avoid fictitious m_W dependence in the ISR kernel). Override
    # with a float for a scheme-variation systematic.
    "alpha_em_isr":           None,
    # ISR quadrature settings. n_quad=200 single-conv (1D) / 32 per leg (2leg);
    # z_min lower-bound on z = x₁x₂ (= 0.10 captures full ISR phase space, the
    # below-threshold contribution to σ̂(zs) is negligible).
    "n_quad":                 200,
    "z_min":                  0.10,
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
    "BEC":  10.0,    # MeV — matches output_xsec/ww/BEC/scan_{p,m}10/
    "BES":  0.1,
    "lumi": 0.01,    # 1% — fit-param unit for the lumi nuisance (LUMI_MODE="nuisance")
    # sw2 nuisance is not implemented at the current order (sin²θ_W is
    # derived from m_W via the OS scheme, so its variation is absorbed
    # into the m_W variation). Re-add when running α(s) / NLO_EW are wired
    # in.
}

# ---------------------------------------------------------------------------
# Lumi-uncertainty treatment ("cov" | "nuisance") — see
# cards/wbwb_default.py for the schema convention.
# ---------------------------------------------------------------------------
LUMI_MODE = "nuisance"

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
    "nominal":    "output_xsec/ww/nominal",
    "scale_vars": "output_xsec/ww/scale_vars",
    "BEC":        "output_xsec/ww/BEC",
    "pseudo":     "output_xsec/ww/pseudo",
}

PLOT_DIR = "fit_output/ww/plots"
SYST_TABLE_PATH = "fit_output/ww/systematics_table.tex"

# ---------------------------------------------------------------------------
# Plot decoration
# ---------------------------------------------------------------------------
PROCESS_LABEL = r"$e^+e^-\rightarrow\mu\nu q\bar q$ at WW threshold"
# Process label without the descriptive suffix, used when combining with the
# chain badge on a single line (see plot_fit_scenario annotation).
PROCESS_LABEL_SHORT = r"$e^+e^-\rightarrow\mu\nu q\bar q$"
GENERATOR_LABEL = (r"BFS N$^{3/2}$LO + NLO loops + dominant NNLO + "
                   r"$\delta_{\rm QCD}$ + Whizard anchor + K$_{\rm C}$ + LL+exp ISR")
# Compact generator badge for the lower-right fit-scenario caption.
GENERATOR_LABEL_SHORT = r"NNLO EFT + LL ISR"
GENERATOR_REF = r"arXiv:0707.0773 + arXiv:0807.0102 (Beneke-Falgari-Schwinn et al.)"
BES_LABEL = r"+ FCC-ee BES"

# Plain LaTeX math symbols (no $, no units) — the atomic piece used to
# compose every other label (axis titles, ratio captions, legend entries).
PARAM_MATH_LABELS = {
    "mass":   r"m_W",
    "width":  r"\Gamma_W",
    "alphas": r"\alpha_s",
}
PARAM_UNITS = {
    "mass":   "GeV",
    "width":  "GeV",
    "alphas": "",
}

# ---------------------------------------------------------------------------
# Plot-output tunables
# ---------------------------------------------------------------------------
# Linear-inflation factor for the Azzurri-style overlay generated from fit
# inputs. Multiplies the (σ_var − σ_nom) deviation read from the templates
# so the band is visible on the σ_WW scale. With PARAMETERS["mass"]["variation"]
# = 0.010 (= ±10 MeV), 100 produces a ±1 GeV-equivalent band — Azzurri
# 2107.04444 Fig. 1 scale.
AZZURRI_OVERLAY_INFLATE = 100

# Clip ``plot_parameter_variations`` to the fit scan range (driven by
# ``_scan_xlim`` in framework/common/plots.py). WW opts in; WbWb does
# not, preserving its legacy full-template view.
RESTRICT_PARAM_VARIATIONS_PLOT_TO_SCAN = True

# ---------------------------------------------------------------------------
# PDG branching ratios — single source of truth for the chain. All derived
# combinations (BR_INCLUSIVE_MUNUQQ etc.) are computed in
# framework/process/ww/xsec_calculator/eft_xsec.py from these primitives.
# ---------------------------------------------------------------------------
BR_W_MUNU = 0.1063   # PDG: BR(W → μν)
BR_W_HAD  = 0.6741   # PDG: BR(W → hadrons), inclusive
BR_W_UD   = 0.3358   # PDG: BR(W → up-type-quark generation), ud̄ + cs̄ ≈ BR_had / 2
