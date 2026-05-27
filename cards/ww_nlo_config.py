"""Physics configuration for the WW BFS-EFT cross-section chain.

Imported by ww_default.py; can also be loaded standalone for generator-only
studies. Physics rationale for each knob → cards/README.md.
"""

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
    # α_em values (σ chain + ISR) live in PARAM_INPUTS — both default to
    # derived (α_Gμ on σ side; α_Gμ(M_W_BFS_REF) on ISR side per BFS
    # prescription).

    # --- Diagnostics --------------------------------------------------------
    # Standalone BFS NLO Coulomb via delta_NLO path.
    # DOUBLE-COUNTS when include_NLO_hard_decay=True — diagnostic only.
    "diagnostic_bfs_coulomb_nlo": False,
}
