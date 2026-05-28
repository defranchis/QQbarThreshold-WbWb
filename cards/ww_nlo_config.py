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
    # NLL ISR correction (BCFS arXiv:1911.12040 eq. NLLsol3, fixed-α, N_F=0).
    # Adds the bracket {1+(α/π)[C_const+C_log·log(1-x)-log²(1-x)]} to the SV
    # piece of the 2-leg radiator.  Forces isr_scheme="2leg" automatically.
    # OFF by default (LL+exp is the production chain; NLL is the upgrade target).
    "isr_nll":                False,
    # eMELA-LL diagnostic: full DGLAP-evolved BETA-scheme LL instead of the
    # analytic β³-truncated formula.  Quantifies LL truncation (~+0.8% at WW).
    # Mutually exclusive with isr_nll.  Forces 2leg.
    "isr_emela_ll":           False,
    # eMELA scheme knobs.  Production default switched 2026-05-28 from
    # ALGMU (α_Gμ ≈ 1/132.1) to ALPMZ (α(M_Z) ≈ 1/128.9): the BFS hard
    # σ̂ and the ISR β_e are formally independent at NLL, and α(M_Z) is
    # what FCC-ee will actually measure (A_FB^μμ off-peak — Riembau).
    # Paired with PARAM_INPUTS["alpha_em_isr"] = 1/128.943.
    # ALGMU revert is the scheme-variation diagnostic (see reference_emela_nll_isr.md).
    "isr_emela_pert_order":   "NLL",
    "isr_emela_fac_scheme":   "DELTA",
    "isr_emela_ren_scheme":   "ALPMZ",
    # ISR factorisation scale ξ. Central Q = √s; symmetric ξ ∈ {0.5,1,2}
    # is the standard factor-2 envelope. Enters the LL log of β_ISR and
    # the eMELA DGLAP scale (Q = ξ·√s). NLL absorbs the leading scale
    # dependence; residual = N²LL theory unc on ISR.
    "isr_scale_factor":       1.0,
    # α_em values (σ chain + ISR) live in PARAM_INPUTS — both default to
    # derived (α_Gμ on σ side; α_Gμ(M_W_BFS_REF) on ISR side per BFS
    # prescription).

    # --- Diagnostics --------------------------------------------------------
    # Standalone BFS NLO Coulomb via delta_NLO path.
    # DOUBLE-COUNTS when include_NLO_hard_decay=True — diagnostic only.
    "diagnostic_bfs_coulomb_nlo": False,
}
