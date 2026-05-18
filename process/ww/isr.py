"""Initial-state radiation convolution for e+e- → μν qq̄ near WW threshold.

Two implementations of the LL+exp electron structure function radiator
in the BETA scheme of Skrzypek (Acta Phys. Pol. B23 (1992) 135) /
Cacciari-Deandrea-Montagna-Nicrosini (Europhys. Lett. 17 (1992) 123),
with full O(α²) exponentiation of soft+virtual. Standard LEP2 Yellow
Report (Beenakker et al., hep-ph/9602351, eq. (GeeLLexp) BETA choice).

(A) **Single-convolution shortcut** (``sigma_ISR_convolution``):

    σ_obs(s) = ∫_{z_min}^1 H(z; s) σ̂(z·s) dz
    H(z;s) = H_SV(β) · β (1-z)^(β-1) + H_NS(z;β)

with β = (2α/π)(L_e-1) and L_e = ln(s/m_e²). This is the LEP2 YR α→2α
shortcut that collapses the two-leg convolution into one variable z = x₁x₂.

(B) **Two-leg form, BFS prescription** (``sigma_ISR_2leg_convolution``):

    σ_obs(s) = ∫₀¹ dx₁ ∫₀¹ dx₂  Γ_ee(x₁) Γ_ee(x₂)  σ̂(x₁ x₂ s)

where each leg carries the *per-leg* BETA-scheme structure function
Γ_ee(x; β) with β = (2α/π)(L_e-1) (same numerical β as in (A) — by LEP2
conventions per-leg β equals the single-conv β; the difference is in
how each form distributes the soft exponent across legs):

    Γ_ee(x; β) = (β/2)(1-x)^(β/2 - 1) · H_SV_per_leg(β)
               + (β/4)(-(1+x))   ← LL non-singular linear-β piece
               - (β²/32) [ (1+3x²)/(1-x) ln(x) + 4(1+x) ln(1-x) + 5 + x ]

    H_SV_per_leg(β) = exp(β(3/4 - γ_E)) / Γ(1+β/2)

This matches BFS's eq. (71) σ_h(s) = ∫∫ dx₁ dx₂ Γ^LL_ee(x₁) Γ^LL_ee(x₂)
σ̂_h^conv(x₁ x₂ s) and is what BFS uses in Table 4 / 5 to obtain the
NLO σ_obs reference numbers (and Table 3's NLO column).

Forms (A) and (B) are formally equivalent at LL+exp; they differ at NLL
by a few permille at √s ≈ 161 GeV. BFS (page 41 around eq. 88) quote
this difference as δm_W ≈ 31 MeV residual ISR uncertainty — the
motivation for the NLL upgrade (analytic Skrzypek-Jadach or eMELA).

Endpoint substitutions (u = (1-z)^β for single-conv, u_i = (1-x_i)^(β/2)
for per-leg) remove the integrable singularity at the soft endpoint;
Gauss-Legendre quadrature on the smoothed integrand.

Validation: 2-leg matches BFS Table 4 σ_obs Born×ISR to <0.5% at 158-170 GeV.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn

from process.ww.eft_xsec import (
    ALPHA_EM_0, M_E,
    ALPHA_S_MW_DEFAULT,
    M_W_DEFAULT, GAMMA_W_DEFAULT, M_W_BFS_REF,
    BFSCorrections,
    alpha_Gmu,
    sigma_partonic_munuqq,
)

EULER_GAMMA = 0.5772156649015329

# BFS prescription (arXiv:0707.0773 line 2514): use α_Gμ in the ISR β. The
# value is evaluated at the BFS reference m_W = 80.377 since the ISR scale
# is the soft/collinear photon, not the W resonance — changing this with
# fit m_W would introduce a fictitious m_W dependence through the ISR
# kernel. Default kept here as a module-level constant; callers can override
# via the explicit ``alpha_em`` argument to ``beta_ISR`` / ``sigma_observed``.
_DEFAULT_ISR_ALPHA = alpha_Gmu(M_W_BFS_REF)   # α_Gμ at the BFS reference m_W; ≈ 1/132.1


# ---------------------------------------------------------------------------
# ISR radiator
# ---------------------------------------------------------------------------

def beta_ISR(s: float, alpha_em: float | None = None) -> float:
    """LL exponent for the e+e- system (both legs combined):
        β = (2α/π) (ln(s/m_e²) - 1).   At √s = 161 GeV: β ≈ 0.113-0.117.

    The α used here is configurable: BFS prescribes "α_Gμ everywhere
    including the initial-state radiation" (arXiv:0707.0773 line 2514).
    The Skrzypek/Cacciari/Beenakker LEP2 YR convention historically uses
    α(0) (Thomson) since the radiated photon is on-shell. The default
    here is α_Gμ at m_W (the BFS prescription) for consistency with the
    rest of the BFS chain — pass ``alpha_em=ALPHA_EM_0`` for the
    historical α(0) convention.
    """
    if alpha_em is None:
        alpha_em = _DEFAULT_ISR_ALPHA
    L_e = np.log(s / (M_E * M_E))
    return (2.0 * alpha_em / np.pi) * (L_e - 1.0)


def H_SV(beta: float) -> float:
    """Soft+virtual exponentiated radiator factor.

    BFS-consistent form (LEP2 YR eq. (GeeLLexp) BETA scheme with the
    α → 2α single-convolution substitution):

        H_SV = exp[β(3/4 − γ_E)] / Γ(1 + β)

    Validation (post-2026-05-18): matches BFS Table 4 σ_obs to better
    than 1 % when convoluted with the corrected H_NS coefficients.

    History: prior versions included an additional β²(9/32 − π²/12)
    term in the exponent, representing the "proper Kuraev-Fadin
    exponentiated" form at NLL accuracy. While that form is more
    accurate at NLL, BFS / LEP2 YR don't use it — keeping it
    produces a ~0.74 % σ_obs deficit vs BFS at β ≈ 0.117. Removed
    to match BFS LL+exp. The full Kuraev-Fadin exponentiation will
    re-enter when the NLL ISR upgrade lands (see
    [[project-followup-nll-isr-plan]]).
    """
    return np.exp(beta * (0.75 - EULER_GAMMA)) / gamma_fn(1.0 + beta)


def H_NS(z, beta: float, one_minus_z=None):
    """Non-singular subleading piece of the radiator.

    Following LEP2 YR eq. (GeeLLexp) with the α → 2α substitution (BETA
    scheme) for the equivalent single-convolution form, BFS-consistent
    LL+exp:

        H_NS(z; β) = -(β/2)(1+z)
                     - (β²/8) [ (1+3z²)/(1-z) ln z
                                + 4(1+z) ln(1-z)
                                + 5 + z ]

    The coefficient 4 on (1+z)ln(1-z) is what comes from the per-leg
    φ(α, x) BETA-scheme T2 coefficient (-(β²/32)·4·(1+x) ln(1-x))
    with α → 2α, i.e. β_per-leg = (2α/π)(L-1) → 2β_per-leg in the
    effective single-convolution → coefficient on (1+z)ln(1-z) is
    -((2β)²/32)·4 = -β²/2 = -(β²/8)·4. Validated by reproducing BFS
    Table 4 σ_obs after ISR convolution.

    H_NS itself diverges logarithmically at z=1 from the -4(1+z) ln(1-z)
    piece; the convolution remains finite because the integrand kernel
    u^{1/β−1} × H_NS suppresses the log as u → 0. Numerically we just
    avoid log(0) by accepting an explicit ``one_minus_z`` argument when
    available (e.g. from the u-substitution), or by clipping (1−z) to
    a floor when only ``z`` is provided.

    Vectorised in ``z``.

    History: prior to 2026-05-18, the coefficient on (1+z)ln(1-z) was
    erroneously 2 (instead of 4), giving a ~1 % σ_obs deficit vs BFS
    Table 4. Fixed and validated 2026-05-18 (validation log item 24).
    """
    z = np.asarray(z, dtype=float)
    if one_minus_z is None:
        one_minus_z = np.maximum(1.0 - z, 1e-300)
    one_minus_z = np.asarray(one_minus_z, dtype=float)
    one_minus_z = np.maximum(one_minus_z, 1e-300)

    log1mz = np.log(one_minus_z)
    # log(z) safely; z near 0 only happens if z_min is very small, irrelevant.
    z_safe = np.maximum(z, 1e-300)
    logz = np.log(z_safe)

    NS1 = -0.5 * beta * (1.0 + z)
    NS2 = -(beta ** 2 / 8.0) * (
        (1.0 + 3.0 * z * z) / one_minus_z * logz
        + 4.0 * (1.0 + z) * log1mz
        + 5.0 + z
    )
    out = NS1 + NS2
    # Below kinematic limit z ≤ 0: no contribution.
    out = np.where(z > 0.0, out, 0.0)

    if np.ndim(z) == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

_LEGGAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _leggauss(n: int):
    cached = _LEGGAUSS_CACHE.get(n)
    if cached is None:
        cached = np.polynomial.legendre.leggauss(n)
        _LEGGAUSS_CACHE[n] = cached
    return cached


def _quad_nodes(n: int, lo: float, hi: float):
    """Gauss-Legendre nodes & weights on [lo, hi] (cached on ``n``)."""
    nodes, weights = _leggauss(n)
    pts = 0.5 * (hi - lo) * (nodes + 1.0) + lo
    wts = 0.5 * (hi - lo) * weights
    return pts, wts


def sigma_ISR_convolution(sqrt_s,
                          sigma_partonic_fn,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          z_min: float = 0.10,
                          n_quad: int = 200,
                          alpha_em: float | None = None,
                          **sigma_kwargs):
    """
    σ_obs(√s) = ∫_{z_min}^1 H(z; s) σ̂(z·s) dz.

    Endpoint substitution u = (1-z)^β → z = 1 − u^{1/β}, dz = −(1/β) u^{1/β−1} du.
    Integrand on [0, u_max] = [0, (1−z_min)^β] is smooth.

    ``alpha_em`` selects the α used to build the LL exponent β_e. Default is
    α_Gμ at m_W (BFS prescription, line 2514 of arXiv:0707.0773); pass
    ``ALPHA_EM_0`` for the historical α(0) Thomson convention.

    Returns σ_obs in pb. ``sigma_partonic_fn`` must accept array-like ``s``.
    Vectorised in ``sqrt_s``: scalar or array.
    """
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)

    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s, alpha_em=alpha_em)
        H_sv = H_SV(beta)
        u_max = (1.0 - z_min) ** beta

        u, w = _quad_nodes(n_quad, 0.0, u_max)
        # z(u) = 1 − u^{1/β};  carry one_minus_z = u^{1/β} explicitly so the
        # H_NS log(1-z) factor never sees a catastrophic 1.0 - 1.0 = 0.
        one_minus_z = u ** (1.0 / beta)
        z_vals = 1.0 - one_minus_z
        s_hat = z_vals * s
        sigma_hat = np.asarray(
            sigma_partonic_fn(s_hat, mW, gammaW, **sigma_kwargs), dtype=float
        )

        # Singular piece: H_sv × σ̂  (Jacobian β(1-z)^{β-1} dz = du absorbed)
        integrand_sing = H_sv * sigma_hat

        # Non-singular piece: (1/β) u^{1/β − 1} × H_NS(z;β) × σ̂
        with np.errstate(over="ignore", invalid="ignore"):
            jac = np.where(u > 1e-300, u ** (1.0 / beta - 1.0) / beta, 0.0)
        NS_vals = H_NS(z_vals, beta, one_minus_z=one_minus_z)
        integrand_ns = jac * NS_vals * sigma_hat

        out[idx] = np.sum(w * (integrand_sing + integrand_ns))

    if np.ndim(sqrt_s) == 0:
        return float(out[0])
    return out


# ---------------------------------------------------------------------------
# Two-leg double convolution (BFS prescription, eq. 71 of 0707.0773)
# ---------------------------------------------------------------------------

def _Gee_per_leg_NS(x, beta: float, one_minus_x=None):
    """Non-singular (linear and β² polynomial) piece of the per-leg
    BETA-scheme radiator, evaluated at x:

        Γ_ee^NS(x; β) = -(β/4)(1+x)
                       - (β²/32) [ (1+3x²)/(1-x) ln(x)
                                   + 4(1+x) ln(1-x) + 5 + x ]

    The (1+x) ln(1-x) coefficient is -β²/8 (= -(β²/32)·4) per-leg,
    matching the LEP2 YR BETA-scheme normalisation.

    Vectorised in x. Pass ``one_minus_x`` explicitly when 1-x is small
    (avoids 1.0 - 1.0 = 0 cancellation from u-substitution).
    """
    x = np.asarray(x, dtype=float)
    if one_minus_x is None:
        one_minus_x = np.maximum(1.0 - x, 1e-300)
    one_minus_x = np.asarray(one_minus_x, dtype=float)
    one_minus_x = np.maximum(one_minus_x, 1e-300)
    x_safe = np.maximum(x, 1e-300)
    log1mx = np.log(one_minus_x)
    logx = np.log(x_safe)

    NS_1 = -(beta / 4.0) * (1.0 + x)
    # LEP2 YR Beenakker hep-ph/9602351 eq. (67): per-leg β² coefficient is
    # −1/(4²·2!) β² = −β²/32 on the bracket [...]; in BETA scheme β_H = β.
    NS_2 = -(beta ** 2 / 32.0) * (
        (1.0 + 3.0 * x * x) / one_minus_x * logx
        + 4.0 * (1.0 + x) * log1mx
        + 5.0 + x
    )
    out = NS_1 + NS_2
    out = np.where(x > 0.0, out, 0.0)
    if np.ndim(x) == 0:
        return float(out)
    return out


def _H_SV_per_leg(beta: float) -> float:
    """Per-leg soft+virtual factor, LEP2 YR Beenakker hep-ph/9602351 eq. (67):

        F(β) = exp(-½ γ_E β + (3/8) β) / Γ(1 + β/2)
             = exp(½ β (3/4 - γ_E)) / Γ(1 + β/2)

    Note the factor ½ in the exponent — NOT β·(3/4-γ_E) as in the
    α→2α single-conv form (where the β there is β_combined = 2 β_per_leg).
    """
    return np.exp(0.5 * beta * (0.75 - EULER_GAMMA)) / gamma_fn(1.0 + beta / 2.0)


def sigma_ISR_2leg_convolution(sqrt_s,
                               sigma_partonic_fn,
                               mW: float = M_W_DEFAULT,
                               gammaW: float = GAMMA_W_DEFAULT,
                               x_min: float = 0.55,
                               n_quad: int = 32,
                               alpha_em: float | None = None,
                               **sigma_kwargs):
    """Two-leg double-convolution ISR (BFS eq. 71):

        σ_obs(s) = ∫_{x_min}^1 dx₁ ∫_{x_min}^1 dx₂  Γ_ee(x₁;β) Γ_ee(x₂;β)
                                                    σ̂(x₁ x₂ s)

    with per-leg β = (2α/π)(L_e - 1).

    For each leg, split Γ_ee = S + NS where S(x) = (β/2)(1-x)^(β/2-1) · H_SV
    is the soft endpoint and NS is the non-singular polynomial. Endpoint
    substitution u_i = (1-x_i)^(β/2) on each leg removes the soft
    singularity:
       dx_i = (2/β) u_i^(2/β - 1) du_i
       S(x_i) dx_i  →  H_SV du_i           (uniform Jacobian × singular ↔ smooth)
       NS(x_i) dx_i →  NS · (2/β) u_i^(2/β-1) du_i

    The total integrand has 4 pieces:
        S·S, S·NS, NS·S, NS·NS
    each a 2D smooth integrand in (u₁, u₂) on [0, u_max]².

    ``x_min`` is the per-leg lower cutoff (default 0.55 so √(x₁ x₂ s) is
    cut at ~0.3·√s — well below threshold, matching the LEP2 YR convention
    of z_min = 0.1 used in the single-convolution form).

    Vectorised in ``sqrt_s``.
    """
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)

    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s, alpha_em=alpha_em)
        H_sv = _H_SV_per_leg(beta)
        half_b = beta / 2.0
        u_max = (1.0 - x_min) ** half_b

        u, w = _quad_nodes(n_quad, 0.0, u_max)
        # 1D vectorisation: build per-leg arrays
        one_minus_x = u ** (1.0 / half_b)
        x_vals = 1.0 - one_minus_x
        # u-side jacobian for NS piece: (2/β) u^(2/β - 1) = u^(1/half_b - 1) / half_b
        with np.errstate(over="ignore", invalid="ignore"):
            jac_NS = np.where(u > 1e-300,
                              u ** (1.0 / half_b - 1.0) / half_b,
                              0.0)
        NS_vals = _Gee_per_leg_NS(x_vals, beta, one_minus_x=one_minus_x)

        # Build 2D meshes:  x₁ on rows, x₂ on cols
        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        s_hat_grid = X1 * X2 * s
        sigma_hat = np.asarray(
            sigma_partonic_fn(s_hat_grid.ravel(), mW, gammaW, **sigma_kwargs),
            dtype=float,
        ).reshape(s_hat_grid.shape)

        # 4-piece integrand decomposition
        # SS:     H_sv * H_sv * σ̂              integrated du₁ du₂
        # SNS:    H_sv * NS₂ * jacNS₂ * σ̂      (S on leg 1, NS on leg 2)
        # NSS:    NS₁ * jacNS₁ * H_sv * σ̂      (symmetric)
        # NSNS:   NS₁·jacNS₁ * NS₂·jacNS₂ * σ̂
        W1, W2 = np.meshgrid(w, w, indexing="ij")
        NS_jac = (NS_vals * jac_NS)  # 1D array per leg
        NS1_grid, NS2_grid = np.meshgrid(NS_jac, NS_jac, indexing="ij")

        integrand = (H_sv * H_sv
                     + H_sv * NS2_grid
                     + NS1_grid * H_sv
                     + NS1_grid * NS2_grid) * sigma_hat

        out[idx] = np.sum(W1 * W2 * integrand)

    if np.ndim(sqrt_s) == 0:
        return float(out[0])
    return out


# ---------------------------------------------------------------------------
# Top-level wrappers
# ---------------------------------------------------------------------------

def sigma_observed_munuqq(sqrt_s,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          channel: str = "inclusive",
                          z_min: float = 0.10,
                          n_quad: int = 200,
                          include_coulomb: bool = True,
                          bfs: BFSCorrections | None = None,
                          br_convention: str = "pdg-constant",
                          # Defaults below are the project's "best calculation"
                          # — full BFS NLO chain + δ_QCD + Whizard anchor + LL+exp ISR.
                          # Single-conv is the default ISR scheme (matches 2-leg
                          # to <0.1% per 2026-05-18 validation, faster 1D quadrature).
                          # alpha_em_isr=None uses module default α_Gμ(M_W_BFS_REF)
                          # per BFS prescription (avoids fictitious m_W-dep in ISR).
                          include_NLO_hard_decay: bool = True,
                          apply_delta_QCD: bool = True,
                          alpha_s: float = ALPHA_S_MW_DEFAULT,
                          alpha_em_isr: float | None = None,
                          apply_whizard_anchor: bool = True,
                          isr_scheme: str = "single_conv"):
    """
    Observed σ(e+e- → μν qq̄) after ISR convolution, in pb. Vectorised in
    ``sqrt_s``. ``br_convention`` and ``include_NLO_hard_decay`` are
    forwarded to ``sigma_partonic_munuqq``.

    ``include_NLO_hard_decay=True`` adds the BFS NLO hard+soft+collinear
    correction (eq. finalcross bracket × √) plus the decay correction
    (eq. 49) to the BFS Born expansion. Combined with the existing K_C
    Coulomb resummation (toggle via ``include_coulomb``) and the optional
    Coulomb NLO α² subleading piece (toggle via ``bfs``), this gives the
    full BFS NLO partonic σ.

    ``isr_scheme`` selects which ISR convolution:

      * ``"single_conv"`` (default) — LEP2 YR α→2α single-convolution
        shortcut, ``sigma_ISR_convolution``. Cheap (1D quadrature) but
        differs from BFS's 2-leg form at NLL (~1.2 pp at 161 GeV).
      * ``"2leg"`` — BFS eq. (71) two-leg double convolution,
        ``sigma_ISR_2leg_convolution``. More accurate, ~n_quad² σ̂ calls.
    """
    common_kwargs = dict(
        mW=mW, gammaW=gammaW,
        alpha_em=alpha_em_isr,
        channel=channel,
        include_coulomb=include_coulomb,
        bfs=bfs,
        br_convention=br_convention,
        include_NLO_hard_decay=include_NLO_hard_decay,
        apply_delta_QCD=apply_delta_QCD,
        alpha_s=alpha_s,
        apply_whizard_anchor=apply_whizard_anchor,
    )
    if isr_scheme == "single_conv":
        return sigma_ISR_convolution(
            sqrt_s, sigma_partonic_munuqq,
            z_min=z_min, n_quad=n_quad,
            **common_kwargs,
        )
    elif isr_scheme == "2leg":
        # Convert single-conv z_min (lower bound on z = x₁ x₂) to a per-leg
        # x_min by taking √z_min — both legs equal at the cutoff edge.
        x_min = float(np.sqrt(z_min))
        # n_quad for 2-leg is per-leg; default to 32 (≈ 1024 σ̂ evals) if
        # the user passed the single-conv default of 200.
        n_q_2leg = 32 if n_quad >= 100 else n_quad
        return sigma_ISR_2leg_convolution(
            sqrt_s, sigma_partonic_munuqq,
            x_min=x_min, n_quad=n_q_2leg,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown isr_scheme: {isr_scheme!r}")


# Legacy alias preserving the previous chat's naming.
def sigma_observed_munuud(sqrt_s, mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT, **kwargs):
    """Legacy alias: σ_obs for the μ⁻ν̄_μ ud̄ channel."""
    return sigma_observed_munuqq(sqrt_s, mW, gammaW, channel="munuud", **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 80)
    print("LL ISR convolution: σ_observed vs σ_partonic for inclusive μν qq̄")
    print("=" * 80)
    print(f"  √s    β_ISR    σ_partonic [pb]   σ_observed [pb]    ratio")
    grid = np.array([157.0, 158.0, 160.0, 161.0, 162.0, 162.5, 163.0, 165.0, 240.0])

    t0 = time.time()
    sigma_o = sigma_observed_munuqq(grid)
    dt = time.time() - t0
    sigma_p = sigma_partonic_munuqq(grid ** 2)

    for sq, sp, so in zip(grid, sigma_p, sigma_o):
        b = beta_ISR(sq * sq)
        ratio = so / sp if sp > 0 else 0.0
        print(f"  {sq:6.2f}  {b:.4f}     {sp:.5f}          {so:.5f}       {ratio:.4f}")
    print(f"\nVectorised convolution over {len(grid)} points: {dt*1000:.1f} ms total")

    print("\n" + "=" * 80)
    print("H_NS values near z=1 — diverges logarithmically (expected)")
    print("=" * 80)
    beta = beta_ISR(161.0 ** 2)
    z_test = np.array([0.5, 0.9, 0.99, 0.999, 0.9999])
    h_ns = H_NS(z_test, beta)
    print(f"β = {beta:.4f}")
    for z, h in zip(z_test, h_ns):
        print(f"  z={z:.6f}   H_NS={h:.6f}")
    print("\nIn the convolution, the (1/β) u^{1/β-1} kernel suppresses the\n"
          "log divergence (integrable singularity).")
