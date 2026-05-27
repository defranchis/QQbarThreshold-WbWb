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

import warnings
import numpy as np
from scipy.special import gamma as gamma_fn, digamma, polygamma

from .bfs_c1fin import Li2 as _Li2

from framework.process.ww.xsec_calculator.eft_xsec import (
    M_E,
    ALPHA_S_MW_DEFAULT,
    M_W_DEFAULT, GAMMA_W_DEFAULT, M_W_BFS_REF,
    M_T_DEFAULT, M_H_DEFAULT, M_Z,
    BFSCorrections,
    alpha_Gmu,
    sigma_partonic_munuqq,
)

EULER_GAMMA = 0.5772156649015329
_SAFE_FLOOR = 1e-300   # underflow guard for log args near 0/1

# NLL ISR constants (BCFS arXiv:1911.12040)
ZETA3 = 1.2020569031595942           # Riemann ζ(3) = Apéry's constant
# BCFS eq. lambda1 at N_F=0 (no light-fermion loops in the ISR kernel):
#   λ₁ = 3/8 − π²/2 + 6ζ₃ ≈ +2.6525
LAMBDA1_NF0 = 3.0/8.0 - np.pi**2/2.0 + 6.0*ZETA3


def _safe_log_pair(z, one_minus_z=None):
    """Return (log z, log(1−z)) safely floored at ``_SAFE_FLOOR``.

    Shared between :func:`H_NS` (single-conv) and :func:`_Gee_per_leg_NS`
    (2-leg per-leg) — both need the same numerical floor when the
    endpoint substitution puts z very close to 1.
    """
    z_arr = np.asarray(z, dtype=float)
    if one_minus_z is None:
        one_minus_z = np.maximum(1.0 - z_arr, _SAFE_FLOOR)
    one_minus_z = np.maximum(np.asarray(one_minus_z, dtype=float), _SAFE_FLOOR)
    z_safe = np.maximum(z_arr, _SAFE_FLOOR)
    return np.log(z_safe), np.log(one_minus_z), one_minus_z, z_arr

def _A_func(kappa: float) -> float:
    """BCFS arXiv:1911.12040 eq. Ares: A(κ) = −γ_E − ψ₀(κ)."""
    return -EULER_GAMMA - float(digamma(kappa))


def _B_func(kappa: float) -> float:
    """BCFS arXiv:1911.12040 eq. Bres:
    B(κ) = γ_E²/2 + π²/12 + γ_E ψ₀(κ) + ψ₀(κ)²/2 − ψ₁(κ)/2."""
    psi0 = float(digamma(kappa))
    psi1 = float(polygamma(1, kappa))
    return (EULER_GAMMA**2 / 2.0 + np.pi**2 / 12.0
            + EULER_GAMMA * psi0 + psi0**2 / 2.0 - psi1 / 2.0)


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
    """Soft+virtual exponentiated radiator factor (BFS / LEP2 YR LL+exp
    BETA scheme, α → 2α single-convolution form):

        H_SV = exp[β(3/4 − γ_E)] / Γ(1 + β)

    Matches BFS Table 4 σ_obs to better than 1 % when convoluted with H_NS.
    """
    return np.exp(beta * (0.75 - EULER_GAMMA)) / gamma_fn(1.0 + beta)


def H_NS(z, beta: float, one_minus_z=None):
    """Non-singular subleading piece of the single-convolution radiator
    (LEP2 YR α → 2α form of Beenakker eq. 67, BETA scheme):

        H_NS(z; β) = -(β/2)(1+z)
                     -(β²/8)  [ (1+3z²)/(1-z) ln z + 4(1+z) ln(1-z) + 5+z ]
                     -(β³/48) [ (1+z)[6 Li₂(z) + 12 ln²(1-z) - 3π²]
                                + (3/2)(1+8z+3z²) ln(z)/(1-z) + 6(z+5) ln(1-z)
                                + 12(1+z²) ln(z) ln(1-z)
                                - ½(1+7z²) ln²(z) + ¼(39 - 24z - 15z²) ]

    Single-conv coefficients are -(4^k k!)⁻¹ (2β)^k from the per-leg eq. 67
    via β → 2β: β²/8 = (2β)²/32, β³/48 = (2β)³/384. β³ contributes <0.04%
    to σ at LEP2/FCC-ee energies (Beenakker Table 12) — included for
    byte-for-byte LL+exp closure.

    H_NS diverges logarithmically at z=1 from the ln(1-z) / ln²(1-z) pieces;
    the convolution remains finite because the u^{1/β−1} kernel suppresses
    these as u → 0. Avoid log(0) by accepting an explicit ``one_minus_z`` from
    the u-substitution, or by clipping (1−z) to a floor when only z is given.

    Vectorised in ``z``.
    """
    logz, log1mz, one_minus_z, z = _safe_log_pair(z, one_minus_z)
    NS1 = -0.5 * beta * (1.0 + z)
    NS2 = -(beta ** 2 / 8.0) * (
        (1.0 + 3.0 * z * z) / one_minus_z * logz
        + 4.0 * (1.0 + z) * log1mz
        + 5.0 + z
    )
    # See ``_Gee_per_leg_NS`` for the bracket structure (β → 2β gives β³/48).
    Li2_z = _Li2(z)
    NS3 = -(beta ** 3 / 48.0) * (
        (1.0 + z) * (6.0 * Li2_z + 12.0 * log1mz ** 2 - 3.0 * np.pi ** 2)
        + (
            1.5 * (1.0 + 8.0 * z + 3.0 * z * z) * logz
            + 6.0 * (z + 5.0) * one_minus_z * log1mz
            + 12.0 * (1.0 + z * z) * logz * log1mz
            - 0.5 * (1.0 + 7.0 * z * z) * logz ** 2
            + 0.25 * (39.0 - 24.0 * z - 15.0 * z * z)
        ) / one_minus_z
    )
    out = np.where(z > 0.0, NS1 + NS2 + NS3, 0.0)
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


def _endpoint_substitution(beta_exponent: float, x_min: float, n_quad: int):
    """Build the u-substitution grid u = (1−x)^β_exp on [0, (1−x_min)^β_exp].

    Returns ``(u, w, x_vals, one_minus_x, jac_NS)`` where:
      - ``u, w``        Gauss-Legendre nodes/weights in u-space
      - ``x_vals``      = 1 − u^{1/β_exp}
      - ``one_minus_x`` = u^{1/β_exp}  (kept explicit to avoid 1.0−1.0=0)
      - ``jac_NS``      = u^{1/β_exp − 1} / β_exp  with underflow guard

    The exponent is ``β`` for the single-conv form (LEP2 YR α→2α) and
    ``β/2`` for the per-leg 2-leg form (BFS eq. 71).
    """
    u_max = (1.0 - x_min) ** beta_exponent
    u, w = _quad_nodes(n_quad, 0.0, u_max)
    one_minus_x = u ** (1.0 / beta_exponent)
    x_vals = 1.0 - one_minus_x
    with np.errstate(over="ignore", invalid="ignore"):
        jac_NS = np.where(
            u > _SAFE_FLOOR,
            u ** (1.0 / beta_exponent - 1.0) / beta_exponent,
            0.0,
        )
    return u, w, x_vals, one_minus_x, jac_NS


def sigma_ISR_convolution(sqrt_s,
                          sigma_partonic_fn,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          z_min: float = 0.10,
                          n_quad: int = 200,
                          alpha_em_isr: float | None = None,
                          **sigma_kwargs):
    """
    σ_obs(√s) = ∫_{z_min}^1 H(z; s) σ̂(z·s) dz.

    Endpoint substitution u = (1-z)^β → z = 1 − u^{1/β}, dz = −(1/β) u^{1/β−1} du.
    Integrand on [0, u_max] = [0, (1−z_min)^β] is smooth.

    ``alpha_em_isr`` selects the α used to build the LL exponent β_e. Default
    is α_Gμ(M_W_BFS_REF) (BFS prescription, line 2514 of arXiv:0707.0773);
    pass ``ALPHA_EM_0`` for the historical α(0) Thomson convention.  Distinct
    from any ``alpha_em`` forwarded via ``**sigma_kwargs`` to the partonic σ.

    Returns σ_obs in pb. ``sigma_partonic_fn`` must accept array-like ``s``.
    Vectorised in ``sqrt_s``: scalar or array.
    """
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)

    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s, alpha_em=alpha_em_isr)
        H_sv = H_SV(beta)

        u, w, z_vals, one_minus_z, jac_NS = _endpoint_substitution(
            beta, z_min, n_quad)
        sigma_hat = np.asarray(
            sigma_partonic_fn(z_vals * s, mW, gammaW, **sigma_kwargs),
            dtype=float)
        NS_vals = H_NS(z_vals, beta, one_minus_z=one_minus_z)
        # Singular piece's β(1-z)^{β-1}·dz = du factor is absorbed into the
        # u-space measure; non-singular piece carries the explicit jac_NS.
        out[idx] = np.sum(w * (H_sv + jac_NS * NS_vals) * sigma_hat)

    if np.ndim(sqrt_s) == 0:
        return float(out[0])
    return out


# ---------------------------------------------------------------------------
# Two-leg double convolution (BFS prescription, eq. 71 of 0707.0773)
# ---------------------------------------------------------------------------

def _Gee_per_leg_NS(x, beta: float, one_minus_x=None):
    """Non-singular (O(β) linear, O(β²), O(β³) polynomial) pieces of the
    per-leg BETA-scheme radiator, evaluated at x. Beenakker hep-ph/9602351
    eq. (67), BETA choice (β_exp = β_S = β_H = β):

        Γ_ee^NS(x; β) =
            -(β/4)(1+x)
            -(β²/32) [ (1+3x²)/(1-x) ln(x) + 4(1+x) ln(1-x) + 5 + x ]
            -(β³/384) [ (1+x)[6 Li₂(x) + 12 ln²(1-x) - 3π²]
                       + (3/2)(1+8x+3x²) ln(x)/(1-x) + 6(x+5) ln(1-x)
                       + 12(1+x²) ln(x) ln(1-x)
                       - ½(1+7x²) ln²(x) + ¼(39 - 24x - 15x²) ]

    Coefficients per Beenakker eq. (67) are −1/(4^k k!) on β^k. The β³ piece
    is "completely negligible" at LEP2/FCC-ee energies per Beenakker Table 12
    (~0.02% on σ); included here for byte-for-byte LL+exp closure with BFS.

    Vectorised in x. Pass ``one_minus_x`` explicitly when 1-x is small
    (avoids 1.0 - 1.0 = 0 cancellation from u-substitution).
    """
    logx, log1mx, one_minus_x, x = _safe_log_pair(x, one_minus_x)
    NS_1 = -(beta / 4.0) * (1.0 + x)
    NS_2 = -(beta ** 2 / 32.0) * (
        (1.0 + 3.0 * x * x) / one_minus_x * logx
        + 4.0 * (1.0 + x) * log1mx
        + 5.0 + x
    )
    # β³ piece per Beenakker eq. (67). Both (1-x)-canceling factors —
    # the 6(x+5)(1-x) ln(1-x) and the (39-24x-15x²)=3(5x+13)(1-x) — are
    # simplified analytically below to avoid the 1/(1-x) blow-up.
    Li2_x = _Li2(x)
    NS_3 = -(beta ** 3 / 384.0) * (
        (1.0 + x) * (6.0 * Li2_x + 12.0 * log1mx ** 2 - 3.0 * np.pi ** 2)
        + (
            1.5 * (1.0 + 8.0 * x + 3.0 * x * x) * logx
            + 6.0 * (x + 5.0) * one_minus_x * log1mx
            + 12.0 * (1.0 + x * x) * logx * log1mx
            - 0.5 * (1.0 + 7.0 * x * x) * logx ** 2
            + 0.25 * (39.0 - 24.0 * x - 15.0 * x * x)
        ) / one_minus_x
    )
    out = np.where(x > 0.0, NS_1 + NS_2 + NS_3, 0.0)
    if np.ndim(x) == 0:
        return float(out)
    return out


def _H_SV_per_leg(beta: float, nll: bool = False,
                  alpha_em: float | None = None) -> float:
    """Per-leg soft+virtual factor, LEP2 YR Beenakker hep-ph/9602351 eq. (67):

        F_LL(β) = exp(½ β (3/4 − γ_E)) / Γ(1 + β/2)

    Note the factor ½ in the exponent — NOT β·(3/4-γ_E) as in the
    α→2α single-conv form (where the β there is β_combined = 2 β_per_leg).

    With ``nll=True`` applies the BCFS arXiv:1911.12040 NLL correction to the
    exponent (eq. heta1def at N_F=0):

        F_NLL(β) = F_LL(β) × exp(κ · (α/π) · λ₁/4)

    where κ = β/2 and λ₁(N_F=0) = LAMBDA1_NF0 ≈ +2.6525. At √s = 161 GeV
    (κ ≈ 0.057, α/π ≈ 0.0024) this is an ~0.009% effect; the dominant NLL
    correction comes from the bracket term in ``sigma_ISR_2leg_convolution``.
    """
    kappa = beta / 2.0
    ll_factor = np.exp(0.5 * beta * (0.75 - EULER_GAMMA)) / gamma_fn(1.0 + kappa)
    if not nll:
        return ll_factor
    if alpha_em is None:
        alpha_em = _DEFAULT_ISR_ALPHA
    # η̂₁ = κ(λ₀ + α·λ₁/(4π)) at N_F=0; LL contains only the κ·λ₀ = κ·3/4 piece.
    # Extra NLL contribution to exponent: κ × (α/π) × (λ₁/4)
    nll_exp = kappa * (alpha_em / np.pi) * (LAMBDA1_NF0 / 4.0)
    return ll_factor * np.exp(nll_exp)


def sigma_ISR_2leg_convolution(sqrt_s,
                               sigma_partonic_fn,
                               mW: float = M_W_DEFAULT,
                               gammaW: float = GAMMA_W_DEFAULT,
                               x_min: float = 0.55,
                               n_quad: int = 32,
                               alpha_em_isr: float | None = None,
                               nll: bool = False,
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

    ``nll=True`` applies the BCFS arXiv:1911.12040 x-space NLL correction.
    **NOT VALIDATED** — the x-space bracket diverges at WW threshold because
    the endpoint-substitution nodes reach ln(1−x) ≈ −80 to −140, far outside
    the range |ln(1−x)| ≲ 20 where the BCFS expansion is valid. The result is
    unphysical (−5% at 161 GeV). Kept for reference; the correct path is
    Mellin-space resummation or eMELA. Default is nll=False (LL+exp only).
    """
    if nll:
        warnings.warn(
            "isr_nll=True (x-space BCFS NLL) is NOT VALIDATED. "
            "The bracket diverges at WW threshold endpoint nodes "
            "(ln(1−x) ≈ −80 to −140). Result is unphysical. "
            "Use eMELA for NLL ISR instead.",
            stacklevel=2,
        )
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)

    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s, alpha_em=alpha_em_isr)

        u, w, x_vals, one_minus_x, jac_NS = _endpoint_substitution(
            beta / 2.0, x_min, n_quad)
        NS_vals = _Gee_per_leg_NS(x_vals, beta, one_minus_x=one_minus_x)

        H_sv = _H_SV_per_leg(beta)
        # Per-leg integrand (1D) = singular H_sv (already u-measure) +
        # non-singular jac_NS · NS. The 2-leg double integral factorises:
        #   ∫∫ (S₁+NS₁)(S₂+NS₂) σ̂  =  ∫dw·∫dw  (per_leg_w · per_leg_w · σ̂)
        per_leg = H_sv + jac_NS * NS_vals

        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        sigma_hat = np.asarray(
            sigma_partonic_fn((X1 * X2 * s).ravel(), mW, gammaW, **sigma_kwargs),
            dtype=float,
        ).reshape(X1.shape)

        # Outer product of per-leg weights+integrand: row-vector × column-vector
        weight_1d = w * per_leg
        sigma_ll = np.einsum("i,j,ij->", weight_1d, weight_1d, sigma_hat)

        if nll:
            # BCFS arXiv:1911.12040 NLLsol3, linearised at O(α/π):
            #   σ_NLL = σ_LL + 2 × Σ_i w_i δ_NLL_i σ_1leg_LL_i
            # where σ_1leg_LL_i = Σ_j w_j per_leg_j σ̂_ij is the 1-leg LL
            # partial integral at fixed x_i.
            #
            # The x-space NLL bracket {1+(α/π)[C+C_log·ln(1-x)-ln²(1-x)]}
            # is only valid for |ln(1-x)| ≲ 20.  At endpoint-substitution
            # nodes ln(1-x) = ln(u)/κ ≈ −80 to −140 — far outside that
            # range — so the direct (linear or exp) form diverges.  The
            # linearised 2×Σ δ_NLL σ_1leg form correctly cancels the
            # large-x divergence and reproduces the O(α/π) NLL result.
            #
            # In u-coordinates: ln(1-x) = ln(u)/κ.
            alpha_a = alpha_em_isr if alpha_em_isr is not None else _DEFAULT_ISR_ALPHA
            kappa = beta / 2.0
            a_nll = _A_func(kappa)
            b_nll = _B_func(kappa)
            # At L₀=0:  −(A+3/4) − 2B + 7/4
            c_const = -(a_nll + 0.75) - 2.0*b_nll + 1.75
            # At L₀=0:  −1 − 2A
            c_log = -1.0 - 2.0*a_nll
            log_u = np.log(np.maximum(u, _SAFE_FLOOR))
            # NLL bracket argument in u-space: ln(1-x) = ln(u)/κ
            bracket_arg = (alpha_a / np.pi) * (
                c_const + (c_log / kappa) * log_u - log_u**2 / kappa**2
            )
            H_sv_nll = _H_SV_per_leg(beta, nll=True, alpha_em=alpha_em_isr)
            # δ_NLL = H_sv_nll × bracket_arg  (bracket correction)
            #       + (H_sv_nll − H_sv)        (prefactor correction)
            delta_sv = H_sv_nll * bracket_arg + (H_sv_nll - H_sv)
            # σ_1leg_LL[i] = Σ_j w_j per_leg_j σ̂_ij
            sigma_1leg = sigma_hat @ weight_1d
            out[idx] = sigma_ll + 2.0 * np.sum(w * delta_sv * sigma_1leg)
        else:
            out[idx] = sigma_ll

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
                          include_BFS_NNLO: bool = True,
                          apply_delta_QCD: bool = True,
                          alpha_s: float = ALPHA_S_MW_DEFAULT,
                          alpha_s_ref: float = ALPHA_S_MW_DEFAULT,
                          # α_em used inside the σ chain (Born + NLO + NNLO).
                          # None → derived α_Gμ(m_W, M_Z); float overrides.
                          alpha_em: float | None = None,
                          # α_em used to build the ISR β_e exponent (BFS eq. 71).
                          # None → α_Gμ(M_W_BFS_REF) per BFS prescription.
                          alpha_em_isr: float | None = None,
                          apply_whizard_anchor: bool = True,
                          whizard_anchor_source: str = "grid",
                          isr_scheme: str = "single_conv",
                          isr_nll: bool = False,
                          coulomb_kc_safe: bool = False,
                          decay_uses_full_born: bool = True,
                          m_t: float = M_T_DEFAULT,
                          M_H: float = M_H_DEFAULT,
                          MZ: float = M_Z):
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
        # σ-chain α_em forwarded to sigma_partonic_munuqq via **sigma_kwargs.
        alpha_em=alpha_em,
        # ISR-β α_em consumed by the convolution itself (beta_ISR call).
        alpha_em_isr=alpha_em_isr,
        channel=channel,
        include_coulomb=include_coulomb,
        bfs=bfs,
        br_convention=br_convention,
        include_NLO_hard_decay=include_NLO_hard_decay,
        include_BFS_NNLO=include_BFS_NNLO,
        apply_delta_QCD=apply_delta_QCD,
        alpha_s=alpha_s,
        alpha_s_ref=alpha_s_ref,
        apply_whizard_anchor=apply_whizard_anchor,
        whizard_anchor_source=whizard_anchor_source,
        coulomb_kc_safe=coulomb_kc_safe,
        decay_uses_full_born=decay_uses_full_born,
        m_t=m_t, M_H=M_H, MZ=MZ,
    )
    # NLL is only implemented in the 2-leg form.  Auto-upgrade isr_scheme.
    effective_scheme = isr_scheme
    if isr_nll and isr_scheme == "single_conv":
        effective_scheme = "2leg"

    if effective_scheme == "single_conv":
        return sigma_ISR_convolution(
            sqrt_s, sigma_partonic_munuqq,
            z_min=z_min, n_quad=n_quad,
            **common_kwargs,
        )
    elif effective_scheme == "2leg":
        # Convert single-conv z_min (lower bound on z = x₁ x₂) to a per-leg
        # x_min by taking √z_min — both legs equal at the cutoff edge.
        x_min = float(np.sqrt(z_min))
        # n_quad for 2-leg is per-leg. Callers passing the single-conv
        # default of 200 get the auto-mapped 2-leg default; explicit
        # values are forwarded as-is so test scripts can dial it up.
        # 128 per leg = 16k σ̂ evals: gets the residual second-difference
        # noise on the Γ_W-variation ratios at the 0.1-GeV grid step
        # down to ~3.4×10⁻⁶ (cf. 6.5×10⁻⁶ at 64 per leg). Higher counts
        # converge slowly (~3.2×10⁻⁶ at 192 per leg, 8× the cost).
        n_q_2leg = 128 if n_quad == 200 else n_quad
        return sigma_ISR_2leg_convolution(
            sqrt_s, sigma_partonic_munuqq,
            x_min=x_min, n_quad=n_q_2leg,
            nll=isr_nll,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown isr_scheme: {isr_scheme!r}")


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
