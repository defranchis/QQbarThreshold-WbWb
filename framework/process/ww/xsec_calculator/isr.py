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

import concurrent.futures
import functools
import math
import multiprocessing
import os

import numpy as np
from scipy.special import gamma as gamma_fn

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

# ISR-quadrature defaults.  Picked by the convergence study at
# scripts/investigations/nll_isr/convergence_study.py (2026-05-28): any
# z_min ≪ z_kin = (2m_W/√s)² ≈ 0.985 is below the WW kinematic threshold
# where σ̂ vanishes, and tighter cutoffs waste GL nodes.
_Z_MIN_DEFAULT      = 0.30                # single-conv lower bound on z = x₁x₂
_X_MIN_2LEG_DEFAULT = math.sqrt(_Z_MIN_DEFAULT)   # per-leg lower bound

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

def beta_ISR(s: float, alpha_em: float | None = None,
             isr_scale_factor: float = 1.0) -> float:
    """LL exponent for the e+e- system (both legs combined):
        β = (2α/π) (ln(ξ²s/m_e²) - 1).   At √s = 161 GeV, ξ=1: β ≈ 0.113-0.117.

    The α used here is configurable: BFS prescribes "α_Gμ everywhere
    including the initial-state radiation" (arXiv:0707.0773 line 2514).
    The Skrzypek/Cacciari/Beenakker LEP2 YR convention historically uses
    α(0) (Thomson) since the radiated photon is on-shell. The default
    here is α_Gμ at m_W (the BFS prescription) for consistency with the
    rest of the BFS chain — pass ``alpha_em=ALPHA_EM_0`` for the
    historical α(0) convention.

    ``isr_scale_factor`` = ξ rescales the ISR factorisation scale
    Q² → ξ² s inside the LL log; symmetric ξ ∈ {0.5, 1, 2} is the
    standard factor-2 scale-variation envelope. The leading log captures
    the bulk of the scale dependence at LL; the eMELA NLL path (in
    sigma_ISR_2leg_convolution) absorbs much of it in the DGLAP
    evolution, leaving the residual N²LL piece as the theory uncertainty.
    """
    if alpha_em is None:
        alpha_em = _DEFAULT_ISR_ALPHA
    L_e = np.log(isr_scale_factor * isr_scale_factor * s / (M_E * M_E))
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
                          z_min: float = _Z_MIN_DEFAULT,
                          n_quad: int = 200,
                          alpha_em_isr: float | None = None,
                          isr_scale_factor: float = 1.0,
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
        beta = beta_ISR(s, alpha_em=alpha_em_isr,
                        isr_scale_factor=isr_scale_factor)
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
                               x_min: float = _X_MIN_2LEG_DEFAULT,
                               n_quad: int = 128,
                               alpha_em_isr: float | None = None,
                               nll: bool = False,
                               emela_ll: bool = False,
                               # eMELA scheme knobs.  Production default
                               # (DELTA + ALPMZ + α(M_Z)) since 2026-05-28
                               # matches the card; ALGMU is the
                               # scheme-variation diagnostic (BFS prescription;
                               # see reference_emela_nll_isr).
                               emela_pert_order: str = "NLL",
                               emela_fac_scheme: str = "DELTA",
                               emela_ren_scheme: str = "ALPMZ",
                               # ISR factorisation scale ξ. Q² → ξ²·s in the
                               # LL log (β_ISR) and Q → ξ·√s in eMELA DGLAP
                               # evolution.  Symmetric ξ ∈ {0.5,1,2} envelope.
                               isr_scale_factor: float = 1.0,
                               n_jobs: int | None = None,
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

    ``x_min`` is the per-leg lower cutoff.  Default √0.30 ≈ 0.5477 so that
    x₁ x₂ ≥ 0.30 at the corner of the integration domain — well below the
    WW kinematic threshold z_kin = (2m_W/√s)² ≈ 0.985 at √s = 161 GeV,
    so σ̂ vanishes for any z < z_kin anyway.  Tighter cutoffs only worsen
    quadrature accuracy by spreading GL nodes over the empty region.

    Vectorised in ``sqrt_s``.

    ``nll=True`` uses the eMELA library (arXiv:1911.12040, DELTA factorisation
    + ALGMU renormalisation) to replace the per-leg LL+exp weight with the
    full NLL electron ePDF.  The per-leg integrand in u-space becomes
    D_NLL(x_i, Q) × |dx/du|_i = CodePdf(11, x_i, omx_i, Q) / x_i × jac_NS_i.

    ``emela_ll=True`` uses the same eMELA library but calls LLPDF(1) (the
    BETA-scheme LL radiator as solved by eMELA's full DGLAP) instead of
    CodePdf.  This differs from the analytic default by the full DGLAP sea
    evolution that our β³-truncated formula omits (~+0.8% at threshold).
    Useful for diagnosing the LL truncation error independently of the NLL
    correction.  Mutually exclusive with ``nll=True``; if both are set,
    ``nll`` takes precedence.

    Near x→1 (omx underflows below 1e-15): the analytic limit H_SV / H_SV_NLL
    is substituted (those nodes contribute negligibly to the sum).
    eMELA must be importable (libeMELApy.so installed via
    scripts/investigations/nll_isr/build_emela_wrapper.sh).

    ``n_jobs`` parallelises the outer √s loop using
    ``ProcessPoolExecutor(mp_context='fork')`` ONLY for the eMELA path
    (``nll`` or ``emela_ll``); the analytic LL+exp path is fast enough that
    fork+pool overhead dominates.  Each child inherits the parent's eMELA
    C++ global state via COW, so no re-init is needed.  ``n_jobs=1`` forces
    serial execution.  ``n_jobs=None`` (default) consults ``WW_ISR_NJOBS``
    in the environment and falls back to 6 when unset — letting batch
    wrappers pin the worker count to ``request_cpus`` without touching
    callers.
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get("WW_ISR_NJOBS", "6"))
    if nll and emela_ll:
        raise ValueError(
            "sigma_ISR_2leg_convolution: nll=True and emela_ll=True are "
            "mutually exclusive (CodePdf vs LLPDF select different eMELA "
            "PDFs); set exactly one."
        )

    # Initialise eMELA once in the parent before any fork (cached globally).
    _use_emela = nll or emela_ll
    if _use_emela:
        from . import emela_wrapper as _emela
        alpha_a = alpha_em_isr if alpha_em_isr is not None else _DEFAULT_ISR_ALPHA
        _emela.initialize(pert_order=emela_pert_order,
                          fac_scheme=emela_fac_scheme,
                          ren_scheme=emela_ren_scheme,
                          alpha=alpha_a)

    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))

    # Parallel dispatch only when the per-point cost justifies the fork+pool
    # overhead — i.e. the eMELA path.  Analytic LL stays serial.
    if _use_emela and n_jobs > 1 and len(sqrt_s_arr) > 1:
        _one = functools.partial(
            sigma_ISR_2leg_convolution,
            sigma_partonic_fn=sigma_partonic_fn,
            mW=mW, gammaW=gammaW, x_min=x_min, n_quad=n_quad,
            alpha_em_isr=alpha_em_isr, nll=nll, emela_ll=emela_ll,
            emela_pert_order=emela_pert_order,
            emela_fac_scheme=emela_fac_scheme,
            emela_ren_scheme=emela_ren_scheme,
            isr_scale_factor=isr_scale_factor,
            n_jobs=1,
            **sigma_kwargs,
        )
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=ctx) as ex:
            return np.array(list(ex.map(_one, sqrt_s_arr)))

    out = np.zeros_like(sqrt_s_arr)

    for idx, sq in enumerate(sqrt_s_arr):
        s = sq * sq
        beta = beta_ISR(s, alpha_em=alpha_em_isr,
                        isr_scale_factor=isr_scale_factor)

        u, w, x_vals, one_minus_x, jac_NS = _endpoint_substitution(
            beta / 2.0, x_min, n_quad)
        NS_vals = _Gee_per_leg_NS(x_vals, beta, one_minus_x=one_minus_x)

        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        sigma_hat = np.asarray(
            sigma_partonic_fn((X1 * X2 * s).ravel(), mW, gammaW, **sigma_kwargs),
            dtype=float,
        ).reshape(X1.shape)

        if _use_emela:
            # eMELA per-leg integrand in u-space:
            #   per_leg[i] = D(x_i, Q) × |dx/du|_i
            #              = xD(x_i, Q) / x_i × jac_NS_i
            # nll=True  → CodePdf (NLL DELTA+ALGMU ePDF)
            # emela_ll  → LLPDF(1) (eMELA full DGLAP LL in BETA scheme)
            # Limit x_i → 1 (omx_i → 0): substitute analytic H_SV limit.
            H_sv_em = _H_SV_per_leg(beta, nll=nll, alpha_em=alpha_a)
            per_leg_em = np.empty_like(x_vals)
            Q = float(sq) * isr_scale_factor
            for i in range(len(x_vals)):
                omx_i = float(one_minus_x[i])
                if omx_i < 1e-15:
                    per_leg_em[i] = H_sv_em
                else:
                    x_i = float(x_vals[i])
                    xD = (_emela.code_pdf(x_i, omx_i, Q) if nll
                          else _emela.ll_pdf(1, x_i, omx_i, Q))
                    per_leg_em[i] = xD / x_i * float(jac_NS[i])
            weight = w * per_leg_em
        else:
            # Per-leg integrand (1D) = singular H_sv (already u-measure) +
            # non-singular jac_NS · NS. The 2-leg double integral factorises:
            #   ∫∫ (S₁+NS₁)(S₂+NS₂) σ̂  =  ∫dw·∫dw  (per_leg_w · per_leg_w · σ̂)
            per_leg = _H_SV_per_leg(beta) + jac_NS * NS_vals
            weight = w * per_leg

        out[idx] = np.einsum("i,j,ij->", weight, weight, sigma_hat)

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
                          z_min: float = _Z_MIN_DEFAULT,
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
                          # eMELA-LL diagnostic: replace analytic β³-truncated
                          # LL+exp with eMELA's DGLAP-evolved BETA-scheme LL.
                          # Quantifies the truncation error (~+0.8% at WW).
                          # Mutually exclusive with isr_nll.
                          isr_emela_ll: bool = False,
                          # eMELA scheme knobs (only used when isr_nll or
                          # isr_emela_ll is True).  Default = BFS prescription.
                          # ren_scheme="ALPMZ" is the α_em_isr nuisance variation.
                          isr_emela_pert_order: str = "NLL",
                          isr_emela_fac_scheme: str = "DELTA",
                          isr_emela_ren_scheme: str = "ALGMU",
                          # ISR factorisation scale ξ ∈ {0.5, 1, 2}. Affects
                          # both LL log (β_ISR) and eMELA DGLAP Q = ξ·√s.
                          isr_scale_factor: float = 1.0,
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
        # ISR factorisation scale ξ consumed by beta_ISR + eMELA Q.
        isr_scale_factor=isr_scale_factor,
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
    # NLL / eMELA-LL are only implemented in the 2-leg form.  Auto-upgrade.
    effective_scheme = isr_scheme
    if (isr_nll or isr_emela_ll) and isr_scheme == "single_conv":
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
            emela_ll=isr_emela_ll,
            emela_pert_order=isr_emela_pert_order,
            emela_fac_scheme=isr_emela_fac_scheme,
            emela_ren_scheme=isr_emela_ren_scheme,
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
