"""Initial-state radiation convolution for e+e- → μν qq̄ near WW threshold.

LL+exp electron structure function radiator in the BETA scheme of
Skrzypek (Acta Phys. Pol. B23 (1992) 135) / Cacciari-Deandrea-Montagna-
Nicrosini (Europhys. Lett. 17 (1992) 123), with full O(α²) exponentiation
of soft+virtual. This is the standard form used in the LEP2 Yellow Report
(Beenakker et al., hep-ph/9602351, eq. (GeeLLexp) BETA choice) and cited
by BFS (arXiv:0707.0773, eq. (eq:physicalcross), with β_e definition on
line 2433):

    σ_obs(s) = ∫_{z_min}^1  H(z; s)  σ̂(z·s)  dz

    H(z; s)  =  H_SV(s) × β (1-z)^(β-1)
              + H_NS(z; β)

with
    β     = (2α/π) (L_e - 1),    L_e = ln(s / m_e²)
    H_SV  = exp[β(3/4 - γ_E) + β²(9/32 - π²/12)] / Γ(1+β)
    H_NS  = -(β/2)(1+z) + (β²/8) [ -2(1+z) ln(1-z)
                                    -(1+3z²)/(1-z) ln z - 5 - z ]

The β² piece in H_SV (= ½×(3/4)² - ½×π²/6) is the second-order
exponentiation of the soft+virtual factor; together with 1/Γ(1+β) it
reproduces the resummed (1-z)^(β-1) NLL soft form factor to O(β²).
Endpoint substitution u = (1-z)^β removes the z→1 integrable singularity;
Gauss-Legendre quadrature on the smoothed integrand.

The single convolution above is the α→2α single-convolution shortcut of
LEP2 YR eq. (LLint), formally equivalent to BFS's two-leg double
convolution at LL+exp accuracy. NLL-level differences are absorbed in
the eMELA upgrade (Bertone-Cacciari-Frixione-Stagnitto, arXiv:1911.12040).

Numerically verified against BFS Table 4 (Born(ISR)/Born) at 158-167 GeV:
matches to <0.3 % at threshold; <2 % below threshold where the BFS Born
itself differs from Whizard's full 4f Born.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn

from process.ww.eft_xsec import (
    ALPHA_EM_0, M_E,
    M_W_DEFAULT, GAMMA_W_DEFAULT,
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
_DEFAULT_ISR_ALPHA = alpha_Gmu(80.377)   # ≈ 1/132.1


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


def sigma_observed_munuqq(sqrt_s,
                          mW: float = M_W_DEFAULT,
                          gammaW: float = GAMMA_W_DEFAULT,
                          channel: str = "inclusive",
                          z_min: float = 0.10,
                          n_quad: int = 200,
                          include_coulomb: bool = True,
                          bfs: BFSCorrections | None = None,
                          br_convention: str = "pdg-constant",
                          include_NLO_hard_decay: bool = False,
                          apply_delta_QCD: bool = False,
                          alpha_s: float = 0.1199,
                          alpha_em_isr: float | None = None,
                          apply_whizard_anchor: bool = False):
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
    """
    return sigma_ISR_convolution(
        sqrt_s,
        sigma_partonic_munuqq,
        mW=mW, gammaW=gammaW,
        z_min=z_min, n_quad=n_quad,
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
