"""MoCaNLO beta-scheme ISR radiator + quadrature convolution.

This is the ISR layer of the **independent, BFS-free** WW-threshold line-shape
calculation.  It convolves a fixed-order NLO-EW *partonic* cross section
σ̂(√ŝ) — produced by an unmodified MoCaNLO run with ``pdf_set=none`` (no beam
ISR) — with the LL+exp electron structure function, which carries ALL the
initial-state radiation (see "No O(α) re-subtraction" below).

Provenance / independence
--------------------------
The radiator ``D(x)`` is transcribed **directly from the MoCaNLO source**,
``src/mocanlo/pdfs/lepton_pdfs.F90`` subroutine ``lepton_pdf_convolution``
(case ``LO_beta`` etc.), itself appendix A of arXiv:2207.03265 (Bertone,
Cacciari, Frixione, Stagnitto, Zaro, Zhao).  The analytic LL chain deliberately
does **not** import the project's BFS ``xsec_calculator.isr`` module and shares
no code or numerical input with the BFS-EFT calculation.  (The two radiators
are mathematically the same LEP-YR structure function, so a numerical
cross-check against ``isr.sigma_ISR_2leg_convolution`` is a useful
*validation*, performed in ``tests``/``__main__`` — but it is not a
dependency.)  The NLL path is the one deliberate exception: it shares the
eMELA library (``emela_wrapper``) and the single λ₁(N_F=0) constant
(``LAMBDA1_NF0``, imported in ``_norm_nll_endpoint``) with the BFS chain —
same NLL convention by design, not an independence leak in the σ̂ inputs.

Why decoupled quadrature
-------------------------
MoCaNLO's *native* lepton-PDF ISR (``pdf_set=LO_beta``) is shipped untested and
its importance sampling of the ``(1-x)^(β-1)`` soft endpoint is not implemented
→ NaN weights, σ→0.  The structure-function *formula* is correct; only its MC
sampling is broken.  So we take MoCaNLO's validated ``pdf_set=none`` partonic
σ̂ and do the x-integral here by deterministic Gauss-Legendre quadrature with an
endpoint substitution that tames the soft singularity analytically.

Conventions
-----------
* Per-leg LL exponent, MoCaNLO ``LO_beta``:
      β = (α/π)(2 ln(μ_F/m_e) − 1)
  α is the EW-scheme α (Gμ → α_Gμ = 7.5553e-3 by default; switchable for
  theory uncertainty).  m_e is the physical electron mass (the ISR regulator).
* Factorisation scale μ_F: default ``μ_F = √s`` (the textbook e+e- ISR choice).
  MoCaNLO's native run tied μ_F to the card ``factorization_scale`` (= m_W in
  our cards); the μ_F choice is carried as an ISR theory-uncertainty knob
  (``mu_F_factor`` rescales √s; ``mu_F_abs`` pins an absolute scale).
* Two-leg double convolution (independent radiation off e⁻ and e⁺):
      σ_obs(s) = ∫∫ dx₁ dx₂ D(x₁) D(x₂) σ̂(x₁ x₂ s).

No O(α) re-subtraction
----------------------
The original design assumed σ̂_NLO(ŝ) from ``pdf_set=none`` to be the
mass-regularised fixed-order result, with the explicit O(α) initial-state
collinear log ln(ŝ/m_e²) inside — and subtracted the overlap C₁[σ̂_Born]
(the O(α) piece of the radiator) to avoid double counting.

The production grids show that premise is FALSE: MoCaNLO's ``idip`` run
applies the initial-state collinear counterterm, so the ISR log is
factorised *out* of σ̂.  Evidence (lnuqq, √s=163): real = +4.2 %,
idip = +1.7 %, σ̂_NLO/σ̂_Born = +8.3 % — no trace of the −30 % radiative
tail — and σ̂_NLO agrees with the ISR-free BFS-EFT partonic to 1.2 %.
The observed line shape is therefore the plain double convolution

      σ_obs(s) = ∫∫ D D σ̂_NLO(x₁x₂s)

which counts O(α) ISR exactly once: the collinear log lives in D, the
finite remnant in σ̂_NLO.  Subtracting C₁ on top (the pre-2026-06-12
behaviour) cancels the physical ISR damping — it inflated σ_obs by ~48 %
(e.g. σ_WW(161.3 GeV) ≈ 5.4 pb against the LEP measurement 3.69±0.45 pb;
fixed value 3.52 pb, and 5.41 pb at 163 GeV vs BFS-NLL 5.30 / YFSWW3 ≈5.5).

A residual O(α) finite-scheme remnant remains between σ̂_NLO's MoCaNLO
counterterm (idip/CS) convention and the radiator's O(α) finite part — DELTA
for the production eMELA-NLL radiator, the ``LO_beta`` D₁ on the LL diagnostic
path.  The rate-closure comparisons above bound only its NORMALISATION (≲ few
%), and that piece is lumi/σ-norm-degenerate, so it does NOT bias m_W.  The
SHAPE component — the part that *can* bias m_W — is NOT constrained by rate
closure; it is bounded separately by the ISR scheme/order shape studies in the
theory budget (DELTA↔MSBAR ≈0.14 MeV; eMELA-LL↔eMELA-NLL truncation ≈0.98 MeV
shape on the indep chain) — see report sec:val-isr-cross.
Diagnostics: ``scripts/investigations/three_calc_xsec/``.

All cross sections are in **fb** (MoCaNLO's unit).  ``sigma_hat_fn`` callables
take √ŝ in GeV and return σ̂ in fb; they are typically interpolators over the
MoCaNLO partonic grid.
"""

from __future__ import annotations

import glob
import hashlib
import math
import os
import pickle
import socket
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma_fn
from scipy.special import spence as _spence


# ---------------------------------------------------------------------------
# Physical constants & scheme α values (independent of the BFS chain)
# ---------------------------------------------------------------------------

#: Physical electron mass [GeV] — the ISR collinear regulator (PDG).  Matches
#: MoCaNLO's default ``pdf_starting_scale``/electron mass 0.511e-3 used to set β.
M_E = 0.51099895069e-3

EULER_GAMMA = 0.5772156649015329
_PI = math.pi

#: EW-scheme fine-structure constants used to build the LL exponent β.
#: Same numerical values MoCaNLO assigns to ``sm_parameters%alpha`` per
#: ``<scheme_alpha>`` (see lepton_pdfs.F90: ``alpha_pdf = ...%alpha``).
ALPHA_GMU = 7.5552976871e-3      # Gμ scheme (production default)
ALPHA_0 = 7.2973525693e-3        # α(0) Thomson
ALPHA_MZ = 7.7983970817e-3       # α(M_Z), MoCaNLO's lepton-PDF value (1/128.232)

#: α(M_Z) for the eMELA NLL radiator (ALPMZ scheme): the PDG value 1/128.943,
#: paired with eMELA's ALPMZ renormalisation and matching the BFS production
#: chain (cards/ww_default.py PARAM_INPUTS["alpha_em_isr"], isr.py). This is the
#: documented production α(M_Z), NOT MoCaNLO's lepton-PDF ``ALPHA_MZ`` above —
#: the two α(M_Z) values differ by 0.55%. The NLL path is an ISR object (QED off
#: the e± line) independent of the σ̂ grid's EW scheme (gf), so it takes the
#: standard PDG/eMELA α(M_Z), not MoCaNLO's internal one.
#: Kept as a separate literal BY DESIGN (not hand-sync drift): this independent
#: chain shares no code with the BFS chain (see module docstring), so it does not
#: import eft_xsec.ALPHA_MZ_PDG. Same physical value (PDG α(M_Z)); if PDG updates,
#: both must move together.
ALPHA_MZ_EMELA = 1.0 / 128.943   # α(M_Z) PDG, eMELA ALPMZ-paired

_ALPHA_BY_SCHEME = {
    "gf": ALPHA_GMU,
    "gmu": ALPHA_GMU,
    "alpha0": ALPHA_0,
    "alpha(0)": ALPHA_0,
    "alphaz": ALPHA_MZ,
    "alpha(mz)": ALPHA_MZ,
}

#: Recognised MoCaNLO LL PDF scheme tags (lepton_pdfs.F90 ``pdf_set``).
ISR_SCHEMES = ("LO_beta", "LO_eta", "LO_mixed", "LO_collinear")

_SAFE_FLOOR = 1e-300


def alpha_for_scheme(scheme: str) -> float:
    key = scheme.strip().lower()
    if key not in _ALPHA_BY_SCHEME:
        raise ValueError(
            f"unknown EW α scheme {scheme!r}; choose from {sorted(_ALPHA_BY_SCHEME)}")
    return _ALPHA_BY_SCHEME[key]


def _Li2(x):
    """Real dilogarithm Li₂(x) for x ≤ 1, via scipy.special.spence.

    scipy's ``spence(z)`` = ∫₁ᶻ ln t/(1−t) dt = Li₂(1−z), so Li₂(x)=spence(1−x).
    Transcription of MoCaNLO's ``real(dilog_cll(complex(x,0)))`` on the physical
    sheet (x ∈ [0,1]).
    """
    return _spence(1.0 - np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# β exponents — MoCaNLO lepton_pdfs.F90 cases 1-4
# ---------------------------------------------------------------------------

def beta_components(mu_F: float, *, scheme: str = "LO_beta",
                    alpha: float = ALPHA_GMU, m_e: float = M_E,
                    pdf_starting_scale: float = M_E):
    """(β_e, β_s, β_h) per leg at factorisation scale ``mu_F``.

    Direct transcription of ``lepton_pdf_convolution`` (lepton_pdfs.F90 124-138):

      LO_beta:      β_e = β_s = β_h = (α/π)(2 ln(μ_F/m_e) − 1)
      LO_eta:       β_e = β_s = (α/π)(2 ln−1);  β_h = (α/π)(2 ln)
      LO_mixed:     β_e = (α/π)(2 ln−1);  β_s = β_h = (α/π)(2 ln)
      LO_collinear: β_e = β_s = β_h = (α/π)(2 ln(μ_F/pdf_starting_scale))

    β_e drives the soft-photon ``(1−x)^(β_e−1)`` endpoint, β_s the
    soft+virtual exponentiation prefactor, β_h the hard-collinear polynomial.
    """
    a = alpha / _PI
    L1 = 2.0 * math.log(mu_F / m_e)            # 2 ln(μ_F/m_e)
    if scheme == "LO_beta":
        b = a * (L1 - 1.0)
        return b, b, b
    if scheme == "LO_eta":
        be = a * (L1 - 1.0)
        return be, be, a * L1
    if scheme == "LO_mixed":
        be = a * (L1 - 1.0)
        bh = a * L1
        return be, bh, bh
    if scheme == "LO_collinear":
        b = a * (2.0 * math.log(mu_F / pdf_starting_scale))
        return b, b, b
    raise ValueError(f"unknown ISR scheme {scheme!r}; choose from {ISR_SCHEMES}")


def _radiator_norm(beta_e: float, beta_s: float) -> float:
    """Soft+virtual exponentiated prefactor of the singular endpoint:
        N = exp(3 β_s/4 − γ_E β_e) / Γ(1 + β_e)
    (lepton_pdfs.F90 line 154 prefactor).
    """
    return math.exp(0.75 * beta_s - EULER_GAMMA * beta_e) / _gamma_fn(1.0 + beta_e)


def _norm_nll_endpoint(beta_e: float, norm: float, alpha: float) -> float:
    """Exact soft+virtual per-leg NLL endpoint: the LL+exp prefactor ``norm``
    (= :func:`_radiator_norm`) times the BCFS NLL exponent correction
    ``exp(β_e·(α/π)·λ₁/4)`` (arXiv:1911.12040; λ₁ = ``xsec_calculator.isr.LAMBDA1_NF0``).
    SINGLE SOURCE shared by ``_per_leg_emela_nll`` / ``_per_leg_grid_nll`` here and
    ``isr_lumi._norm_nll``: the per-leg radiator → this value as x→1, and the
    luminosity ρ̃(v) → β_e·this as v→0.  (The BFS-side ``xsec_calculator.isr`` writes
    the same physics with κ=β_e/2; keep that port in sync separately.)"""
    from framework.process.ww.xsec_calculator.isr import LAMBDA1_NF0
    return norm * math.exp(beta_e * (alpha / _PI) * (LAMBDA1_NF0 / 4.0))


def _radiator_NS(x, beta_h: float, *, one_minus_x=None):
    """Non-singular (hard-collinear) part of the single-leg radiator D(x).

    Transcription of lepton_pdfs.F90 lines 155-160 (the β_h¹, β_h², β_h³
    polynomial pieces; the singular β_e·(1−x)^(β_e−1) term is handled by the
    endpoint substitution, not here):

        D_NS(x) = − β_h (1+x)/2
                  − β_h²/8 [ (1+3x²)/(1−x) ln x + 4(1+x) ln(1−x) + 5 + x ]
                  − β_h³/48 [ (1+x)(6 Li₂(x) + 12 ln²(1−x) − 3π²)
                              + ( 1.5(1+8x+3x²) ln x + 6(x+5)(1−x) ln(1−x)
                                  + 12(1+x²) ln x ln(1−x) − 0.5(1+7x²) ln²x
                                  + 0.25(39−24x−15x²) ) / (1−x) ]

    The ``/(1−x)`` terms are individually integrable (ln x ∼ −(1−x) and
    39−24x−15x² = 3(1−x)(5x+13) cancel the pole); ``one_minus_x`` is passed
    explicitly from the u-substitution so 1−x is never the catastrophic
    ``1.0 − 1.0``.  Vectorised in x.
    """
    x = np.asarray(x, dtype=float)
    if one_minus_x is None:
        one_minus_x = 1.0 - x
    omx = np.asarray(one_minus_x, dtype=float)
    omx_safe = np.maximum(omx, _SAFE_FLOOR)
    x_safe = np.maximum(x, _SAFE_FLOOR)
    logx = np.log(x_safe)
    log1mx = np.log(omx_safe)

    NS1 = -beta_h * (1.0 + x) / 2.0
    NS2 = -beta_h ** 2 / 8.0 * (
        (1.0 + 3.0 * x * x) / omx_safe * logx
        + 4.0 * (1.0 + x) * log1mx
        + 5.0 + x
    )
    Li2x = _Li2(x)
    NS3 = -beta_h ** 3 / 48.0 * (
        (1.0 + x) * (6.0 * Li2x + 12.0 * log1mx ** 2 - 3.0 * _PI ** 2)
        + (
            1.5 * (1.0 + 8.0 * x + 3.0 * x * x) * logx
            + 6.0 * (x + 5.0) * omx * log1mx
            + 12.0 * (1.0 + x * x) * logx * log1mx
            - 0.5 * (1.0 + 7.0 * x * x) * logx ** 2
            + 0.25 * (39.0 - 24.0 * x - 15.0 * x * x)
        ) / omx_safe
    )
    out = np.where(x > 0.0, NS1 + NS2 + NS3, 0.0)
    return float(out) if np.ndim(x) == 0 else out


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature with soft-endpoint substitution
# ---------------------------------------------------------------------------

_LEGGAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _leggauss(n: int):
    cached = _LEGGAUSS_CACHE.get(n)
    if cached is None:
        cached = np.polynomial.legendre.leggauss(n)
        _LEGGAUSS_CACHE[n] = cached
    return cached


def _quad_nodes(n: int, lo: float, hi: float):
    nodes, weights = _leggauss(n)
    pts = 0.5 * (hi - lo) * (nodes + 1.0) + lo
    wts = 0.5 * (hi - lo) * weights
    return pts, wts


def _endpoint_grid(beta_e: float, x_min: float, n_quad: int):
    """u = (1−x)^β_e substitution on a single leg.

    Returns (u, w, x_vals, one_minus_x, jac_NS) where for the smooth NS part
    ∫ NS(x) dx = ∫ NS(x(u)) · jac_NS du with jac_NS = u^{1/β_e−1}/β_e, and the
    singular part ∫ N·β_e(1−x)^{β_e−1} dx = ∫ N du (uniform in u).
    """
    u_max = (1.0 - x_min) ** beta_e
    u, w = _quad_nodes(n_quad, 0.0, u_max)
    one_minus_x = u ** (1.0 / beta_e)
    x_vals = 1.0 - one_minus_x
    with np.errstate(over="ignore", invalid="ignore"):
        jac_NS = np.where(u > _SAFE_FLOOR,
                          u ** (1.0 / beta_e - 1.0) / beta_e, 0.0)
    return u, w, x_vals, one_minus_x, jac_NS


# ---------------------------------------------------------------------------
# Radiator configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISRConfig:
    """Knobs for the beta-scheme radiator + matching.

    scheme        MoCaNLO LL PDF tag (LO_beta default; eta/mixed/collinear
                  are the ISR-scheme theory-uncertainty variations).
    alpha         EW-scheme α in β.  Default α_Gμ; α(0)/α(M_Z) are the
                  EW-scheme theory variations.  ``alpha=None`` + ``ew_scheme``
                  resolves it from the scheme name.
    ew_scheme     name used only when ``alpha is None`` (gf/alpha0/alphaz).
    mu_F_factor   μ_F = mu_F_factor · √s  (ISR factorisation scale; ξ knob).
    mu_F_abs      if set (>0), pins μ_F to this absolute value (GeV), ignoring
                  mu_F_factor — e.g. m_W to mimic MoCaNLO's native μ_F choice.
    m_e           ISR regulator mass.
    x_min         per-leg lower cutoff (x₁x₂ ≥ x_min² stays below WW threshold).
    n_quad        GL nodes per leg.
    nll           replace the analytic LL+exp per-leg radiator with eMELA's
                  NLL electron ePDF (DGLAP-evolved); production for the NLL
                  chain since 74f110a (dofit_indep default).  There is NO O(α)
                  matching subtraction (see module docstring "No O(α) re-
                  subtraction"): the σ̂ grids are beam-ISR-free, so the radiator
                  alone supplies the ISR and σ_obs = ∫∫ D D σ̂_NLO.
                  Configure with alpha=ALPHA_MZ_EMELA (PDG α(M_Z)=1/128.943, NOT
                  MoCaNLO's lepton-PDF ALPHA_MZ) + emela_ren_scheme="ALPMZ" to
                  match the BFS production NLL convention; the precomputed grid
                  must be baked at the same α/scheme (enforced by the guard in
                  ``_per_leg_grid_nll``).
    """
    scheme: str = "LO_beta"
    alpha: float | None = ALPHA_GMU
    ew_scheme: str = "gf"
    mu_F_factor: float = 1.0
    mu_F_abs: float = 0.0
    m_e: float = M_E
    x_min: float = math.sqrt(0.30)
    n_quad: int = 128
    nll: bool = False
    emela_fac_scheme: str = "DELTA"
    emela_ren_scheme: str = "ALPMZ"
    #: Path to a precomputed eMELA grid (.npz from isr_emela_grid.build_and_write).
    #: When set AND nll=True, the per-leg NLL radiator interpolates x·D from the
    #: grid instead of calling eMELA's DGLAP solver per node — √s/μ_F/quadrature-
    #: independent, no eMELA runtime dep.  The analytic norm_nll endpoint
    #: (omx<1e-15) is unchanged.  "" = direct eMELA.  PROD_EMELA_GRID via the
    #: generator is the production NLL route (feeds the luminosity form).
    emela_grid: str = ""
    #: Route the two-leg convolution through the 1-D LUMINOSITY self-convolution
    #: (``isr_lumi``) instead of the 2-D ``convolve_2leg`` einsum.  Requires
    #: ``nll=True`` AND ``emela_grid`` set (the per-leg ρ̃ source).  Faithful to
    #: the 2-D (reproduces its ripple-free many-n_quad mean to ~tens of ppm) but
    #: RIPPLE-FREE and smoother: the 2-D's plain Gauss-Legendre straddles the σ̂
    #: grid-edge step at √ŝ=156 (point-wise ripple ∝1/n_quad), whereas the
    #: luminosity makes that step the outer integration LIMIT.  PRODUCTION for
    #: the NLL chain since 74f110a: the field default stays False, but
    #: WWGeneratorMoCaNLO (isr_lumi=None auto) routes NLL fits through it —
    #: scheme-variation callers building their own cfg keep the 2-D unless they
    #: opt in.
    lumi: bool = False

    def resolved_alpha(self) -> float:
        return self.alpha if self.alpha is not None else alpha_for_scheme(self.ew_scheme)

    def mu_F(self, sqrt_s: float) -> float:
        return self.mu_F_abs if self.mu_F_abs > 0.0 else self.mu_F_factor * sqrt_s

    def betas(self, sqrt_s: float):
        return beta_components(self.mu_F(sqrt_s), scheme=self.scheme,
                               alpha=self.resolved_alpha(), m_e=self.m_e)


# ---------------------------------------------------------------------------
# Two-leg convolution
# ---------------------------------------------------------------------------

def _per_leg_emela_nll(cfg, be, norm, x_vals, one_minus_x, jac_NS, sqrt_s):
    """Per-leg eMELA NLL radiator weight in u-space (EXPLORATORY).

    Mirrors ``isr.sigma_ISR_2leg_convolution``'s eMELA handling: the full ePDF
    xD(x,Q) replaces the analytic norm+NS split.  ``jac_NS`` here is the FULL,
    universal |dx/du| = u^(1/β_e−1)/β_e (the name is historical — for the LL path
    it multiplies only the NS polynomial, but for the NLL ePDF it multiplies the
    whole xD/x).  xD/x · jac_NS is finite at the soft endpoint — the ePDF's
    (1-x)^(β-1) singularity cancels the jacobian's u^(1/β-1).  At x→1
    (omx<1e-15) the analytic NLL soft+virtual limit is used (norm × the BCFS
    arXiv:1911.12040 exponent correction exp(β_e·(α/π)·λ₁/4)).
    Imports eMELA lazily so the default LL path keeps no BFS/eMELA dependency.

    KNOWN NLL SYSTEMATIC (omx<1e-15 endpoint substitution, shared with the BFS-
    side isr.py — do NOT change one without the other or the validated machine-
    precision port closure breaks).  Because u=omx^β_e with β_e≈0.06 (1/β_e≈17),
    the smallest GL nodes reach omx≈1e-66, so ~30/128 nodes fall below the 1e-15
    cutoff and carry ~13 % of the per-leg integral weight; there the flat analytic
    ``norm_nll`` is used instead of ``code_pdf``.

    QUANTIFIED 2026-06-03 — the cutoff is CORRECT, do NOT lower it.  As x→1 the
    integrand MUST approach the exact soft+virtual constant ``norm_nll``; eMELA's
    ``code_pdf`` agrees with it to ~0.2 % where its numerics are valid (omx≈1e-7
    to 1e-9) but then DIVERGES monotonically away, reaching code/analytic≈+6 % at
    omx≈1e-66 (the divergence sets in exactly where x underflows to 1.0).  So the
    analytic ``norm_nll`` is the trusted endpoint value and the 1e-15 cutoff
    shields the result from eMELA's x→1 grid-edge artifact.  LOWERING the cutoff
    (calling code_pdf on the soft nodes) would IMPORT that artifact: +0.53 % on
    the line shape, a shape-only m_W bias of only −0.11 MeV (lumi-weighted) but in
    the WRONG direction.  Net: the cutoff avoids a ~0.1 MeV error; it is not a
    systematic on the current result.  (Same substitution lives in the production
    isr.py — keep them in sync.)  scripts/investigations/bfs_match/ analysis.
    """
    from framework.process.ww.xsec_calculator import emela_wrapper as _emela
    alpha = cfg.resolved_alpha()
    _emela.initialize(pert_order="NLL", fac_scheme=cfg.emela_fac_scheme,
                      ren_scheme=cfg.emela_ren_scheme, alpha=alpha)
    Q = cfg.mu_F(sqrt_s)
    norm_nll = _norm_nll_endpoint(be, norm, alpha)
    per_leg = np.empty_like(x_vals)
    for i in range(len(x_vals)):
        omx_i = float(one_minus_x[i])
        if omx_i < 1e-15:
            per_leg[i] = norm_nll
        else:
            x_i = float(x_vals[i])
            per_leg[i] = _emela.code_pdf(x_i, omx_i, Q) / x_i * float(jac_NS[i])
    return per_leg


def _per_leg_grid_nll(cfg, be, norm, x_vals, one_minus_x, jac_NS, sqrt_s):
    """Per-leg NLL radiator weight via the LHAPDF-style precomputed grid
    (EXPLORATORY; cfg.emela_grid set).  Numerically the same construction as
    ``_per_leg_emela_nll`` — same analytic ``norm_nll`` soft+virtual endpoint for
    omx<1e-15, same xD/x·|dx/du| in the mid region — except x·D comes from
    ``isr_emela_grid`` interpolation instead of a per-node eMELA DGLAP call.  The
    whole mid region is evaluated in ONE vectorised spline call.  Keep this in
    lockstep with ``_per_leg_emela_nll`` (endpoint cutoff, norm_nll formula)."""
    from framework.process.ww.indep import isr_emela_grid as _grid
    alpha = cfg.resolved_alpha()
    grid = _grid.load_grid(cfg.emela_grid)
    # Fail loud if the precomputed grid's baked scheme/α differ from this cfg:
    # the soft endpoint (norm_nll) is built from cfg.α while xD comes from the
    # grid's α, so a mismatch silently splits α within the radiator (e.g. a grid
    # built at MoCaNLO's 1/128.232 used with cfg at the PDG 1/128.943). Rebuild
    # the grid (isr_emela_grid.build_and_write) at the cfg's α/scheme to fix.
    # A grid with absent provenance meta CANNOT be certified, so treat any
    # MISSING key as a mismatch — defaulting a missing key to the cfg value
    # would let an old/meta-stripped grid baked at the wrong α slip through
    # silently (the very case this guard exists to catch).
    gm = grid.meta or {}
    _missing = [k for k in ("alpha", "fac_scheme", "ren_scheme") if k not in gm]
    if (_missing
            or abs(gm["alpha"] - alpha) > 1e-9 * alpha
            or gm["fac_scheme"] != cfg.emela_fac_scheme
            or gm["ren_scheme"] != cfg.emela_ren_scheme):
        raise ValueError(
            f"eMELA grid {cfg.emela_grid!r} provenance cannot be certified "
            f"against cfg: baked meta={gm or 'EMPTY'} "
            f"(missing keys {_missing}) "
            f"vs cfg (α={alpha:.10g}, {cfg.emela_fac_scheme}/{cfg.emela_ren_scheme}); "
            "rebuild the grid (isr_emela_grid.build_and_write) at the cfg's α/scheme.")
    Q = cfg.mu_F(sqrt_s)
    norm_nll = _norm_nll_endpoint(be, norm, alpha)
    per_leg = np.empty_like(x_vals)
    soft = one_minus_x < _grid.OMX_FLOOR
    per_leg[soft] = norm_nll
    nz = ~soft
    if np.any(nz):
        xD = grid.xfxQ(x_vals[nz], one_minus_x[nz], Q)
        per_leg[nz] = xD / x_vals[nz] * jac_NS[nz]
    return per_leg


#: Per-leg radiator setup cache.  The radiator weight D(x)·|dx/du| depends ONLY
#: on (√s, cfg) — NOT on σ̂ — yet a morph build calls convolve_2leg ~60× (every
#: varpoint × channel) with the same √s grid and cfg.  Caching the setup turns
#: the (expensive, eMELA-NLL) per-leg build into a one-off; the σ̂ mesh eval +
#: einsum still run per call.  Keyed by (grid-content, cfg-fingerprint).
_RADIATOR_CACHE: dict = {}

#: DISK persistence of the eMELA-NLL radiator setup.  eMELA's ``code_pdf`` is
#: ~22 ms/call (it re-evolves DGLAP per query) and a morph build issues ~19k
#: queries → ~400 s, ALL of it σ̂-independent.  We persist the (x_vals, w, per_leg)
#: setups so the build is a one-off *across processes/sessions*: the next fit,
#: cross-fit, fork-pool worker or condor job loads them in ms instead of
#: rebuilding.  EXACT (the cached arrays are byte-identical to a fresh build) and
#: gated to the NLL path ONLY — the analytic LL path is fast and never touches
#: disk, so the production-default line shape is unaffected.  Bump
#: ``_RADIATOR_DISK_VERSION`` if the NLL radiator math (``_per_leg_emela_nll`` /
#: the eMELA library / the scheme conventions) changes, or clear the cache dir.
#: Location: ``$WW_ISR_RADIATOR_CACHE`` (default ~/.cache/ww_isr_radiator);
#: set it empty to disable disk caching.
_RADIATOR_DISK_VERSION = 1


def _cfg_fingerprint(cfg: ISRConfig) -> tuple:
    # Continuous fields rounded well below physical resolution so a value reaching
    # the key by two different float paths hashes identically (cache HIT, not a
    # silent rebuild that defeats prewarm); scheme α's differ at 1e-4 → no collision.
    base = (cfg.scheme, round(cfg.resolved_alpha(), 15), round(cfg.mu_F_factor, 12),
            round(cfg.mu_F_abs, 12), cfg.m_e, round(cfg.x_min, 12), cfg.n_quad,
            cfg.nll, cfg.emela_fac_scheme, cfg.emela_ren_scheme)
    # The luminosity (1-D) path is a distinct convolution at the same radiator, so
    # it must hash distinctly from the 2-D — but ONLY append the marker when active,
    # so the default (2-D) fingerprint and its validated production cache are
    # byte-unchanged (no spurious rebuild).
    lumi_tag = ("lumi",) if getattr(cfg, "lumi", False) else ()
    # Append the grid tag ONLY when the LHAPDF-grid path is active, so the
    # direct-eMELA fingerprint (and its validated production cache) is unchanged.
    grid = getattr(cfg, "emela_grid", "")
    if not grid:
        return base + lumi_tag
    # Fold the grid file's identity (mtime_ns + size) into the key so a same-path
    # rebuild invalidates the in-memory radiator, mirroring isr_emela_grid.load_grid
    # (which keys on the same stat fields).  Without this an L1 hit serves a
    # radiator built from the OLD grid contents after an in-process regeneration.
    try:
        st = os.stat(grid)
        return base + ("grid:" + grid, st.st_mtime_ns, st.st_size) + lumi_tag
    except OSError:
        return base + ("grid:" + grid,) + lumi_tag


def _radiator_key(sqrt_s_arr: np.ndarray, cfg: ISRConfig):
    """(√s-grid identity, cfg fingerprint) — the shared in-memory + disk cache
    key.  Used by both ``_radiator_setup`` (read/write) and ``prewarm`` (build)
    so the two never disagree on what a given (grid, cfg) maps to."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    return ((a.shape, hashlib.sha1(a.tobytes()).hexdigest()), _cfg_fingerprint(cfg))


def _radiator_cache_dir() -> str:
    return os.environ.get(
        "WW_ISR_RADIATOR_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "ww_isr_radiator"))


_EMELA_LIB_TAG: str | None = None


def _emela_lib_tag() -> str:
    """Content fingerprint of the eMELA shared library, folded into the NLL disk
    key so a rebuilt/updated eMELA AUTO-invalidates stale cache files (otherwise a
    silent eMELA change would keep serving old radiators).  Computed once per
    process; falls back to a constant if eMELA isn't locatable."""
    global _EMELA_LIB_TAG
    if _EMELA_LIB_TAG is None:
        try:
            from framework.process.ww.xsec_calculator import emela_wrapper as _e
            with open(_e._LIB_PATH, "rb") as fh:
                _EMELA_LIB_TAG = hashlib.sha1(fh.read()).hexdigest()[:16]
        except Exception:
            _EMELA_LIB_TAG = "noemela"
    return _EMELA_LIB_TAG


def _radiator_disk_path(key) -> str | None:
    """File for a (√s-grid, cfg) radiator setup, or None if disk caching is off.
    The key also carries the eMELA library content hash so an eMELA rebuild
    invalidates stale entries without a manual ``_RADIATOR_DISK_VERSION`` bump."""
    cache_dir = _radiator_cache_dir()
    if not cache_dir:
        return None
    h = hashlib.sha1(repr(
        (_RADIATOR_DISK_VERSION, _emela_lib_tag(), key)).encode()).hexdigest()
    return os.path.join(cache_dir, f"rad_{h}.pkl")


def _radiator_setup(sqrt_s_arr: np.ndarray, cfg: ISRConfig):
    """List of (x_vals, w, per_leg) per √s — σ̂-independent, cached (in-memory +,
    for the eMELA-NLL path, on disk; see ``_RADIATOR_CACHE`` / ``_radiator_disk_path``)."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    if cfg.nll and getattr(cfg, "lumi", False):
        # Luminosity path: the 2-D per-leg setups are unused (convolve_2leg routes
        # to isr_lumi before reaching here).  Warm the σ̂-independent luminosity
        # cache instead, so a parent prewarm benefits the fork pool via COW, then
        # return an empty list (never consumed on this path).
        from framework.process.ww.indep import isr_lumi
        isr_lumi.prewarm(a, cfg)
        return []
    key = _radiator_key(a, cfg)
    cached = _RADIATOR_CACHE.get(key)
    if cached is not None:
        return cached

    # Disk cache: worthwhile only for the expensive direct-eMELA NLL path.  The
    # LHAPDF-grid path is already a fast table interpolation and its result
    # depends on the grid FILE contents (not fingerprinted), so it is NOT
    # disk-persisted — in-memory caching only, no staleness risk.
    disk = _radiator_disk_path(key) if (cfg.nll and not cfg.emela_grid) else None
    if disk is not None and os.path.exists(disk):
        try:
            with open(disk, "rb") as fh:
                setups = pickle.load(fh)
            _RADIATOR_CACHE[key] = setups
            return setups
        except Exception:
            pass     # corrupt/partial/incompatible → fall through and rebuild

    setups = []
    for sq in a:
        be, bs, bh = cfg.betas(float(sq))
        norm = _radiator_norm(be, bs)
        u, w, x_vals, one_minus_x, jac_NS = _endpoint_grid(be, cfg.x_min, cfg.n_quad)
        if cfg.nll and cfg.emela_grid:
            per_leg = _per_leg_grid_nll(cfg, be, norm, x_vals, one_minus_x,
                                        jac_NS, float(sq))
        elif cfg.nll:
            per_leg = _per_leg_emela_nll(cfg, be, norm, x_vals, one_minus_x,
                                         jac_NS, float(sq))
        else:
            NS_vals = _radiator_NS(x_vals, bh, one_minus_x=one_minus_x)
            per_leg = norm + jac_NS * NS_vals        # D(x)·|dx/du| in u-space
        setups.append((x_vals, w, per_leg))
    _RADIATOR_CACHE[key] = setups

    if disk is not None:
        # Best-effort, atomic (tmp + os.replace) so concurrent fork-pool/condor
        # writers never leave a partial file; caching must NEVER break the calc.
        try:
            os.makedirs(_radiator_cache_dir(), exist_ok=True)
            # NODE-unique tmp: PID is not unique across nodes, so two condor jobs
            # on different machines could otherwise collide on the same shared-AFS
            # tmp file.  host+pid+random makes the tmp collision-proof.
            tok = f"{socket.gethostname()}.{os.getpid()}.{os.urandom(4).hex()}"
            tmp = f"{disk}.tmp.{tok}"
            with open(tmp, "wb") as fh:
                pickle.dump(setups, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, disk)
        except Exception:
            pass
    return setups


def convolve_2leg(sqrt_s, sigma_hat_fn, cfg: ISRConfig = ISRConfig()):
    """σ_obs(s) = ∫∫ D(x₁) D(x₂) σ̂(√(x₁x₂)·√s) dx₁ dx₂  [fb].

    ``sigma_hat_fn(sqrt_shat)`` returns σ̂ [fb] for an array of √ŝ [GeV].
    Vectorised in ``sqrt_s`` (scalar or array).  ``cfg.nll`` swaps the analytic
    LL+exp per-leg radiator for eMELA's NLL ePDF.  The σ̂-independent radiator is
    cached across calls (see ``_radiator_setup``).

    When ``cfg.lumi`` (opt-in; requires ``cfg.nll`` and ``cfg.emela_grid``) the
    convolution is computed by the RIPPLE-FREE 1-D luminosity self-convolution
    (``isr_lumi.sigma_obs``) instead of the 2-D einsum — same observable, smoother
    line shape; see ``ISRConfig.lumi``.
    """
    if cfg.nll and getattr(cfg, "lumi", False):
        # Lazy import (isr_lumi imports isr_beta) → route to the luminosity form.
        from framework.process.ww.indep import isr_lumi
        return isr_lumi.sigma_obs(sqrt_s, sigma_hat_fn, cfg)

    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)
    setups = _radiator_setup(sqrt_s_arr, cfg)

    for idx, sq in enumerate(sqrt_s_arr):
        x_vals, w, per_leg = setups[idx]
        X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
        sqrt_shat = np.sqrt(X1 * X2) * float(sq)
        sigma_hat = np.asarray(sigma_hat_fn(sqrt_shat.ravel()),
                               dtype=float).reshape(X1.shape)

        weight = w * per_leg
        out[idx] = np.einsum("i,j,ij->", weight, weight, sigma_hat)

    return float(out[0]) if np.ndim(sqrt_s) == 0 else out


# ---------------------------------------------------------------------------
# O(α) ISR subtraction — DIAGNOSTIC ONLY, not part of the production observable
# ---------------------------------------------------------------------------

def oalpha_isr_subtraction(sqrt_s, sigma_born_fn, cfg: ISRConfig = ISRConfig()):
    """C₁[σ̂_Born](s) — the O(α) piece of the two-leg radiator convolution [fb].

    NOT used in the production observable (2026-06-12): the σ̂ grids are
    collinear-counterterm-subtracted, so there is no O(α) ISR in σ̂_NLO to
    match against — subtracting C₁ from ∫∫DD σ̂_NLO cancels the physical ISR
    damping (see module docstring).  Retained for the C₁ scheme diagnostics
    under ``scripts/investigations/bfs_match/``, which study exactly this
    kernel.  It would only re-enter the observable for grids that genuinely
    carry the mass-regularised O(α) ISR log.

    This is the O(α) expansion of ``convolve_2leg(σ̂_Born)``, built so that the
    artefacts of the *finite* x_min cutoff cancel exactly between it and
    ``convolve_2leg(σ̂_NLO)`` in the matched result.  Per leg, expanding the
    truncated convolution to O(α):

        I[g] − g(s) = (3/4)β_s g(s)  +  β_e g(s) ln(1−x_min)      [endpoint norm]
                      + β_e ∫_{x_min}^1 (g(xs) − g(s))/(1−x) dx   [[1/(1−x)]₊]
                      − (β_h/2) ∫_{x_min}^1 (1+x) g(xs) dx        [hard collinear]

    where the ``β_e g(s) ln(1−x_min)`` term comes from ∫_{x_min}^1 β_e(1−x)^{β_e−1}
    = (1−x_min)^{β_e} = 1 + β_e ln(1−x_min) + …  (the singular weight no longer
    integrates to 1 once the leg is cut at x_min).  Two legs → ×2.  For a
    physical σ̂ that vanishes below threshold this term is harmless on its own,
    but it MUST be kept here so the matching cancels the corresponding piece of
    ``convolve_2leg`` to O(α²).  Without it the matched line shape carries a
    spurious O(α) ISR remainder.

    From D₁(x) = β_e[1/(1−x)]₊ + (3/4)β_s δ(1−x) − (β_h/2)(1+x).  The plus
    integrand's numerator vanishes as x→1, cancelling the 1/(1−x) pole; both
    integrals are smooth, done by GL quadrature on [x_min,1].
    ``sigma_born_fn(sqrt_shat)`` returns σ̂_Born [fb].  Vectorised in ``sqrt_s``.
    """
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.zeros_like(sqrt_s_arr)

    x_nodes, x_w = _quad_nodes(cfg.n_quad, cfg.x_min, 1.0)
    log_omx_min = math.log(1.0 - cfg.x_min)

    for idx, sq in enumerate(sqrt_s_arr):
        be, bs, bh = cfg.betas(float(sq))
        s_full = float(sq)
        sig_s = float(np.asarray(sigma_born_fn(np.array([s_full])), dtype=float)[0])
        sqrt_shat = np.sqrt(x_nodes) * s_full
        sig_x = np.asarray(sigma_born_fn(sqrt_shat), dtype=float)

        plus_int = np.sum(x_w * (sig_x - sig_s) / (1.0 - x_nodes))
        hard_int = np.sum(x_w * (1.0 + x_nodes) * sig_x)
        out[idx] = 2.0 * (0.75 * bs * sig_s
                          + be * sig_s * log_omx_min
                          + be * plus_int
                          - 0.5 * bh * hard_int)

    return float(out[0]) if np.ndim(sqrt_s) == 0 else out


def sigma_observed(sqrt_s, sigma_nlo_fn, cfg: ISRConfig = ISRConfig()):
    """Observed line shape [fb]:   σ_obs(s) = ∫∫ D D σ̂_NLO(x₁x₂s).

    The σ̂ grids are beam-ISR-free (``pdf_set=none`` + collinear counterterm,
    see module docstring), so the radiator alone supplies the initial-state
    radiation: the collinear log enters via D, the finite O(α) remnant via
    σ̂_NLO — each counted once.  Do NOT subtract ``oalpha_isr_subtraction``
    here: that cancels the ISR damping and inflates σ_obs by ~48 %
    (pre-2026-06-12 bug).  ``sigma_nlo_fn`` is a σ̂_NLO interpolator
    (√ŝ → fb).  Vectorised in ``sqrt_s``.
    """
    return convolve_2leg(sqrt_s, sigma_nlo_fn, cfg)


# ---------------------------------------------------------------------------
# Campaign prewarm — populate the NLL radiator disk cache before fanning out
# ---------------------------------------------------------------------------

def _coerce_grid_list(grids) -> list[np.ndarray]:
    """Accept either a single √s grid (1-D array-like of scalars) or a sequence
    of such grids, and return a list of contiguous float arrays."""
    if isinstance(grids, np.ndarray):
        return [np.ascontiguousarray(grids, dtype=float)]
    grids = list(grids)
    if not grids:
        return []
    if np.ndim(grids[0]) == 0:                      # sequence of scalars → one grid
        return [np.ascontiguousarray(grids, dtype=float)]
    return [np.ascontiguousarray(g, dtype=float) for g in grids]


def _sweep_stale_tmp(cache_dir, max_age_s: float = 6 * 3600):
    """Best-effort removal of orphaned ``rad_*.pkl.tmp.*`` files left behind when a
    writer is hard-killed between ``open(tmp)`` and ``os.replace``.  The tmp name is
    host+pid+random unique and is never the live ``.pkl``, so a stale one is
    harmless litter; this keeps the shared cache dir tidy.  Only tmps older than
    ``max_age_s`` are removed, so an in-flight concurrent write is never touched.
    Skips ``rad_bfs_*`` (owned by the BFS chain, which sweeps its own)."""
    if not cache_dir:
        return
    try:
        now = time.time()
        for p in glob.glob(os.path.join(cache_dir, "rad_*.pkl.tmp.*")):
            if os.path.basename(p).startswith("rad_bfs_"):
                continue
            try:
                if now - os.path.getmtime(p) > max_age_s:
                    os.remove(p)
            except OSError:
                pass
    except Exception:
        pass


def _prewarm_build_one(grid_cfg):
    """Worker: ensure the radiator disk file for one (√s-grid, cfg) exists.

    Returns (status, disk_path) with status ∈ {'built','exists','no-disk'} — same
    (status, path) order as the ``existing`` list in ``prewarm`` so the two merge
    into one homogeneous ``files`` list.  Idempotent — an existing file is left
    untouched (not even re-read), so a re-run / resume only pays for what is
    still missing."""
    grid, cfg = grid_cfg
    a = np.ascontiguousarray(grid, dtype=float)
    path = _radiator_disk_path(_radiator_key(a, cfg))
    if path is None:
        return ("no-disk", None)
    if os.path.exists(path):
        return ("exists", path)
    _radiator_setup(a, cfg)        # builds via eMELA + atomically writes `path`
    return ("built", path)


def prewarm(sqrt_s_grids, cfgs, *, n_workers: int = 1, verbose: bool = True):
    """Pre-build the eMELA-NLL radiator **disk** cache for a whole campaign in a
    single up-front pass, so a subsequent parallel fan-out (theory variations,
    scenario fits, cross-fits, condor jobs across nodes) only ever *reads* the
    cache — never races to rebuild it.

    Why this and not a lock
    -----------------------
    The radiator is σ̂-independent and its disk cache lives on shared AFS visible
    to every condor node, but AFS has no reliable cross-node file lock.  N cold
    jobs starting together would each spend the full ~22 ms × ~19k-query
    (~minutes) eMELA build computing byte-identical arrays and then atomically
    over-write the same ``rad_*.pkl``.  The write is already safe; the *redundant
    compute* (and the eMELA/grid first-touch stampede) is the waste.  Building
    once, here, before the fan-out converts an N-way build stampede into one
    build + N millisecond reads — the only concurrency-safe shape available
    across AFS nodes.

    What multiplies the cache (and what does NOT)
    ---------------------------------------------
    One file is one ``(√s-grid, ISR-cfg)``.  Only fields the radiator depends on
    multiply it: the √s-grid contents and ``_cfg_fingerprint`` (scheme, α,
    mu_F_factor, mu_F_abs, m_e, x_min, n_quad, emela_*).  The radiator does NOT
    depend on σ̂, so an entire partonic ladder (LO/NLO/NNLO/δ_QCD) at one ISR cfg
    collapses to a *single* file; and the analytic LL path (``nll=False``) never
    touches disk, so only direct-eMELA NLL cfgs are built — LL cfgs are counted
    as ``skipped_LL`` and the LHAPDF-grid path (``emela_grid`` set, in-memory
    only) as ``skipped_grid``; both are ignored.  The prewarm set is therefore
    ``{√s grids} × {NLL ISR cfgs}`` — typically O(10) files for a full
    theory-variation campaign, not hundreds.

    Parameters
    ----------
    sqrt_s_grids : one √s array, or an iterable of them.
    cfgs         : one ISRConfig, or an iterable of them.
    n_workers    : build the de-duplicated work list with this many *processes*
                   (fork pool; each build is independent).  Default 1 (serial,
                   deterministic — best for validation).  For a real campaign run
                   on fcc-ironic with n_workers≈8-16 (node cap 48).
    verbose      : print one line per file + a final tally.

    Returns
    -------
    dict: ``built`` / ``exists`` / ``skipped_LL`` / ``no_disk`` counts,
    ``n_unique`` (de-duplicated build targets), ``files`` (list of (status,
    path)), and ``cache_dir``.

    Idempotent and resumable: a re-run only builds what is still missing.  Must
    COMPLETE before the fan-out launches — a job that starts mid-prewarm and
    finds its file missing will rebuild it (correct, just defeats the purpose).
    For condor, run prewarm as a DAG parent / held-then-released pre-step.
    """
    grids = _coerce_grid_list(sqrt_s_grids)
    cfgs = [cfgs] if isinstance(cfgs, ISRConfig) else list(cfgs)
    _sweep_stale_tmp(_radiator_cache_dir())     # clear orphaned *.tmp from prior kills

    # De-duplicate by disk path: distinct (grid, cfg) that map to the same file
    # (they can't, by construction, but a caller may pass duplicates) build once.
    work: dict[str, tuple] = {}
    skipped_LL = 0
    skipped_grid = 0
    no_disk = 0
    for cfg in cfgs:
        if not cfg.nll:
            skipped_LL += len(grids)          # LL path never disk-caches
            continue
        if getattr(cfg, "emela_grid", ""):
            # LHAPDF-grid path is a fast in-memory interpolation and is NOT
            # disk-cached (see _radiator_setup) — prewarming it would write
            # nothing yet falsely report 'built'.  Its one-off artifact is the
            # grid file itself (isr_emela_grid.build_and_write), not a rad_*.pkl.
            skipped_grid += len(grids)
            continue
        for grid in grids:
            path = _radiator_disk_path(_radiator_key(grid, cfg))
            if path is None:
                no_disk += len(grids)
                break                          # disk caching off entirely
            work.setdefault(path, (grid, cfg))

    existing = [p for p in work if os.path.exists(p)]
    to_build = [(p, gc) for p, gc in work.items() if p not in existing]

    if verbose:
        print(f"[prewarm] {len(work)} unique (grid,cfg) file(s): "
              f"{len(existing)} already cached, {len(to_build)} to build "
              f"(skipped_LL={skipped_LL}, skipped_grid={skipped_grid}); "
              f"cache={_radiator_cache_dir()!r}")

    files = [("exists", p) for p in existing]
    if to_build:
        import time as _time
        import multiprocessing as _mp
        t0 = _time.time()
        items = [gc for _p, gc in to_build]
        if n_workers > 1 and len(items) > 1:
            ctx = _mp.get_context("fork")
            with ctx.Pool(min(n_workers, len(items))) as pool:
                results = pool.map(_prewarm_build_one, items)
        else:
            results = []
            for i, gc in enumerate(items, 1):
                results.append(_prewarm_build_one(gc))
                if verbose:
                    print(f"[prewarm]   built {i}/{len(items)}  {results[-1][1]}")
        files.extend(results)
        if verbose:
            print(f"[prewarm] built {len(items)} file(s) in {_time.time()-t0:.1f}s")

    report = {
        "built": sum(1 for s, _ in files if s == "built"),
        "exists": sum(1 for s, _ in files if s == "exists"),
        "skipped_LL": skipped_LL,
        "skipped_grid": skipped_grid,
        "no_disk": no_disk,
        "n_unique": len(work),
        "files": files,
        "cache_dir": _radiator_cache_dir(),
    }
    if verbose:
        print(f"[prewarm] done: built={report['built']} exists={report['exists']} "
              f"skipped_LL={report['skipped_LL']} skipped_grid={report['skipped_grid']} "
              f"no_disk={report['no_disk']}")
    return report
