"""MoCaNLO beta-scheme ISR radiator + quadrature convolution + O(α) matching.

This is the ISR layer of the **independent, BFS-free** WW-threshold line-shape
calculation.  It convolves a fixed-order NLO-EW *partonic* cross section
σ̂(√ŝ) — produced by an unmodified MoCaNLO run with ``pdf_set=none`` (no beam
ISR) — with the LL+exp electron structure function, applying the standard
O(α) matching subtraction so the O(α) initial-state radiation is counted once.

Provenance / independence
--------------------------
The radiator ``D(x)`` is transcribed **directly from the MoCaNLO source**,
``src/mocanlo/pdfs/lepton_pdfs.F90`` subroutine ``lepton_pdf_convolution``
(case ``LO_beta`` etc.), itself appendix A of arXiv:2207.03265 (Bertone,
Cacciari, Frixione, Stagnitto, Zaro, Zhao).  It deliberately does **not**
import the project's BFS ``xsec_calculator.isr`` module — this chain shares no
code or numerical input with the BFS-EFT calculation.  (The two radiators are
mathematically the same LEP-YR structure function, so a numerical cross-check
against ``isr.sigma_ISR_2leg_convolution`` is a useful *validation*, performed
in ``tests``/``__main__`` — but it is not a dependency.)

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

O(α) matching
-------------
σ̂_NLO(ŝ) from ``pdf_set=none`` already contains the explicit O(α) initial-state
collinear log ln(ŝ/m_e²) (mass-regularised, *not* factorised into a PDF).
Convolving it with the resummed radiator would double-count that O(α) ISR.  The
matched result subtracts the overlap:

      σ_obs(s) = ∫∫ D D σ̂_NLO(x₁x₂s)  −  C₁[σ̂_Born](s)

      C₁[σ̂_Born] = 2 ∫₀¹ dx D₁(x) σ̂_Born(x s),

      D₁(x) = β_e [1/(1−x)]₊ + (3/4)β_s δ(1−x) − (β_h/2)(1+x)

the O(α) piece of one leg of the radiator.  Expanding ∫∫ D D σ̂_NLO to O(α)
gives σ̂_NLO + C₁[σ̂_Born]; the subtraction restores the exact fixed-order NLO at
O(α) while keeping the all-orders LL resummation.  (For ``LO_beta``,
D₁ = β·½[(1+x²)/(1−x)]₊ + ¾β δ(1−x), the familiar P_ee splitting + soft const.)

All cross sections are in **fb** (MoCaNLO's unit).  ``sigma_hat_fn`` callables
take √ŝ in GeV and return σ̂ in fb; they are typically interpolators over the
MoCaNLO partonic grid.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
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
ALPHA_MZ = 7.7983970817e-3       # α(M_Z)

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
    nll           EXPLORATORY: replace the analytic LL+exp per-leg radiator with
                  eMELA's NLL electron ePDF (DGLAP-evolved).  The O(α) matching
                  subtraction stays the analytic LL form → leading-log-exact
                  with a residual O(α) NLL/DELTA-scheme constant (the #1 NLL
                  refinement: derive the subtraction from eMELA's own O(α)).
                  Configure with alpha=ALPHA_MZ + emela_ren_scheme="ALPMZ" to
                  match the BFS production NLL convention.
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
    from framework.process.ww.xsec_calculator.isr import LAMBDA1_NF0
    alpha = cfg.resolved_alpha()
    _emela.initialize(pert_order="NLL", fac_scheme=cfg.emela_fac_scheme,
                      ren_scheme=cfg.emela_ren_scheme, alpha=alpha)
    Q = cfg.mu_F(sqrt_s)
    norm_nll = norm * math.exp(be * (alpha / _PI) * (LAMBDA1_NF0 / 4.0))
    per_leg = np.empty_like(x_vals)
    for i in range(len(x_vals)):
        omx_i = float(one_minus_x[i])
        if omx_i < 1e-15:
            per_leg[i] = norm_nll
        else:
            x_i = float(x_vals[i])
            per_leg[i] = _emela.code_pdf(x_i, omx_i, Q) / x_i * float(jac_NS[i])
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
    return (cfg.scheme, cfg.resolved_alpha(), cfg.mu_F_factor, cfg.mu_F_abs,
            cfg.m_e, round(cfg.x_min, 12), cfg.n_quad, cfg.nll,
            cfg.emela_fac_scheme, cfg.emela_ren_scheme)


def _radiator_cache_dir() -> str:
    return os.environ.get(
        "WW_ISR_RADIATOR_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "ww_isr_radiator"))


def _radiator_disk_path(key) -> str | None:
    """File for a (√s-grid, cfg) radiator setup, or None if disk caching is off."""
    cache_dir = _radiator_cache_dir()
    if not cache_dir:
        return None
    h = hashlib.sha1(repr((_RADIATOR_DISK_VERSION, key)).encode()).hexdigest()
    return os.path.join(cache_dir, f"rad_{h}.pkl")


def _radiator_setup(sqrt_s_arr: np.ndarray, cfg: ISRConfig):
    """List of (x_vals, w, per_leg) per √s — σ̂-independent, cached (in-memory +,
    for the eMELA-NLL path, on disk; see ``_RADIATOR_CACHE`` / ``_radiator_disk_path``)."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    key = ((a.shape, hashlib.sha1(a.tobytes()).hexdigest()), _cfg_fingerprint(cfg))
    cached = _RADIATOR_CACHE.get(key)
    if cached is not None:
        return cached

    # Disk cache: worthwhile only for the expensive eMELA-NLL path.
    disk = _radiator_disk_path(key) if cfg.nll else None
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
        if cfg.nll:
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
            tmp = f"{disk}.tmp{os.getpid()}"
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
    """
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
# O(α) matching subtraction
# ---------------------------------------------------------------------------

def oalpha_isr_subtraction(sqrt_s, sigma_born_fn, cfg: ISRConfig = ISRConfig()):
    """C₁[σ̂_Born](s) — the O(α) ISR overlap already in the fixed-order NLO [fb].

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


def sigma_observed_matched(sqrt_s, sigma_nlo_fn, sigma_born_fn,
                           cfg: ISRConfig = ISRConfig()):
    """Matched observed line shape [fb]:

        σ_obs(s) = ∫∫ D D σ̂_NLO(x₁x₂s)  −  C₁[σ̂_Born](s).

    Reproduces fixed-order NLO-EW at O(α) and resums LL ISR to all orders, with
    no O(α) ISR double-counting.  ``sigma_nlo_fn``/``sigma_born_fn`` are σ̂_NLO
    and σ̂_Born interpolators (√ŝ → fb).  Vectorised in ``sqrt_s``.
    """
    conv = convolve_2leg(sqrt_s, sigma_nlo_fn, cfg)
    sub = oalpha_isr_subtraction(sqrt_s, sigma_born_fn, cfg)
    return conv - sub
