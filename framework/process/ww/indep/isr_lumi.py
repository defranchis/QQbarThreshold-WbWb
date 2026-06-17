"""Luminosity (1-D) form of the two-leg eMELA-NLL ISR convolution.

The production observable is the two-leg double convolution
``σ_obs(s) = ∫∫ D(x₁) D(x₂) σ̂(√(x₁x₂)·√s) dx₁ dx₂`` (``isr_beta.convolve_2leg``),
evaluated by a plain Gauss-Legendre einsum over the (x₁, x₂) mesh.  That 2-D rule
*straddles* the σ̂ grid-edge step at √ŝ = ``SIGMA_GRID_LO`` (σ̂ ≡ 0 below) and so
**ripples** ∝ 1/n_quad as √s sweeps (point-wise up to ~0.65 % at the production
n_quad = 128; it only converges in a many-n_quad average).

This module collapses the double convolution onto the single luminosity variable
``z = x₁ x₂``::

    σ_obs(s) = ∫ L(z; μ_F) σ̂(√z·√s) dz ,   L(z) = ∫ (dx/x) D(x) D(z/x) ,

the two-leg LUMINOSITY = the radiator self-convolution.  In ``V = −ln z`` the
self-convolution is additive and its soft ``V→0`` behaviour factorises::

    L(z) dz = L_V(V) dV ,   L_V(V) = V^{2β_e−1} · L̃(V; μ_F) ,   L̃ smooth,

with the per-leg density ρ(v) = x·D (eMELA's x·D) and ``L̃`` the Beta-type
double-soft self-convolution
``L̃(V) = ∫₀¹ t^{β_e−1}(1−t)^{β_e−1} ρ̃(Vt) ρ̃(V(1−t)) dt`` integrated EXACTLY by
Gauss-Jacobi(β_e−1, β_e−1), where ``ρ̃(v) = x·D·v^{1−β_e}`` is smooth.

The outer V-integral uses a SINGLE Gauss-Jacobi(2β_e−1, 0) panel on ``[0, V_top]``
with ``V_top = 2 ln(√s / SIGMA_GRID_LO)``: the σ̂ grid-edge step is the integration
*limit* (never an interior node), so the line shape carries NO Gauss-Legendre
ripple.  This is WHY the luminosity is SMOOTHER than the 2-D.

Why DIRECT L̃, not a precomputed (V, μ_F) spline table
-----------------------------------------------------
``L̃(V)`` carries a mild ``V·lnV`` cusp at V→0 (genuine NLL ePDF structure); a
cubic spline over a finite V-grid cannot represent it, and the outer ``V^{2β−1}``
weight *amplifies* the small-V interpolation error — so a precomputed-table line
shape does NOT converge cleanly in the outer node count (measured shape residual
spiked to ~1700 ppm at an unlucky n_out).  Computing ``L̃`` DIRECTLY at the outer
nodes by Gauss-Jacobi self-convolution is exact at every node and stable.  L̃ is
σ̂-INDEPENDENT (only (V, μ_F)), so the per-√s outer-node setup is built ONCE and
cached (``_lumi_setup``) — reused across every channel / varpoint / scan call,
exactly as ``isr_beta._radiator_setup`` caches the 2-D radiator.  ρ̃ comes from the
already-persisted eMELA grid (``isr_emela_grid``); no separate artifact is needed.
(See ``scripts/investigations/nll_isr/RESULTS_lumi_convolution_2026-06-17.md`` and
``validate_lumi_production.py`` for the faithfulness validation: this reproduces
the 2-D's ripple-free mean to ~tens of ppm and is smoother than it.)

Soft endpoint (the V→0 anchor)
------------------------------
``ρ̃(v) = x·D(x,Q)·v^{1−β_e}`` from the eMELA grid for ``omx = 1−x ≥ OMX_FLOOR``,
and the exact analytic soft+virtual constant ``ρ̃ → norm_nll·β_e`` for
``omx < OMX_FLOOR`` — IDENTICAL to what the 2-D (`isr_beta._per_leg_grid_nll`)
integrates, where ``x = 1−omx`` underflows to 1.0 in float64 and the grid breaks.
This makes the double-soft limit exact: ``L̃(V→0) → (norm_nll·β_e)²·B(β_e, β_e)``.

Provenance: the eMELA grid's baked α / fac-scheme / ren-scheme MUST match the
consuming ``ISRConfig`` (the soft anchor uses cfg's α while x·D uses the grid's);
enforced by the same guard as ``isr_beta._per_leg_grid_nll``.

This is the production form; the investigation prototypes (``lumi_faithful.py``,
``lumi_grid.py``, ``luminosity_prototype.py``) are the validation harness.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep import isr_emela_grid as _eg
from framework.process.ww.indep import grid as _grid

#: Deep-endpoint cutoff — MUST match ``isr_emela_grid.OMX_FLOOR`` /
#: ``isr_beta._per_leg_grid_nll``'s 1e-15 switch to the analytic ``norm_nll``.
OMX_FLOOR = _eg.OMX_FLOOR

#: σ̂ grid bottom (σ̂ ≡ 0 below): caps the V-range at V_top = 2 ln(√s/SIGMA_GRID_LO)
#: so the σ̂ grid-edge step is always the outer integration LIMIT, never interior.
#: SINGLE-SOURCED from ``grid.ECM_MIN`` (the partonic σ̂-grid floor, =156 GeV) — the
#: σ̂ step sits at that floor by construction, so binding here makes V_top track it
#: automatically if the MoCaNLO grid is ever regenerated at a different floor (no
#: hand-synced literal that could silently drift from the real edge).
SIGMA_GRID_LO = _grid.ECM_MIN

#: Production quadrature for the luminosity convolution (module constants, like
#: the 2-D's n_quad).  ``LUMI_N_JAC`` (inner self-conv): its deepest Gauss-Jacobi
#: node reaches omx ≈ 3e-8 — the soft tail below that (down to OMX_FLOOR) is the
#: analytic ``norm_nll`` anchor (ρ̃→norm_nll·β_e), so the deep endpoint is exact
#: regardless of n_jac; the surviving n_jac dependence is a ~few-100-ppm NORM
#: offset that is lumi-degenerate (cancels in the morph σ/σ_nom ratio → ≲30 ppm
#: residual SHAPE, sub-0.02 MeV).  ``LUMI_N_OUT`` (outer panel): the line-SHAPE is
#: converged (≲20 ppm vs n_out≥384 in the 157–163 physics window).  Both are a
#: ONE-TIME per-√s cost (σ̂-independent, cached); the per-channel/varpoint cost is
#: only ``LUMI_N_OUT`` σ̂-evals (vs the 2-D's n_quad²).  NB the Gauss-Jacobi
#: self-conv is NOT monotone at large n_jac (scipy roots_jacobi degrades for the
#: near-singular α=β≈−0.94 weight); these production values are validated, do not
#: raise n_jac blindly for "accuracy".
LUMI_N_OUT = 192
LUMI_N_JAC = 400


def _jac01(n: int, a: float, b: float):
    """Nodes t∈[0,1] and weights for ∫₀¹ t^a (1−t)^b f(t) dt (Gauss-Jacobi)."""
    from scipy.special import roots_jacobi
    x, w = roots_jacobi(n, b, a)              # scipy weight (1−x)^b (1+x)^a
    return 0.5 * (x + 1.0), w / 2.0 ** (a + b + 1.0)


def _norm_nll(cfg: isr_beta.ISRConfig, Q: float):
    """(β_e, norm_nll) — the exact analytic soft+virtual per-leg endpoint value,
    matching ``isr_beta._per_leg_grid_nll`` (and the BFS-side isr.py): the LL+exp
    prefactor ``_radiator_norm`` times the BCFS NLL exponent correction
    ``exp(β_e·(α/π)·λ₁/4))``.  ρ̃(v) → norm_nll·β_e as v → 0.  The endpoint value is
    the SINGLE-SOURCED ``isr_beta._norm_nll_endpoint`` (shared with the 2-D path)."""
    be, bs, _bh = cfg.betas(Q)
    alpha = cfg.resolved_alpha()
    norm = isr_beta._radiator_norm(be, bs)
    return be, isr_beta._norm_nll_endpoint(be, norm, alpha)


def _check_grid_provenance(grid, cfg: isr_beta.ISRConfig):
    """Fail loud if the eMELA grid's baked α/scheme differ from cfg — the soft
    anchor (norm_nll) uses cfg's α while x·D comes from the grid, so a mismatch
    silently splits α within the radiator.  Same guard as ``_per_leg_grid_nll``."""
    alpha = cfg.resolved_alpha()
    gm = grid.meta or {}
    missing = [k for k in ("alpha", "fac_scheme", "ren_scheme") if k not in gm]
    if (missing
            or abs(gm["alpha"] - alpha) > 1e-9 * alpha
            or gm["fac_scheme"] != cfg.emela_fac_scheme
            or gm["ren_scheme"] != cfg.emela_ren_scheme):
        raise ValueError(
            f"eMELA grid {cfg.emela_grid!r} provenance cannot be certified against "
            f"cfg: baked meta={gm or 'EMPTY'} (missing keys {missing}) vs cfg "
            f"(α={alpha:.10g}, {cfg.emela_fac_scheme}/{cfg.emela_ren_scheme}); "
            "rebuild the grid (isr_emela_grid.build_and_write) at the cfg's α/scheme.")


def _rho_tilde_factory(cfg: isr_beta.ISRConfig, Q: float):
    """ρ̃(v) = x·D(x,Q)·v^{1−β_e}, x = e^{−v}, x·D the eMELA grid ePDF.

    The integrable soft singularity x·D ~ omx^{β_e−1} is divided out, so ρ̃ is
    smooth and ρ̃(0) = norm_nll·β_e (the exact double-soft anchor).  Identical to
    what ``isr_beta._per_leg_grid_nll`` integrates: grid ePDF for omx ≥ OMX_FLOOR,
    the analytic soft+virtual constant below (where x = 1−omx underflows to 1.0)."""
    grid = _eg.load_grid(cfg.emela_grid)
    _check_grid_provenance(grid, cfg)
    be, norm_nll = _norm_nll(cfg, Q)
    rt_soft = norm_nll * be                                # ρ̃(v→0)

    def rho_tilde(v):
        v = np.asarray(v, dtype=float)
        x = np.exp(-v)
        omx = -np.expm1(-v)                               # 1−x, accurate small v
        soft = omx < OMX_FLOOR
        omx_c = np.clip(omx, OMX_FLOOR, None)
        xD = grid.xfxQ(x, omx_c, Q)
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = np.where(v > 0.0, xD * v ** (1.0 - be), 0.0)
        return np.where(soft, rt_soft, rt)                # analytic soft anchor
    return rho_tilde, be


def _ltilde_at(rho_tilde, be: float, V: np.ndarray, n_jac: int) -> np.ndarray:
    """L̃(V) = ∫₀¹ t^{β−1}(1−t)^{β−1} ρ̃(Vt) ρ̃(V(1−t)) dt, vectorised over V, by
    Gauss-Jacobi(β−1, β−1).  V=0 → ρ̃≡ρ̃(0) → L̃ = ρ̃(0)²·B(β,β) (exact)."""
    t, wj = _jac01(n_jac, be - 1.0, be - 1.0)
    V = np.atleast_1d(np.asarray(V, dtype=float))
    Vt = np.outer(V, t)                                    # (nV, n_jac)
    rt1 = rho_tilde(Vt.ravel()).reshape(Vt.shape)
    rt2 = rho_tilde((V[:, None] * (1.0 - t)).ravel()).reshape(Vt.shape)
    return (rt1 * rt2) @ wj


#: Per-√s luminosity setup cache: σ̂-INDEPENDENT outer-node weights, keyed by
#: (√s-grid identity, cfg fingerprint, n_out, n_jac).  Mirrors
#: ``isr_beta._RADIATOR_CACHE`` — a morph build reuses one setup across all
#: channels & varpoints.  In-memory only (the self-conv reads the fast eMELA
#: spline grid, so cross-process disk persistence is unnecessary; cf. the eMELA-
#: grid 2-D path, also in-memory only).
_LUMI_CACHE: dict = {}


def _lumi_setup(sqrt_s_arr: np.ndarray, cfg: isr_beta.ISRConfig,
                n_out: int, n_jac: int):
    """List of (shat_arg, outer_w) per √s such that
    ``σ_obs(√s) = Σ outer_w · σ̂(shat_arg)`` — the σ̂-independent luminosity
    weights (outer Gauss-Jacobi panel × the directly-computed L̃ at its nodes)."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    key = (isr_beta._radiator_key(a, cfg), int(n_out), int(n_jac))
    cached = _LUMI_CACHE.get(key)
    if cached is not None:
        return cached
    # Grid x-coverage guard: the inner self-conv samples ρ̃ at v up to V_top, i.e.
    # 1−x up to 1−(SIGMA_GRID_LO/√s)² — for large √s this exceeds the eMELA grid's
    # omx_hi (the threshold production √s≤170 reaches only ~0.16, well within 0.5,
    # but e.g. a 240 GeV anchor needs ~0.58).  Fail EARLY with the √s ceiling rather
    # than deep inside the vectorised xfxQ spline call.
    omx_hi = float(_eg.load_grid(cfg.emela_grid).omx[-1])
    setups = []
    for sq in a:
        sq = float(sq)
        Q = cfg.mu_F(sq)
        rho_tilde, be = _rho_tilde_factory(cfg, Q)
        V_top = 2.0 * math.log(sq / SIGMA_GRID_LO)
        if V_top <= 0.0:                                   # √s ≤ σ̂ floor → no support
            setups.append((np.array([sq]), np.array([0.0])))
            continue
        omx_need = 1.0 - (SIGMA_GRID_LO / sq) ** 2         # deepest 1−x the self-conv reaches
        if omx_need > omx_hi * (1.0 + 1e-9):
            sq_max = SIGMA_GRID_LO / math.sqrt(1.0 - omx_hi)
            raise ValueError(
                f"isr_lumi: √s={sq:.4g} GeV needs the ePDF down to 1−x={omx_need:.4g}, "
                f"beyond the eMELA grid's omx_hi={omx_hi:.4g} (max √s≈{sq_max:.1f} GeV "
                f"for this grid). Rebuild the grid with a larger omx_hi, or use the 2-D "
                f"direct-eMELA path (isr_lumi=False).")
        tj, wj = _jac01(n_out, 2.0 * be - 1.0, 0.0)        # ∫₀¹ τ^{2β−1}
        V = V_top * tj
        Lt = _ltilde_at(rho_tilde, be, V, n_jac)
        outer_w = V_top ** (2.0 * be) * wj * Lt            # σ̂-independent weight
        shat_arg = np.exp(-V / 2.0) * sq
        setups.append((shat_arg, outer_w))
    _LUMI_CACHE[key] = setups
    return setups


def prewarm(sqrt_s_arr, cfg: isr_beta.ISRConfig,
            n_out: int = LUMI_N_OUT, n_jac: int = LUMI_N_JAC):
    """Build + cache the σ̂-independent per-√s luminosity setup at the production
    resolution, so a parent process pays the one-time L̃ self-convolution once and
    a fork pool (morph build) inherits it via COW.  Called by
    ``isr_beta._radiator_setup`` on the lumi path; idempotent (returns the cache)."""
    a = np.ascontiguousarray(sqrt_s_arr, dtype=float)
    return _lumi_setup(a, cfg, n_out, n_jac)


def sigma_obs(sqrt_s, sigma_hat_fn, cfg: isr_beta.ISRConfig,
              n_out: int = LUMI_N_OUT, n_jac: int = LUMI_N_JAC):
    """σ_obs(√s) [fb] via the luminosity (single Gauss-Jacobi outer panel; L̃ by
    direct self-conv of the eMELA-grid ρ̃, cached per √s).

    Drop-in for ``isr_beta.convolve_2leg`` on the NLL path: ``sigma_hat_fn(√ŝ)``
    returns σ̂ [fb] for an array of √ŝ.  Vectorised in ``sqrt_s``.  Requires
    ``cfg.nll`` and ``cfg.emela_grid`` set (the per-leg ρ̃ source)."""
    if not (cfg.nll and cfg.emela_grid):
        raise ValueError("isr_lumi.sigma_obs requires cfg.nll=True and "
                         "cfg.emela_grid set (the per-leg ρ̃ source)")
    sqrt_s_arr = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    out = np.empty_like(sqrt_s_arr)
    setups = _lumi_setup(sqrt_s_arr, cfg, n_out, n_jac)
    for idx in range(len(sqrt_s_arr)):
        shat_arg, outer_w = setups[idx]
        out[idx] = float(np.sum(outer_w
                                * np.asarray(sigma_hat_fn(shat_arg), dtype=float)))
    return float(out[0]) if np.ndim(sqrt_s) == 0 else out


if __name__ == "__main__":
    import os
    from scipy.special import beta as _Beta
    from framework.process.ww.indep import grid as _grid
    assert SIGMA_GRID_LO == _grid.ECM_MIN, (SIGMA_GRID_LO, _grid.ECM_MIN)
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "..", "..", ".."))
    _PG = os.path.join(_REPO, "framework/process/ww/indep/grids/"
                              "emela_nll_delta_alpmz.npz")
    cfg = isr_beta.ISRConfig(nll=True, alpha=isr_beta.ALPHA_MZ_EMELA,
                             emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ",
                             emela_grid=_PG)
    # V=0 double-soft anchor exact:
    rho_tilde, be = _rho_tilde_factory(cfg, cfg.mu_F(160.0))
    L0 = _ltilde_at(rho_tilde, be, np.array([0.0]), 200)[0]
    _be, norm_nll = _norm_nll(cfg, cfg.mu_F(160.0))
    anchor = (norm_nll * be) ** 2 * _Beta(be, be)
    print(f"L̃(0)={L0:.6e} vs analytic (norm_nll·β)²·B(β,β)={anchor:.6e} "
          f"(rel {L0/anchor - 1.0:+.1e})")
