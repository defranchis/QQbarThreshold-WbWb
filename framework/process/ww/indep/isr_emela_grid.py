"""LHAPDF-grid realization of the eMELA NLL electron ISR structure function.

The default independent-WW NLL path calls eMELA's ``code_pdf(x, omx, Q)`` once per
quadrature node — a Mellin-space DGLAP *re-evolution* at ~22 ms/query.  This
module is the alternative discussed as "the LHAPDF approach": sample eMELA's
x·D(x,Q) **once** onto an (x, Q) grid, persist it (LHAPDF ``lhagrid1`` format, so
the artifact is standard / inspectable / lhapdf-readable), and at runtime replace
the DGLAP solver with a fast table interpolation.  Unlike ``prewarm`` (which
caches our *quadrature* at fixed √s-grids), this caches the *PDF itself* and is
therefore independent of √s, μ_F and the quadrature — one grid serves every scan
point, every scale variation, the whole partonic ladder.

Why grid in 1-x, not x
----------------------
A textbook LHAPDF x-grid is dense as x→0; e+e- ISR lives entirely at x→1, where
the structure function carries the integrable soft singularity x·D ~ omx^(β-1).
So we lay knots in ``omx = 1-x`` (dense toward the endpoint) and interpolate
``ln(x·D)`` against ``(ln omx, ln Q)`` — the variables in which the resummed soft
tail is *nearly a straight line*, so even a cubic spline is sub-permille.  The
knots run down to ``OMX_FLOOR`` (deep endpoint; x = 1−omx underflows to exactly
1.0 there, which is fine — eMELA takes omx explicitly and applies its own soft
asymptotic internally).  BELOW the grid's own deepest knot, ``xfxQ`` continues
``ln(x·D)`` log-linearly with the spline's edge slope: the genuine NLL soft
drift is log-linear in these variables (measured through omx≈1e-66), so the
continuation IS the deep-endpoint physics.  No analytic substitution anywhere —
the pre-2026-07-03 ``norm_nll`` endpoint patch was the indep sibling of the BFS
9a8862b MAJOR (it truncated the genuine NLL soft enhancement: −0.19 % per-leg
radiator mass, −0.45 % σ_obs) and has been removed.

Standard LHAPDF interpolation (log-x, value-cubic) is the *wrong* variable for
this endpoint and degrades there — see the cross-check in
``scripts/investigations/isr_prewarm_lhapdf/``.  Our omx/log interpolation on the
same standard grid is the endpoint-aware fix; runtime needs only numpy+scipy (no
lhapdf dependency on worker nodes).
"""
from __future__ import annotations

import ast
import math
import os

import numpy as np

#: Default DEEP EDGE of built grids (``default_omx_knots``'s omx_lo) — a grid
#: floor only, NO physics substitution below it (2026-07-03 endpoint fix; the
#: old value 1e-15 was the boundary of the removed norm_nll substitution in
#: isr_beta/isr_lumi).  Consumers continue log-linearly below the grid's own
#: deepest knot (``EmelaGrid.xfxQ``), and 1e-70 is deep enough that the
#: continued region carries <1e-4 of the per-leg quadrature weight even at
#: n_quad=256 (deepest node omx≈1e-76) — with the log-linear drift exact, the
#: residual error there is negligible.
OMX_FLOOR = 1e-70

#: PDG id eMELA returns for the electron structure function.
ELECTRON_ID = 11


# ---------------------------------------------------------------------------
# Grid construction (sample eMELA once)
# ---------------------------------------------------------------------------

def default_omx_knots(omx_lo: float = OMX_FLOOR, omx_hi: float = 0.5,
                      per_decade: int = 8) -> np.ndarray:
    """Log-spaced omx knots from ``omx_lo`` to ``omx_hi`` (ascending)."""
    decades = math.log10(omx_hi) - math.log10(omx_lo)
    n = max(8, int(round(per_decade * decades)) + 1)
    return np.logspace(math.log10(omx_lo), math.log10(omx_hi), n)


def default_q_knots(q_lo: float = 150.0, q_hi: float = 175.0,
                    n: int = 12) -> np.ndarray:
    """Log-spaced Q knots [GeV] over the scan window (ascending)."""
    return np.logspace(math.log10(q_lo), math.log10(q_hi), n)


def sample_emela_table(omx_knots: np.ndarray, q_knots: np.ndarray, *,
                       fac_scheme: str, ren_scheme: str, alpha: float,
                       pert_order: str = "NLL"):
    """Sample x·D(x,Q) over (omx, Q) by calling eMELA once per knot.

    Returns ``table[i_omx, i_q] = x·D`` with x = 1 - omx.  ~per_decade×15 × n_q
    eMELA calls (≈ 1.5k × 22 ms ≈ 30 s) — paid once per (scheme, α) grid.
    """
    from framework.process.ww.xsec_calculator import emela_wrapper as _emela
    _emela.initialize(pert_order=pert_order, fac_scheme=fac_scheme,
                      ren_scheme=ren_scheme, alpha=alpha)
    omx = np.asarray(omx_knots, dtype=float)
    q = np.asarray(q_knots, dtype=float)
    table = np.empty((omx.size, q.size), dtype=float)
    for i, om in enumerate(omx):
        x = 1.0 - om
        for j, Q in enumerate(q):
            table[i, j] = _emela.code_pdf(x, float(om), float(Q))
    return table


# ---------------------------------------------------------------------------
# Runtime grid object (omx/log-cubic interpolation; no lhapdf dependency)
# ---------------------------------------------------------------------------

class EmelaGrid:
    """Interpolating x·D(x,Q) from a precomputed (omx, Q) table.

    Interpolation: cubic B-spline of ``ln(x·D)`` over ``(ln omx, ln Q)`` (via
    scipy RectBivariateSpline).  ``xfxQ(x, omx, Q)`` is vectorised over the
    nodes; Q is a scalar per convolution call (the per-leg radiator is built at a
    single μ_F).  Above the omx-knot range xfxQ fails loud (bulk edge is below
    x_min by construction); BELOW the deepest knot it continues ``ln(x·D)``
    log-linearly with the spline's edge slope — the genuine NLL soft drift is
    log-linear there, so the continuation is exact deep-endpoint physics
    (2026-07-03 fix; the pre-fix edge CLAMP existed only to backstop the
    norm_nll substitution, now removed)."""

    def __init__(self, omx_knots, q_knots, table, meta=None):
        from scipy.interpolate import RectBivariateSpline
        omx = np.ascontiguousarray(omx_knots, dtype=float)
        q = np.ascontiguousarray(q_knots, dtype=float)
        tab = np.ascontiguousarray(table, dtype=float)
        order = np.argsort(omx)
        self.omx = omx[order]
        self.q = q
        self.table = tab[order, :]
        self.meta = dict(meta or {})
        self._ln_omx = np.log(self.omx)
        self._ln_q = np.log(self.q)
        if np.any(self.table <= 0.0):
            raise ValueError("x*D must be positive to interpolate in log; "
                             "grid contains non-positive samples")
        ky = 3 if self.q.size > 3 else max(1, self.q.size - 1)
        self._spline = RectBivariateSpline(
            self._ln_omx, self._ln_q, np.log(self.table), kx=3, ky=ky)

    # -- persistence (fast native format) --
    def save_npz(self, path):
        np.savez(path, omx=self.omx, q=self.q, table=self.table,
                 meta=np.array(repr(self.meta)))

    @classmethod
    def load_npz(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = {}
        try:
            # literal_eval, NOT eval: the npz is loaded allow_pickle=False to avoid
            # executing file content; eval() would reintroduce arbitrary-code-exec.
            meta = ast.literal_eval(str(d["meta"]))
        except (ValueError, SyntaxError):
            pass
        return cls(d["omx"], d["q"], d["table"], meta)

    def xfxQ(self, x, omx, Q):
        """x·D(x,Q).  ``x``/``omx`` array-like (same shape), ``Q`` scalar.

        FAILS LOUD if a query lands ABOVE the omx knot range or OUTSIDE the Q
        range (beyond a tiny tolerance): silently edge-clamping there would bias
        the convolution with no error.  This bites if ``ISRConfig.x_min`` is
        lowered below the grid's ``1-omx_hi`` (max one_minus_x exceeds omx_hi) or
        if μ_F=mu_F_factor·√s leaves the grid's Q window (e.g. a ξ scale
        variation) — rebuild the grid wider.  BELOW the deepest omx knot,
        ``ln(x·D)`` is continued log-linearly in ``ln(omx)`` with the spline's
        edge slope (= β_e−1−δ, δ the genuine NLL soft-drift exponent ≈ 4e-4 —
        see :meth:`deep_slope`); the drift is log-linear through the deep
        endpoint, so the continuation is exact there (no flat clamp, no
        analytic substitution — 2026-07-03 fix)."""
        omx = np.asarray(omx, dtype=float)
        tol = 1e-9
        omx_max = float(np.max(omx)) if omx.size else self.omx[-1]
        if omx_max > self.omx[-1] * (1.0 + tol):
            raise ValueError(
                f"omx={omx_max:.4g} exceeds grid omx_hi={self.omx[-1]:.4g} "
                f"(x_min too low for this grid); rebuild with a larger omx_hi")
        Qf = float(Q)
        if not (self.q[0] * (1.0 - tol) <= Qf <= self.q[-1] * (1.0 + tol)):
            raise ValueError(
                f"Q={Qf:.4g} outside grid Q∈[{self.q[0]:.4g},{self.q[-1]:.4g}]; "
                f"rebuild the grid to cover μ_F=mu_F_factor·√s (incl. ξ variations)")
        # 1e-320 floor: keeps ln finite for an exact omx=0 query (measure-zero
        # in every consumer; the continued value there is finite and irrelevant).
        ln_omx = np.log(np.maximum(omx, 1e-320))
        lo = self._ln_omx[0]
        ln_q = math.log(min(max(Qf, self.q[0]), self.q[-1]))
        out_ln = self._spline(np.clip(ln_omx, lo, self._ln_omx[-1]), ln_q,
                              grid=False)
        below = ln_omx < lo
        if np.any(below):
            edge = float(self._spline(lo, ln_q, grid=False))
            slope = float(self._spline(lo, ln_q, dx=1, grid=False))
            out_ln = np.where(below, edge + slope * (ln_omx - lo), out_ln)
        out = np.exp(out_ln)
        return float(out) if np.ndim(omx) == 0 else out

    def deep_slope(self, Q) -> float:
        """d ln(x·D)/d ln(omx) at the grid's DEEPEST omx knot, at scale ``Q`` —
        the log-linear soft exponent the deep continuation in :meth:`xfxQ` uses.
        Equals β_e − 1 − δ with δ ≈ 4e-4 the genuine NLL soft-drift exponent
        (the integrand ∝ omx^(−δ) enhancement the 2026-07-03 endpoint fix
        restored); ``isr_lumi`` reads δ from here to absorb the drift into its
        Gauss-Jacobi weights."""
        ln_q = math.log(min(max(float(Q), self.q[0]), self.q[-1]))
        return float(self._spline(self._ln_omx[0], ln_q, dx=1, grid=False))


# ---------------------------------------------------------------------------
# Build + cache helpers
# ---------------------------------------------------------------------------

def build_grid(*, fac_scheme: str, ren_scheme: str, alpha: float,
               omx_knots=None, q_knots=None, pert_order: str = "NLL",
               verbose: bool = False) -> EmelaGrid:
    """Sample eMELA and return an in-memory ``EmelaGrid``."""
    omx = default_omx_knots() if omx_knots is None else np.asarray(omx_knots, float)
    q = default_q_knots() if q_knots is None else np.asarray(q_knots, float)
    if verbose:
        print(f"[emela_grid] sampling {omx.size}×{q.size} = {omx.size*q.size} "
              f"eMELA points ({fac_scheme}/{ren_scheme}, α={alpha:.6e})")
    table = sample_emela_table(omx, q, fac_scheme=fac_scheme,
                               ren_scheme=ren_scheme, alpha=alpha,
                               pert_order=pert_order)
    meta = dict(fac_scheme=fac_scheme, ren_scheme=ren_scheme, alpha=alpha,
                pert_order=pert_order, omx_lo=float(omx.min()),
                omx_hi=float(omx.max()), q_lo=float(q.min()), q_hi=float(q.max()))
    return EmelaGrid(omx, q, table, meta)


_GRID_CACHE: dict = {}


def load_grid(path: str) -> EmelaGrid:
    """Load (and in-process cache) an ``EmelaGrid`` from a ``.npz`` produced by
    ``EmelaGrid.save_npz`` / ``build_and_write``.  This is the singleton the
    isr_beta NLL-grid path pulls from.  Keyed by (abspath, mtime, size) so a
    grid REBUILT at the same path within one process invalidates the stale
    object instead of silently serving the old scheme/α."""
    ap = os.path.abspath(path)
    st = os.stat(ap)
    key = (ap, st.st_mtime_ns, st.st_size)
    g = _GRID_CACHE.get(key)
    if g is None:
        g = EmelaGrid.load_npz(ap)
        _GRID_CACHE[key] = g
    return g


def build_and_write(npz_path: str, *, fac_scheme: str, ren_scheme: str,
                    alpha: float, omx_knots=None, q_knots=None,
                    pert_order: str = "NLL", write_lhagrid1: bool = True,
                    verbose: bool = False) -> EmelaGrid:
    """Build a grid and persist it as ``.npz`` (runtime) and, optionally, a
    standard LHAPDF ``lhagrid1`` set alongside (portability / cross-check).
    Keep ``npz_path`` OFF the AFS work volume (use /tmp or EOS)."""
    g = build_grid(fac_scheme=fac_scheme, ren_scheme=ren_scheme, alpha=alpha,
                   omx_knots=omx_knots, q_knots=q_knots, pert_order=pert_order,
                   verbose=verbose)
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
    g.save_npz(npz_path)
    if write_lhagrid1:
        setdir = os.path.splitext(npz_path)[0] + "_lha"
        write_lhapdf_set(setdir, g)
        g.meta["lhagrid1_set"] = setdir
    if verbose:
        print(f"[emela_grid] wrote {npz_path} "
              f"({os.path.getsize(npz_path)/1024:.0f} KiB)")
    return g


# ---------------------------------------------------------------------------
# LHAPDF lhagrid1 writer (standard, so real lhapdf can read the same grid)
# ---------------------------------------------------------------------------

def write_lhapdf_set(setdir: str, grid: EmelaGrid, setname: str | None = None):
    """Write ``grid`` as a 1-member, 1-flavor LHAPDF6 ``lhagrid1`` set under
    ``setdir`` (``<set>.info`` + ``<set>_0000.dat``).  x knots ascending (=omx
    descending).  Used for the real-lhapdf cross-check and as a portable artifact.
    """
    setname = setname or os.path.basename(setdir.rstrip("/"))
    os.makedirs(setdir, exist_ok=True)
    x = (1.0 - grid.omx)[::-1]               # ascending x
    q = grid.q
    tab = grid.table[::-1, :]                # match x order; [i_x, i_q] = x·D
    # The densest endpoint omx knots (omx ≲ few ulp of 1) collapse under x=1−omx to
    # duplicate float64 values; lhagrid1 requires STRICTLY ascending x, so drop the
    # non-increasing nodes (keep the first occurrence) before writing x and tab.
    keep = np.concatenate(([True], np.diff(x) > 0.0))
    x, tab = x[keep], tab[keep, :]
    xmin, xmax = float(x.min()), float(x.max())
    qmin, qmax = float(q.min()), float(q.max())

    info = (
        f"SetDesc: eMELA NLL electron ISR structure function (omx-gridded)\n"
        f"SetIndex: 90000\n"
        f"Authors: WW-threshold indep chain (auto-generated)\n"
        f"Reference: eMELA arXiv:1911.12040, 2207.03265\n"
        f"Format: lhagrid1\n"
        f"DataVersion: 1\n"
        f"NumMembers: 1\n"
        f"Particle: 11\n"
        f"Flavors: [11]\n"
        f"OrderQCD: 0\n"
        f"FlavorScheme: fixed\n"
        f"NumFlavors: 0\n"
        f"ErrorType: replicas\n"
        f"XMin: {xmin:.16e}\nXMax: {xmax:.16e}\n"
        f"QMin: {qmin:.16e}\nQMax: {qmax:.16e}\n"
        f"MZ: 91.1876\n"
        f"AlphaS_MZ: 0.118\nAlphaS_OrderQCD: 0\nAlphaS_Type: ipol\n"
        f"AlphaS_Qs: [{qmin:.6e}, {qmax:.6e}]\nAlphaS_Vals: [0.118, 0.118]\n"
    )
    with open(os.path.join(setdir, f"{setname}.info"), "w") as fh:
        fh.write(info)

    lines = ["PdfType: central", "Format: lhagrid1", "---"]
    lines.append(" ".join(f"{v:.16e}" for v in x))
    lines.append(" ".join(f"{v:.16e}" for v in q))
    lines.append("11")
    for ix in range(x.size):                 # x outer, Q inner
        for iq in range(q.size):
            lines.append(f"{tab[ix, iq]:.16e}")
    lines.append("---")
    with open(os.path.join(setdir, f"{setname}_0000.dat"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return setdir
