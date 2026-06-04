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
deep endpoint ``omx < OMX_FLOOR`` is deliberately NOT gridded: there
``isr_beta`` uses the exact analytic soft+virtual constant (``norm_nll``),
identical to the direct path.  The grid only ever serves the mid region
``OMX_FLOOR ≤ omx ≤ omx_hi`` (≈ the convolution's x∈[x_min, 1-OMX_FLOOR]).

Standard LHAPDF interpolation (log-x, value-cubic) is the *wrong* variable for
this endpoint and degrades there — see the cross-check in
``scripts/investigations/isr_prewarm_lhapdf/``.  Our omx/log interpolation on the
same standard grid is the endpoint-aware fix; runtime needs only numpy+scipy (no
lhapdf dependency on worker nodes).
"""
from __future__ import annotations

import math
import os

import numpy as np

#: Deep-endpoint cutoff — MUST match isr_beta._per_leg_emela_nll's 1e-15 switch
#: to the analytic norm_nll.  The grid is built down to a slightly smaller omx so
#: the spline never extrapolates at the cutoff.
OMX_FLOOR = 1e-15

#: PDG id eMELA returns for the electron structure function.
ELECTRON_ID = 11


# ---------------------------------------------------------------------------
# Grid construction (sample eMELA once)
# ---------------------------------------------------------------------------

def default_omx_knots(omx_lo: float = 1e-16, omx_hi: float = 0.5,
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
    single μ_F).  Outside the omx-knot range the spline is clamped to the edge
    knot (the convolution never queries there — the endpoint is analytic and the
    bulk edge is below x_min)."""

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
            meta = eval(str(d["meta"]))            # our own repr, trusted
        except Exception:
            pass
        return cls(d["omx"], d["q"], d["table"], meta)

    def xfxQ(self, x, omx, Q):
        """x·D(x,Q).  ``x``/``omx`` array-like (same shape), ``Q`` scalar.

        FAILS LOUD if a query lands ABOVE the omx knot range or OUTSIDE the Q
        range (beyond a tiny tolerance): silently edge-clamping there would bias
        the convolution with no error.  This bites if ``ISRConfig.x_min`` is
        lowered below the grid's ``1-omx_hi`` (max one_minus_x exceeds omx_hi) or
        if μ_F=mu_F_factor·√s leaves the grid's Q window (e.g. a ξ scale
        variation) — rebuild the grid wider.  Below-range omx is clamped (it is
        below the analytic endpoint cutoff and never actually contributes)."""
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
        ln_omx = np.log(np.clip(omx, self.omx[0], self.omx[-1]))
        ln_q = math.log(min(max(Qf, self.q[0]), self.q[-1]))
        out = np.exp(self._spline(ln_omx, ln_q, grid=False))
        return float(out) if np.ndim(omx) == 0 else out


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
