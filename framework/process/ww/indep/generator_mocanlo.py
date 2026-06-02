"""Independent (BFS-free) WW line-shape generator — MoCaNLO + decoupled ISR.

Drop-in for the BFS ``WWGenerator`` (same ``file_name`` / ``do_scan`` contract,
consumed by ``common.fit_core``).  Pipeline:

  1. MoCaNLO NLO-EW partonic σ̂_Born(√ŝ), σ̂_NLO(√ŝ) per channel × varpoint
     (``partonic_grid.load_grids`` → smooth interpolators).
  2. Per channel: ISR-matched observed line shape σ_obs(√s) =
     ∫∫ D D σ̂_NLO − C₁[σ̂_Born]  (``isr_beta.sigma_observed_matched``).
  3. Assemble the 6 blocks with flavour/colour multiplicities
     (``channels.assemble_total``) → σ_tot(√s) per varpoint.
  4. Quadratic + bilinear morph over the 6 (m_W, Γ_W) varpoints → σ_tot at the
     requested fit point.  Output in **pb** (BFS convention; MoCaNLO is fb ×1e-3).

This shares NO code or numerical input with the BFS-EFT chain: the goal is an
independent σ(m_W)/σ(Γ_W)/ρ, not number-matching.

Output √s outside the σ̂ generation grid [ECM_MIN, ECM_MAX] (156–164 GeV) is set
to 0 — the WW-threshold scan window lives inside it; the fit must not place scan
points above ECM_MAX (no σ̂ there to convolve).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.channels import (
    BLOCKS, BLOCKS_BY_KEY, PURE_WW_WEIGHTS,
)
from framework.process.ww.indep.varpoints import MW0, GW0, VARPOINTS
from framework.process.ww.indep.partonic_grid import (
    load_grids, DEFAULT_RESULTS_DIR, ChannelVarGrid,
)
from framework.process.ww.indep import grid as gridmod

FB_TO_PB = 1.0e-3

# Output grid: mirror the BFS WWGenerator (155–170 @0.1 + 240) so the fit reads
# an identical-shape CSV; values are 0 outside the σ̂ coverage.
ECM_FINE_MIN, ECM_FINE_MAX, ECM_FINE_STEP = 155.0, 170.0, 0.1
ECM_LAST = 240.0


def _build_fine_grid() -> np.ndarray:
    n = int(round((ECM_FINE_MAX - ECM_FINE_MIN) / ECM_FINE_STEP)) + 1
    g = ECM_FINE_MIN + ECM_FINE_STEP * np.arange(n)
    return np.concatenate([g, [ECM_LAST]])


@dataclass
class WWGeneratorMoCaNLO:
    """Independent WW template producer (MoCaNLO partonic ⊗ beta-scheme ISR)."""
    results_dir: str = DEFAULT_RESULTS_DIR
    scheme_alpha: str = "gf"
    lepton_cut: float | None = None     # None=inclusive(pure-WW); 0.95=fiducial
    isr_cfg: isr_beta.ISRConfig = field(default_factory=isr_beta.ISRConfig)
    order: int = 1                      # NLO-EW → filename tag "1"
    smooth: float | None = None         # σ̂ spline smoothing factor (None=auto)
    _grids: dict = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def _load(self):
        if self._grids is None:
            self._grids = load_grids(self.results_dir, self.scheme_alpha,
                                     self.lepton_cut)
            if not self._grids:
                raise FileNotFoundError(
                    f"no σ̂ grids under {self.results_dir} (scheme {self.scheme_alpha})")
        return self._grids

    def _weights(self) -> dict[str, float]:
        """Channel→multiplicity map for the active definition.

        Inclusive 'pure-WW' (lepton_cut is None): the 3 stable channels
        12·lnuqq + 4·qqqq + 9·mutau (channels.PURE_WW_WEIGHTS).  Fiducial
        (lepton_cut set): the original 6 blocks with their multiplicities.
        """
        if self.lepton_cut is None:
            return dict(PURE_WW_WEIGHTS)
        return {b.key: b.weight for b in BLOCKS}

    def _varpoint_lineshape(self, varpoint: str, sqrt_s: np.ndarray) -> np.ndarray:
        """Assembled σ_tot(√s) [pb] for one varpoint (cached)."""
        ck = (varpoint, id(sqrt_s))
        if ck in self._cache:
            return self._cache[ck]
        grids = self._load()
        weights = self._weights()
        sigma_tot = np.zeros_like(sqrt_s, dtype=float)
        for key, w in weights.items():
            g: ChannelVarGrid = grids[(key, varpoint)]
            obs = isr_beta.sigma_observed_matched(
                sqrt_s, g.nlo_fn(self.smooth), g.born_fn(self.smooth), self.isr_cfg)
            sigma_tot = sigma_tot + w * obs
        out = sigma_tot * FB_TO_PB           # fb → pb (BFS convention)
        self._cache[ck] = out
        return out

    def _fit_morph(self, sqrt_s: np.ndarray):
        """Least-squares quadratic+bilinear morph coefficients over all varpoints.

        Fits, per √s,  σ(Δm,Δw) = c0 + c1·Δm + c2·Δw + c3·Δm² + c4·Δw² + c5·ΔmΔw
        (Δ in MeV) to the assembled line shapes at every varpoint present in the
        grid.  Over-determined (~20 points, 6 coeffs) ⇒ the per-point MC noise is
        averaged down and the wide lever arm pins the slopes.  Returns coeffs of
        shape (6, len(sqrt_s)).  Cached per sqrt_s identity.
        """
        ck = ("coeffs", id(sqrt_s))
        if ck in self._cache:
            return self._cache[ck]
        grids = self._load()
        channels = list(self._weights())
        rows, rhs = [], []
        for v in VARPOINTS:
            if all((ch, v.key) in grids for ch in channels):
                dm, dw = v.dmW_MeV, v.dgW_MeV
                rows.append([1.0, dm, dw, dm * dm, dw * dw, dm * dw])
                rhs.append(self._varpoint_lineshape(v.key, sqrt_s))
        A = np.asarray(rows)                       # (n_vp, 6)
        Y = np.asarray(rhs)                        # (n_vp, n_s)
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)   # (6, n_s)
        self._cache[ck] = coeffs
        return coeffs

    def _morphed(self, mW: float, gW: float, sqrt_s: np.ndarray) -> np.ndarray:
        """σ_tot(√s; m_W, Γ_W) [pb] via the fitted quad+bilinear morph."""
        coeffs = self._fit_morph(sqrt_s)
        dm = (mW - MW0) * 1e3      # MeV
        dw = (gW - GW0) * 1e3
        basis = np.array([1.0, dm, dw, dm * dm, dw * dw, dm * dw])
        return basis @ coeffs

    # ------------------------------------------------------------------
    # WWGenerator contract
    # ------------------------------------------------------------------
    def file_tag(self, values: dict) -> str:
        mW = float(values["mass"]); gW = float(values["width"])
        return f"mass{mW:.3f}_width{gW:.3f}"

    def file_name(self, values: dict, *, mass_scale=None, width_scale=None,
                  mass_scheme: str = "OS", indir: str = ".") -> str:
        return os.path.join(indir, f"WW_{self.order}_{self.file_tag(values)}.txt")

    def do_scan(self, values: dict, *, mass_scale: float = 1.0,
                width_scale: float = 1.0, mass_scheme: str = "OS",
                outdir: str = "output_xsec/ww_indep/nominal",
                ecm_shift_MeV: float = 0.0) -> str:
        mW = float(values["mass"]); gW = float(values["width"])
        ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3

        # only convolve inside σ̂ coverage; 0 elsewhere
        inside = (ecm_grid >= gridmod.ECM_MIN) & (ecm_grid <= gridmod.ECM_MAX)
        sigma = np.zeros_like(ecm_grid)
        if inside.any():
            sigma[inside] = self._morphed(mW, gW, ecm_grid[inside])

        os.makedirs(outdir, exist_ok=True)
        path = self.file_name(values, indir=outdir)
        with open(path, "w") as fh:
            fh.write(f"# generator: WWGeneratorMoCaNLO (independent, BFS-free)\n")
            fh.write(f"# scheme_alpha: {self.scheme_alpha}\n")
            fh.write(f"# isr_scheme: {self.isr_cfg.scheme}  "
                     f"mu_F_factor: {self.isr_cfg.mu_F_factor}\n")
            fh.write(f"# mass: {mW:.4f}  width: {gW:.4f}  units: pb\n")
            for ecm, sig in zip(ecm_grid, sigma):
                fh.write(f"{ecm:.4f}, {sig:.8f}\n")
        return path
