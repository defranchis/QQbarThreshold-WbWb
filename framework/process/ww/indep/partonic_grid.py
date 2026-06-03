"""Read the MoCaNLO partonic σ̂ grid and build smooth σ̂(√ŝ) interpolators.

Consumes the per-point result CSVs written by ``scripts/indep_mocanlo/
run_point.py`` (one file per channel×varpoint×√ŝ), groups them by
(channel, varpoint), and exposes denoised interpolators σ̂_Born(√ŝ) and
σ̂_NLO(√ŝ) [fb] for the ISR convolution.

Denoising: a weighted smoothing spline (scipy ``UnivariateSpline``, weights =
1/err) absorbs the per-point MC fluctuations (~0.3–0.5 %); with 33 points the
smooth curve is far more precise than any single point.  σ̂ is clamped to ≥0 and
returns 0 outside the grid (the radiator is peaked at √ŝ≈√s, and σ̂≈0 below the
WW turn-on, so the off-grid region contributes negligibly).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline


DEFAULT_RESULTS_DIR = ("/eos/user/m/mdefranc/FCC/QQbar_threshold/"
                       "grid_gen/results")


def _read_csv(path: str) -> dict:
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        row = fh.readline().strip().split(",")
    rec = dict(zip(header, row))
    for k in ("ecm", "mW", "gW", "sigma_born", "err_born", "sigma_nlo",
              "err_nlo", "sigma_virt", "sigma_real", "sigma_idip"):
        if k in rec:
            rec[k] = float(rec[k])
    return rec


@dataclass
class ChannelVarGrid:
    channel: str
    varpoint: str
    ecm: np.ndarray
    sigma_born: np.ndarray
    err_born: np.ndarray
    sigma_nlo: np.ndarray
    err_nlo: np.ndarray

    def _spline(self, y, err, smooth: float | None):
        # weights = 1/err; default smoothing factor s = len(points) (χ²≈N).
        w = 1.0 / np.maximum(err, 1e-12 * np.maximum(np.abs(y), 1.0))
        s = len(self.ecm) if smooth is None else smooth
        order = min(3, len(self.ecm) - 1)
        spl = UnivariateSpline(self.ecm, y, w=w, k=order, s=s, ext="zeros")
        lo, hi = self.ecm[0], self.ecm[-1]

        def fn(sqrt_shat):
            x = np.asarray(sqrt_shat, dtype=float)
            val = spl(np.clip(x, lo, hi))
            val = np.where((x >= lo) & (x <= hi), val, 0.0)
            return np.clip(val, 0.0, None)
        return fn

    def born_fn(self, smooth: float | None = None):
        return self._spline(self.sigma_born, self.err_born, smooth)

    def nlo_fn(self, smooth: float | None = None):
        return self._spline(self.sigma_nlo, self.err_nlo, smooth)


def load_grids(results_dir: str = DEFAULT_RESULTS_DIR,
               scheme_alpha: str = "gf",
               lepton_cut: float | None = None
               ) -> dict[tuple[str, str], ChannelVarGrid]:
    """Return {(channel, varpoint): ChannelVarGrid} from result CSVs.

    ``lepton_cut`` selects the campaign: ``None`` → inclusive (no-cut) files
    ``*_<scheme>.csv``; a value (e.g. 0.95) → fiducial files
    ``*_<scheme>_cut<NN>.csv``.
    """
    suffix = (f"_{scheme_alpha}" if lepton_cut is None
              else f"_{scheme_alpha}_cut{int(round(lepton_cut*100))}")
    rows: dict[tuple[str, str], list[dict]] = {}
    for path in glob.glob(os.path.join(results_dir, f"*{suffix}.csv")):
        rec = _read_csv(path)
        key = (rec["channel"], rec["varpoint"])
        rows.setdefault(key, []).append(rec)

    grids: dict[tuple[str, str], ChannelVarGrid] = {}
    for key, recs in rows.items():
        recs.sort(key=lambda r: r["ecm"])
        ecm = np.array([r["ecm"] for r in recs])
        grids[key] = ChannelVarGrid(
            channel=key[0], varpoint=key[1], ecm=ecm,
            sigma_born=np.array([r["sigma_born"] for r in recs]),
            err_born=np.array([r["err_born"] for r in recs]),
            sigma_nlo=np.array([r["sigma_nlo"] for r in recs]),
            err_nlo=np.array([r["err_nlo"] for r in recs]),
        )
    return grids
