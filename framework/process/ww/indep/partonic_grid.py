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


def _read_csv(path: str) -> dict | None:
    """Parse one result CSV (header + first data row), or ``None`` if the file is
    empty/truncated/malformed.  EOS productions occasionally leave a header-only
    or zero-byte CSV when a worker is killed mid-write; such a point is treated
    as missing (one σ̂ √ŝ hole the smoothing spline absorbs) rather than crashing
    the whole load."""
    with open(path) as fh:
        header_line = fh.readline()
        row_line = fh.readline()
    if not header_line.strip() or not row_line.strip():
        return None
    rec = dict(zip(header_line.strip().split(","), row_line.strip().split(",")))
    if "channel" not in rec or "varpoint" not in rec:
        return None
    try:
        for k in ("ecm", "mW", "gW", "sigma_born", "err_born", "sigma_nlo",
                  "err_nlo", "sigma_virt", "sigma_real", "sigma_idip"):
            if k in rec:
                rec[k] = float(rec[k])
    except ValueError:
        return None
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


def fiducial_suffix(scheme_alpha: str, lepton_cut: float | None,
                    lepton_pt_min: float | None = None,
                    lepton_mll_min: float | None = None) -> str:
    """Result-CSV suffix for a campaign — mirrors ``run_point.py`` / ``submit_grid``.

    Inclusive (``lepton_cut is None``) → ``_<scheme>``; fiducial →
    ``_<scheme>_cut<NN>[pt<PT>][mll<MLL>]`` (the pt/mll tokens only when set).
    """
    if lepton_cut is None:
        return f"_{scheme_alpha}"
    suffix = f"_{scheme_alpha}_cut{int(round(lepton_cut * 100))}"
    if lepton_pt_min is not None:
        suffix += f"pt{int(round(lepton_pt_min))}"
    if lepton_mll_min is not None:
        suffix += f"mll{int(round(lepton_mll_min))}"
    return suffix


def load_grids(results_dir: str = DEFAULT_RESULTS_DIR,
               scheme_alpha: str = "gf",
               lepton_cut: float | None = None,
               lepton_pt_min: float | None = None,
               lepton_mll_min: float | None = None
               ) -> dict[tuple[str, str], ChannelVarGrid]:
    """Return {(channel, varpoint): ChannelVarGrid} from result CSVs.

    ``lepton_cut`` selects the campaign: ``None`` → inclusive (no-cut) files
    ``*_<scheme>.csv``; a value (e.g. 0.97) → fiducial files
    ``*_<scheme>_cut<NN>[pt<PT>][mll<MLL>].csv`` (the production fiducial set is
    ``cut97pt10mll10``).  The glob anchors on the full suffix so an inclusive
    load never picks up a fiducial file and vice-versa.
    """
    suffix = fiducial_suffix(scheme_alpha, lepton_cut,
                             lepton_pt_min, lepton_mll_min)
    rows: dict[tuple[str, str], list[dict]] = {}
    skipped = 0
    for path in glob.glob(os.path.join(results_dir, f"*{suffix}.csv")):
        rec = _read_csv(path)
        if rec is None:                      # empty/truncated → treat as missing
            skipped += 1
            continue
        key = (rec["channel"], rec["varpoint"])
        rows.setdefault(key, []).append(rec)
    if skipped:
        import warnings
        warnings.warn(f"load_grids: skipped {skipped} empty/malformed CSV(s) "
                      f"matching *{suffix}.csv under {results_dir}", stacklevel=2)

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
