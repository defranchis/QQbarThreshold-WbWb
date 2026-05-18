"""WW threshold-scan template generator (EFT-based).

Computes σ(e+e- → μν qq̄, inclusive) on a fine ECM grid spanning the WW
threshold scan and writes a two-column CSV (``ecm, xsec``) consumed by
``common.fit_core.FitCore``.

Physics layers (assembled in :mod:`process.ww.eft_xsec` and :mod:`process.ww.isr`):

  • Doubly-resonant CC03 Born, calibrated against a RACOONWW Born grid
    (161.33–500 GeV; 156–161 GeV is a power-law BW-tail model — see the
    TODO in eft_xsec.py for the chief physics gap).
  • Finite-Γ_W complex-velocity prescription (EFT smoothing across threshold).
  • Coulomb K-factor (Fadin-Khoze-Martin + Bardin-Riemann O(α²)).
  • LL ISR convolution with YFS soft+virtual exponentiation and the β²
    non-singular piece (Cacciari et al. NPB 451).
  • BFS NLO / dominant-NNLO hooks (currently disabled — to be filled from
    arXiv:0707.0773 + 0807.0102 for MeV precision on m_W).

Precision today: ~1% on σ in the WW peak region, ~5–10% below threshold,
limited by (i) BFS NLO/NNLO not yet filled in and (ii) the power-law
BW-tail extrapolation 156–161 GeV.

The ``mass_scale`` / ``width_scale`` parameters are recorded in the
filename but have no effect on the LO+Coulomb+LL-ISR cross section (no
renormalisation scale to vary at this order). They become meaningful
when the BFS NLO hooks and NLL ISR are activated.
"""

from __future__ import annotations

import os

import numpy as np

from process.ww.eft_xsec import (
    BFSCorrections,
    M_W_DEFAULT, GAMMA_W_DEFAULT,
)
from process.ww.isr import sigma_observed_munuqq


# ---------------------------------------------------------------------------
# Fine ECM grid for the template (mirrors the WbWb convention)
# ---------------------------------------------------------------------------
# Wider than the analysis scan window so BES convolution has clean margin.
ECM_FINE_MIN  = 155.0
ECM_FINE_MAX  = 170.0
ECM_FINE_STEP = 0.1
ECM_LAST      = 240.0


def _build_fine_grid() -> np.ndarray:
    """Fine ECM grid: 155.0–170.0 step 0.1 GeV, plus ``ECM_LAST`` appended."""
    n = int(round((ECM_FINE_MAX - ECM_FINE_MIN) / ECM_FINE_STEP)) + 1
    grid = ECM_FINE_MIN + ECM_FINE_STEP * np.arange(n)
    return np.concatenate([grid, [ECM_LAST]])


class WWGenerator:
    """LO + Coulomb + LL-ISR template producer for the WW threshold fit."""

    def __init__(self, *, order: int = 2, channel: str = "inclusive",
                 include_coulomb: bool = True, bfs: BFSCorrections | None = None,
                 n_quad: int = 200, z_min: float = 0.10,
                 # BFS-NLO knobs (defaults match cards/ww_default.py NLO_CONFIG):
                 include_NLO_hard_decay: bool = True,
                 apply_delta_QCD: bool = False,
                 br_convention: str = "pdg-constant",
                 alpha_s: float = 0.1199,
                 apply_whizard_anchor: bool = False):
        self.order = order              # informational; recorded in filename
        self.channel = channel
        self.include_coulomb = include_coulomb
        self.bfs = bfs if bfs is not None else BFSCorrections(enabled=False)
        self.n_quad = n_quad
        self.z_min = z_min
        self.include_NLO_hard_decay = include_NLO_hard_decay
        self.apply_delta_QCD = apply_delta_QCD
        self.br_convention = br_convention
        self.alpha_s = alpha_s
        self.apply_whizard_anchor = apply_whizard_anchor

    # ------------------------------------------------------------------
    # Filenames (match the stub pattern so existing harness still works)
    # ------------------------------------------------------------------
    @staticmethod
    def _order_str(order: int) -> str:
        return {0: "LO", 1: "NLO", 2: "NNLO", 3: "N3LO"}.get(order, f"O{order}")

    def file_tag(self, values: dict) -> str:
        parts = []
        for name, val in values.items():
            label = "asVar" if name == "alphas" else name
            decimals = 4 if name == "alphas" else 3
            parts.append(f"{label}{val:.{decimals}f}")
        return "_".join(parts)

    def file_name(self, values: dict, *, mass_scale: float, width_scale: float,
                  mass_scheme: str = "OS", indir: str = ".") -> str:
        body = self.file_tag(values)
        scales = f"scaleM{mass_scale:.1f}_scaleW{width_scale:.1f}"
        return os.path.join(indir, f"WW_{self._order_str(self.order)}_{body}_{scales}.txt")

    # ------------------------------------------------------------------
    # Template production
    # ------------------------------------------------------------------
    def do_scan(self, values: dict, *, mass_scale: float, width_scale: float,
                mass_scheme: str = "OS", outdir: str = "output_WW",
                ecm_shift_MeV: float = 0.0) -> str:
        """Compute σ_obs(√s; m_W, Γ_W) on the fine ECM grid and write CSV.

        ``ecm_shift_MeV`` shifts every √s in the output grid by the given
        amount (used by the BEC nuisance machinery, which expects templates
        in ``BEC_variations_WW/scan_{p,m}{var}/``).

        Returns the output file path.
        """
        mW     = float(values["mass"])
        gammaW = float(values["width"])
        # ``alphas`` in the steering card is the OFFSET from the nominal
        # α_s(M_W) (matches the WbWb convention). Add it to the generator's
        # nominal α_s when computing the effective δ_QCD(α_s) at this fit point.
        alpha_s_eff = self.alpha_s + float(values.get("alphas", 0.0))

        ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3
        sigma_obs = sigma_observed_munuqq(
            ecm_grid,
            mW=mW, gammaW=gammaW,
            channel=self.channel,
            include_coulomb=self.include_coulomb,
            bfs=self.bfs,
            n_quad=self.n_quad,
            z_min=self.z_min,
            include_NLO_hard_decay=self.include_NLO_hard_decay,
            apply_delta_QCD=self.apply_delta_QCD,
            alpha_s=alpha_s_eff,
            br_convention=self.br_convention,
            apply_whizard_anchor=self.apply_whizard_anchor,
        )

        os.makedirs(outdir, exist_ok=True)
        path = self.file_name(values, mass_scale=mass_scale, width_scale=width_scale,
                              mass_scheme=mass_scheme, indir=outdir)
        # CSV: ecm, xsec (no header) — matches FitCore.read_csv contract.
        with open(path, "w") as fh:
            for ecm, sigma in zip(ecm_grid, sigma_obs):
                fh.write(f"{ecm:.4f}, {sigma:.8f}\n")
        return path
