"""Generation grid for the independent WW partonic σ̂ campaign.

The "Full" grid (user sign-off 2026-06-02): partonic √ŝ ∈ [156, 164] @ 0.25 GeV
(33 points) × 6 POI variation points × 6 channel blocks = 1188 MoCaNLO 4-run
jobs.  With ``pdf_set=none`` the MoCaNLO ``cms_beam_energy`` IS √ŝ, so this grid
is scanned directly.  The observed line shape σ(√s) is later obtained by
convolving the smooth σ̂(√ŝ) with the ISR radiator (framework.process.ww.indep.
isr_beta) and assembling the 6 blocks (channels.assemble_total).

√ŝ spans the full scan so the radiator (peaked at √ŝ ≈ √s) is well sampled at
every output √s; σ̂ ≈ 0 below the off-shell WW turn-on (~158 GeV) but is kept on
the grid for a clean interpolation.
"""

from __future__ import annotations

import numpy as np

from framework.process.ww.indep.channels import BLOCKS
from framework.process.ww.indep.varpoints import VARPOINTS

#: Partonic √ŝ scan [GeV].
ECM_MIN = 156.0
ECM_MAX = 164.0
ECM_STEP = 0.25


def ecm_grid() -> np.ndarray:
    n = int(round((ECM_MAX - ECM_MIN) / ECM_STEP)) + 1
    return np.round(ECM_MIN + ECM_STEP * np.arange(n), 4)


def enumerate_points():
    """Yield (channel_key, varpoint_key, ecm) for every grid point."""
    grid = ecm_grid()
    for block in BLOCKS:
        for vp in VARPOINTS:
            for ecm in grid:
                yield block.key, vp.key, float(ecm)


def n_points() -> int:
    return len(BLOCKS) * len(VARPOINTS) * len(ecm_grid())


if __name__ == "__main__":
    g = ecm_grid()
    print(f"√ŝ grid: {len(g)} pts {g[0]}..{g[-1]} @ {ECM_STEP} GeV")
    print(f"channels={len(BLOCKS)} varpoints={len(VARPOINTS)} "
          f"→ total points = {n_points()}")
