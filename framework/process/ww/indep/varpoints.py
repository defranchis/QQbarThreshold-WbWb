"""POI variation points for the independent WW template grid.

Each partonic σ̂ grid is generated at these six (m_W, Γ_W) points so the fit can
build the morphing response (linear in m_W and Γ_W plus the bilinear m_W×Γ_W
cross term, mirroring the BFS chain's CROSS_TERMS).  m_W is the **OS** mass (the
project POI; MoCaNLO converts OS→pole internally and consistently).  Steps are
±10 MeV, matching the BFS bilinear-morph corner.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Central (nominal) OS values [GeV] — MoCaNLO ons_w_mass / ons_w_width.
MW0 = 80.379
GW0 = 2.085

#: Morphing step [GeV] = 10 MeV.
STEP_MW = 0.010
STEP_GW = 0.010


@dataclass(frozen=True)
class VarPoint:
    key: str
    mW: float
    gW: float
    label: str


VARPOINTS: tuple[VarPoint, ...] = (
    VarPoint("nominal", MW0,           GW0,           "m_W, Γ_W nominal"),
    VarPoint("massUp",  MW0 + STEP_MW, GW0,           "m_W +10 MeV"),
    VarPoint("massDn",  MW0 - STEP_MW, GW0,           "m_W −10 MeV"),
    VarPoint("widthUp", MW0,           GW0 + STEP_GW, "Γ_W +10 MeV"),
    VarPoint("widthDn", MW0,           GW0 - STEP_GW, "Γ_W −10 MeV"),
    VarPoint("cross",   MW0 + STEP_MW, GW0 + STEP_GW, "m_W +10 & Γ_W +10 (cross)"),
)

VARPOINTS_BY_KEY: dict[str, VarPoint] = {v.key: v for v in VARPOINTS}
