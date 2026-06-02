"""POI variation points for the independent WW template grid.

Rich lever-arm grid (2026-06-02): m_W and Γ_W each at ±10/±20/±30/±50 MeV plus a
few cross points, so the morph is a least-squares quadratic + bilinear fit over
~20 points rather than a single ±10 MeV finite difference — the wider lever arm
(×5) and N-point averaging beat the per-point MC noise, and combined with
**correlated seeds** (same MC seed across varpoints at each √ŝ → the statistical
fluctuation cancels in the morph differences) give clean, mirror-symmetric
slopes.  m_W is the **OS** mass (project POI; MoCaNLO OS→pole internally).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Central (nominal) OS values [GeV] — MoCaNLO ons_w_mass / ons_w_width.
MW0 = 80.379
GW0 = 2.085

# legacy aliases (some callers import these)
STEP_MW = 0.010
STEP_GW = 0.010


@dataclass(frozen=True)
class VarPoint:
    key: str
    mW: float          # GeV
    gW: float          # GeV
    dmW_MeV: float     # m_W shift wrt nominal [MeV] (design-matrix coord)
    dgW_MeV: float     # Γ_W shift wrt nominal [MeV]
    label: str


def _mk(dm: float, dw: float) -> VarPoint:
    if dm == 0 and dw == 0:
        key = "nominal"
    elif dw == 0:
        key = f"m{'p' if dm > 0 else 'm'}{abs(int(dm))}"
    elif dm == 0:
        key = f"w{'p' if dw > 0 else 'm'}{abs(int(dw))}"
    else:
        key = (f"x{'p' if dm > 0 else 'm'}{abs(int(dm))}"
               f"{'p' if dw > 0 else 'm'}{abs(int(dw))}")
    return VarPoint(key, MW0 + dm * 1e-3, GW0 + dw * 1e-3, float(dm), float(dw),
                    f"Δm_W={dm:+g}, ΔΓ_W={dw:+g} MeV")


_MASS_SHIFTS = (10, 20, 30, 50)
_WIDTH_SHIFTS = (10, 20, 30, 50)
_CROSS = ((20, 20), (-20, 20), (50, 50))   # for the bilinear m_W×Γ_W term

VARPOINTS: tuple[VarPoint, ...] = (
    (_mk(0, 0),)
    + tuple(_mk(s, 0) for d in _MASS_SHIFTS for s in (d, -d))
    + tuple(_mk(0, s) for d in _WIDTH_SHIFTS for s in (d, -d))
    + tuple(_mk(dm, dw) for dm, dw in _CROSS)
)

VARPOINTS_BY_KEY: dict[str, VarPoint] = {v.key: v for v in VARPOINTS}


if __name__ == "__main__":
    print(f"{len(VARPOINTS)} varpoints:")
    for v in VARPOINTS:
        print(f"  {v.key:10s}  mW={v.mW:.5f}  gW={v.gW:.5f}  "
              f"(Δm={v.dmW_MeV:+g}, Δw={v.dgW_MeV:+g} MeV)")
