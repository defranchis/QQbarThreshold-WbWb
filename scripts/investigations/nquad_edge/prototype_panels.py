"""Prototype: edge-aware per-leg u-panels for sigma_ISR_2leg_convolution.

The 2-leg tensor GL rule straddles the sigma-hat support structure --- the
BFS floor ramp on [149, 150] GeV (quintic, C2) with sigma-hat == 0 below ---
producing a few-100-ppm n_quad ripple at the low scan edge (measured +494 ppm
n128 vs n256 at 157.5, morph anchor, HEAD c9b0072).

Idea: the per-leg endpoint substitution u = (1-x)^(beta/2) is global, so the
per-leg integrand value at ANY u is the same formula (H_SV + jac*NS, or the
eMELA per-leg weight); only the NODE LAYOUT changes. Concatenating GL panels
in u-space [0, u_1], [u_1, u_2], ... puts clustered GL endpoints exactly at
the images of the sigma-hat kink lines, which is where the tensor rule loses
its accuracy.

Variants (per-leg breakpoints given as partonic-floor energies F -> the
DIAGONAL crossing x_split = F/sqrt(s), u_split = (1-x_split)^(beta/2)):

  V0  status quo: single panel
  V1  split at F=149 (support edge)
  V2  split at F=150 (ramp top)
  V3  splits at F=149 and F=150 (ramp isolated in its own panel)
  V4  V3 + drop nodes below the MARGINAL dead edge x < (149/sqrt s)^2
      (outer band where the whole row/col is dead regardless of the other leg)

All at total n_quad nodes (panels get proportional shares, min 8/panel).
Reference: V3 at n=768.

Usage: PYTHONPATH=.:$PYTHONPATH python3 scripts/investigations/nquad_edge/prototype_panels.py
"""

import numpy as np

from framework.process.ww.xsec_calculator.isr import (
    _Gee_per_leg_NS, _H_SV_per_leg, _quad_nodes, beta_ISR,
)
from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq

MW, GW = 80.379, 2.085
X_MIN = np.sqrt(0.30)
SIGMA_KW = dict(channel="inclusive", include_coulomb=False,
                whizard_anchor_source="morph")


def _panel_grid(beta_half, x_min, breaks_x, n_total):
    """GL nodes/weights on u-panels defined by per-leg x breakpoints.

    breaks_x: descending x values in (x_min, 1) -> ascending u breakpoints.
    Returns (x_vals, one_minus_x, jac_NS, w) concatenated over panels."""
    u_max = (1.0 - x_min) ** beta_half
    edges = [0.0] + sorted((1.0 - bx) ** beta_half for bx in breaks_x
                           if x_min < bx < 1.0) + [u_max]
    edges = sorted(set(e for e in edges if 0.0 <= e <= u_max))
    n_panel = len(edges) - 1
    lengths = np.diff(edges)
    # proportional node allocation, min 8, sum == n_total
    n_i = np.maximum(8, np.round(n_total * lengths / lengths.sum()).astype(int))
    while n_i.sum() > n_total:
        n_i[np.argmax(n_i)] -= 1
    while n_i.sum() < n_total:
        n_i[np.argmax(lengths / n_i)] += 1
    us, ws = [], []
    for k in range(n_panel):
        u, w = _quad_nodes(int(n_i[k]), edges[k], edges[k + 1])
        us.append(u); ws.append(w)
    u = np.concatenate(us); w = np.concatenate(ws)
    one_minus_x = u ** (1.0 / beta_half)
    x_vals = 1.0 - one_minus_x
    with np.errstate(over="ignore", invalid="ignore"):
        jac_NS = np.where(u > 1e-300, u ** (1.0 / beta_half - 1.0) / beta_half, 0.0)
    return x_vals, one_minus_x, jac_NS, w


def sigma_2leg_panels(sq, n_quad, breaks_F=(), drop_dead=False):
    """Analytic-LL 2-leg convolution with per-leg u-panels at the diagonal
    images of the partonic floors ``breaks_F`` [GeV]."""
    s = sq * sq
    beta = beta_ISR(s)
    breaks_x = [F / sq for F in breaks_F]
    x_vals, omx, jac_NS, w = _panel_grid(beta / 2.0, X_MIN, breaks_x, n_quad)
    NS = _Gee_per_leg_NS(x_vals, beta, one_minus_x=omx)
    per_leg = _H_SV_per_leg(beta) + jac_NS * NS
    weight = w * per_leg
    if drop_dead:
        # a node is useless if even with the other leg at x=1 the product
        # is below the true support edge (149 GeV)
        alive = x_vals >= (149.0 / sq) ** 2
        x_vals, weight = x_vals[alive], weight[alive]
    X1, X2 = np.meshgrid(x_vals, x_vals, indexing="ij")
    sh = np.asarray(sigma_partonic_munuqq((X1 * X2 * s).ravel(), MW, GW,
                                          **SIGMA_KW), float).reshape(X1.shape)
    return float(np.einsum("i,j,ij->", weight, weight, sh))


def main():
    variants = {
        "V0 single": dict(breaks_F=()),
        "V1 F=149": dict(breaks_F=(149.0,)),
        "V2 F=150": dict(breaks_F=(150.0,)),
        "V3 149+150": dict(breaks_F=(149.0, 150.0)),
        "V4 V3+drop": dict(breaks_F=(149.0, 150.0), drop_dead=True),
    }
    for sq in (157.5, 161.0, 162.5):
        ref = sigma_2leg_panels(sq, 768, breaks_F=(149.0, 150.0))
        print(f"\nsqrt_s = {sq}  (ref V3@768 = {ref:.9f} pb)")
        hdr = f"  {'variant':12s}" + "".join(f"  n={n:<4d}" for n in (64, 128, 256))
        print(hdr + "   [ppm vs ref]")
        for name, kw in variants.items():
            cells = []
            for n in (64, 128, 256):
                v = sigma_2leg_panels(sq, n, **kw)
                cells.append(f"{1e6*(v/ref-1):+8.1f}")
            print(f"  {name:12s}" + "".join(f"  {c}" for c in cells))


if __name__ == "__main__":
    main()
