"""Prototype 2: inner-leg edge-as-LIMIT quadrature for the 2-leg convolution.

For fixed outer x1, sigma-hat's support/kink lines map to inner-x2 points:
    x2 = Z149/x1   (support edge: sigma-hat == 0 below sqrt(s-hat)=149)
    x2 = Z150/x1   (ramp top: quintic smoothstep ends; C-inf above)
Making these the inner integration LIMITS (dead region dropped entirely,
ramp region its own panel) removes every interior kink from the quadrature
domain — the 2-D tensor rule's straddle ripple should collapse.

Analytic-LL radiator (closed form -> inner nodes can sit anywhere free).
Compare V0 (status quo tensor) vs EDGE (inner limits) at n=64/96/128/256
against EDGE@512 references.

Usage: PYTHONPATH=.:$PYTHONPATH python3 scripts/investigations/nquad_edge/prototype_inner_limits.py
"""

import numpy as np

from framework.process.ww.xsec_calculator.isr import (
    _Gee_per_leg_NS, _H_SV_per_leg, _endpoint_substitution, _quad_nodes,
    beta_ISR,
)
from framework.process.ww.xsec_calculator.eft_xsec import sigma_partonic_munuqq

MW, GW = 80.379, 2.085
X_MIN = np.sqrt(0.30)
F_LO, F_HI = 149.0, 150.0     # BFS floor / ramp top [GeV]
SIGMA_KW = dict(channel="inclusive", include_coulomb=False,
                whizard_anchor_source="morph")


def _perleg(x, omx, jac, beta):
    return _H_SV_per_leg(beta) + jac * _Gee_per_leg_NS(x, beta, one_minus_x=omx)


def _u_panel(beta_half, u_lo, u_hi, n):
    """GL nodes on [u_lo, u_hi] in u-space -> (x, omx, jac, w)."""
    u, w = _quad_nodes(n, u_lo, u_hi)
    omx = u ** (1.0 / beta_half)
    x = 1.0 - omx
    with np.errstate(over="ignore", invalid="ignore"):
        jac = np.where(u > 1e-300, u ** (1.0 / beta_half - 1.0) / beta_half, 0.0)
    return x, omx, jac, w


def sigma_2leg_edge(sq, n_quad, n_ramp=16):
    """Inner-leg edge-as-limit variant (outer plain, inner split at the
    per-outer-node images of the 149/150 floors)."""
    s = sq * sq
    beta = beta_ISR(s)
    bh = beta / 2.0
    z149, z150 = (F_LO / sq) ** 2, (F_HI / sq) ** 2

    # outer leg: the standard full-range endpoint grid
    u1, w1, x1v, omx1, jac1 = _endpoint_substitution(bh, X_MIN, n_quad)
    pl1 = _perleg(x1v, omx1, jac1, beta)

    total = 0.0
    for x1, wt1 in zip(x1v, w1 * pl1):
        if x1 <= 0:
            continue
        x2_edge = z149 / x1          # below: sigma-hat == 0 exactly
        x2_ramp = z150 / x1          # below: inside the quintic ramp
        if x2_edge >= 1.0:
            continue                 # whole inner range dead
        # smooth live panel: x2 in [min(1, max(x2_ramp, X_MIN)), 1]
        panels = []
        x_lo_live = min(1.0, max(x2_ramp, X_MIN))
        panels.append((x_lo_live, 1.0, n_quad))
        # ramp panel: [max(x2_edge, X_MIN), min(x2_ramp, 1)] — clip at 1 for
        # outer nodes whose ramp image extends past the domain (x1 < z150)
        a_r, b_r = max(x2_edge, X_MIN), min(x2_ramp, 1.0)
        if b_r > a_r:
            panels.append((a_r, b_r, n_ramp))
        inner = 0.0
        for a, b, n in panels:
            if b <= a:
                continue
            u_hi = (1.0 - a) ** bh
            u_lo = (1.0 - b) ** bh if b < 1.0 else 0.0
            x2, omx2, jac2, w2 = _u_panel(bh, u_lo, u_hi, n)
            pl2 = _perleg(x2, omx2, jac2, beta)
            sh = np.asarray(sigma_partonic_munuqq(x1 * x2 * s, MW, GW,
                                                  **SIGMA_KW), float)
            inner += float(np.sum(w2 * pl2 * sh))
        total += wt1 * inner
    return total


def sigma_2leg_v0(sq, n_quad):
    s = sq * sq
    beta = beta_ISR(s)
    u, w, xv, omx, jac = _endpoint_substitution(beta / 2.0, X_MIN, n_quad)
    pl = _perleg(xv, omx, jac, beta)
    weight = w * pl
    X1, X2 = np.meshgrid(xv, xv, indexing="ij")
    sh = np.asarray(sigma_partonic_munuqq((X1 * X2 * s).ravel(), MW, GW,
                                          **SIGMA_KW), float).reshape(X1.shape)
    return float(np.einsum("i,j,ij->", weight, weight, sh))


def main():
    for sq in (157.5, 161.0, 162.5):
        ref = sigma_2leg_edge(sq, 512, n_ramp=48)
        rows = []
        for n in (64, 96, 128, 256):
            v0 = sigma_2leg_v0(sq, n)
            ve = sigma_2leg_edge(sq, n)
            rows.append(f"  n={n:4d}  V0 {1e6*(v0/ref-1):+8.1f} ppm   EDGE {1e6*(ve/ref-1):+8.1f} ppm")
        print(f"sqrt_s = {sq}   (ref EDGE@512 = {ref:.9f} pb)")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
