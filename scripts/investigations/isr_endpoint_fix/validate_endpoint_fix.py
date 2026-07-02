"""Validate the 2026-07-02 eMELA-NLL endpoint fix (isr.py _build_emela_radiator).

The former omx<1e-15 branch replaced eMELA's NLL density with the analytic LL
endpoint constant over ~13% of the u-space GL weight, truncating 0.19% of the
per-leg radiator mass (sigma_obs -0.445% at 157.5).  After the fix every node
is a genuine eMELA query (omx passed explicitly).

Checks (reference values = the 2026-07-02 review's "direct-endpoint" variant):
  1. per-leg radiator integral at 157.5:  0.997762 (old) -> 0.999647 (new)
  2. sigma_obs(157.5) 0.166236 -> 0.166975 pb (+0.445%); 161/162.5 recorded
  3. substituted-node census: count of omx<1e-15 nodes and their GL weight
  4. n_quad 128->256 stability at 157.5 / 162.5 (ppm)
  5. disk-cache invalidation: _RADIATOR_DISK_VERSION=2 -> fresh rad_bfs_* hash

Production knobs: isr_nll=True, DELTA/ALPMZ/alpha(M_Z), include_coulomb=False,
full BFS sigma-hat, inclusive munuqq channel.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from framework.process.ww.xsec_calculator import isr
from framework.process.ww.xsec_calculator.isr import (
    _build_emela_radiator, _endpoint_substitution, _resolve_isr_alpha,
    beta_ISR, sigma_observed_munuqq, _RADIATOR_DISK_VERSION,
)

REF = {  # 2026-07-02 review, direct-endpoint variant (expected AFTER fix)
    "perleg_157.5": 0.999647,
    "sigma_157.5": 0.166975,
    "old_perleg_157.5": 0.997762,
    "old_sigma_157.5": 0.166236,
    "old_sigma_162.5_n128": 0.68674361,
}

PROD = dict(nll=True, emela_pert_order="NLL", emela_fac_scheme="DELTA",
            emela_ren_scheme="ALPMZ")
SIG = dict(isr_nll=True, include_coulomb=False, isr_scheme="2leg")


def perleg_integral(sq, n_quad=128):
    alpha_a = _resolve_isr_alpha(None, "ALPMZ")
    x, w, per_leg = _build_emela_radiator(
        sq, alpha_a=alpha_a, isr_scale_factor=1.0,
        x_min=np.sqrt(0.30), n_quad=n_quad, **PROD)
    return float(np.sum(w * per_leg))


def node_census(sq, n_quad=128):
    alpha_a = _resolve_isr_alpha(None, "ALPMZ")
    beta = beta_ISR(sq * sq, alpha_em=alpha_a)
    u, w, x, omx, jac = _endpoint_substitution(beta / 2.0, np.sqrt(0.30), n_quad)
    m = omx < 1e-15
    return int(m.sum()), float(w[m].sum() / w.sum())


def main():
    print(f"_RADIATOR_DISK_VERSION = {_RADIATOR_DISK_VERSION} (expect 2)")

    n, wfrac = node_census(161.0)
    print(f"\n[3] omx<1e-15 nodes at 161 GeV, n_quad=128: {n}/128 "
          f"({100*wfrac:.2f}% of GL weight)  [review: 30/128, 13.1%]")

    ipl = perleg_integral(157.5)
    print(f"\n[1] per-leg radiator integral @157.5: {ipl:.6f}  "
          f"[expect {REF['perleg_157.5']:.6f}, old {REF['old_perleg_157.5']:.6f}]")

    grid = np.array([157.5, 161.0, 162.5])
    s128 = sigma_observed_munuqq(grid, n_quad=128, **SIG)
    s256 = sigma_observed_munuqq(grid, n_quad=256, **SIG)
    print("\n[2] sigma_obs (inclusive munuqq, production NLL chain):")
    for sq, a, b in zip(grid, s128, s256):
        print(f"    {sq:6.1f}  n128 = {a:.6f} pb   n256 = {b:.6f} pb   "
              f"(n256/n128-1 = {1e6*(b/a-1):+.1f} ppm)")
    d = s128[0] / REF["old_sigma_157.5"] - 1.0
    print(f"    157.5 vs OLD (0.166236): {100*d:+.3f}%  [review: +0.445%]")
    print(f"    157.5 vs expected new  : {1e6*(s128[0]/REF['sigma_157.5']-1):+.1f} ppm")
    print(f"    162.5 vs OLD (n128 {REF['old_sigma_162.5_n128']:.6f}): "
          f"{100*(s128[2]/REF['old_sigma_162.5_n128']-1):+.3f}%")

    print("\n[4] n_quad stability (fix does NOT address the sigma-hat-edge "
          "sensitivity at the scan floor; expected to persist at O(-300 ppm) "
          "at 157.5 -- recorded, see review finding #2)")

    print("\ndone.")


if __name__ == "__main__":
    main()
