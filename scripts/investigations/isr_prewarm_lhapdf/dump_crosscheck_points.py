"""Stage 1 of the real-LHAPDF cross-check (run in the FRAMEWORK python, which has
eMELA).  Builds the eMELA grid, writes it as a standard lhagrid1 set, and dumps a
set of test points with the eMELA ground truth + our omx-interpolator values, so
stage 2 (crosscheck_lhapdf_real.py, run under LCG_109 lhapdf) can compare the
STANDARD lhapdf log-x interpolation against the same truth — demonstrating where
vanilla LHAPDF interpolation degrades (the soft endpoint) and where it is fine.

Artifacts (on /tmp, never AFS):
    /tmp/ww_xcheck_<pid>.npz            our grid (npz)
    /tmp/ww_xcheck_<pid>_lha/           standard lhagrid1 set (for real lhapdf)
    /tmp/ww_xcheck_points.npz           test points + eMELA truth + our values
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from framework.process.ww.indep import isr_beta as ib            # noqa: E402
from framework.process.ww.indep import isr_emela_grid as eg       # noqa: E402
from framework.process.ww.xsec_calculator import emela_wrapper as em  # noqa: E402

ALPHA = ib.ALPHA_MZ
NPZ = "/tmp/ww_xcheck_grid.npz"
POINTS = "/tmp/ww_xcheck_points.npz"


def main():
    grid = eg.build_and_write(NPZ, fac_scheme="DELTA", ren_scheme="ALPMZ",
                              alpha=ALPHA, q_knots=eg.default_q_knots(150.0, 168.0, 14),
                              verbose=True)
    setdir = grid.meta["lhagrid1_set"]

    em.initialize(pert_order="NLL", fac_scheme="DELTA", ren_scheme="ALPMZ", alpha=ALPHA)
    # off-knot test omx, spanning bulk → soft endpoint, at a fixed Q.
    Q = 161.0
    omx = np.array([3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
                    1e-7, 1e-8, 1e-9, 1e-11, 1e-13])
    x = 1.0 - omx
    truth = np.array([em.code_pdf(float(xi), float(oi), Q) for xi, oi in zip(x, omx)])
    ours = grid.xfxQ(x, omx, Q)
    np.savez(POINTS, omx=omx, x=x, Q=Q, truth=truth, ours=ours,
             setdir=np.array(setdir), setname=np.array(os.path.basename(setdir)))
    print(f"set dir   : {setdir}")
    print(f"points    : {POINTS}  ({omx.size} pts at Q={Q})")
    print(f"LHAPDF_DATA_PATH should include: {os.path.dirname(setdir)}")


if __name__ == "__main__":
    main()
