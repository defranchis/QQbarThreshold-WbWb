"""Build the PRODUCTION eMELA-NLL ISR grid for the independent WW chain.

Samples eMELA's x*D(x,Q) once in the production NLL convention and persists it as
an .npz (runtime) + standard lhagrid1 set under
``framework/process/ww/indep/grids/``.  The artifact is tiny (~tens of KiB) so it
lives on AFS next to the code; point ``ISRConfig.emela_grid`` at the .npz to use
it (opt-in — the default path is direct eMELA).

Convention (must match generator_mocanlo._isr_cfg's production NLL branch):
    pert_order=NLL, fac_scheme=DELTA, ren_scheme=ALPMZ, alpha=ALPHA_MZ(=alpha(M_Z)).

Grid coverage:
  omx in [3e-16, 0.5] at 16/decade  — 3e-16 sits just below isr_beta's 1e-15
        analytic-norm cutoff (so the spline never extrapolates at the cutoff) yet
        keeps x=1-omx < 1.0 (avoids the float64 x->1 underflow).
  Q in [75, 350] GeV, 16 log knots — covers mu_F = mu_F_factor*sqrt(s) for the
        scan sqrt(s) ~ [150,172] with xi=0.5/2 scale variations AND the
        mu_F_abs=m_W(~80.4) absolute-scale option.

Run:  PYTHONPATH=$PWD python3 scripts/investigations/isr_prewarm_lhapdf/build_production_grid.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WW_ISR_RADIATOR_CACHE", "")   # no radiator pkls (build samples code_pdf directly)

from framework.process.ww.indep import isr_beta as ib            # noqa: E402
from framework.process.ww.indep import isr_emela_grid as eg       # noqa: E402

OUTDIR = os.path.join(REPO, "framework", "process", "ww", "indep", "grids")
NPZ = os.path.join(OUTDIR, "emela_nll_delta_alpmz.npz")

omx_knots = eg.default_omx_knots(omx_lo=3e-16, omx_hi=0.5, per_decade=16)
q_knots = eg.default_q_knots(75.0, 350.0, 16)

print(f"[build] {omx_knots.size}x{q_knots.size} = {omx_knots.size*q_knots.size} "
      f"eMELA samples -> {NPZ}")
g = eg.build_and_write(NPZ, fac_scheme="DELTA", ren_scheme="ALPMZ",
                       alpha=ib.ALPHA_MZ_EMELA, omx_knots=omx_knots, q_knots=q_knots,
                       pert_order="NLL", write_lhagrid1=True, verbose=True)
print(f"[build] meta = {g.meta}")

# quick self-check: reload from disk and spot-check a few nodes vs eMELA truth
from framework.process.ww.xsec_calculator import emela_wrapper as em   # noqa: E402
em.initialize(pert_order="NLL", fac_scheme="DELTA", ren_scheme="ALPMZ", alpha=ib.ALPHA_MZ_EMELA)
g2 = eg.load_grid(NPZ)
import numpy as np                                                     # noqa: E402
worst = 0.0
for Q in (80.385, 157.5, 162.5, 168.0):
    for omx in (1e-14, 1e-10, 1e-6, 1e-3, 1e-1, 0.4):
        x = 1.0 - omx
        truth = em.code_pdf(x, omx, Q)
        got = float(g2.xfxQ(np.array([x]), np.array([omx]), Q)[0])
        rel = abs(got / truth - 1.0)
        worst = max(worst, rel)
print(f"[build] reload spot-check worst rel-diff vs eMELA = {worst:.2e}")
print(f"[build] npz size = {os.path.getsize(NPZ)/1024:.1f} KiB")
