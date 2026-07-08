"""Build the 4 eMELA NLL grids for the alpha-stability scheme scan (step 1).

Grid matrix: {DELTA, MSBAR} x {alpha(M_Z)=1/128.943 (PDG), 1/128.232 (MoCaNLO)}.
All four share the PRODUCTION coverage (omx in [3e-16,0.5] @16/decade, Q in
[75,350] @16 knots, ren_scheme=ALPMZ, pert_order=NLL) so the ONLY differences are
the factorisation scheme and alpha input.  Written to /tmp (throwaway scan inputs,
kept off the AFS work volume).

Run:  PYTHONPATH=$PWD python3 scripts/investigations/nll_isr/build_scheme_scan_grids.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WW_ISR_RADIATOR_CACHE", "")

from framework.process.ww.indep import isr_beta as ib            # noqa: E402
from framework.process.ww.indep import isr_emela_grid as eg       # noqa: E402

OUTDIR = "/tmp/ww_nll_scheme_scan/grids"
os.makedirs(OUTDIR, exist_ok=True)

# (scheme, alpha, alpha-tag) — the 2x2 matrix.
ALPHAS = [("alpmz", ib.ALPHA_MZ_EMELA), ("moca", ib.ALPHA_MZ)]
SCHEMES = ["DELTA", "MSBAR"]

# omx_lo tracks eg.OMX_FLOOR (deep edge, 2026-07-03 endpoint fix); existing
# shallow grids remain usable — xfxQ continues log-linearly below their edge.
# EXCEPTION: MSBAR-fac is endpoint-pathological and code_pdf goes NEGATIVE
# below omx ≈ 1e-17 (measured 2026-07-03: xD(1e-20) = −1.4e17), so its grids
# keep the old 3e-16 edge — EmelaGrid would (rightly) refuse the negative
# samples; below the edge the xfxQ log-linear continuation stays positive.
omx_knots = eg.default_omx_knots(omx_hi=0.5, per_decade=16)
omx_knots_msbar = eg.default_omx_knots(omx_lo=3e-16, omx_hi=0.5, per_decade=16)
q_knots = eg.default_q_knots(75.0, 350.0, 16)


def grid_path(scheme, atag):
    return os.path.join(OUTDIR, f"emela_nll_{scheme.lower()}_{atag}.npz")


def main():
    print(f"coverage: {omx_knots.size}x{q_knots.size} = "
          f"{omx_knots.size*q_knots.size} samples/grid")
    for scheme in SCHEMES:
        for atag, alpha in ALPHAS:
            npz = grid_path(scheme, atag)
            print(f"\n[build] {scheme} / ALPMZ / alpha={alpha:.6e} "
                  f"(1/{1.0/alpha:.3f}) -> {npz}")
            ok = omx_knots_msbar if scheme == "MSBAR" else omx_knots
            g = eg.build_and_write(npz, fac_scheme=scheme, ren_scheme="ALPMZ",
                                   alpha=alpha, omx_knots=ok,
                                   q_knots=q_knots, pert_order="NLL",
                                   write_lhagrid1=False, verbose=True)
            print(f"[build] meta = {g.meta}")
    print("\nDONE. Grid paths:")
    for scheme in SCHEMES:
        for atag, _ in ALPHAS:
            print(" ", grid_path(scheme, atag))


if __name__ == "__main__":
    main()
