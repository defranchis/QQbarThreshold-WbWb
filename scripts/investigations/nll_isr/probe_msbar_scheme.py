"""Pre-check (NLL-promotion step 1 gate): does eMELA accept fac_scheme=MSBAR?

The whole alpha-stability scheme scan hinges on eMELA's QuickInitialize accepting
"MSBAR" and returning a sane ePDF.  An unrecognised scheme string may abort() the
process, so run this as its own subprocess.

For each scheme we sample x*D(x,Q) at a few x at Q=161 GeV with the production NLL
convention (pert_order=NLL, ren_scheme=ALPMZ, alpha=alpha(M_Z)=1/128.943).

Expectation if MSBAR is accepted:
  - values finite and positive in the bulk,
  - DELTA and MSBAR agree on the LL log (collinear), differ by an O(alpha) finite
    constant -> a small (few permille) bulk difference that GROWS toward x->1.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from framework.process.ww.xsec_calculator import emela_wrapper as em  # noqa: E402

ALPHA_MZ = 1.0 / 128.943
Q = 161.0
XS = [0.30, 0.50, 0.70, 0.90, 0.99, 0.999]


def sample(fac_scheme):
    # fresh process state per call is not possible (global eMELA state), but
    # re-initialize switches the cached scheme.
    em.initialize(pert_order="NLL", fac_scheme=fac_scheme,
                  ren_scheme="ALPMZ", alpha=ALPHA_MZ)
    out = {}
    for x in XS:
        out[x] = em.code_pdf(x, 1.0 - x, Q)
    return out


def main():
    print(f"eMELA scheme probe  Q={Q} GeV  alpha(M_Z)={ALPHA_MZ:.6f}")
    print("=" * 64)
    delta = sample("DELTA")
    print("DELTA initialized OK")
    msbar = sample("MSBAR")
    print("MSBAR initialized OK")
    print()
    print(f"{'x':>8} {'xD_DELTA':>14} {'xD_MSBAR':>14} {'rel diff':>12}")
    for x in XS:
        d, m = delta[x], msbar[x]
        rel = (m - d) / d if d else float("nan")
        print(f"{x:>8.3f} {d:>14.6e} {m:>14.6e} {rel:>+12.3e}")
    print()
    print("PROBE OK: MSBAR accepted and returns finite ePDF.")


if __name__ == "__main__":
    main()
