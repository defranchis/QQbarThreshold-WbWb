"""Decouple ISR scale variation into β-only and Q-only contributions.

Question: in the eMELA paths, varying isr_scale_factor scales BOTH
  (i)  the LL log inside beta_ISR (integration-grid distribution + x->1 fallback),
  (ii) the eMELA DGLAP scale Q = xi * sqrt(s) (the physics).

The status-quo smoke test (smoke_test_scale_alphaMZ.py) sees ~equal envelopes
across analytic-LL / eMELA-LL / eMELA-NLL.  Hypothesis: (i) contaminates the
eMELA paths through grid-sampling residuals, making NLL look as variable as LL
when its physical scale dependence (Q only) is actually smaller.

Diagnostic: for each eMELA path, redo the xi sweep with beta_ISR held at the
xi=1 value while Q still scales.  Compare against the full status-quo sweep.

Expected if the hypothesis holds:
  - Q-only NLL envelope < Q-only LL envelope  (the canonical hierarchy)
  - Q-only envelope < full (β+Q) envelope     (grid contamination removed)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "..")))

import numpy as np

from cards import ww_default
from framework.process.ww.generator import WWGenerator
from framework.process.ww.xsec_calculator import isr as isr_mod
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq


def envelope(sigs):
    sig_lo, sig_c, sig_hi = sigs
    return max(abs(sig_hi - sig_c), abs(sig_c - sig_lo)) / sig_c * 100.0


class FreezeBetaScale:
    """Context manager that monkey-patches beta_ISR to ignore isr_scale_factor.

    Inside the with-block, beta_ISR(s, alpha_em, isr_scale_factor=anything)
    returns beta at xi=1.  Used to isolate the Q-only scale dependence in
    the eMELA paths.
    """
    def __enter__(self):
        self._orig = isr_mod.beta_ISR

        def _beta_no_scale(s, alpha_em=None, isr_scale_factor=1.0):
            return self._orig(s, alpha_em=alpha_em, isr_scale_factor=1.0)

        isr_mod.beta_ISR = _beta_no_scale
        return self

    def __exit__(self, *a):
        isr_mod.beta_ISR = self._orig


def sweep(sqrt_s, common, path_kwargs, scales, n_quad=None):
    kw = dict(path_kwargs)
    if n_quad is not None:
        kw["n_quad"] = n_quad
    return [
        float(np.atleast_1d(sigma_observed_munuqq(
            sqrt_s, **common, **kw, isr_scale_factor=xi))[0])
        for xi in scales
    ]


def main():
    sqrt_s = np.array([162.0])
    mW, gammaW = 80.379, 2.085

    gen = WWGenerator.from_card(ww_default)
    common = {
        "mW": mW, "gammaW": gammaW,
        "channel": gen.channel,
        "include_coulomb": gen.include_coulomb, "bfs": gen.bfs,
        "include_NLO_hard_decay": gen.include_NLO_hard_decay,
        "include_BFS_NNLO": gen.include_BFS_NNLO,
        "apply_delta_QCD": gen.apply_delta_QCD,
        "alpha_s": gen.alpha_s, "alpha_s_ref": gen.alpha_s,
        "br_convention": gen.br_convention,
        "apply_whizard_anchor": gen.apply_whizard_anchor,
        "whizard_anchor_source": gen.whizard_anchor_source,
        "isr_emela_pert_order": gen.isr_emela_pert_order,
        "isr_emela_fac_scheme": gen.isr_emela_fac_scheme,
        "isr_emela_ren_scheme": gen.isr_emela_ren_scheme,
        "alpha_em": gen.alpha_em, "alpha_em_isr": gen.alpha_em_isr,
        "coulomb_kc_safe": gen.coulomb_kc_safe,
        "decay_uses_full_born": gen.decay_uses_full_born,
        "m_t": gen.m_t, "M_H": gen.M_H, "MZ": gen.MZ,
    }
    scales = [0.5, 1.0, 2.0]

    paths = [
        ("analytic LL+exp",      {"isr_scheme": "single_conv",
                                  "isr_nll": False, "isr_emela_ll": False}),
        ("eMELA LL (BETA-DGLAP)", {"isr_scheme": "2leg",
                                  "isr_nll": False, "isr_emela_ll": True}),
        ("eMELA NLL (CodePdf)",   {"isr_scheme": "2leg",
                                  "isr_nll": True, "isr_emela_ll": False}),
    ]

    # n_quad sweep: pass the per-leg value to sigma_observed_munuqq.  The 2-leg
    # auto-map (n_quad=200 -> 128) is bypassed when we pass any other value.
    n_quads = [128, 192, 256]

    print(f"=== Scale-decouple diagnostic at sqrt(s) = {sqrt_s[0]} GeV ===")
    print(f"    xi in {scales}")
    print(f"    Card alpha_em_isr = {gen.alpha_em_isr:.6e}  (1/alpha = {1.0/gen.alpha_em_isr:.3f})")
    print(f"    Card ren_scheme   = {gen.isr_emela_ren_scheme}")
    print()

    for nq in n_quads:
        print(f"--- n_quad = {nq} (per leg in 2-leg paths) ---")
        print(f"{'Path':<28} {'full (beta+Q)':>14} {'Q-only':>10} {'beta-resid':>12}")
        print("-" * 70)

        for name, kw in paths:
            # Analytic LL+exp uses single_conv (n_quad means full conv);
            # eMELA paths use 2-leg (n_quad per leg).  Same call signature.
            sigs_full = sweep(sqrt_s, common, kw, scales, n_quad=nq)
            env_full = envelope(sigs_full)

            with FreezeBetaScale():
                sigs_Q = sweep(sqrt_s, common, kw, scales, n_quad=nq)
            env_Q = envelope(sigs_Q)

            beta_only_est = env_full - env_Q
            print(f"{name:<28} {env_full:>13.3f}% {env_Q:>9.3f}% {beta_only_est:>11.3f}%")
        print()

    print("Reading:")
    print("  - analytic LL+exp: only beta carries scale dep -> beta-resid=full, Q-only=0.")
    print("  - eMELA paths:    if grid is converged, beta-resid ~ 0 and full = Q-only.")
    print("                     non-zero beta-resid signals grid-residual contamination.")
    print("  - Pick smallest n_quad where eMELA beta-resid is below ~0.05% (i.e. small")
    print("    vs the 0.25 MeV FCC-ee m_W target equivalent at threshold).")


if __name__ == "__main__":
    main()
