#!/usr/bin/env python3
"""Independent (BFS-free) WW Asimov fit — σ(m_W) / σ(Γ_W) / ρ.

Generates the morph template set with :class:`WWGeneratorMoCaNLO` (MoCaNLO
NLO-EW partonic σ̂ ⊗ decoupled beta-scheme ISR — no BFS code or numbers) and
runs the SAME 2-POI cov-lumi Asimov fit as the BFS theory ladder, so the
resulting sensitivities are directly comparable to the BFS Asimov column
(both read the same card scenario, incl. the selection efficiency ε that
inflates the per-point stat by 1/√ε) while being computed from a fully
independent line shape. The placeholder cross-section systematics
(card.XSEC_SYST) are NOT activated here — this stays a clean line-shape fit.

This is the STEP-4 entry point of the independent cross-check: it answers
"what σ(m_W)/σ(Γ_W)/ρ does the independent calculation give?", not
"does it match BFS number-for-number".

Usage:
  dofit_indep.py [--scheme gf] [--lepton-cut 0.95] [--isr-scheme single_conv]
                 [--keep] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import types

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import WWGeneratorMoCaNLO

DEFAULT_OUTDIR = ("/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/"
                  "mocanlo/grid_gen/fit_templates")


def _card_2poi():
    """SimpleNamespace clone of the WW card, reduced to a 2-POI cov-lumi fit
    (no nuisances/constraints; lumi handled in the covariance) — identical
    scenario to the BFS theory-ladder Asimov column."""
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"],
                    "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def _generate_templates(gen, params, outdir):
    os.makedirs(outdir, exist_ok=True)
    mass_scale, width_scale = 80.0, 80.0       # no-op for WW (scaleM/scaleW)
    for tag in params.tags:
        path = gen.do_scan(params.values(tag), mass_scale=mass_scale,
                           width_scale=width_scale, mass_scheme="OS",
                           outdir=outdir)
        print(f"  template {tag:18s} → {os.path.basename(path)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="gf",
                    choices=["gf", "alpha0", "alphaz", "alphamsbar"],
                    help="EW input scheme (the σ̂ grid must exist for it)")
    ap.add_argument("--lepton-cut", type=float, default=None,
                    help="fiducial |cosθ_l|<COS grid (e.g. 0.97); default "
                         "inclusive pure-WW")
    ap.add_argument("--lepton-pt-min", type=float, default=None,
                    help="fiducial p_T,ℓ>PT GeV (production set: 10); pairs with "
                         "--lepton-cut")
    ap.add_argument("--lepton-mll-min", type=float, default=None,
                    help="fiducial m_ℓℓ>MLL GeV on same-flavour OS pairs "
                         "(production set: 10)")
    ap.add_argument("--isr-scheme", default="LO_beta",
                    choices=list(isr_beta.ISR_SCHEMES))
    ap.add_argument("--isr-nll", action=argparse.BooleanOptionalAction, default=True,
                    help="eMELA NLL radiator (α(M_Z)/ALPMZ/DELTA) — the PRODUCTION "
                         "DEFAULT for the independent chain (ripple-free 1-D "
                         "luminosity convolution, see --isr-lumi). --no-isr-nll "
                         "reverts to the analytic LL+exp radiator (--isr-scheme). "
                         "When NLL is active --isr-scheme is ignored.")
    ap.add_argument("--isr-lumi", action=argparse.BooleanOptionalAction, default=None,
                    help="ISR convolution form for --isr-nll: default (auto) uses "
                         "the ripple-free 1-D luminosity self-convolution on the "
                         "production eMELA grid; --no-isr-lumi forces the 2-D einsum "
                         "(direct-eMELA, pre-flip path)")
    ap.add_argument("--mu-F-factor", type=float, default=1.0,
                    help="ISR factorisation scale μ_F / √s")
    ap.add_argument("--br-convention", default="off-shell",
                    choices=["off-shell", "pdg-constant"],
                    help="off-shell = native σ(4f)∝BR²; pdg-constant = divide out "
                         "BR(m_W,Γ_W) (BFS convention) → Γ_W line-shape-only while "
                         "KEEPING the m_W rate handle (no shape-only needed)")
    ap.add_argument("--munuqq", action="store_true",
                    help="restrict the inclusive assembly to the μνqq channel "
                         "scope (lnuqq weight 4 → B≈0.143, the BFS-comparable "
                         "single channel). σ drops to ~0.143× the full pure-WW "
                         "sum, so the per-point statistics scale automatically. "
                         "Templates are written to a separate '_munuqq' dir to "
                         "avoid clobbering the shared full-assembly templates. "
                         "Composes with --br-convention / --flatConst / --shapeOnly.")
    ap.add_argument("--shapeOnly", action="store_true",
                    help="open the lumi prior wide (overall normalisation "
                         "unconstrained) so m_W/Γ_W come from the line-shape "
                         "SHAPE only — removes the off-shell BR²(Γ_W) rate "
                         "handle, giving the apples-to-apples comparison with "
                         "the BFS pdg-constant ρ sign.")
    ap.add_argument("--flatConst", action="store_true",
                    help="add a free-floating, energy-INDEPENDENT additive σ "
                         "term (the SAME constant c at every √s, fully correlated "
                         "across ECM, no prior) and profile it. Reports the "
                         "sensitivity σ(c) and the degradation of σ(m_W)/σ(Γ_W) "
                         "once c is floated. Probes how much of the line-shape "
                         "constraint relies on the absolute σ pedestal.")
    ap.add_argument("--flatConstScale", type=float, default=None,
                    help="absolute σ scale (pb) carried per unit of the cFlat "
                         "parameter; default = mean nominal σ over the scan "
                         "(only affects parameter conditioning, not the physics).")
    ap.add_argument("--statCorrScan", action="store_true",
                    help="after the fit, sweep the point-to-point statistical "
                         "correlation ρ (0→1, stat-only) and save σ(m_W)/σ(Γ_W) "
                         "vs ρ to --outdir")
    ap.add_argument("--scan-min", type=float, default=None,
                    help="override card.SCENARIO scan_min (GeV); default from card")
    ap.add_argument("--scan-max", type=float, default=None,
                    help="override card.SCENARIO scan_max (GeV); default from card")
    ap.add_argument("--scan-step", type=float, default=None,
                    help="override card.SCENARIO scan_step (GeV); default from card")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    cfg = isr_beta.ISRConfig(scheme=args.isr_scheme, mu_F_factor=args.mu_F_factor)
    # μνqq scope: keep the inclusive (lepton_cut=None) pure-WW grids but reweight
    # the assembly to the single lnuqq block (×4 = inclusive μνqq, B≈0.143).
    munuqq_weights = {"lnuqq": 4.0} if args.munuqq else None
    gen = WWGeneratorMoCaNLO(scheme_alpha=args.scheme, lepton_cut=args.lepton_cut,
                             lepton_pt_min=args.lepton_pt_min,
                             lepton_mll_min=args.lepton_mll_min,
                             pure_ww_weights=munuqq_weights,
                             isr_cfg=cfg, br_convention=args.br_convention,
                             isr_nll=args.isr_nll, isr_lumi=args.isr_lumi)
    # μνqq templates (σ ≈ 0.143× full) MUST NOT overwrite the shared full-assembly
    # templates — the filenames encode only (order, m_W, Γ_W), not the channel
    # scope. Route them to a sibling '_munuqq' dir when the user kept the default.
    if args.munuqq and args.outdir == DEFAULT_OUTDIR:
        args.outdir = DEFAULT_OUTDIR + "_munuqq"
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)

    label = ("μνqq scope (B≈0.143, BFS channel)" if args.munuqq
             else "inclusive pure-WW" if args.lepton_cut is None
             else (f"fiducial |cosθ|<{args.lepton_cut}"
                   + (f" pt>{args.lepton_pt_min:g}" if args.lepton_pt_min else "")
                   + (f" mll>{args.lepton_mll_min:g}" if args.lepton_mll_min else "")))
    isr_lbl = (("eMELA-NLL" + (" 2D" if args.isr_lumi is False else " lumi"))
               if args.isr_nll else args.isr_scheme)
    print(f"[indep fit] generator: MoCaNLO {args.scheme}, {label}, "
          f"ISR {isr_lbl} (μ_F/√s={args.mu_F_factor})")
    print(f"[indep fit] generating {len(params.tags)} morph templates → {args.outdir}")
    _generate_templates(gen, params, args.outdir)

    fit = WWFit(c, gen, input_dir=args.outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    scan_min = args.scan_min if args.scan_min is not None else S["scan_min"]
    scan_max = args.scan_max if args.scan_max is not None else S["scan_max"]
    scan_step = args.scan_step if args.scan_step is not None else S["scan_step"]
    fit.init_scenario(scan_min=scan_min, scan_max=scan_max,
                      scan_step=scan_step, total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"])

    if args.shapeOnly:                      # cov mode: open the normalisation
        fit.lumi_uncorr = 0.0
        fit.lumi_corr = 1.0

    fit.fit_parameters()
    res = fit.fit_results(printout=False)   # [mass, width] ufloat, GeV

    mass, width = res[0], res[1]
    rho = float(unc.correlation_matrix([mass, width])[0, 1])
    lumi_lbl = "shape-only (lumi free)" if args.shapeOnly else "cov-lumi"
    print("\n" + "=" * 64)
    print(f"[indep] INDEPENDENT WW Asimov fit (2-POI, {lumi_lbl}, {S['total_lumi']/1e6:.1f} ab⁻¹)")
    print(f"[indep] scan {scan_min}–{scan_max} GeV step {scan_step}")
    print(f"[indep] σ(m_W)  = {mass.s * 1e3:6.2f} MeV")
    print(f"[indep] σ(Γ_W)  = {width.s * 1e3:6.2f} MeV")
    print(f"[indep] ρ(m,Γ)  = {rho:+.3f}")
    print("=" * 64)

    if args.flatConst:
        # Re-fit with one extra free parameter: a fully-correlated, energy-
        # independent additive σ pedestal c (no prior). Reuses the same
        # templates/scenario — only the χ² model and parameter set change.
        fit.add_flat_const(scale=args.flatConstScale)
        fit.fit_parameters()                       # re-init Minuit w/ cFlat
        res2 = fit.fit_results(printout=False)
        i_cf = fit.param_names.index("cFlat")
        mass2, width2, cflat = res2[0], res2[1], res2[i_cf]
        rho2 = float(unc.correlation_matrix([mass2, width2])[0, 1])
        cm = unc.correlation_matrix([mass2, width2, cflat])
        mean_sigma = float(np.mean(fit.pseudo_data_scenario))   # pb
        print("\n" + "=" * 64)
        print("[indep] + FREE energy-INDEPENDENT additive σ pedestal "
              "c (correlated across ECM, no prior)")
        print(f"[indep] mean nominal σ over scan = {mean_sigma:.4f} pb "
              f"(cFlat scale = {fit._flat_const_scale:.4f} pb)")
        print("-" * 64)
        print(f"[indep] sensitivity σ(c)  = {cflat.s * 1e3:7.3f} fb "
              f"= {cflat.s / mean_sigma * 100:6.3f}% of mean σ "
              f"(c = {cflat.n * 1e3:+.3f} fb)")
        print("-" * 64)
        print(f"[indep] σ(m_W) : {mass.s * 1e3:6.2f} →{mass2.s * 1e3:6.2f} MeV"
              f"   (Δ = {(mass2.s - mass.s) * 1e3:+.2f} MeV,"
              f" ×{mass2.s / mass.s:.2f})")
        print(f"[indep] σ(Γ_W) : {width.s * 1e3:6.2f} →{width2.s * 1e3:6.2f} MeV"
              f"   (Δ = {(width2.s - width.s) * 1e3:+.2f} MeV,"
              f" ×{width2.s / width.s:.2f})")
        print(f"[indep] ρ(m,Γ) : {rho:+.3f} →{rho2:+.3f}")
        print(f"[indep] corr(c, m_W) = {cm[2, 0]:+.3f}   "
              f"corr(c, Γ_W) = {cm[2, 1]:+.3f}")
        print("=" * 64)

    if args.statCorrScan:
        from framework.common import scans
        print(f"[indep] ρ-scan (stat-only): σ(m_W)/σ(Γ_W) vs stat correlation → {args.outdir}")
        scans.scan_stat_correlation(fit, outdir=args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
