#!/usr/bin/env python3
"""Task 2: 240 GeV anchor (ZH-run 10.8/ab) impact on the WW threshold fit.

Adds a high-statistics sigma_WW(240) rate anchor (full MoCaNLO NLO-EW, the ZH-run
luminosity 10.8/ab) to the 2-POI cov-lumi threshold fit and reports the change in
sigma(m_W) / sigma(Gamma_W) / rho.  The 240 point pins the absolute normalisation,
decorrelating the steeply-rising threshold rate from m_W.

Morph at 240: sigma_WW(240) is weakly m_W-dependent; its residual sensitivity
comes from the ISR radiative tail reaching the threshold region.  So the 6
varpoints {nominal, mp50/mm50, wp50/wm50, xp50p50} use the REAL threshold grids
(where the m_W shape lives) and NOMINAL sigma_hat above threshold (m_W-flat
there), built into one quad+bilinear morph spanning [threshold, 240].  Fit A
(no anchor) and Fit B (with anchor) use the SAME morph -> their difference
isolates the anchor effect (the absolute numbers use the coarser 6-varpoint
morph, not the headline 20-varpoint one).
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import types

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import uncertainties as unc

from cards import ww_default as card
from framework.common.parameters import Parameters
from framework.process.ww.fit import WWFit
from framework.process.ww.indep import isr_beta
from framework.process.ww.indep.generator_mocanlo import (
    WWGeneratorMoCaNLO, _build_fine_grid)
from framework.process.ww.indep.varpoints import MW0, GW0, VARPOINTS_BY_KEY

PROD = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results"
HIEN = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_hienergy"
ANCHOR = "/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_anchor"
TEMPL = "/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/mocanlo/grid_gen/fit_templates_anchor"
CHANS = ("lnuqq", "qqqq", "mutau")
ANCHOR_VPS = ("mp50", "mm50", "wp50", "wm50", "xp50p50")
LAST_ECM = 240.0


def _link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


def _synthetic(src_nominal_csv, varpoint, dst, scale):
    """Write a high-sqrt(s) varpoint CSV from the nominal one, relabelled and
    scaled by the off-shell decay factor BR^2 = (mW/MW0)^6 (GW0/gW)^2.

    Above threshold the PRODUCTION sigma_WW is ~m_W/Gamma_W-flat, but sigma_hat(4f)
    still carries the decay branching BR^2 ∝ m_W^6/Gamma_W^2 at every energy
    (verified against the real 240 grids to <0.2%).  Copying the nominal sigma_hat
    unscaled (the earlier bug) dropped it, leaving a spurious Gamma_W rate handle
    in the pdg-constant convention where _br_factor would otherwise cancel it."""
    with open(src_nominal_csv) as f:
        rows = list(csv.reader(f))
    rows[1][1] = varpoint                       # column 1 = varpoint
    for i in range(7, 17):                       # sigma_*/err_* columns
        rows[1][i] = repr(float(rows[1][i]) * scale)
    with open(dst, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def build_anchor_dir():
    os.makedirs(ANCHOR, exist_ok=True)
    for f in glob.glob(os.path.join(ANCHOR, "*.csv")):
        os.unlink(f)
    n_link = n_syn = 0
    # nominal: real threshold + real high-E
    for src in (PROD, HIEN):
        for ch in CHANS:
            for p in glob.glob(os.path.join(src, f"{ch}_nominal_ecm*_gf.csv")):
                _link(p, os.path.join(ANCHOR, os.path.basename(p))); n_link += 1
    # anchor varpoints: real threshold grids (the m_W threshold SHAPE lives here)
    # + real high-E grids where generated (the 240 batch points) + BR^2-scaled
    # synthetic for the remaining high-E nominal points (the gap-fill).
    for vp in ANCHOR_VPS:
        v = VARPOINTS_BY_KEY[vp]
        br2 = (v.mW / MW0) ** 6 * (GW0 / v.gW) ** 2
        for ch in CHANS:
            real_hi = set()
            for src in (PROD, HIEN):
                for p in glob.glob(os.path.join(src, f"{ch}_{vp}_ecm*_gf.csv")):
                    _link(p, os.path.join(ANCHOR, os.path.basename(p))); n_link += 1
                    real_hi.add(os.path.basename(p).split("_ecm")[1].split("_gf")[0])
            for p in glob.glob(os.path.join(HIEN, f"{ch}_nominal_ecm*_gf.csv")):
                e = os.path.basename(p).split("_ecm")[1].split("_gf")[0]
                if e in real_hi:                 # real varpoint grid exists (e.g. 240)
                    continue
                dst = os.path.join(ANCHOR, f"{ch}_{vp}_ecm{e}_gf.csv")
                _synthetic(p, vp, dst, br2); n_syn += 1
    print(f"[anchor] dir {ANCHOR}: {n_link} linked + {n_syn} BR^2-scaled synthetic CSVs")


class AnchorGen(WWGeneratorMoCaNLO):
    """Generator that fills the LAST_ECM (240) overflow point with a REAL
    sigma_WW(240) instead of the production zero.

    _build_fine_grid() is already [uniform body] + [240 overflow]; production
    do_scan zeroes everything above 164 (no sigma_hat there).  Here the combined
    grid (results_anchor) extends sigma_hat to 365, so we compute the morph up to
    370 GeV -> the 240 overflow carries the real anchor cross section.  Below
    156 GeV stays zero (below the sigma_hat grid)."""
    def do_scan(self, values, *, mass_scale=1.0, width_scale=1.0,
                mass_scheme="OS", outdir=".", ecm_shift_MeV=0.0):
        mW = float(values["mass"]); gW = float(values["width"])
        ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3   # body + 240 overflow
        inside = (ecm_grid >= 156.0) & (ecm_grid <= 370.0)
        sigma = np.zeros_like(ecm_grid)
        sigma[inside] = self._morphed(mW, gW, ecm_grid[inside])
        sigma *= self._br_factor(mW, gW)
        os.makedirs(outdir, exist_ok=True)
        path = self.file_name(values, indir=outdir)
        with open(path, "w") as fh:
            fh.write("# generator: AnchorGen (MoCaNLO indep + 240 anchor)\n")
            fh.write(f"# mass: {mW:.4f}  width: {gW:.4f}  units: pb\n")
            for ecm, sig in zip(ecm_grid, sigma):
                fh.write(f"{ecm:.4f}, {sig:.8f}\n")
        return path


def _card_2poi():
    c = types.SimpleNamespace(**{k: getattr(card, k) for k in dir(card)
                                 if not k.startswith("__")})
    c.PARAMETERS = {"mass": card.PARAMETERS["mass"], "width": card.PARAMETERS["width"]}
    c.SYSTEMATICS = {}
    c.LUMI_MODE = "cov"
    c.CROSS_TERMS = [("mass", "width")]
    return c


def run_fit(add_anchor: bool, br_convention: str):
    c = _card_2poi()
    params = Parameters(c.PARAMETERS, cross_terms=c.CROSS_TERMS)
    gen = AnchorGen(results_dir=ANCHOR, scheme_alpha="gf", lepton_cut=None,
                    isr_cfg=isr_beta.ISRConfig(scheme="LO_beta"),
                    br_convention=br_convention)
    outdir = os.path.join(TEMPL, br_convention)
    for tag in params.tags:
        gen.do_scan(params.values(tag), outdir=outdir)
    fit = WWFit(c, gen, input_dir=outdir, asimov=True, mass_scheme="OS")
    S = card.SCENARIO
    fit.init_scenario(scan_min=S["scan_min"], scan_max=S["scan_max"],
                      scan_step=S["scan_step"], total_lumi=S["total_lumi"],
                      last_lumi=S["last_lumi"], add_last_ecm=add_anchor)
    fit.fit_parameters()
    res = fit.fit_results(printout=False)
    m, w = res[0], res[1]
    rho = float(unc.correlation_matrix([m, w])[0, 1])
    return m.s * 1e3, w.s * 1e3, rho


def main():
    build_anchor_dir()
    print(f"\n240 GeV anchor (ZH-run {card.SCENARIO['last_lumi']/1e6:.1f}/ab) "
          f"on the 2-POI cov-lumi threshold fit")
    print(f"{'convention':16s} {'anchor':8s} {'s(mW)':>7s} {'s(GW)':>7s} {'rho':>7s}")
    print("-" * 52)
    for br in ("pdg-constant", "off-shell"):
        for add in (False, True):
            sm, sw, rho = run_fit(add, br)
            print(f"{br:16s} {'+240' if add else 'base':8s} "
                  f"{sm:7.2f} {sw:7.2f} {rho:+7.3f}")


if __name__ == "__main__":
    main()
