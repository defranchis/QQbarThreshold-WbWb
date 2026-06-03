"""Independent (BFS-free) WW line-shape generator — MoCaNLO + decoupled ISR.

Drop-in for the BFS ``WWGenerator`` (same ``file_name`` / ``do_scan`` contract,
consumed by ``common.fit_core``).  Pipeline:

  1. MoCaNLO NLO-EW partonic σ̂_Born(√ŝ), σ̂_NLO(√ŝ) per channel × varpoint
     (``partonic_grid.load_grids`` → smooth interpolators).
  2. Per channel: ISR-matched observed line shape σ_obs(√s) =
     ∫∫ D D σ̂_NLO − C₁[σ̂_Born]  (``isr_beta.sigma_observed_matched``).
  3. Assemble the 6 blocks with flavour/colour multiplicities
     (``channels.assemble_total``) → σ_tot(√s) per varpoint.
  4. Quadratic + bilinear morph over the 6 (m_W, Γ_W) varpoints → σ_tot at the
     requested fit point.  Output in **pb** (BFS convention; MoCaNLO is fb ×1e-3).

This shares NO code or numerical input with the BFS-EFT chain: the goal is an
independent σ(m_W)/σ(Γ_W)/ρ, not number-matching.

Output √s outside the σ̂ generation grid [ECM_MIN, ECM_MAX] (156–164 GeV) is set
to 0 — the WW-threshold scan window lives inside it; the fit must not place scan
points above ECM_MAX (no σ̂ there to convolve).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from framework.process.ww.indep import isr_beta
from framework.process.ww.indep import match_bfs
from framework.process.ww.indep.channels import (
    BLOCKS, BLOCKS_BY_KEY, PURE_WW_WEIGHTS,
)
from framework.process.ww.indep.mocanlo_cards import SMInputs
from framework.process.ww.indep.varpoints import (
    MW0, GW0, VARPOINTS, VARPOINTS_BY_KEY,
)
from framework.process.ww.indep.partonic_grid import (
    load_grids, DEFAULT_RESULTS_DIR, ChannelVarGrid,
)
from framework.process.ww.indep import grid as gridmod

FB_TO_PB = 1.0e-3

# Output grid: mirror the BFS WWGenerator (155–170 @0.1 + 240) so the fit reads
# an identical-shape CSV; values are 0 outside the σ̂ coverage.
ECM_FINE_MIN, ECM_FINE_MAX, ECM_FINE_STEP = 155.0, 170.0, 0.1
ECM_LAST = 240.0


def _build_fine_grid() -> np.ndarray:
    n = int(round((ECM_FINE_MAX - ECM_FINE_MIN) / ECM_FINE_STEP)) + 1
    g = ECM_FINE_MIN + ECM_FINE_STEP * np.arange(n)
    return np.concatenate([g, [ECM_LAST]])


@dataclass
class WWGeneratorMoCaNLO:
    """Independent WW template producer (MoCaNLO partonic ⊗ beta-scheme ISR)."""
    results_dir: str = DEFAULT_RESULTS_DIR
    scheme_alpha: str = "gf"
    lepton_cut: float | None = None     # None=inclusive(pure-WW); 0.95=fiducial
    isr_cfg: isr_beta.ISRConfig = field(default_factory=isr_beta.ISRConfig)
    order: int = 1                      # NLO-EW → filename tag "1"
    smooth: float | None = None         # σ̂ spline smoothing factor (None=auto)
    br_convention: str = "off-shell"    # "off-shell" (native σ(4f)∝BR²) or
                                        # "pdg-constant" (divide out BR(m_W,Γ_W);
                                        # mirrors BFS — Γ_W becomes line-shape-only
                                        # WITHOUT discarding the m_W rate handle)
    # --- EXPLORATORY BFS matching (NOT production; opt-in) ----------------
    # Add the BFS pieces genuinely missing from MoCaNLO's complete NLO-EW:
    # δ_NNLO threshold block (production K-factor on the Born) + δ_QCD on
    # hadronic decay.  Strictly-NLO BFS pieces are NOT added (already in NLO).
    # See framework.process.ww.indep.match_bfs.
    match_bfs: bool = False
    match_bfs_nnlo: bool = True         # add BFS NNLO threshold (Coulomb α²/v² + hard/decay)
    match_bfs_dqcd: bool = True         # add δ_QCD to hadronic-decay channels
    alpha_s: float = 0.1199             # α_s(M_W) for δ_QCD (BFS reference)
    sm: SMInputs = field(default_factory=SMInputs)   # mt/MH/MZ for δ_NNLO
    # EXPLORATORY: upgrade the ISR convolution from analytic LL+exp to eMELA NLL
    # (α(M_Z)/ALPMZ/DELTA — the BFS production NLL convention).  Independent of
    # match_bfs; the O(α) matching subtraction stays analytic-LL (see isr_beta).
    isr_nll: bool = False
    _grids: dict = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _br_factor(self, mW: float, gW: float) -> float:
        """Multiplicative factor converting the native off-shell σ(4f) ∝ BR²
        to the BFS pdg-constant convention (BR held fixed at the reference
        m_W, Γ_W).  BR ∝ Γ_partial(m_W)/Γ_W with Γ_partial ∝ m_W³, so
        BR² ∝ m_W⁶/Γ_W²; dividing it out is universal (per-channel partial-width
        constants cancel).  =1 at the reference point and for 'off-shell'."""
        if self.br_convention == "off-shell":
            return 1.0
        if self.br_convention == "pdg-constant":
            return (gW / GW0) ** 2 * (MW0 / mW) ** 6
        raise ValueError(f"unknown br_convention {self.br_convention!r}")

    # ------------------------------------------------------------------
    def _load(self):
        if self._grids is None:
            self._grids = load_grids(self.results_dir, self.scheme_alpha,
                                     self.lepton_cut)
            if not self._grids:
                raise FileNotFoundError(
                    f"no σ̂ grids under {self.results_dir} (scheme {self.scheme_alpha})")
        return self._grids

    def _weights(self) -> dict[str, float]:
        """Channel→multiplicity map for the active definition.

        Inclusive 'pure-WW' (lepton_cut is None): the 3 stable channels
        12·lnuqq + 4·qqqq + 9·mutau (channels.PURE_WW_WEIGHTS).  Fiducial
        (lepton_cut set): the original 6 blocks with their multiplicities.
        """
        if self.lepton_cut is None:
            return dict(PURE_WW_WEIGHTS)
        return {b.key: b.weight for b in BLOCKS}

    def _isr_cfg(self) -> "isr_beta.ISRConfig":
        """Resolve the ISR config; ``isr_nll`` upgrades the analytic LL+exp
        radiator to eMELA NLL in the BFS production convention (α(M_Z)/ALPMZ/
        DELTA), preserving the user's μ_F / x_min / n_quad knobs."""
        if self.isr_nll and not self.isr_cfg.nll:
            return isr_beta.ISRConfig(
                nll=True, alpha=isr_beta.ALPHA_MZ, ew_scheme="alphaz",
                mu_F_factor=self.isr_cfg.mu_F_factor, mu_F_abs=self.isr_cfg.mu_F_abs,
                m_e=self.isr_cfg.m_e, x_min=self.isr_cfg.x_min,
                n_quad=self.isr_cfg.n_quad,
                emela_fac_scheme="DELTA", emela_ren_scheme="ALPMZ")
        return self.isr_cfg

    def _varpoint_lineshape(self, varpoint: str, sqrt_s: np.ndarray) -> np.ndarray:
        """Assembled σ_tot(√s) [pb] for one varpoint (cached)."""
        ck = (varpoint, id(sqrt_s))
        if ck in self._cache:
            return self._cache[ck]
        grids = self._load()
        weights = self._weights()

        # Exploratory BFS matching: δ_NNLO(√ŝ) production K-factor at THIS
        # varpoint's (m_W, Γ_W), added on MoCaNLO's Born; the morph then carries
        # its m_W/Γ_W dependence.  δ_QCD is a per-channel decay-side factor.
        dnnlo_fn = None
        if self.match_bfs and self.match_bfs_nnlo:
            vp = VARPOINTS_BY_KEY[varpoint]
            dnnlo_fn = match_bfs.delta_nnlo_interp(
                vp.mW, vp.gW, mt=self.sm.mt, MH=self.sm.mH, MZ=self.sm.mZ)

        cfg = self._isr_cfg()
        sigma_tot = np.zeros_like(sqrt_s, dtype=float)
        for key, w in weights.items():
            g: ChannelVarGrid = grids[(key, varpoint)]
            born = g.born_fn(self.smooth)
            nlo = g.nlo_fn(self.smooth)
            if dnnlo_fn is not None:
                # σ̂_comb = σ̂_NLO + δ_NNLO·σ̂_Born.  The O(α) ISR matching
                # subtraction (3rd arg) stays MoCaNLO's Born ONLY — the BFS
                # NNLO term is ISR-naked, MoCaNLO's σ̂_NLO carries the O(α) ISR.
                def nlo_eff(sh, _nlo=nlo, _born=born, _d=dnnlo_fn):
                    return _nlo(sh) + _d(sh) * _born(sh)
            else:
                nlo_eff = nlo
            obs = isr_beta.sigma_observed_matched(sqrt_s, nlo_eff, born, cfg)
            if self.match_bfs and self.match_bfs_dqcd:
                obs = obs * match_bfs.delta_qcd_channel_factor(
                    BLOCKS_BY_KEY[key].outgoing, self.alpha_s)
            sigma_tot = sigma_tot + w * obs
        out = sigma_tot * FB_TO_PB           # fb → pb (BFS convention)
        self._cache[ck] = out
        return out

    def _fit_morph(self, sqrt_s: np.ndarray):
        """Least-squares quadratic+bilinear morph coefficients over all varpoints.

        Fits, per √s,  σ(Δm,Δw) = c0 + c1·Δm + c2·Δw + c3·Δm² + c4·Δw² + c5·ΔmΔw
        (Δ in MeV) to the assembled line shapes at every varpoint present in the
        grid.  Over-determined (~20 points, 6 coeffs) ⇒ the per-point MC noise is
        averaged down and the wide lever arm pins the slopes.  Returns coeffs of
        shape (6, len(sqrt_s)).  Cached per sqrt_s identity.
        """
        ck = ("coeffs", id(sqrt_s))
        if ck in self._cache:
            return self._cache[ck]
        grids = self._load()
        channels = list(self._weights())
        rows, rhs = [], []
        for v in VARPOINTS:
            if all((ch, v.key) in grids for ch in channels):
                dm, dw = v.dmW_MeV, v.dgW_MeV
                rows.append([1.0, dm, dw, dm * dm, dw * dw, dm * dw])
                rhs.append(self._varpoint_lineshape(v.key, sqrt_s))
        A = np.asarray(rows)                       # (n_vp, 6)
        Y = np.asarray(rhs)                        # (n_vp, n_s)
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)   # (6, n_s)
        self._cache[ck] = coeffs
        return coeffs

    def _morphed(self, mW: float, gW: float, sqrt_s: np.ndarray) -> np.ndarray:
        """σ_tot(√s; m_W, Γ_W) [pb] via the fitted quad+bilinear morph."""
        coeffs = self._fit_morph(sqrt_s)
        dm = (mW - MW0) * 1e3      # MeV
        dw = (gW - GW0) * 1e3
        basis = np.array([1.0, dm, dw, dm * dm, dw * dw, dm * dw])
        return basis @ coeffs

    # ------------------------------------------------------------------
    # WWGenerator contract
    # ------------------------------------------------------------------
    def file_tag(self, values: dict) -> str:
        mW = float(values["mass"]); gW = float(values["width"])
        return f"mass{mW:.3f}_width{gW:.3f}"

    def file_name(self, values: dict, *, mass_scale=None, width_scale=None,
                  mass_scheme: str = "OS", indir: str = ".") -> str:
        # "m" suffix on the order tag keeps BFS-matched templates from
        # colliding with the pure-MoCaNLO ones (exploratory).
        tag = f"{self.order}m" if self.match_bfs else f"{self.order}"
        return os.path.join(indir, f"WW_{tag}_{self.file_tag(values)}.txt")

    def do_scan(self, values: dict, *, mass_scale: float = 1.0,
                width_scale: float = 1.0, mass_scheme: str = "OS",
                outdir: str = "output_xsec/ww_indep/nominal",
                ecm_shift_MeV: float = 0.0) -> str:
        mW = float(values["mass"]); gW = float(values["width"])
        ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3

        # only convolve inside σ̂ coverage; 0 elsewhere
        inside = (ecm_grid >= gridmod.ECM_MIN) & (ecm_grid <= gridmod.ECM_MAX)
        sigma = np.zeros_like(ecm_grid)
        if inside.any():
            sigma[inside] = self._morphed(mW, gW, ecm_grid[inside])
        sigma *= self._br_factor(mW, gW)   # off-shell→pdg-constant if requested

        os.makedirs(outdir, exist_ok=True)
        path = self.file_name(values, indir=outdir)
        with open(path, "w") as fh:
            tag = ("MoCaNLO+BFS-matched (EXPLORATORY)" if self.match_bfs
                   else "MoCaNLO (independent, BFS-free)")
            fh.write(f"# generator: WWGeneratorMoCaNLO — {tag}\n")
            fh.write(f"# scheme_alpha: {self.scheme_alpha}\n")
            if self.match_bfs:
                fh.write(f"# match_bfs: nnlo={self.match_bfs_nnlo} "
                         f"dqcd={self.match_bfs_dqcd} alpha_s={self.alpha_s}\n")
            _cfg = self._isr_cfg()
            fh.write(f"# isr: {'eMELA-NLL' if _cfg.nll else _cfg.scheme}  "
                     f"mu_F_factor: {_cfg.mu_F_factor}\n")
            fh.write(f"# mass: {mW:.4f}  width: {gW:.4f}  units: pb\n")
            for ecm, sig in zip(ecm_grid, sigma):
                fh.write(f"{ecm:.4f}, {sig:.8f}\n")
        return path
