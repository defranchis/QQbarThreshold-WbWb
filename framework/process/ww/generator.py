"""WW threshold-scan template generator (BFS-EFT chain).

Computes σ(e+e- → μν qq̄, inclusive) on a fine ECM grid spanning the WW
threshold scan and writes a two-column CSV (``ecm, xsec``) consumed by
``common.fit_core.FitCore``.

Physics chain — defaults are the project's "best calculation":

  • BFS-EFT N^(3/2)LO Born expansion (arXiv:0707.0773 eq. 17+33+37+39
    incl. h4-h7 single-resonant); finite-Γ_W complex-velocity smoothing.
  • Whizard 4f Born anchor f(δ, Γ_W) — BFS sec. 6.2 prescription
    (apply_whizard_anchor=True).
  • BFS NLO loops: hard+soft+collinear (eq. 56), NLO Coulomb (eq. 62),
    EW decay (eq. 60) — include_NLO_hard_decay=True.
  • BFS dominant NNLO (arXiv:0807.0102 eq. 49): C×[S+H] + NLO-C + C×decay
    + C×res + C3 — include_BFS_NNLO=True. ~+1 fb at peak (~3 MeV m_W
    impact per BFS sec. 6.4 after ISR convolution).
  • δ_QCD multiplier 1 + α_s/π + 1.409(α_s/π)² — apply_delta_QCD=True.
  • Coulomb K-factor resummation (Fadin-Khoze-Martin + Bardin-Riemann α²).
  • LL+exp ISR (LEP2 YR BETA scheme); single-conv default, 2-leg available.
  • RACOONWW CC03 spline above √s = 170 GeV (calibration region only;
    threshold-scan grid 157-163 GeV stays in the BFS-EFT region).

Validation: scripts/validate_bfs_nlo.py. Closure to BFS Tables 1+2 at
4-5 digits (Born); to BFS Table 3+4 at 0.4-1.0 % (NLO + ISR, residual is
NLL beyond LL+exp — BFS's own 31 MeV systematic). NNLO pieces match
arXiv:0807.0102 Table 1 to 4 digits via
scripts/investigations/bfs_nnlo/check_closed_form_pieces.py.

Remaining open work: NLL ISR (eMELA / Skrzypek-Jadach) — see
project-followup-nll-isr-plan memory.

The ``mass_scale`` / ``width_scale`` parameters are recorded in the
filename but currently have no effect on the cross section (no
renormalisation scale to vary at this order).
"""

from __future__ import annotations

import os

import numpy as np

from framework.process.ww.xsec_calculator.eft_xsec import (
    ALPHA_S_MW_DEFAULT, M_T_DEFAULT, M_H_DEFAULT, M_Z,
    BFSCorrections,
)
from framework.process.ww.xsec_calculator.isr import sigma_observed_munuqq
from framework.process.ww.template_metadata import compose_header


# ---------------------------------------------------------------------------
# Card → chain-kwargs helpers (single source of truth)
# ---------------------------------------------------------------------------

def _derive_order(card) -> int:
    """Informational order tag (NNLO=2 / NLO=1 / LO=0) auto-derived from
    NLO_CONFIG flags. Used for the ``WW_<order>_...`` filename suffix and
    the ``describe()`` line — never to gate any physics."""
    nlo = getattr(card, "NLO_CONFIG", {})
    if nlo.get("include_BFS_NNLO", False):
        return 2
    if nlo.get("include_NLO_hard_decay", False):
        return 1
    return 0


def partonic_kwargs_from_card(card) -> dict:
    """Card NLO_CONFIG + PARAM_INPUTS → kwargs for ``sigma_partonic_munuqq``.

    All chain knobs that drive the *partonic* (pre-ISR) σ. Used by the plot
    script, WWGenerator.from_card, validation scripts — anywhere a card-aware
    call to ``sigma_partonic_munuqq`` is needed.
    """
    nlo = getattr(card, "NLO_CONFIG", {})
    theory = getattr(card, "PARAM_INPUTS", {})
    return dict(
        channel=str(nlo.get("channel", "inclusive")),
        include_coulomb=bool(nlo.get("include_coulomb", True)),
        br_convention=str(nlo.get("br_convention", "pdg-constant")),
        include_NLO_hard_decay=bool(nlo.get("include_NLO_hard_decay", True)),
        include_BFS_NNLO=bool(nlo.get("include_BFS_NNLO", True)),
        apply_delta_QCD=bool(nlo.get("apply_delta_QCD", True)),
        alpha_s=float(theory.get("alpha_s_MW", ALPHA_S_MW_DEFAULT)),
        apply_whizard_anchor=bool(nlo.get("apply_whizard_anchor", True)),
        whizard_anchor_source=str(nlo.get("whizard_anchor_source", "grid")),
        coulomb_kc_safe=bool(nlo.get("coulomb_kc_safe", False)),
        decay_uses_full_born=bool(nlo.get("decay_uses_full_born", True)),
        m_t=float(theory.get("m_t", M_T_DEFAULT)),
        M_H=float(theory.get("M_H", M_H_DEFAULT)),
        MZ=float(theory.get("M_Z", M_Z)),
        alpha_em=theory.get("alpha_em", None),
    )


def chain_summary_latex(kwargs: dict) -> str:
    """LaTeX-friendly summary of the chain configuration, built **dynamically**
    from a kwargs dict matching the signature of ``sigma_observed_munuqq`` /
    ``sigma_partonic_munuqq``. Used by the diagnostic plots so the chain
    label always reflects the actual flags driving the calculation.

    ISR fragment is included iff ``isr_scheme`` is present in ``kwargs`` —
    i.e. observed-level kwargs produce ``... + LL+exp ISR (single-conv)``
    while partonic-level kwargs stop at the partonic chain.

    Order of pieces matches the physics build-up: Born → NLO loops →
    NNLO → δ_QCD → anchor → K_C → ISR.
    """
    parts = [r"BFS N$^{3/2}$LO"]
    if kwargs.get("include_NLO_hard_decay", False):
        nlo_label = "NLO"
        if kwargs.get("coulomb_kc_safe", False):
            nlo_label = r"NLO ($K_{\rm C}$-safe Coul.)"
        if not kwargs.get("decay_uses_full_born", True):
            nlo_label = nlo_label + r" ($\delta_{\rm dec}\!\times\!\sigma^{(0)}$)"
        parts.append(nlo_label)
    if kwargs.get("include_BFS_NNLO", False):
        parts.append("NNLO")
    if kwargs.get("apply_delta_QCD", False):
        # For pdg-constant the QCD correction lives in the BR (α_s-aware
        # BR_PDG × δ_QCD(α_s)/δ_QCD(α_s^ref)); for bfs-eft it multiplies σ.
        if kwargs.get("br_convention") == "pdg-constant":
            parts.append(r"$\delta_{\rm QCD}^{\rm (BR)}$")
        else:
            parts.append(r"$\delta_{\rm QCD}$")
    if kwargs.get("apply_whizard_anchor", False):
        parts.append("anchor")
    if kwargs.get("include_coulomb", False):
        parts.append(r"$K_{\rm C}$")
    if "isr_scheme" in kwargs:
        scheme = str(kwargs["isr_scheme"]).replace("_", "-")
        if kwargs.get("isr_nll", False):
            isr_label = "NLL ISR (eMELA)"
        elif kwargs.get("isr_emela_ll", False):
            isr_label = "LL ISR (eMELA DGLAP)"
        else:
            isr_label = "LL+exp ISR"
        parts.append(rf"{isr_label} ({scheme})")
    return " + ".join(parts)


def observed_kwargs_from_card(card) -> dict:
    """Card NLO_CONFIG + PARAM_INPUTS → kwargs for ``sigma_observed_munuqq``
    (partonic kwargs + ISR-only physics knobs). Numerical quadrature controls
    (n_quad, z_min) are not card-exposed; sensible defaults live in isr.py.
    """
    nlo = getattr(card, "NLO_CONFIG", {})
    theory = getattr(card, "PARAM_INPUTS", {})
    return {
        **partonic_kwargs_from_card(card),
        "isr_scheme":            str(nlo.get("isr_scheme", "single_conv")),
        "isr_nll":               bool(nlo.get("isr_nll", False)),
        "isr_emela_ll":          bool(nlo.get("isr_emela_ll", False)),
        "isr_emela_pert_order":  str(nlo.get("isr_emela_pert_order", "NLL")),
        "isr_emela_fac_scheme":  str(nlo.get("isr_emela_fac_scheme", "DELTA")),
        "isr_emela_ren_scheme":  str(nlo.get("isr_emela_ren_scheme", "ALGMU")),
        "alpha_em_isr":          theory.get("alpha_em_isr", None),
    }


# ---------------------------------------------------------------------------
# Fine ECM grid for the template (mirrors the WbWb convention)
# ---------------------------------------------------------------------------
# Wider than the analysis scan window so BES convolution has clean margin.
ECM_FINE_MIN  = 155.0
ECM_FINE_MAX  = 170.0
ECM_FINE_STEP = 0.1
ECM_LAST      = 240.0


def _build_fine_grid() -> np.ndarray:
    """Fine ECM grid: 155.0–170.0 step 0.1 GeV, plus ``ECM_LAST`` appended."""
    n = int(round((ECM_FINE_MAX - ECM_FINE_MIN) / ECM_FINE_STEP)) + 1
    grid = ECM_FINE_MIN + ECM_FINE_STEP * np.arange(n)
    return np.concatenate([grid, [ECM_LAST]])


class WWGenerator:
    """LO + Coulomb + LL-ISR template producer for the WW threshold fit."""

    def __init__(self, *, order: int = 2, channel: str = "inclusive",
                 include_coulomb: bool = True, bfs: BFSCorrections | None = None,
                 # Defaults below are the project's "best calculation"; see
                 # cards/ww_default.py + sigma_observed_munuqq in isr.py for
                 # the per-knob rationale.
                 include_NLO_hard_decay: bool = True,
                 include_BFS_NNLO: bool = True,
                 apply_delta_QCD: bool = True,
                 br_convention: str = "pdg-constant",
                 alpha_s: float = ALPHA_S_MW_DEFAULT,
                 apply_whizard_anchor: bool = True,
                 whizard_anchor_source: str = "grid",
                 isr_scheme: str = "single_conv",
                 isr_nll: bool = False,
                 isr_emela_ll: bool = False,
                 isr_emela_pert_order: str = "NLL",
                 isr_emela_fac_scheme: str = "DELTA",
                 isr_emela_ren_scheme: str = "ALGMU",
                 alpha_em_isr: float | None = None,
                 coulomb_kc_safe: bool = False,
                 decay_uses_full_born: bool = True,
                 # Theory inputs entering the BFS matching coefficients
                 # (c_p,LR / c_d,l / c_d,h analytic in m_t, M_H, M_Z) and the
                 # EW couplings (sin²θ_W = 1 − (m_W/M_Z)² via OS scheme).
                 m_t: float = M_T_DEFAULT, M_H: float = M_H_DEFAULT,
                 MZ: float = M_Z,
                 # σ-chain α_em override (None → α_Gμ derived from m_W, M_Z).
                 alpha_em: float | None = None,
                 # Steering card reference — ``template_fingerprint`` reads
                 # PDG BRs + PARAMETERS variation magnitudes from it.
                 card=None):
        self.order = order              # informational; recorded in filename
        self.channel = channel
        self.include_coulomb = include_coulomb
        self.bfs = bfs if bfs is not None else BFSCorrections()
        self.include_NLO_hard_decay = include_NLO_hard_decay
        self.include_BFS_NNLO = include_BFS_NNLO
        self.apply_delta_QCD = apply_delta_QCD
        self.br_convention = br_convention
        self.alpha_s = alpha_s
        self.apply_whizard_anchor = apply_whizard_anchor
        self.whizard_anchor_source = whizard_anchor_source
        self.isr_scheme = isr_scheme
        self.isr_nll = isr_nll
        self.isr_emela_ll = isr_emela_ll
        self.isr_emela_pert_order = isr_emela_pert_order
        self.isr_emela_fac_scheme = isr_emela_fac_scheme
        self.isr_emela_ren_scheme = isr_emela_ren_scheme
        self.alpha_em_isr = alpha_em_isr
        self.coulomb_kc_safe = coulomb_kc_safe
        self.decay_uses_full_born = decay_uses_full_born
        self.m_t = m_t
        self.M_H = M_H
        self.MZ = MZ
        self.alpha_em = alpha_em
        self.card = card

    @classmethod
    def from_card(cls, card, *, bfs: BFSCorrections | None = None):
        """Build a generator from a steering card's PARAM_INPUTS + NLO_CONFIG.

        Single source of truth: any card edit propagates to every entry
        point (compute_xsec_ww, doFit_ww, scripts/fit_2107_*, etc.) that
        uses this factory.

        ``order`` is auto-derived from the chain knobs (NNLO if BFS NNLO is
        on; NLO if hard+decay loops are on; LO otherwise) — it is purely
        informational, used for the ``WW_<order>_...`` filename tag and
        the ``describe()`` line.

        If ``bfs`` is None and the card sets the diagnostic flag
        NLO_CONFIG["diagnostic_bfs_coulomb_nlo"], a
        ``BFSCorrections(enabled_coulomb_NLO=True)`` is constructed
        automatically. CLI overrides (e.g. ``--diagnostic-bfs-coulomb-nlo``)
        should pass an explicit ``bfs`` to take precedence.
        """
        nlo_cfg = getattr(card, "NLO_CONFIG", {})
        if bfs is None and bool(nlo_cfg.get("diagnostic_bfs_coulomb_nlo", False)):
            bfs = BFSCorrections(enabled_coulomb_NLO=True)
        return cls(
            order=int(getattr(card, "ORDER", _derive_order(card))),
            bfs=bfs,
            card=card,
            **observed_kwargs_from_card(card),
        )

    def _chain_kwargs(self) -> dict:
        return {
            "include_NLO_hard_decay": self.include_NLO_hard_decay,
            "include_BFS_NNLO":       self.include_BFS_NNLO,
            "apply_delta_QCD":        self.apply_delta_QCD,
            "br_convention":          self.br_convention,
            "apply_whizard_anchor":   self.apply_whizard_anchor,
            "whizard_anchor_source":  self.whizard_anchor_source,
            "include_coulomb":        self.include_coulomb,
            "coulomb_kc_safe":        self.coulomb_kc_safe,
            "decay_uses_full_born":   self.decay_uses_full_born,
            "isr_scheme":             self.isr_scheme,
            "isr_nll":                self.isr_nll,
            "isr_emela_ll":           self.isr_emela_ll,
        }

    def chain_label(self) -> str:
        """LaTeX summary of *this generator's* chain configuration — the
        full, traceability-grade string stamped into the template header
        by ``do_scan``. Never abbreviated; the long form is the integrity
        check that ties every plot back to the templates it consumes."""
        return chain_summary_latex(self._chain_kwargs())

    def template_fingerprint(self) -> dict:
        """Stringified snapshot of every input that affects σ_template.
        Stamped into the template header by ``do_scan``; the fit's
        :func:`_check_template_freshness` reads it back and refuses to
        run if any field disagrees with the live card. This is the
        integrity check that catches "I edited the card but forgot to
        regenerate templates" mistakes."""
        fp = {
            "chain":           self.chain_label(),
            "channel":         self.channel,
            "br_convention":   self.br_convention,
            "alpha_s":         f"{self.alpha_s:.5f}",
            "alpha_em":        ("auto" if self.alpha_em is None
                                else f"{self.alpha_em:.6e}"),
            "alpha_em_isr":    ("auto" if self.alpha_em_isr is None
                                else f"{self.alpha_em_isr:.6e}"),
            "isr_scheme":      self.isr_scheme,
            "isr_nll":         str(bool(self.isr_nll)),
            "isr_emela_ll":    str(bool(self.isr_emela_ll)),
            "isr_emela_pert_order": self.isr_emela_pert_order,
            "isr_emela_fac_scheme": self.isr_emela_fac_scheme,
            "isr_emela_ren_scheme": self.isr_emela_ren_scheme,
            "m_t":             f"{self.m_t:.3f}",
            "M_H":             f"{self.M_H:.3f}",
            "M_Z":             f"{self.MZ:.4f}",
            "anchor_source":   self.whizard_anchor_source,
            "coulomb_kc_safe": str(bool(self.coulomb_kc_safe)),
            "decay_uses_full_born": str(bool(self.decay_uses_full_born)),
            # Where δ_QCD enters the chain. "in_br" = α_s-aware BR_PDG
            # × δ_QCD(α_s)/δ_QCD(α_s_ref) (pdg-constant); "on_sigma" =
            # multiplicative on σ per BFS §6.1 (bfs-eft). Old templates
            # lack this field → freshness check fires on first read.
            "delta_qcd_routing":
                ("in_br" if (self.apply_delta_QCD and
                             self.br_convention == "pdg-constant")
                 else "on_sigma" if self.apply_delta_QCD
                 else "off"),
        }
        if self.card is not None:
            # PDG branching-ratio primitives (PDG-constant chain uses BR_INCLUSIVE_MUNUQQ;
            # the Azzurri overlay divides by 2·BR_W_MUNU·BR_W_HAD).
            for name in ("BR_W_MUNU", "BR_W_HAD", "BR_W_UD"):
                if hasattr(self.card, name):
                    fp[name] = f"{getattr(self.card, name):.5f}"
            # PARAMETERS variation magnitudes — change the variation
            # template's δ and the fit's morphing inputs change.
            for poi, spec in getattr(self.card, "PARAMETERS", {}).items():
                fp[f"{poi}_variation"] = f"{float(spec['variation']):.6f}"
        return fp

    def describe(self) -> str:
        """One-line summary of the chain configuration (for log lines)."""
        a_isr = (f"{self.alpha_em_isr:.6e}" if self.alpha_em_isr is not None
                 else "α_Gμ(M_W_BFS_REF) [BFS default]")
        a_chain = (f"{self.alpha_em:.6e}" if self.alpha_em is not None
                   else "α_Gμ(m_W) [derived]")
        return (
            f"WWGenerator order={self.order} channel={self.channel}  "
            f"BR={self.br_convention}  "
            f"NLO_loops={self.include_NLO_hard_decay} "
            f"(K_C-safe Coul={self.coulomb_kc_safe}, "
            f"decay×σ_Born={self.decay_uses_full_born})  "
            f"NNLO={self.include_BFS_NNLO}  "
            f"δ_QCD={self.apply_delta_QCD} (α_s={self.alpha_s})  "
            f"anchor={self.apply_whizard_anchor}[{self.whizard_anchor_source}]  "
            f"ISR={self.isr_scheme} NLL={self.isr_nll} eMELA_LL={self.isr_emela_ll} "
            f"({self.isr_emela_pert_order}/{self.isr_emela_fac_scheme}/{self.isr_emela_ren_scheme}; "
            f"α_em_isr={a_isr})  "
            f"α_em_chain={a_chain}  "
            f"m_t={self.m_t} M_H={self.M_H} M_Z={self.MZ}"
        )

    # ------------------------------------------------------------------
    # Filenames (match the stub pattern so existing harness still works)
    # ------------------------------------------------------------------
    @staticmethod
    def _order_str(order: int) -> str:
        return {0: "LO", 1: "NLO", 2: "NNLO", 3: "N3LO"}.get(order, f"O{order}")

    def file_tag(self, values: dict) -> str:
        parts = []
        for name, val in values.items():
            label = "asVar" if name == "alphas" else name
            decimals = 4 if name == "alphas" else 3
            parts.append(f"{label}{val:.{decimals}f}")
        return "_".join(parts)

    def file_name(self, values: dict, *, mass_scale: float, width_scale: float,
                  mass_scheme: str = "OS", indir: str = ".") -> str:
        body = self.file_tag(values)
        scales = f"scaleM{mass_scale:.1f}_scaleW{width_scale:.1f}"
        return os.path.join(indir, f"WW_{self._order_str(self.order)}_{body}_{scales}.txt")

    # ------------------------------------------------------------------
    # Template production
    # ------------------------------------------------------------------
    def do_scan(self, values: dict, *, mass_scale: float, width_scale: float,
                mass_scheme: str = "OS", outdir: str = "output_xsec/ww/nominal",
                ecm_shift_MeV: float = 0.0) -> str:
        """Compute σ_obs(√s; m_W, Γ_W) on the fine ECM grid and write CSV.

        ``ecm_shift_MeV`` shifts every √s in the output grid by the given
        amount (used by the BEC nuisance machinery, which expects templates
        in ``output_xsec/ww/BEC/scan_{p,m}{var}/``).

        Returns the output file path.
        """
        mW     = float(values["mass"])
        gammaW = float(values["width"])
        # ``alphas`` in the steering card is the OFFSET from the nominal
        # α_s(M_W) (matches the WbWb convention). Add it to the generator's
        # nominal α_s when computing the effective δ_QCD(α_s) at this fit point.
        alpha_s_eff = self.alpha_s + float(values.get("alphas", 0.0))

        ecm_grid = _build_fine_grid() + ecm_shift_MeV * 1e-3
        sigma_obs = sigma_observed_munuqq(
            ecm_grid,
            mW=mW, gammaW=gammaW,
            channel=self.channel,
            include_coulomb=self.include_coulomb,
            bfs=self.bfs,
            include_NLO_hard_decay=self.include_NLO_hard_decay,
            include_BFS_NNLO=self.include_BFS_NNLO,
            apply_delta_QCD=self.apply_delta_QCD,
            alpha_s=alpha_s_eff,
            alpha_s_ref=self.alpha_s,
            br_convention=self.br_convention,
            apply_whizard_anchor=self.apply_whizard_anchor,
            whizard_anchor_source=self.whizard_anchor_source,
            isr_scheme=self.isr_scheme,
            isr_nll=self.isr_nll,
            isr_emela_ll=self.isr_emela_ll,
            isr_emela_pert_order=self.isr_emela_pert_order,
            isr_emela_fac_scheme=self.isr_emela_fac_scheme,
            isr_emela_ren_scheme=self.isr_emela_ren_scheme,
            alpha_em=self.alpha_em,
            alpha_em_isr=self.alpha_em_isr,
            coulomb_kc_safe=self.coulomb_kc_safe,
            decay_uses_full_born=self.decay_uses_full_born,
            m_t=self.m_t, M_H=self.M_H, MZ=self.MZ,
        )

        os.makedirs(outdir, exist_ok=True)
        path = self.file_name(values, mass_scale=mass_scale, width_scale=width_scale,
                              mass_scheme=mass_scheme, indir=outdir)
        # CSV: optional ``# key: value`` preamble + (ecm, xsec) rows. The
        # preamble carries the active chain configuration so consumers
        # (fit plot footers, downstream tooling) can label themselves
        # against the *actual* templates rather than the live card.
        header = compose_header(self.template_fingerprint())
        with open(path, "w") as fh:
            fh.write(header)
            for ecm, sigma in zip(ecm_grid, sigma_obs):
                fh.write(f"{ecm:.4f}, {sigma:.8f}\n")
        return path
