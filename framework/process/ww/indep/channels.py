"""Channel blocks and the total-σ assembly for the independent WW calculation.

The total e+e- → WW → 4f cross section is built as an explicit weighted sum of
six representative partonic final states ("blocks"), each generated once with
MoCaNLO.  The weights are pure flavour/colour multiplicities (massless quarks +
CKM unitarity ⇒ 2 up-types per W; massless leptons ⇒ τ ≈ μ); the colour factor
N_c per quark line lives *inside* each block's σ̂ (MoCaNLO sums final colours).

Blocks (MoCaNLO ``<outgoing>`` strings; e=electron has extra t-channel
single-W diagrams, μ/τ do not):

    enuqq   e- ve~ u d~     electron semileptonic   (t-channel)
    lnuqq   mu- vm~ u d~    μ/τ semileptonic        (no t-channel)
    qqqq    u d~ s c~       hadronic, mixed flavour (pure WW, no ZZ/identical)
    enuenu  e- ve~ e+ ve    electron leptonic        (t-channel)
    lnulnu  mu- vm~ mu+ vm  μ/τ leptonic             (no t-channel)
    emu     e- ve~ mu+ vm   mixed e/μ leptonic       (t-channel one side)

Assembly (verified by the colour/flavour sum-rule below):

    σ_tot = 4·enuqq + 8·lnuqq + 4·qqqq + 1·enuenu + 4·emu + 4·lnulnu

  Semileptonic  = 4·enuqq + 8·lnuqq   (12 = 2 W-leptonic × 3 families × 2 up-types)
  Hadronic      = 4·qqqq              (4  = 2 up-types × 2 up-types)
  Leptonic      = enuenu + 4·emu + 4·lnulnu  (9 = 3 × 3 family combinations)

Sum-rule cross-check: with σ̂ ∝ (colour weight), w(q-pair)=N_c=3, w(lepton)=1,
each block's relative size is the product of its two W weights; Σ weights·size
= (Σ_W weight)² = (6_had + 3_lep)² = 9² = 81 — i.e. the total factorises into
two independent W decays, as it must.  ``check_sum_rule`` asserts this.

ZZ/neutral-current contamination of the same-flavour leptonic blocks (enuenu,
lnulnu) is kinematically suppressed at the WW threshold (√s ≈ 161 ≪ 2 m_Z),
so the μμ representative for lnulnu (and ee for enuenu) is adequate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelBlock:
    key: str            # short tag used in filenames / grid keys
    outgoing: str       # MoCaNLO proc_card <outgoing> (Born/virt level)
    weight: float       # multiplicity in σ_tot
    has_tchannel: bool  # electron in final state ⇒ extra single-W diagrams
    label: str          # human-readable

    @property
    def outgoing_real(self) -> str:
        """Real-emission <outgoing> = Born outgoing + a hard photon."""
        return f"{self.outgoing} a"


#: The six blocks and their σ_tot multiplicities.
BLOCKS: tuple[ChannelBlock, ...] = (
    ChannelBlock("enuqq",  "e- ve~ u d~",    4.0, True,  "e ν qq (semileptonic, t-ch)"),
    ChannelBlock("lnuqq",  "mu- vm~ u d~",   8.0, False, "μ/τ ν qq (semileptonic)"),
    ChannelBlock("qqqq",   "u d~ s c~",      4.0, False, "qqqq (hadronic)"),
    ChannelBlock("enuenu", "e- ve~ e+ ve",   1.0, True,  "e ν e ν (leptonic, t-ch)"),
    ChannelBlock("emu",    "e- ve~ mu+ vm",  4.0, True,  "e ν μ ν (mixed leptonic, t-ch)"),
    ChannelBlock("lnulnu", "mu- vm~ mu+ vm", 4.0, False, "μ/τ ν μ/τ ν (leptonic)"),
)

#: Clean doubly-resonant-WW leptonic representative: a MIXED non-electron
#: flavour pair (μ⁻ν̄_μ τ⁺ν_τ).  No same-flavour ℓ⁺ℓ⁻ pair ⇒ no γ*→ℓℓ low-mass
#: pole; no electron ⇒ no forward t-channel.  By flavour universality its σ
#: equals the pure-WW σ(WW→ℓνℓν), so the full leptonic sector (9 family combos)
#: is 9·mutau — WITHOUT the neutral-current backgrounds that contaminate the
#: same-flavour (enuenu, lnulnu) channels.  Used for the inclusive "pure-WW"
#: definition; the fiducial definition instead cuts the original 6 blocks.
MUTAU = ChannelBlock("mutau", "mu- vm~ ta+ vt", 9.0, False,
                     "μ ν τ ν (clean WW leptonic, no NC bkg)")

#: Pure-WW (doubly-resonant, background-free, NO cut) uses only the 3 stable
#: channels: σ_WW = 12·lnuqq + 4·qqqq + 9·mutau.
PURE_WW_WEIGHTS: dict[str, float] = {"lnuqq": 12.0, "qqqq": 4.0, "mutau": 9.0}

BLOCKS_BY_KEY: dict[str, ChannelBlock] = {b.key: b for b in BLOCKS}
BLOCKS_BY_KEY[MUTAU.key] = MUTAU


def assemble_pure_ww(block_sigmas: dict[str, float]) -> float:
    """Inclusive pure-WW σ_tot from the 3 stable channels (no cut)."""
    return sum(w * block_sigmas[k] for k, w in PURE_WW_WEIGHTS.items())

#: Grouping for reporting partial cross sections.
GROUPS: dict[str, tuple[str, ...]] = {
    "semileptonic": ("enuqq", "lnuqq"),
    "hadronic": ("qqqq",),
    "leptonic": ("enuenu", "emu", "lnulnu"),
}


def assemble_total(block_sigmas: dict[str, float]) -> float:
    """σ_tot from a dict {block_key: σ_block} using the multiplicities."""
    return sum(b.weight * block_sigmas[b.key] for b in BLOCKS)


def assemble_group(group: str, block_sigmas: dict[str, float]) -> float:
    """Partial σ for a flavour group ('semileptonic'/'hadronic'/'leptonic')."""
    keys = GROUPS[group]
    return sum(BLOCKS_BY_KEY[k].weight * block_sigmas[k] for k in keys)


def check_sum_rule(tol: float = 1e-9) -> None:
    """Assert the multiplicities reproduce the factorised (Σ_W weight)² = 81.

    Model each block's σ̂ as the product of its two W-decay colour weights
    (N_c=3 per quark line, 1 per lepton line); the weighted sum must equal
    (6_had + 3_lep)² = 81.
    """
    NC = 3.0
    # per-W weight of each leg of a block (q-pair → N_c, lepton → 1)
    leg = {
        "enuqq":  (1.0, NC),       # (e ν)(u d̄)
        "lnuqq":  (1.0, NC),       # (μ ν)(u d̄)
        "qqqq":   (NC, NC),        # (u d̄)(s c̄)
        "enuenu": (1.0, 1.0),
        "emu":    (1.0, 1.0),
        "lnulnu": (1.0, 1.0),
    }
    total = sum(b.weight * leg[b.key][0] * leg[b.key][1] for b in BLOCKS)
    expected = (2.0 * NC + 3.0) ** 2   # 9² = 81
    if abs(total - expected) > tol:
        raise AssertionError(
            f"channel multiplicity sum-rule FAILED: got {total}, expected {expected}")


if __name__ == "__main__":
    check_sum_rule()
    print("channel sum-rule OK: Σ weight·(colour size) = 81 = (2·N_c + 3)²")
    print(f"σ_tot = " + " + ".join(f"{b.weight:g}·{b.key}" for b in BLOCKS))
    for g, keys in GROUPS.items():
        print(f"  {g:13s}: " + " + ".join(f"{BLOCKS_BY_KEY[k].weight:g}·{k}" for k in keys))
