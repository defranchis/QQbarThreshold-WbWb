"""WW-process plot-decoration helpers.

The framework's plot machinery (``framework/common/plots.py``) dispatches
here via ``card.PROCESS_ID == "ww"`` so the steering card stays free of
hard-coded LaTeX strings. Channel-aware labels are built from
``card.NLO_CONFIG["channel"]``; the dynamic chain summary comes from
``chain_summary_latex(observed_kwargs_from_card(card))``.
"""

from __future__ import annotations


# LaTeX final-state strings keyed by ``NLO_CONFIG["channel"]``.
_CHANNEL_FINAL_STATE = {
    "inclusive": r"\mu\nu q\bar q",
    "munuud":    r"\mu^- \bar\nu_\mu u\bar d",
}

THRESHOLD_NAME = "WW"
GENERATOR_REF = (r"arXiv:0707.0773 + arXiv:0807.0102 "
                 r"(Beneke-Falgari-Schwinn et al.)")

# Highest perturbative-order label as a function of the NLO_CONFIG flags.
# Tracked here (not in the card) so flipping a chain knob is reflected
# automatically in the fit-scenario caption.
_NLO_ORDER_TAG = (
    ("include_BFS_NNLO",       "NNLO"),
    ("include_NLO_hard_decay", "NLO"),
)


def _order_tag(card) -> str:
    nlo = getattr(card, "NLO_CONFIG", {})
    for flag, tag in _NLO_ORDER_TAG:
        if nlo.get(flag, False):
            return tag
    return r"N$^{3/2}$LO"


# The analytic BETA-scheme single-conv / 2-leg radiators are LL+exp;
# when ``isr_nll=True`` the convolution is performed with the eMELA NLL
# structure functions instead (production default since 2026-05-29).
_ISR_SCHEME_TAG = {
    "single_conv": "LL ISR",
    "2leg":        "LL ISR",
}


def _isr_tag(card) -> str:
    nlo = getattr(card, "NLO_CONFIG", {})
    if nlo.get("isr_nll", False):
        return "NLL ISR"
    scheme = nlo.get("isr_scheme", "single_conv")
    return _ISR_SCHEME_TAG.get(scheme, "ISR")


def generator_label_short(card) -> str:
    """Compact lower-right badge — e.g. ``"NNLO EFT + LL ISR"`` — auto-
    derived from the card's chain configuration. EFT marker is fixed
    (the WW chain is BFS-EFT by construction)."""
    return f"{_order_tag(card)} EFT + {_isr_tag(card)}"


def _final_state(card) -> str:
    channel = card.NLO_CONFIG.get("channel", "inclusive")
    return _CHANNEL_FINAL_STATE.get(channel, _CHANNEL_FINAL_STATE["inclusive"])


def process_label(card) -> str:
    return rf"$e^+e^-\rightarrow {_final_state(card)}$ at {THRESHOLD_NAME} threshold"


def process_label_short(card) -> str:
    return rf"$e^+e^-\rightarrow {_final_state(card)}$"


def generator_label(card) -> str:
    """Dynamic ``BFS N^(3/2)LO + ...`` string from the card's chain config.
    Used as the static fallback when no template-preamble chain label is
    available."""
    from framework.process.ww.generator import (
        chain_summary_latex, observed_kwargs_from_card,
    )
    return chain_summary_latex(observed_kwargs_from_card(card))
