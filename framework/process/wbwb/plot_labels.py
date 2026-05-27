"""WbWb-process plot-decoration helpers.

Dispatched from ``framework/common/plots.py`` via ``card.PROCESS_ID == "wbwb"``.
The labels are static here (no channel knob like WW has); the static chain
label is the same string used historically.
"""

from __future__ import annotations


GENERATOR_NAME = "QQbar_Threshold"
GENERATOR_REF = r"[JHEP 02 (2018) 125]"

# card.ORDER → "LO" / "NLO" / "NNLO" / "N^{n}LO" badge.
_ORDER_TAGS = {0: "LO", 1: "NLO", 2: "NNLO"}


def _order_tag(card) -> str:
    order = int(getattr(card, "ORDER", 3))
    return _ORDER_TAGS.get(order, rf"N$^{{{order}}}$LO")


def _generator_chain_label(card) -> str:
    return f"{GENERATOR_NAME} {_order_tag(card)}+ISR"


def process_label(card) -> str:
    return f"WbWb at {_order_tag(card)}+ISR"


def process_label_short(card) -> str:
    # WbWb historically didn't define a short form distinct from the full one.
    return process_label(card)


def generator_label(card) -> str:
    return _generator_chain_label(card)


def generator_label_short(card) -> str:
    """Compact lower-right badge — auto-derived from ``card.ORDER``."""
    return _generator_chain_label(card)
