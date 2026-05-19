"""Lightweight metadata header for WW template CSV files.

Each template written by :class:`WWGenerator.do_scan` carries a short
``# key: value`` preamble before the (ecm, xsec) rows. The preamble is
ignored by :class:`pandas.read_csv` when ``comment='#'`` is set, so the
existing fit-side reader keeps working.

The intent is to make the cross section file self-describing:

  * ``chain`` — LaTeX-friendly summary of the active physics chain
    (from :func:`chain_summary_latex`), used by plot footers.
  * ``isr_scheme`` — quick text marker for which ISR convolution was
    used. Lets the fit-side warn on cross-template mismatches.

Header lines are limited to ``key: value`` pairs (one per line). Adding
a new key is a one-line edit to :func:`compose_header` and consumers.
"""

from __future__ import annotations

from typing import Mapping


COMMENT_PREFIX = "# "


def compose_header(metadata: Mapping[str, str]) -> str:
    """Return a ``# key: value\\n`` preamble (with trailing newline).

    Non-string values are coerced via ``str()``. Empty / None values are
    skipped so callers can pass ``None`` for missing optional keys.
    """
    lines = []
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        lines.append(f"{COMMENT_PREFIX}{key}: {value}")
    return "\n".join(lines) + ("\n" if lines else "")


def read_header(path: str) -> dict[str, str]:
    """Parse the ``# key: value`` preamble of a template CSV.

    Stops at the first non-comment / non-blank line. Returns an empty
    dict if there is no preamble (back-compat with templates written
    before metadata was introduced).
    """
    out: dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            body = stripped.lstrip("#").strip()
            if ":" not in body:
                continue
            key, _, value = body.partition(":")
            out[key.strip()] = value.strip()
    return out
