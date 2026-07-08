"""Byte-identity gate for the systematics-table emitters (NEED-4 dedup).

Exercises framework/common/systematics.py::_print_table and ::_write_latex on
mock cards that hit every formatting trap:
  - WW-like   : 2 POIs, no THEORY_UNC, a NaN diagonal cell  -> dash, no theory row
  - WbWb-like : 3 POIs incl. relative-yukawa, THEORY_UNC set -> theory row + divide

Run BEFORE the refactor to write the golden, AFTER to compare:
    python3 -m scripts.investigations.syst_table_dedup.check_byte_identical write   # golden
    python3 -m scripts.investigations.syst_table_dedup.check_byte_identical check   # diff
"""
from __future__ import annotations

import io
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

from framework.common import systematics as S

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.txt")


def _ww_card():
    return SimpleNamespace(
        POI_DISPLAY={"mass": {"unit": "MeV"}, "width": {"unit": "MeV"}},
        SYST_TABLE_PATH="",
    )  # no THEORY_UNC attribute -> theory row absent


def _wbwb_card():
    return SimpleNamespace(
        POI_DISPLAY={
            "mass": {"unit": "MeV"},
            "width": {"unit": "MeV"},
            "yukawa": {"unit": "%", "relative": True},
        },
        THEORY_UNC={"mass": 5, "width": 4, "yukawa": 3},
        SYST_TABLE_PATH="",
    )


def _inputs(pois, with_nan):
    # syst: ordered dict {syst_name -> {poi -> raw}}; NaN on the diagonal of stat
    systs = ["stat", "lumi_corr", "bec_corr", "alphas"]
    syst = {}
    for i, poi in enumerate(pois):
        col = {}
        for j, name in enumerate(systs):
            val = 0.5 * (i + 1) + 0.1 * j
            if with_nan and name == "stat" and i == 0:
                val = float("nan")
            col[name] = val
        syst[poi] = col
    totals = {poi: 1.0 + 0.3 * k for k, poi in enumerate(pois)}
    centrals = {poi: 80.3 + k for k, poi in enumerate(pois)}
    return syst, totals, centrals


def _capture(card, pois):
    syst, totals, centrals = _inputs(pois, with_nan=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        S._print_table(card, syst, totals, centrals)
    with tempfile.NamedTemporaryFile("r+", suffix=".tex", delete=False) as fh:
        tex_path = fh.name
    S._write_latex(card, syst, totals, centrals, tex_path)
    with open(tex_path) as fh:
        tex = fh.read()
    os.unlink(tex_path)
    return buf.getvalue() + "\n===TEX===\n" + tex


def render_all():
    out = []
    out.append("##### WW-like (2 POI, no theory, NaN diagonal) #####")
    out.append(_capture(_ww_card(), ["mass", "width"]))
    out.append("##### WbWb-like (3 POI, theory, relative yukawa, NaN) #####")
    out.append(_capture(_wbwb_card(), ["mass", "width", "yukawa"]))
    return "\n".join(out)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    rendered = render_all()
    if mode == "write":
        with open(GOLDEN, "w") as fh:
            fh.write(rendered)
        print(f"[golden] wrote {GOLDEN} ({len(rendered)} bytes)")
        return
    with open(GOLDEN) as fh:
        golden = fh.read()
    if rendered == golden:
        print("[check] BYTE-IDENTICAL to golden  ✓")
        sys.exit(0)
    print("[check] *** DIFFERS FROM GOLDEN ***")
    import difflib
    for line in difflib.unified_diff(
            golden.splitlines(), rendered.splitlines(),
            "golden", "current", lineterm=""):
        print(line)
    sys.exit(1)


if __name__ == "__main__":
    main()
