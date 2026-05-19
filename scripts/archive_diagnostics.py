#!/usr/bin/env python
"""Snapshot the current WW diagnostic plots to EOS under archive/<date>_<sha>/ww_diagnostics/.

Re-stamps whatever is currently in plots/ww_diagnostics/ (the output dir of
scripts/plot_ww_diagnostics.py) — no local copy is kept. Tag is derived from
`git rev-parse --short HEAD` plus today's date; `-dirty` is appended if the
working tree has uncommitted changes."""

from framework.common.eos_publish import archive_tag, publish

PLOT_DIR = "fit_output/ww/diagnostics"

if __name__ == "__main__":
    publish(PLOT_DIR, f"archive/{archive_tag()}/ww/diagnostics")
