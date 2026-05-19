#!/usr/bin/env python
"""Snapshot the current WbWb fit plots to EOS under archive/<date>_<sha>/wbwb/.

Re-stamps whatever is currently in cards.wbwb_default.PLOT_DIR — no local
copy is kept. Tag is derived from `git rev-parse --short HEAD` plus today's
date; `-dirty` is appended if the working tree has uncommitted changes."""

from cards import wbwb_default as card
from common.eos_publish import archive_tag, publish

if __name__ == "__main__":
    publish(card.PLOT_DIR, f"archive/{archive_tag()}/wbwb/plots")
