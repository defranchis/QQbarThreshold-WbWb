#!/bin/bash
# Build the BFS-implementation report PDF and mirror it to the project EOS
# web area. Idempotent — safe to re-run after every .tex edit.
#
# Usage:
#   ./publish.sh            (from report/)
#   make publish            (from report/, equivalent)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TEX=ww_bfs_implementation
PDF=$TEX.pdf
EOS_DEST=${EOS_REPORT_DEST:-/eos/user/m/mdefranc/www/WW_threshold/report}
EOS_URL=${EOS_REPORT_URL:-https://mdefranc.web.cern.ch/WW_threshold/report}

# Refresh investigation figures embedded in the report: morphing-scheme
# validation lives in plots/whizard_grid_highstats/ (gitignored). Copy the
# 3 PDFs the .tex references into figs/ with a `morph_` prefix.
MORPH_SRC="../plots/whizard_grid_highstats"
if [[ -d "$MORPH_SRC" ]]; then
    mkdir -p figs
    # Source filename → report-figs filename. `morph_` prefix on dest names
    # tags them in the figs/ namespace.
    declare -A MORPH_FIGS=(
        [morphing_bilinear]=morph_morphing_bilinear
        [4d_morphing_loo]=morph_4d_morphing_loo
        [morph_azzurri]=morph_azzurri
    )
    for src in "${!MORPH_FIGS[@]}"; do
        dst=${MORPH_FIGS[$src]}
        if [[ -f "$MORPH_SRC/$src.pdf" ]]; then
            cp -p "$MORPH_SRC/$src.pdf" "figs/$dst.pdf"
        else
            echo "[publish] WARN: $MORPH_SRC/$src.pdf missing"
        fi
    done
else
    echo "[publish] WARN: $MORPH_SRC not found — morphing figs may be stale"
fi

# Build twice so references resolve.
echo "[publish] building $TEX.pdf"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex" >/dev/null

if [[ ! -s "$PDF" ]]; then
    echo "[publish] ERROR: $PDF missing or empty after pdflatex" >&2
    exit 1
fi

# Copy to EOS. The dest dir is expected to exist (created on first publish).
if [[ ! -d "$EOS_DEST" ]]; then
    echo "[publish] creating $EOS_DEST"
    mkdir -p "$EOS_DEST"
fi
cp -p "$PDF" "$EOS_DEST/$PDF"

echo "[publish] wrote $EOS_DEST/$PDF ($(stat -c %s "$PDF") bytes)"
echo "[publish] URL: $EOS_URL/$PDF"
