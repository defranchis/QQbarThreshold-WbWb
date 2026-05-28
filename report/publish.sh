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
# PDFs the appendix references into figs/ with a `morph_` prefix.
MORPH_SRC="../plots/whizard_grid_highstats"
if [[ -d "$MORPH_SRC" ]]; then
    mkdir -p figs
    # Source filename → report-figs filename. `morph_` prefix on dest names
    # tags them in the figs/ namespace (sources already so-prefixed keep it).
    declare -A MORPH_FIGS=(
        [grid_overview]=morph_grid_overview
        [axis_fits]=morph_axis_fits
        [cross_term]=morph_cross_term
        [sqrts_quantities]=morph_sqrts_quantities
        [sqrts_interpolation]=morph_sqrts_interpolation
        [sqrts_validation]=morph_sqrts_validation
        [4d_morphing_loo]=morph_4d_morphing_loo
        [validate_substep]=morph_validate_substep
        [validate_fine_mW]=morph_validate_fine_mW
        [validate_fine_gW]=morph_validate_fine_gW
        [validate_bfs_tables]=morph_validate_bfs_tables
        [morph_smooth_sigma_sqrts]=morph_smooth_sigma_sqrts
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

# Coulomb-double-count appendix figures: produced by the scripts under
# scripts/investigations/coulomb_double_count/, output dropped in
# plots/coulomb_double_count/ (gitignored). Mirror into figs/ with the
# `coul_` prefix.
COUL_SRC="../plots/coulomb_double_count"
if [[ -d "$COUL_SRC" ]]; then
    mkdir -p figs
    declare -A COUL_FIGS=(
        [kc_vs_term1_overlap]=coul_kc_vs_term1_overlap
        [compare_kc_safe_chain]=coul_compare_kc_safe_chain
    )
    for src in "${!COUL_FIGS[@]}"; do
        dst=${COUL_FIGS[$src]}
        if [[ -f "$COUL_SRC/$src.pdf" ]]; then
            cp -p "$COUL_SRC/$src.pdf" "figs/$dst.pdf"
        else
            echo "[publish] WARN: $COUL_SRC/$src.pdf missing"
        fi
    done
else
    echo "[publish] WARN: $COUL_SRC not found — Coulomb figs may be stale"
fi

# ISR diagnostic figure (3-way scheme comparison): produced by
# scripts/investigations/nll_isr/plot_isr_comparison.py, output at
# plots/isr_comparison.pdf (gitignored).
ISR_SRC="../plots/isr_comparison.pdf"
if [[ -f "$ISR_SRC" ]]; then
    mkdir -p figs
    cp -p "$ISR_SRC" "figs/isr_comparison.pdf"
else
    echo "[publish] WARN: $ISR_SRC missing — ISR-scheme figure will be stale"
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
