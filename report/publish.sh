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

# Scan-scenario comparison figures: produced by `doFit_ww.py --compareScenarios`
# (framework/process/ww/scenario_compare.py), output under plots/scenario_compare/
# (gitignored). Mirror the set into figs/ verbatim — the report includes them by
# their basenames (layout, ellipses, per-POI syst bars, systematic sweeps).
mkdir -p figs
_scen_n=0
for f in ../plots/scenario_compare/scenario_compare_layout.pdf \
         ../plots/scenario_compare/scenario_compare_ellipses.pdf \
         ../plots/scenario_compare/scenario_compare_syst_mW.pdf \
         ../plots/scenario_compare/scenario_compare_syst_gW.pdf \
         ../plots/scenario_compare/scenario_compare_scan_*.pdf; do
    if [[ -f "$f" ]]; then cp -p "$f" "figs/$(basename "$f")"; _scen_n=$((_scen_n+1)); fi
done
[[ $_scen_n -gt 0 ]] || echo "[publish] WARN: no plots/scenario_compare/*.pdf — scenario figs stale"

# Cross-section diagnostic figures (σ-chain vs √s, dσ/dm_W & dσ/dΓ_W sensitivity,
# lineshape ratios, Azzurri-style overlays): produced by
# scripts/plot_ww_diagnostics.py into fit_output/ww/diagnostics/ (gitignored).
# The report embeds them as diag/<name>; mirror the set into figs/diag/ so the
# published PDF tracks the LIVE σ chain. (These were previously a hand-copied
# snapshot that drifted stale — e.g. predating NLL ISR — because nothing wired
# the regenerated figures into the report build.)
DIAG_SRC="../fit_output/ww/diagnostics"
if [[ -d "$DIAG_SRC" ]]; then
    mkdir -p figs/diag
    _diag_n=0
    for name in xsec_vs_sqrts sensitivity_vs_sqrts ratios_mW_GammaW \
                azzurri_style_pm1GeV azzurri_style_overlay; do
        if [[ -f "$DIAG_SRC/$name.pdf" ]]; then
            cp -p "$DIAG_SRC/$name.pdf" "figs/diag/$name.pdf"; _diag_n=$((_diag_n+1))
        else
            echo "[publish] WARN: $DIAG_SRC/$name.pdf missing — diag fig stale"
        fi
    done
    [[ $_diag_n -gt 0 ]] || echo "[publish] WARN: no diagnostics PDFs mirrored"
else
    echo "[publish] WARN: $DIAG_SRC not found — diagnostic figs may be stale"
fi

# ISR diagnostic figure (3-way scheme comparison): produced by
# scripts/investigations/nll_isr/plot_isr_comparison.py, output at
# plots/nll_isr/isr_comparison.pdf (gitignored).
ISR_SRC="../plots/nll_isr/isr_comparison.pdf"
if [[ -f "$ISR_SRC" ]]; then
    mkdir -p figs
    cp -p "$ISR_SRC" "figs/isr_comparison.pdf"
else
    echo "[publish] WARN: $ISR_SRC missing — ISR-scheme figure will be stale"
fi

# Independent MoCaNLO cross-check figures (per-channel line-shape response,
# off-shell BR convention with the BR-rescaled BFS overlay): produced by
# scripts/investigations/indep_mocanlo/plot_channel_response.py into
# plots/indep_mocanlo/ (gitignored). The report embeds them as
# indep_response_{mass,width}; mirror so the published PDF tracks the live grid.
INDEP_SRC="../plots/indep_mocanlo"
if [[ -d "$INDEP_SRC" ]]; then
    mkdir -p figs
    declare -A INDEP_FIGS=(
        [channel_response_mass]=indep_response_mass
        [channel_response_width]=indep_response_width
    )
    for src in "${!INDEP_FIGS[@]}"; do
        dst=${INDEP_FIGS[$src]}
        if [[ -f "$INDEP_SRC/$src.pdf" ]]; then
            cp -p "$INDEP_SRC/$src.pdf" "figs/$dst.pdf"
        else
            echo "[publish] WARN: $INDEP_SRC/$src.pdf missing — indep fig stale"
        fi
    done
else
    echo "[publish] WARN: $INDEP_SRC not found — indep cross-check figs may be stale"
fi

# Statistical-correlation scan (sigma(m_W)/sigma(Gamma_W) vs the point-to-point
# stat correlation rho): produced by `doFit_ww.py --statCorrScan`
# (framework/common/scans.py:scan_stat_correlation) into fit_output/ww/plots/.
STATCORR_SRC="../fit_output/ww/plots/uncert_mass_width_vs_statcorr.pdf"
if [[ -f "$STATCORR_SRC" ]]; then
    mkdir -p figs
    cp -p "$STATCORR_SRC" "figs/uncert_mass_width_vs_statcorr.pdf"
else
    echo "[publish] WARN: $STATCORR_SRC missing — stat-correlation figure will be stale"
fi

# Cross-section-systematic scan (sigma(m_W)/sigma(Gamma_W) vs the per-point
# xsec syst as a percentage of stat, 0->2x, corr + uncorr): produced by
# `doFit_ww.py --xsecSystScan` (framework/common/scans.py:scan_xsec_syst) into
# fit_output/ww/plots/.
XSECSYST_SRC="../fit_output/ww/plots/uncert_mass_width_vs_xsecsyst.pdf"
if [[ -f "$XSECSYST_SRC" ]]; then
    mkdir -p figs
    cp -p "$XSECSYST_SRC" "figs/uncert_mass_width_vs_xsecsyst.pdf"
else
    echo "[publish] WARN: $XSECSYST_SRC missing — xsec-systematic figure will be stale"
fi

# Data tables the report cites (text provenance of the results sections, e.g.
# report/data/scenario_compare.txt referenced in the scenario tables). Mirror
# the human-readable dumps from the untracked plots/ working areas into the
# TRACKED report/data/ so the report is self-contained. (.txt only — the .csv
# machine versions stay in plots/, regenerable.)
mkdir -p data
declare -A DATA_FILES=(
    [../plots/scenario_compare/scenario_compare.txt]=scenario_compare.txt
    [../plots/theory_ladder/theory_ladder.txt]=theory_ladder.txt
    [../plots/channel_extrap/channel_extrap.txt]=channel_extrap.txt
)
for src in "${!DATA_FILES[@]}"; do
    if [[ -f "$src" ]]; then cp -p "$src" "data/${DATA_FILES[$src]}"
    else echo "[publish] WARN: $src missing — report/data table will be stale"; fi
done

# Build three times so all references resolve: with hyperref + TOC + the
# long results tables, a fresh build (no .aux) needs a third pass before
# the cross-references and outlines stop shifting ("Rerun to get
# cross-references right" persists after only two).
echo "[publish] building $TEX.pdf"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX.tex" >/dev/null
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
