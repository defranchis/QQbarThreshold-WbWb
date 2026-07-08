# BFS implementation report

LaTeX summary of the WW-threshold cross-section implementation (BFS
unstable-particle EFT chain) and its validation against the two BFS
papers, [arXiv:0707.0773](https://arxiv.org/abs/0707.0773) and
[arXiv:0807.0102](https://arxiv.org/abs/0807.0102).

## Build

```bash
make            # pdflatex twice (resolves the references)
make publish    # build + mirror PDF to EOS (run after every .tex edit)
make clean      # remove aux files
```

Output: `ww_bfs_implementation.pdf`. Requires `pdflatex`, `amsmath`,
`booktabs`, `hyperref`, `microtype` — all in a vanilla TeX Live.

`make publish` (or `./publish.sh` directly) builds the PDF and copies
it to `/eos/user/m/mdefranc/www/WW_threshold/report/` so the version at
<https://mdefranc.web.cern.ch/WW_threshold/report/ww_bfs_implementation.pdf>
matches the source. Run it any time the .tex changes. The destination
can be overridden via the `EOS_REPORT_DEST` / `EOS_REPORT_URL` env vars.

## Self-contained inputs (tracked)

The report travels with everything it uses, so it is rebuildable from the repo
without re-running any pipeline:

- `figs/` — exactly the PDF figures the `.tex` embeds (no PNGs, no extras).
- `data/` — the human-readable text dumps the report cites for its results
  tables (`scenario_compare.txt`, `theory_ladder.txt`, `channel_extrap.txt`).

Both are **tracked** and kept in sync by `publish.sh`, which mirrors the
report-used subset from the untracked `plots/` and `fit_output/` working areas
on every publish. Only the built `ww_bfs_implementation.pdf` is gitignored.

## Reproducing the validation numbers

Every table in section 3 corresponds to a script in the repo:

| Section | Script | Mirrors |
|---|---|---|
| 3.1 Tables 1/2 (Born)               | `python framework/process/ww/xsec_calculator/bfs_eft.py` | BFS NLO paper §3 |
| 3.2 NLO-loop magnitudes (C/D/E)     | `python scripts/validate_bfs_nlo.py` Scenarios C,D,E   | BFS NLO paper §4 |
| 3.3 Tables 3/4 (Born+ISR + NLO)     | `python scripts/validate_bfs_nlo.py` Scenario F        | BFS NLO paper §6 |
| 3.4 Whizard anchor closure          | `python scripts/validate_bfs_nlo.py` Scenario I        | BFS NLO paper §6.2 |
| 3.5 NNLO Table 1 (each piece + sum) | `python scripts/investigations/bfs_nnlo/check_closed_form_pieces.py` | BFS NNLO paper §4 |
| 3.6 NNLO ISR-improved (Table 2 col) | `python scripts/investigations/bfs_nnlo/check_isr_table2.py`         | BFS NNLO paper §4 |

## Appendix B — morphing-scheme validation

Appendix B documents and validates the WHIZARD-grid morphing predictor
of §3.6.1. Its figures are produced by scripts under
`scripts/investigations/whizard_grid_highstats/` and mirrored into
`figs/` (with a `morph_` prefix) by `publish.sh`:

| Appendix figure | Script |
|---|---|
| input grid (B.2)                          | `plot_grid_overview.py` |
| per-√s quadratic fits (B.3)               | `plot_axis_fits.py` |
| bilinear cross term (B.4)                 | `plot_cross_term.py` |
| √s morph quantities, abs + dev (B.5)      | `plot_sqrts_quantities.py`, `plot_sqrts_interpolation.py` |
| √s thinned-grid LOO blind test (B.5)      | `plot_sqrts_validation.py` |
| 4D √s leave-one-out (B.5)                 | `plot_morph_predictions.py` |
| 1-MeV (m_W,Γ_W) held-out closure (B.6)    | `plot_validate_substep.py` |
| sub-MeV (m_W,Γ_W) held-out closure (B.6)  | `plot_validate_fine.py` |
| BFS-table closure (B.6)                   | `validate_bfs_tables.py` |
| line shape + Azzurri (B.7)                | `plot_morph_predictions.py` |
