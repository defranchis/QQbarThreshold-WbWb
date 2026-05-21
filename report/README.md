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
