# WW threshold paper draft

Draft of the FCC-ee W boson mass and width threshold-scan paper, styled
after the companion tt threshold study (arXiv:2503.18713,
JHEP 11 (2025) 020).

## Status
- **Draft / preliminary.** Detector-level cross-section study (à la the
  WbWb paper reco section) is not yet included and will be added later.
- Some systematic variations are still missing; the assumed integrated
  luminosity (12 ab^-1) and the BEC/BES input uncertainties are
  placeholders. Open items are flagged inline with red `[TODO: ...]`
  notes via the `\TODO{}` macro.
- Author list is a placeholder.

## Build
```
make          # pdflatex -> bibtex -> pdflatex x2 -> main.pdf
make clean    # remove build artifacts
```

## Source of numbers
Fit results (Table 1) and figures come from the current production chain:
- `doFit_ww.py --systTable` for the systematics breakdown and correlation
  (sigma_mW = 2.5 MeV, sigma_GammaW = 4.0 MeV, rho = 0.61, stat-only
  1.1 / 2.3 MeV);
- `doFit_ww.py --chi2scans / --lumiscans / --BECscans / --BESscans /
  --alphaSscan` for the contour and the systematic scans;
- ISR theory uncertainty (5.4 MeV on mW) from the LL->NLL cross-fit
  documented in `report/ww_bfs_implementation.tex`.

Figures live in `figures/`; regenerate them from `fit_output/ww/plots/`,
`fit_output/ww/diagnostics/`, and top-level `plots/`.
