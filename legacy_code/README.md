# Legacy

Pre-refactor entry script + helper scripts. Archived as historical reference;
**not part of the current workflow**.

| File | Replaced by |
|---|---|
| `doFit.py` | `doFit_wbwb.py` (+ `cards/wbwb_default.py` + `common/` + `process/wbwb/`) — monolithic 1450-line script split into the modular framework. |
| `allFits.sh` | New `allFits_wbwb.sh` / `allFits_ww.sh` at the top level, calling the modular entry scripts with the current flag names. |
| `run_fits_nom.sh` | No replacement at top level — pick the relevant subset of `allFits_*.sh` lines if you want a quick run. |
| `copy_plots_paper.sh` | No replacement — output plot directories changed. Re-do per-publication if needed. |
| `compute_singletop.py` | No live caller. Single-top xsec convenience driver; preserved in case a future analysis revives it. |
| `test_BES.py` | No live caller. Standalone BES diagnostic script. |

`scripts/` holds top-pair (WbWb) plotting scripts from before the WW pivot —
no live callers, all reference deprecated output paths (`output/{NLO,NNLO,N3LO}_scan_{MS,PS}_*`):

| File | Purpose |
|---|---|
| `plot.py` | top-pair σ(√s) by α_s order × scheme |
| `plotISR.py` | top-pair ISR-folded σ(√s) overlay |
| `plotScaleVars.py` | top-pair α_s scale-variation comparison |
| `compareISR.py` | shared `convoluteXsecGauss` for `plot.py` / `plotISR.py` / `checkMorphing.py` |
| `checkMorphing.py` | top-pair (PS m_t=171.5 GeV) mass+width morphing diagnostic |
| `mt_mW_uncert.py` | one-off m_t + m_W combined-uncertainty figure (July 2025) |

The cross-check at `extensibility_refactor` (commit `49a5582` onwards) verified
that `doFit_wbwb.py` reproduces the legacy `doFit.py` chi² / constraint /
nuisance / cov / SM-width / pseudodata logic byte-for-byte modulo
documented divergences (RNG fix, dead-code drops, perf optimisations).

Imports inside these files may fail (`utils_fit.fitUtils` etc.) — they're
not on the current `sys.path`.
