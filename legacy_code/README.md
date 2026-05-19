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

The cross-check at `extensibility_refactor` (commit `49a5582` onwards) verified
that `doFit_wbwb.py` reproduces the legacy `doFit.py` chi² / constraint /
nuisance / cov / SM-width / pseudodata logic byte-for-byte modulo
documented divergences (RNG fix, dead-code drops, perf optimisations).

Imports inside these files may fail (`utils_fit.fitUtils` etc.) — they're
not on the current `sys.path`.
