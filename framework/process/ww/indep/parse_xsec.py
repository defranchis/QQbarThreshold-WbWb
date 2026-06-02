"""Parse a MoCaNLO ``result/cross_section.dat`` cross section.

The machine-readable line is the one beginning with ``|``:

    |  <RunID> <Subproc> <Type> <Fsc> <Rsc> <PDFmem> <Xsec> <XsecErr> <relerr> ...

Cross section and error are in fb.  Returns ``(sigma_fb, err_fb)``.
"""

from __future__ import annotations

import glob
import os


def parse_cross_section_dat(path: str) -> tuple[float, float]:
    """Return (σ [fb], err [fb]) from a cross_section.dat file."""
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("|"):
                parts = s.split()
                # parts[0]='|', then RunID,Subproc,Type,Fsc,Rsc,PDFmem,Xsec,Err,...
                return float(parts[7]), float(parts[8])
    raise ValueError(f"no '|' cross-section line in {path}")


def find_latest_result(procdir: str, run_type: str,
                       subprocess: str = "ww",
                       runs_dir: str = "out") -> str:
    """Path to the newest ``cross_section.dat`` for a run type.

    MoCaNLO writes ``<procdir>/<runs_dir>/runs/<subprocess>/<run_type>/run_<ts>/
    result/cross_section.dat``; pick the most recent timestamped run dir.
    """
    pattern = os.path.join(procdir, runs_dir, "runs", subprocess, run_type,
                           "run_*", "result", "cross_section.dat")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no cross_section.dat under {pattern}")
    return matches[-1]


def read_point(procdir: str, subprocess: str = "ww",
               runs_dir: str = "out") -> dict:
    """Collect born/virt/real/idip → σ̂_Born, σ̂_NLO (+ errors in quadrature).

    Returns a dict with per-run σ/err and the derived Born & NLO totals [fb].
    """
    import math
    out: dict[str, float] = {}
    errs = []
    for rtype in ("born", "virt", "real", "idip"):
        path = find_latest_result(procdir, rtype, subprocess, runs_dir)
        sig, err = parse_cross_section_dat(path)
        out[f"sigma_{rtype}"] = sig
        out[f"err_{rtype}"] = err
        errs.append(err)
    out["sigma_born"] = out["sigma_born"]
    out["err_born_tot"] = out["err_born"]
    out["sigma_nlo"] = (out["sigma_born"] + out["sigma_virt"]
                        + out["sigma_real"] + out["sigma_idip"])
    out["err_nlo"] = math.sqrt(sum(e * e for e in errs))
    return out
