"""Diagnostic: σ(e+e- → μνqq̄) vs √s for four ISR scenarios.

All scenarios use Born-level partonic σ (no NLO loops, NNLO, δ_QCD, anchor)
to isolate the ISR treatment.

Scenarios:
  1. Born only       — no ISR convolution
  2. Born + LL ISR   — isr.py 2-leg form (β-scheme LL+exp)
  3. Born + eMELA LL — 2-leg with eMELA LLPDF(β=1) per leg   [emela_ll=True]
  4. Born + eMELA NLL — 2-leg with eMELA CodePdf(NLL/DELTA/ALGMU) [nll=True]

Output panels:
  Upper: σ_obs [fb] vs √s
  Lower: (σ_X / σ_LL_isr.py − 1) × 100 % — relative to isr.py LL as reference

Usage:
  cd <WW_threshold root>
  source setup.sh
  python scripts/investigations/nll_isr/plot_isr_comparison.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress eMELA init banner (written to C-level stdout)
_devnull_fd: int | None = None
_saved_fd:   int | None = None

def _stdout_off() -> None:
    global _devnull_fd, _saved_fd
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_fd   = os.dup(1)
    os.dup2(_devnull_fd, 1)

def _stdout_on() -> None:
    if _saved_fd is not None:
        os.dup2(_saved_fd, 1)
        os.close(_saved_fd)
        os.close(_devnull_fd)

_stdout_off()
from framework.process.ww.xsec_calculator import emela_wrapper as em
from framework.process.ww.xsec_calculator.isr import (
    sigma_ISR_2leg_convolution,
    _DEFAULT_ISR_ALPHA,
)
from framework.process.ww.xsec_calculator.eft_xsec import (
    sigma_partonic_munuqq,
    M_W_DEFAULT,
    GAMMA_W_DEFAULT,
)
em.initialize("NLL", "DELTA", "ALGMU", _DEFAULT_ISR_ALPHA)
_stdout_on()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MW    = M_W_DEFAULT
GW    = GAMMA_W_DEFAULT
X_MIN = float(np.sqrt(0.10))   # per-leg lower bound: z_min = x_min^2 = 0.10
N_QUAD = 64                     # per-leg GL nodes (64² = 4096 σ̂ calls per √s)

# kwargs forwarded to sigma_partonic_munuqq (Born level, no corrections)
BORN_KW = dict(
    include_NLO_hard_decay=False,
    include_BFS_NNLO=False,
    apply_delta_QCD=False,
    apply_whizard_anchor=False,
    br_convention="pdg-constant",
)

SQ_GRID = np.arange(157.0, 170.25, 0.25)

# ---------------------------------------------------------------------------
# Compute the four scenarios — all via framework functions
# ---------------------------------------------------------------------------
print(f"√s range: {SQ_GRID[0]:.1f}–{SQ_GRID[-1]:.1f} GeV  ({len(SQ_GRID)} points)",
      flush=True)

print("1/4  Born (no ISR) ...", flush=True)
sig_born = np.array([
    sigma_partonic_munuqq(sq ** 2, MW, GW, **BORN_KW)
    for sq in SQ_GRID
], dtype=float)

ISR_KW = dict(mW=MW, gammaW=GW, x_min=X_MIN, n_quad=N_QUAD, **BORN_KW)

print("2/4  Born + LL ISR (isr.py 2-leg β-scheme) ...", flush=True)
sig_ll = sigma_ISR_2leg_convolution(
    SQ_GRID, sigma_partonic_munuqq, **ISR_KW)

print("3/4  Born + LL ISR (eMELA LLPDF) ...", flush=True)
sig_emela_ll = sigma_ISR_2leg_convolution(
    SQ_GRID, sigma_partonic_munuqq, emela_ll=True, **ISR_KW)

print("4/4  Born + NLL ISR (eMELA CodePdf) ...", flush=True)
sig_emela_nll = sigma_ISR_2leg_convolution(
    SQ_GRID, sigma_partonic_munuqq, nll=True, **ISR_KW)

# ---------------------------------------------------------------------------
# n_jobs consistency check: serial vs parallel at 3 reference points
# ---------------------------------------------------------------------------
print("\nn_jobs consistency check (n_jobs=6 vs n_jobs=1) ...", flush=True)
CHECK_PTS = np.array([158.0, 161.0, 163.0])
sig_nll_serial = sigma_ISR_2leg_convolution(
    CHECK_PTS, sigma_partonic_munuqq, nll=True, n_jobs=1, **ISR_KW)
sig_nll_parallel = sigma_ISR_2leg_convolution(
    CHECK_PTS, sigma_partonic_munuqq, nll=True, n_jobs=6, **ISR_KW)
max_reldiff = np.max(np.abs(sig_nll_parallel / sig_nll_serial - 1))
print(f"  max |parallel/serial − 1| = {max_reldiff:.2e}  "
      f"({'PASS' if max_reldiff < 1e-12 else 'FAIL'})", flush=True)

# ---------------------------------------------------------------------------
# Numerical summary at key √s points
# ---------------------------------------------------------------------------
check_pts = [158.0, 161.0, 163.0]
print("\n  √s   | Born [fb]  | +LL isr.py | +eMELA LL  | +eMELA NLL | NLL/isr.py−1")
print("  " + "-" * 80)
for sq_c in check_pts:
    idx  = np.argmin(np.abs(SQ_GRID - sq_c))
    b    = sig_born[idx]      * 1e3
    ll   = sig_ll[idx]        * 1e3
    ell  = sig_emela_ll[idx]  * 1e3
    nll  = sig_emela_nll[idx] * 1e3
    diff = (nll / ll - 1) * 100.0
    print(f"  {SQ_GRID[idx]:5.2f}  | {b:10.4f} | {ll:10.4f} | {ell:10.4f} | {nll:10.4f} | {diff:+.4f} %")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 8),
    gridspec_kw={"height_ratios": [2, 1]},
    sharex=True,
)
fig.subplots_adjust(hspace=0.05)

kw_born = dict(color="black",      ls="--", lw=1.5,
               label="Born (no ISR)")
kw_ll   = dict(color="tab:blue",   ls="-",  lw=1.8,
               label=r"Born + LL ISR (isr.py $\beta$-scheme, 2-leg)")
kw_ell  = dict(color="tab:orange", ls="-",  lw=1.8,
               label="Born + LL ISR (eMELA LLPDF, 2-leg)")
kw_nll  = dict(color="tab:red",    ls="-",  lw=1.8,
               label="Born + NLL ISR (eMELA CodePdf, DELTA+ALGMU)")

fb = 1e3  # pb → fb

ax1.plot(SQ_GRID, sig_born      * fb, **kw_born)
ax1.plot(SQ_GRID, sig_ll        * fb, **kw_ll)
ax1.plot(SQ_GRID, sig_emela_ll  * fb, **kw_ell)
ax1.plot(SQ_GRID, sig_emela_nll * fb, **kw_nll)
ax1.set_ylabel(r"$\sigma_{\rm obs}$  [fb]", fontsize=13)
ax1.legend(fontsize=11, loc="upper left")
ax1.set_title(
    r"$e^+e^-\!\to\!\mu\nu q\bar{q}$: ISR scheme comparison"
    "\n(Born-level partonic; inclusive; pdg-constant BR; no NLO/NNLO/δ_QCD/anchor)",
    fontsize=11,
)
ax1.grid(True, alpha=0.3)

ref = sig_ll
ax2.axhline(0, color="tab:blue", ls="-", lw=1.2, label=r"LL ISR (isr.py) [ref]")
ax2.plot(SQ_GRID, (sig_emela_ll  / ref - 1) * 100, **kw_ell)
ax2.plot(SQ_GRID, (sig_emela_nll / ref - 1) * 100, **kw_nll)
ax2.set_ylabel(
    r"$(\sigma_X\,/\,\sigma_{\rm LL,isr.py})\,-\,1$  [%]", fontsize=12)
ax2.set_xlabel(r"$\sqrt{s}$  [GeV]", fontsize=13)
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(True, alpha=0.3)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "plots")
OUT    = os.path.join(OUTDIR, "isr_comparison.pdf")
fig.savefig(OUT, bbox_inches="tight")
print(f"\nSaved: {OUT}", flush=True)
