"""Python ctypes interface to the eMELA NLL electron PDF library.

eMELA (Bertone, Cacciari, Frixione, Stagnitto — arXiv:1911.12040) solves
DGLAP in Mellin space at NLL QED accuracy and returns the electron structure
function D(x, Q) in the DELTA factorisation scheme.

This module wraps libeMELApy.so — a thin extern-"C" layer over the C++
eMELA library (scripts/investigations/nll_isr/emela_c_wrapper.cpp).

Public interface
----------------
initialize(pert_order, fac_scheme, ren_scheme, alpha)
    Must be called once before any PDF queries.  Thread-unsafe (global
    eMELA state).  Re-calling with the same arguments is a no-op.

code_pdf(x, omx, Q) -> float
    Returns x * D(x, Q).  Switches to analytic asymptotic form near x→1
    automatically (eMELA's CodePdf internal logic).

ll_pdf(ll_index, x, omx, Q) -> float
    Returns x * D_LL(x, Q) via eMELA's built-in LL radiator.
    ll_index: 0=collinear, 1=BETA, 2=eta, 3=mixed.  Use 1 for BETA scheme
    (matches our current isr.py LL+exp implementation).

alpha_qed(q2) -> float
    Returns alpha_QED at scale squared q2 (for diagnostics only).
"""

from __future__ import annotations

import ctypes
import os

_QQBAR_INSTALL = os.environ.get(
    "QQBAR_INSTALL",
    "/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install",
)
_LIB_PATH = os.path.join(_QQBAR_INSTALL, "lib", "libeMELApy.so")

_lib: ctypes.CDLL | None = None
_init_args: tuple | None = None   # cache of last QuickInitialize call


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"libeMELApy.so not found at {_LIB_PATH}. "
                "Run scripts/investigations/nll_isr/build_emela_wrapper.sh first."
            )
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.emela_quick_initialize.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double,
        ]
        _lib.emela_quick_initialize.restype = None
        _lib.emela_code_pdf.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        _lib.emela_code_pdf.restype = ctypes.c_double
        _lib.emela_ll_pdf.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ]
        _lib.emela_ll_pdf.restype = ctypes.c_double
        _lib.emela_alpha_qed.argtypes = [ctypes.c_double]
        _lib.emela_alpha_qed.restype = ctypes.c_double
    return _lib


def initialize(pert_order: str = "NLL",
               fac_scheme: str = "DELTA",
               ren_scheme: str = "ALGMU",
               alpha: float = 1.0 / 132.1) -> None:
    """Initialise eMELA.  Must be called once before code_pdf / ll_pdf.

    Defaults match the BFS prescription:
      - DELTA factorisation: σ̂ = σ_Born, no ISR collinear subtractions.
      - ALGMU renormalisation: fixed coupling α_Gμ(M_W) ≈ 1/132.1,
        matching BFS arXiv:0707.0773 line 2514.

    For the scheme-variation nuisance, call again with ren_scheme="ALPMZ"
    and alpha = alpha(M_Z) ≈ 1/128.9 (then re-evaluate σ_obs).

    Calling with identical arguments is a no-op (cached).
    """
    global _init_args
    key = (pert_order, fac_scheme, ren_scheme, alpha)
    if key == _init_args:
        return
    lib = _load()
    lib.emela_quick_initialize(
        pert_order.encode(), fac_scheme.encode(), ren_scheme.encode(),
        ctypes.c_double(alpha),
    )
    _init_args = key


def code_pdf(x: float, omx: float, Q: float) -> float:
    """Return x * D_NLL(x, Q) for the electron (PDG id 11).

    Parameters
    ----------
    x   : momentum fraction
    omx : 1 - x  (pass explicitly; eMELA uses omx for numerical accuracy
                  in the x→1 asymptotic switching)
    Q   : factorisation scale in GeV  (pass sqrt(s) for WW ISR)

    Returns
    -------
    float : x * D(x, Q)  — divide by x to get D(x, Q) for the convolution
    """
    if _init_args is None:
        raise RuntimeError("Call emela_wrapper.initialize() before code_pdf()")
    return float(_load().emela_code_pdf(11, float(x), float(omx), float(Q)))


def ll_pdf(ll_index: int, x: float, omx: float, Q: float) -> float:
    """Return x * D_LL(x, Q) via eMELA's built-in LL radiator.

    ll_index=1 selects the BETA scheme (matches isr.py LL+exp implementation).
    Use for LL-closure validation only.
    """
    if _init_args is None:
        raise RuntimeError("Call emela_wrapper.initialize() before ll_pdf()")
    return float(_load().emela_ll_pdf(ll_index, float(x), float(omx), float(Q)))


def alpha_qed(q2: float) -> float:
    """Return alpha_QED at scale squared q2 (diagnostic)."""
    if _init_args is None:
        raise RuntimeError("Call emela_wrapper.initialize() before alpha_qed()")
    return float(_load().emela_alpha_qed(float(q2)))
