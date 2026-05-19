"""WbWb cross-section generator.

Wraps the existing pybind11 extension ``process.wbwb.xsec_calculator/xsec_calc`` (compiled
from ``ttThresholdScanISR.cpp``) and owns the on-disk file-name convention.

``FitCore`` calls :meth:`file_name` to locate an already-computed template;
:meth:`do_scan` is the entry point used by ``compute_xsec_wbwb.py`` to
produce new templates.
"""

import os

# The pybind11 module is built in place inside the process.wbwb.xsec_calculator/
# directory; importing it works as long as that directory remains on the
# Python path. The compile script lives at process.wbwb.xsec_calculator/compile_calc.sh.
import process.wbwb.xsec_calculator.xsec_calc as _xsec_calc  # type: ignore


# Map from parameter name to the (tag, format) pair used by the C++ side
# when it writes its output filenames.
_FILE_TAG_MAP = {
    "mass":   ("mass",   "{:.2f}"),
    "width":  ("width",  "{:.2f}"),
    "yukawa": ("yukawa", "{:.2f}"),
    "alphas": ("asVar",  "{:.4f}"),  # C++ side calls it "asVar"
}


class WbWbGenerator:
    """File-name convention + ``do_scan`` wrapper for the tt threshold."""

    def __init__(self, *, order=3, isr=True):
        self.order = order
        self.isr = isr

    # ------------------------------------------------------------------
    # Filenames
    # ------------------------------------------------------------------
    @staticmethod
    def _order_str(order):
        return {0: "LO", 1: "NLO", 2: "NNLO", 3: "N3LO"}[order]

    def file_tag(self, values):
        """Build the ``mass171.50_width1.33_...`` portion of the filename."""
        return "_".join(
            f"{label}{fmt.format(values[name])}"
            for name, (label, fmt) in _FILE_TAG_MAP.items()
            if name in values
        )

    def file_name(self, values, *, mass_scale, width_scale, mass_scheme="PS", indir="."):
        scheme = "1S" if mass_scheme == "1S" else "PS"
        isr_tag = "ISR" if self.isr else "noISR"
        body = self.file_tag(values)
        scales = f"scaleM{mass_scale:.1f}_scaleW{width_scale:.1f}"
        return os.path.join(indir, f"{self._order_str(self.order)}_scan_{scheme}_{isr_tag}_{body}_{scales}.txt")

    # ------------------------------------------------------------------
    # Template production
    # ------------------------------------------------------------------
    def do_scan(self, values, *, mass_scale, width_scale, mass_scheme="PS", outdir="output"):
        _xsec_calc.do_scan(
            order=self.order,
            PS_mass=values["mass"],
            width=values["width"],
            mass_scale=mass_scale,
            width_scale=width_scale,
            yukawa=values["yukawa"],
            as_var=values["alphas"],
            outdir=outdir,
            oneS_mass=(mass_scheme == "1S"),
        )
