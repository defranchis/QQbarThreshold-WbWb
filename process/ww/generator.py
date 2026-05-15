"""WW cross-section generator — PLACEHOLDER.

There is no WW analogue of ``QQbar_threshold/xsec_calc`` in this tree; a
proper implementation (Whizard / RACOONWW / MCFM wrapper, or a dedicated
theory parameterisation) needs to be plugged in here.

This stub provides the same ``file_name`` / ``do_scan`` interface the rest
of the code relies on, so the framework can be exercised end-to-end as
soon as a real ``do_scan`` is dropped in.

File-name convention used by the stub:
    ``<outdir>/WW_<order>_mass<m>_width<w>_asVar<as>_scaleM<sM>_scaleW<sW>.txt``
Adjust to match whatever generator you adopt; the only contract is that
``file_name(values, ...)`` returns a path that ``do_scan(values, ...)``
will populate.
"""

import os


class WWGenerator:
    """Stub generator. Replace :meth:`do_scan` with a real implementation."""

    def __init__(self, *, order=2):
        self.order = order

    # ------------------------------------------------------------------
    # Filenames
    # ------------------------------------------------------------------
    @staticmethod
    def _order_str(order):
        return {0: "LO", 1: "NLO", 2: "NNLO", 3: "N3LO"}.get(order, f"O{order}")

    def file_tag(self, values):
        parts = []
        for name, val in values.items():
            label = "asVar" if name == "alphas" else name
            decimals = 4 if name == "alphas" else 3
            parts.append(f"{label}{val:.{decimals}f}")
        return "_".join(parts)

    def file_name(self, values, *, mass_scale, width_scale, mass_scheme="OS", indir="."):
        body = self.file_tag(values)
        scales = f"scaleM{mass_scale:.1f}_scaleW{width_scale:.1f}"
        return os.path.join(indir, f"WW_{self._order_str(self.order)}_{body}_{scales}.txt")

    # ------------------------------------------------------------------
    # Template production — stub
    # ------------------------------------------------------------------
    def do_scan(self, values, *, mass_scale, width_scale, mass_scheme="OS", outdir="output"):
        raise NotImplementedError(
            "WWGenerator.do_scan is a stub; wire in the actual WW threshold "
            "generator (Whizard / RACOONWW / theory parameterisation) before "
            "running cross-section computation."
        )
