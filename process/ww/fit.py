"""WW-specific fit subclass — PLACEHOLDER.

For now this is just ``FitCore`` re-exported under a WW-flavoured name;
add any WW-specific hooks (e.g. mW / mZ correlations, ΓW SM-prediction
constraint) here when the analysis requires them.
"""

from common.fit_core import FitCore


class WWFit(FitCore):
    pass
