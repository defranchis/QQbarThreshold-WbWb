"""Per-layout systematic impact of the aemEW nuisance (and cross-check rows),
using the exact scenario_compare definitions + breakdown method the report
systematics table (tab:scenario-syst) is built from.

Run: python scripts/investigations/aemEW_nuisance/scenario_impacts.py
"""
from framework.process.ww.scenario_compare import build_scenarios, _full_syst_breakdown

ROWS = ["stat", "alphas", "aem_isr", "aemEW", "BES", "BEC", "lumi"]


def _grouped(syst, poi, src):
    """Quadrature-combine all breakdown keys belonging to `src` (mirrors
    scenario_compare._grouped_syst): exact name or `src_*` children."""
    import numpy as np
    d = syst.get(poi, {})
    acc, found = 0.0, False
    for k, v in d.items():
        if (k == src or k.startswith(src + "_")) and v is not None and np.isfinite(v):
            acc += v * v
            found = True
    return acc ** 0.5 if found else None


def main():
    scenarios = build_scenarios()
    out = {}
    for name, scn in scenarios.items():
        res = _full_syst_breakdown(scn)
        if res is None:
            print(f"[skip] {name}: breakdown failed")
            continue
        syst, totals, _ = res
        out[name] = (syst, totals)

    names = list(out.keys())
    hdr = f"{'source':12s}" + "".join(f"{n:>16s}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for src in ROWS:
        cells = []
        for n in names:
            syst, _ = out[n]
            m = _grouped(syst, "mass", src)
            w = _grouped(syst, "width", src)
            cells.append(f"{(m or 0):.3f}/{(w or 0):.3f}")
        print(f"{src:12s}" + "".join(f"{c:>16s}" for c in cells))
    # totals
    cells = []
    for n in names:
        _, totals = out[n]
        cells.append(f"{totals.get('mass', 0):.3f}/{totals.get('width', 0):.3f}")
    print(f"{'total exp':12s}" + "".join(f"{c:>16s}" for c in cells))
    print("\n(cells are  sigma_mW / sigma_GammaW  in MeV)")


if __name__ == "__main__":
    main()
