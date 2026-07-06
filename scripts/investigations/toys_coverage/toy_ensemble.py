"""Toy-ensemble coverage test for the production WW threshold fit.

Everything to date is Asimov; this runs a pseudo-experiment ensemble and
checks that the profiled (m_W, Gamma_W) intervals are Gaussian and cover.

Two fluctuation modes:

* ``stat``  — exactly what ``doFit_ww.py --pseudo`` does today: per-point
  statistical noise only (``FitCore.create_scenario`` with ``asimov=False``,
  persistent ``default_rng`` — NOT the legacy re-seeding path). Nuisance
  truths are NOT fluctuated, so the profiled intervals are expected to
  OVER-cover by sigma_stat/sigma_profiled.
* ``full``  — self-consistent frequentist toys: stat noise PLUS a per-toy
  draw of every penalised nuisance truth (1-D constraints, per-bin +
  correlated binned nuisances incl. the per-point counting rescale, global
  nuisances), injected into the pseudo-data through the fit's own morph
  rows (data generated exactly per the fit model), PLUS a multivariate
  draw of any cov-based xsec systematics. Expected: pull width 1.00,
  coverage 68.3 %.

Configs: ``default`` = the plain doFit_ww fit (POIs + alphas/aem_isr/aemEW
+ lumi nuisance); ``production`` = the --systTable configuration (adds
BEC + BES binned nuisances and the XSEC_SYST cov systematics).

Per-worker RNG: each forkserver worker reseeds ``fit._rng`` (and its own
nuisance rng) from ``base_seed + worker_id`` — otherwise every worker
would inherit seed 42 and draw IDENTICAL toys. Toy uniqueness is asserted
on the first pseudo-data vector of each worker.

Usage:
  PYTHONPATH=.:$PYTHONPATH python3 scripts/investigations/toys_coverage/toy_ensemble.py \
      --config production --fluctuate full --ntoys 400 --workers 16
"""

import argparse
import csv
import os
import sys

import numpy as np

BASE_SEED = 20260702   # overridable via --seed (independent-ensemble checks)


# ---------------------------------------------------------------------------
# Worker: build the fit once, then run a chunk of toys
# ---------------------------------------------------------------------------
_FIT = None
_CFG = None


def _build_fit(config):
    from cards import ww_default as card
    from framework.process.ww.fit import WWFit
    from framework.process.ww.generator import WWGenerator
    gen = WWGenerator.from_card(card)
    fit = WWFit(card, gen, input_dir=card.INPUT_DIRS["nominal"], asimov=False,
                mass_scheme=getattr(card, "MASS_SCHEME", "OS"))
    fit.init_scenario(
        scan_min=card.SCENARIO["scan_min"],
        scan_max=card.SCENARIO["scan_max"],
        scan_step=card.SCENARIO["scan_step"],
        total_lumi=card.SCENARIO["total_lumi"],
        last_lumi=card.SCENARIO["last_lumi"],
        add_last_ecm=False,
    )
    if config == "production":
        fit.add_binned_nuisance("BEC")
        fit.add_binned_nuisance("BES")
        xs = getattr(card, "XSEC_SYST", None)
        if xs:
            fit.set_xsec_systematics(
                corr_frac=xs.get("corr_frac_of_stat", 0.0),
                uncorr_frac=xs.get("uncorr_frac_of_stat", 0.0))
    return fit


def _nuisance_prior_widths(fit):
    """Fit-space Gaussian prior width per parameter (0 for POIs = not
    fluctuated), mirroring FitCore.chi2's penalty terms exactly."""
    widths = np.zeros(len(fit.param_names))
    for name, c in fit._constraints.items():
        if c["active"]:
            widths[fit._idx[name]] = c["sigma"] / fit.parameters.step(name)
    for kind in fit._active_binned_nuisances:
        priors = fit._nuisance_priors[kind]
        prior_u, prior_c = priors["uncorr"], priors["corr"]
        bin_idx = fit._per_kind_bin_idx[kind]
        scale = fit._uncorr_perbin_scale
        if (scale is not None and kind in fit._counting_uncorr_kinds
                and len(scale) == len(bin_idx)):
            widths[bin_idx] = prior_u * scale
        else:
            widths[bin_idx] = prior_u
        widths[fit._idx[kind]] = prior_c
    for kind in fit._active_global_nuisances:
        widths[fit._idx[kind]] = fit._nuisance_priors[kind]["prior"]
    return widths


def _xsec_syst_cov(fit):
    """The cov-based xsec-systematic part of fit.cov (production config),
    reconstructed the way _build_cov adds it; None if not activated."""
    fu = getattr(fit, "xsec_syst_uncorr", 0.0)
    fc = getattr(fit, "xsec_syst_corr", 0.0)
    if not fu and not fc:
        return None
    stat = np.asarray(fit.unc_pseudodata_scenario, dtype=float)
    cov = np.zeros((len(stat), len(stat)))
    if fu:
        cov += np.diag((fu * stat) ** 2)
    if fc:
        cov += np.outer(fc * stat, fc * stat)
    return cov


def _init_worker(config, fluctuate, worker_id, seed=BASE_SEED):
    global _FIT, _CFG
    import matplotlib
    matplotlib.use("Agg")
    _FIT = _build_fit(config)
    # CRITICAL: fresh per-worker stream — otherwise every forked worker
    # inherits default_rng(42) and draws identical toys.
    _FIT._rng = np.random.default_rng(seed + 1000 * worker_id)
    _CFG = dict(fluctuate=fluctuate,
                nuis_rng=np.random.default_rng(seed + 1000 * worker_id + 1))


def _one_toy():
    fit, cfg = _FIT, _CFG
    fit.create_scenario()            # redraw stat noise around pseudo truth
    fit.init_minuit()                # rebuild caches for this toy
    if cfg["fluctuate"] == "full":
        rng = cfg["nuis_rng"]
        widths = _nuisance_prior_widths(fit)
        t = np.where(widths > 0, rng.normal(0.0, np.maximum(widths, 1e-300)), 0.0)
        # inject the nuisance truths through the fit's own morph rows
        factors = np.prod(1 + t[:, None] * fit._morph_matrix, axis=0)
        fit.pseudo_data_scenario = fit.pseudo_data_scenario * factors
        xcov = _xsec_syst_cov(fit)
        if xcov is not None:
            fit.pseudo_data_scenario = fit.pseudo_data_scenario + \
                rng.multivariate_normal(np.zeros(len(xcov)), xcov)
    fit.minuit.migrad()
    valid = bool(fit.minuit.valid)
    res = fit.fit_results(printout=False)
    im, iw = fit._idx["mass"], fit._idx["width"]
    # fit_results returns ABSOLUTE physical values; subtract the pseudo truth
    # (same convention as FitCore._pull_for) to get the per-toy residual.
    m = res[im] - fit.d_params[fit.pseudodata_tag]["mass"]
    w = res[iw] - fit.d_params[fit.pseudodata_tag]["width"]
    first_pd = float(np.sum(fit.pseudo_data_scenario))
    return (valid, m.n * 1e3, m.s * 1e3, w.n * 1e3, w.s * 1e3, first_pd)


def _run_chunk(args):
    config, fluctuate, worker_id, n, seed = args
    _init_worker(config, fluctuate, worker_id, seed)
    return [_one_toy() for _ in range(n)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["default", "production"], default="production")
    ap.add_argument("--fluctuate", choices=["stat", "full"], default="full")
    ap.add_argument("--ntoys", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--outdir", default=os.path.join("plots", "toys_coverage"))
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.config}_{args.fluctuate}"
    if args.seed != BASE_SEED:
        tag += f"_seed{args.seed}"

    import multiprocessing as mp
    ctx = mp.get_context("forkserver")
    per = [args.ntoys // args.workers + (1 if i < args.ntoys % args.workers else 0)
           for i in range(args.workers)]
    jobs = [(args.config, args.fluctuate, i, n, args.seed) for i, n in enumerate(per) if n > 0]
    rows = []
    with ctx.Pool(processes=len(jobs)) as pool:
        for chunk in pool.imap_unordered(_run_chunk, jobs):
            rows.extend(chunk)
            print(f"[toys:{tag}] {len(rows)}/{args.ntoys} done", flush=True)

    rows = np.array(rows, dtype=float)
    valid = rows[:, 0] > 0.5
    n_bad = int((~valid).sum())
    r = rows[valid]
    # toy-uniqueness guard: pseudo-data sums must not repeat. Compare the
    # full-precision checksums — rounding to 6 decimals produced spurious
    # birthday collisions at N≳1000.
    n_dupe = len(r) - len(np.unique(r[:, 5]))

    with open(os.path.join(args.outdir, f"toys_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["valid", "dm_MeV", "sig_m_MeV", "dw_MeV", "sig_w_MeV", "pd_checksum"])
        w.writerows(rows.tolist())

    lines = [f"WW toy ensemble — config={args.config}, fluctuate={args.fluctuate}, "
             f"N={args.ntoys} (invalid: {n_bad}, duplicate pseudo-data: {n_dupe})"]
    for label, dv, sv in (("m_W", r[:, 1], r[:, 2]), ("Gamma_W", r[:, 3], r[:, 4])):
        pull = dv / sv
        n = len(pull)
        mean, std = float(np.mean(pull)), float(np.std(pull, ddof=1))
        mean_err = std / np.sqrt(n)
        std_err = std / np.sqrt(2 * (n - 1))
        cov = float(np.mean(np.abs(dv) <= sv))
        cov_err = np.sqrt(cov * (1 - cov) / n)
        lines.append(
            f"  {label:8s} pull mean = {mean:+.3f} ± {mean_err:.3f}   "
            f"width = {std:.3f} ± {std_err:.3f}   "
            f"coverage(±1σ) = {100*cov:.1f} ± {100*cov_err:.1f} %   "
            f"scatter(Δ) = {np.std(dv, ddof=1):.2f} MeV   "
            f"reported σ = {np.mean(sv):.2f} ± {np.std(sv, ddof=1):.2f} MeV")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.outdir, f"toys_{tag}_summary.txt"), "w") as fh:
        fh.write(text + "\n")

    # pull histograms
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, label, dv, sv in ((axes[0], r"$m_W$", r[:, 1], r[:, 2]),
                              (axes[1], r"$\Gamma_W$", r[:, 3], r[:, 4])):
        pull = dv / sv
        ax.hist(pull, bins=30, range=(-4, 4), histtype="step", color="C0")
        xs = np.linspace(-4, 4, 200)
        binw = 8 / 30
        ax.plot(xs, len(pull) * binw * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi),
                color="C3", lw=1.2)
        ax.set_xlabel(f"pull({label})")
        ax.set_title(f"{label}: width={np.std(pull, ddof=1):.3f}")
    fig.suptitle(f"WW toys — {tag} (N={len(r)})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"toys_{tag}_pulls.png"), dpi=140)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
