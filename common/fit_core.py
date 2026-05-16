"""Generic chi2 / Minuit fit machinery for threshold-scan lineshape fits.

The original ``doFit.py`` rolled cross-section reading, smearing, morphing,
scenario building, the chi2 itself, Minuit, every scan, every plot, and the
systematic-table machinery into a single 1450-line class. This module keeps
the central state in one place (``FitCore``) but delegates two concerns to
collaborators:

  * **the input cross sections** — the ``generator`` parameter is any object
    that exposes ``file_name(values, scales)`` returning the path of an
    already-computed template; this is the only place the WbWb/WW theory
    plumbing leaks in.
  * **the steering values** — a ``card`` module (typically
    ``cards.wbwb_default`` / ``cards.ww_default``) supplies every magic
    number. ``FitCore`` reads from ``card`` but stores per-fit overrides on
    ``self`` so that the same card can drive multiple fits.

Process-specific quirks (the SM-width Yukawa-mass relation, mass-scheme
file-name tweaks) are exposed as hooks on subclasses.
"""

import copy
import os
import sys

import iminuit
import numpy as np
import pandas as pd
import scipy
import uncertainties as unc
from scipy.linalg import cho_factor, cho_solve

from common.parameters import Parameters
from common.smearing import convolute_gauss


def ecm_to_str(ecm):
    return f"{ecm:.1f}"


def quadrature_subtract(total, partial):
    """Return ``sqrt(total**2 - partial**2)`` (clipped at zero), elementwise.

    Works for both scalar and array inputs; the "impact in quadrature" pattern
    that recurs across scans and the syst table.
    """
    total = np.asarray(total)
    partial = np.asarray(partial)
    return np.sqrt(np.maximum(total ** 2 - partial ** 2, 0))


# ---------------------------------------------------------------------------
# Defensive lower bound on the BEC / BES / sw2 priors used in chi2. The
# floor is reached in two real scenarios:
#   - ``reinitialise_to_stat`` (and the ``_TURN_OFF`` map in
#     ``systematics.py``) write ``_OFF = 1e-10`` into the priors for the
#     syst-table flow. The floor brings them to 1e-6 — the chi2 penalty is
#     enormous either way and the parameter is pinned, but the Hessian is
#     better-conditioned at 1e-6.
#   - ``_scan_nuisance(BEC)`` at its leftmost grid point (v = 1e-6) yields
#     a BEC prior of 1e-7 after dividing by ``input_var["BEC"] = 10``.
#     Without the floor the BEC-bin Hessian rows reach ~1e14, close to the
#     conditioning threshold of the cov inversion. (BES doesn't hit this:
#     ``input_var["BES"] = 0.1`` keeps the prior at 1e-5 > floor.)
# alpha_s and Yukawa aren't floored: their _OFF values are reached only in
# one-dimensional penalty terms where the cov stays well-conditioned.
# ---------------------------------------------------------------------------
_PRIOR_FLOOR = 1.0e-6
_OFF = 1.0e-10


def _warn_if_invalid(minuit):
    """Print a one-line stderr warning when migrad did not converge or hesse
    failed. Consumers (``fit_params_with_cov`` / ``results_from_minuit``) call
    this before reading ``minuit.covariance`` — otherwise the post-fit
    uncertainties propagate silently from a garbage covariance.

    Warn-and-continue (rather than raise) because scans that sweep across
    edge cases (e.g. ``scan_lumi`` at ``lumi=0``) routinely hit non-converged
    fits and still need to plot the rest of the curve.
    """
    if not minuit.valid:
        print(f"WARNING: minuit fit not valid (fval={minuit.fval:.3g}); "
              "uncertainties may be unreliable", file=sys.stderr)


class FitCore:
    """Generic threshold-scan fit.

    Construction goes through three explicit steps; each populates a distinct
    family of attributes that the later steps (and ``chi2`` / scans / the
    syst table) consume. Skipping or reordering them is a setup bug.

    1. ``__init__(card, generator, ...)`` — read the card, build the
       parameter grid, load and smear and morph the input cross-section
       templates. Populates ``param_names``, ``parameters``, ``d_params``,
       ``xsec_dict``, ``xsec_dict_smeared``, ``morph_dict``, ``_bec_raw``,
       ``_sw2_raw``, plus card-derived scalars (scales, priors, BES, lumi,
       last_ecm). Nuisance toggles (``bec_nuisances``, ``bes_nuisances``,
       ``sw2_nuisance``) start False.

    2. ``init_scenario(...)`` (or ``init_scenario_custom``) — pick the ecm
       grid, total lumi, optional above-threshold point, and Asimov /
       pseudo-data flavour. Populates ``scenario_dict``, ``scenario``,
       ``xsec_scenario``, ``scale_var_scenario``, ``pseudo_data_scenario``,
       ``unc_pseudodata_scenario``, ``morph_scenario``. Between this step
       and step 3 the ``add_bec_nuisances`` / ``add_bes_nuisances`` /
       ``add_sw2_nuisance`` methods may extend ``param_names`` and
       ``morph_scenario`` with nuisance bins.

    3. ``init_minuit(...)`` — invoked implicitly by ``fit_parameters`` and
       ``update``. Builds the chi2 hot-path caches (``cov``, ``_cov_factor``,
       ``lumi_uncorr_ecm``, ``_idx``, ``_bec_bin_idx``, ``_bes_bin_idx``,
       ``_xsec_base``, ``_morph_matrix``) and the Minuit instance
       ``minuit``. After this step ``chi2`` is callable and
       ``fit_parameters`` / ``fit_results`` work.

    ``update()`` re-runs smearing, morphing, scenario building, and
    ``init_minuit``; call it after mutating any input that feeds the chi2
    cache.

    Subclass per process and override :meth:`physical_fit_params` (and any
    other hook) for process-specific constraints between fit parameters.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        card,
        generator,
        *,
        input_dir=None,
        sm_width=False,
        asimov=True,
        constrain_yukawa=False,
        read_scale_vars=False,
        mass_scheme=None,
        shift_scan=False,
        legacy_pseudo_rng=False,
        debug=False,
    ):
        self.card = card
        self.generator = generator
        self.debug = debug
        self.asimov = asimov
        self.read_scale_vars = read_scale_vars
        self.sm_width = sm_width
        self.constrain_yukawa = constrain_yukawa
        # When the scan list is going to be shifted off the original grid
        # (see scan_shift) the BEC/BES/sw2 templates would have to be
        # interpolated; skip building them entirely instead.
        self.shift_scan = shift_scan
        # Pseudo-data RNG. Default: one PCG64 stream seeded with 42, advanced
        # per ``create_scenario`` call so repeated scenario builds get
        # independent noise realisations. ``legacy_pseudo_rng=True`` falls
        # back to the historical pattern (global ``np.random.seed(42)`` then
        # ``np.random.normal``, every call) which gives every call the *same*
        # noise — kept only for byte-reproducing pre-fix pseudo runs.
        self.legacy_pseudo_rng = legacy_pseudo_rng
        self._rng = None if legacy_pseudo_rng else np.random.default_rng(42)

        # Mass scheme & input/plot directories ------------------------------
        self.mass_scheme = mass_scheme if mass_scheme is not None else card.MASS_SCHEME
        is_1s = self.mass_scheme == "1S"
        if input_dir is None:
            if is_1s:
                input_dir = card.INPUT_DIRS["scale_1S"] if read_scale_vars else card.INPUT_DIRS["nominal_1S"]
            else:
                input_dir = card.INPUT_DIRS["scale_vars"] if read_scale_vars else card.INPUT_DIRS["nominal"]
        self.input_dir = input_dir
        self.plot_dir = card.PLOT_DIR_1S if is_1s else card.PLOT_DIR

        # Parameters --------------------------------------------------------
        card_params = card.PARAMETERS_1S if is_1s else card.PARAMETERS
        self.parameters = Parameters(
            card_params,
            scale_vars=card.RENORM_SCALES["vars"] if read_scale_vars else [],
        )
        self.d_params = self.parameters.as_dict()
        self.param_names = list(self.parameters.names)
        self.pseudodata_tag = "pseudodata" if not self.sm_width else "mass_var"

        # Renormalisation scales -------------------------------------------
        scales = card.RENORM_SCALES
        if read_scale_vars:
            self.mass_scale = scales["alt"]["mass"]
            self.width_scale = scales["alt"]["width"]
        else:
            self.mass_scale = scales["mass"]
            self.width_scale = scales["width"]
        self.scale_vars = self.parameters.scale_vars

        # Beam-energy spectrum ---------------------------------------------
        self.beam_energy_res = card.BEAM_ENERGY_RES
        self.smear_xsec = card.SMEAR_XSEC
        self.peak_ecm = card.PEAK_ECM
        self.last_ecm = card.LAST_ECM

        # Priors -----------------------------------------------------------
        priors = card.PRIORS
        self.input_uncert_SM_width = priors.get("SM_width", {}).get("default", _OFF)
        lumi = priors["lumi"]
        self.lumi_uncorr = lumi["uncorr"]
        self.lumi_corr = lumi["corr"]
        self.input_var = card.INPUT_VAR

        # 1-D Gaussian-constraint registry, built from card.CONSTRAINTS.
        # Each entry holds its current sigma (mutable for the syst-table
        # flow via the legacy `input_uncert_X` property shims) and centre
        # (resolved once from the pseudodata "true" value if the card
        # doesn't specify one). Entries whose parameter is not part of
        # this fit (e.g. yukawa for WW) are skipped. The yukawa entry is
        # gated by ``self.constrain_yukawa`` in ``chi2`` until commit 7
        # generalises CLI gating.
        self._constraints = {}
        for name, spec in card.CONSTRAINTS.items():
            if name not in self.parameters.names:
                continue
            self._constraints[name] = {
                "sigma":     spec["sigma"],
                "center":    spec.get("center", self.d_params[self.pseudodata_tag][name]),
                "always_on": spec["always_on"],
            }

        # Nuisance toggles -------------------------------------------------
        self.bec_nuisances = False
        self.bes_nuisances = False
        self.sw2_nuisance = False
        self.bec_prior_uncorr = self.bec_prior_corr = None
        self.bes_prior_uncorr = self.bes_prior_corr = None
        self.sw2_prior = None
        # ``param_idx -> (kind, bin_idx_in_morph_scenario[kind])`` for per-bin
        # nuisance params registered by ``_expand_per_bin_nuisance``. Single
        # source of truth — readers don't re-parse the "BEC_bin{i}" name.
        self._per_bin_meta = {}

        # chi2 hot-path caches — populated by init_minuit. Listed here so
        # that AttributeError-style failures from calling chi2 before
        # init_minuit show a clear "this attribute is None" instead.
        self._idx = None
        self._bec_bin_idx = None
        self._bes_bin_idx = None
        self._xsec_base = None
        self._morph_matrix = None
        self._cov_factor = None

        # Build templates --------------------------------------------------
        if debug:
            print(f"Input directory: {self.input_dir}")
            print(f"Parameters: {self.param_names}")
            print(f"Beam energy resolution: {self.beam_energy_res}")
            print(f"Smear cross sections: {self.smear_xsec}")
            print(f"Constrain width to SM value: {self.sm_width}")
            print(f"Constrain Yukawa: {self.constrain_yukawa}")
            print(f"Asimov fit: {self.asimov}")

        self._read_cross_sections()
        self._read_aux_templates()
        self._smear_cross_sections()
        self._morph_cross_sections()
        if debug:
            print("Initialization done")

    def _read_aux_templates(self):
        """Pre-load raw BEC and sw2 template DataFrames once.

        These are re-smeared on every ``update()`` (e.g. when beam_energy_res
        changes in scan_beam_resolution), but the on-disk content never
        changes — cache the raw read so we don't re-touch the filesystem
        on every iteration.
        """
        self._bec_raw = None
        self._sw2_raw = None
        if self.read_scale_vars or self.mass_scheme == "1S" or self.shift_scan:
            return
        bec = self._scan_for_tag(
            "nominal",
            indir=os.path.join(self.card.INPUT_DIRS["BEC"], self._bec_var_dir(self.input_var["BEC"])),
        )
        # BEC variation templates are sampled at ECMs shifted by ±input_var
        # MeV (10 MeV here). Snap them back to the nominal 0.1-GeV grid so
        # they line up positionally with the nominal template (the morph
        # computation divides per-row). Constraint: variations must stay
        # <= 40 MeV — beyond that the one-decimal rounding would alias onto
        # the next nominal ECM bin (banker's rounding at .05).
        bec["ecm"] = bec["ecm"].round(1)
        self._bec_raw = bec
        self._sw2_raw = self._scan_for_tag("nominal", indir=self.card.INPUT_DIRS["sw2"])

    # ------------------------------------------------------------------
    # Copy semantics — the steering card is a module and can't be pickled,
    # so deepcopy must skip it (modules are stateless from our POV).
    # ------------------------------------------------------------------
    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k, v in self.__dict__.items():
            if k == "card":
                clone.__dict__[k] = v
            else:
                clone.__dict__[k] = copy.deepcopy(v, memo)
        return clone

    # ------------------------------------------------------------------
    # Compatibility shims for the legacy ``input_uncert_X`` / ``X_center``
    # attribute names. They proxy to ``self._constraints[X]``. Used by
    # ``scan_alphas`` / ``scan_yukawa_constraint`` / ``systematics._TURN_OFF`` /
    # ``reinitialise_to_*`` until commits 3 and 7 rewrite those paths to
    # use ``self._constraints`` directly.
    # ------------------------------------------------------------------
    @property
    def input_uncert_alphas(self):
        return self._constraints["alphas"]["sigma"]

    @input_uncert_alphas.setter
    def input_uncert_alphas(self, v):
        self._constraints["alphas"]["sigma"] = v

    @property
    def input_uncert_yukawa(self):
        if "yukawa" in self._constraints:
            return self._constraints["yukawa"]["sigma"]
        return _OFF

    @input_uncert_yukawa.setter
    def input_uncert_yukawa(self, v):
        if "yukawa" in self._constraints:
            self._constraints["yukawa"]["sigma"] = v
        # silently dropped when yukawa is not a constraint (e.g. the WW card)

    @property
    def alphas_center(self):
        return self._constraints["alphas"]["center"]

    @property
    def yukawa_center(self):
        if "yukawa" in self._constraints:
            return self._constraints["yukawa"]["center"]
        return 0.0

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def physical_fit_params(self, params):
        """Hook for process-specific cross-parameter relations.

        Returns ``(resolved_params, prior_extra)``:

        * ``resolved_params`` — the parameter vector after any
          inter-parameter relations have been resolved (e.g. for WbWb with
          ``--SMwidth``, the width entry is overwritten with the value
          derived from mass + floating theory_knob).
        * ``prior_extra`` — extra Gaussian-prior contribution to chi² beyond
          what ``FitCore.chi2`` already accounts for (typically a prior on
          the otherwise-unconstrained theory knob).

        Callers (``chi2``, ``fit_params_with_cov``, ``results_from_minuit``)
        pre-copy ``params`` so subclass overrides are free to mutate it in
        place — they never touch Minuit's own state array.

        Default: no relations to resolve, no extra prior.
        """
        return params, 0.0

    # ------------------------------------------------------------------
    # File I/O & templates
    # ------------------------------------------------------------------
    def read_xsec(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cross-section template not found: {path}")
        with open(path) as fh:
            return pd.read_csv(fh, header=None, names=["ecm", "xsec"])

    def _scan_for_tag(self, tag, mass_scale=None, width_scale=None, indir=None):
        if mass_scale is None:
            mass_scale = self.mass_scale
        if width_scale is None:
            width_scale = self.width_scale
        if indir is None:
            indir = self.input_dir
        values = self.d_params[tag]
        path = self.generator.file_name(
            values, mass_scale=mass_scale, width_scale=width_scale,
            mass_scheme=self.mass_scheme, indir=indir,
        )
        return self.read_xsec(path)

    def _read_cross_sections(self):
        self.xsec_dict = {tag: self._scan_for_tag(tag) for tag in self.parameters.tags}
        self.l_ecm = [ecm_to_str(e) for e in self.xsec_dict[self.parameters.tags[0]]["ecm"]]
        if not self.read_scale_vars:
            return
        for scale in self.scale_vars:
            for which, kw in (("M", "mass_scale"), ("W", "width_scale")):
                xsec = self._scan_for_tag("nominal", **{kw: scale})
                if len(xsec) < len(self.l_ecm) - 1:
                    print(f"Warning: scale{which}={scale} truncated, skipping")
                    continue
                self.xsec_dict[f"scale{which}_{scale:.1f}"] = xsec

    # ------------------------------------------------------------------
    # Smearing
    # ------------------------------------------------------------------
    def smear(self, xsec, bes=None):
        if not self.smear_xsec:
            return xsec
        if bes is None:
            bes = self.beam_energy_res
        last_ecm_xsec = ecm_to_str(float(xsec["ecm"].iloc[-1]))
        last_is_overflow = last_ecm_xsec == ecm_to_str(self.last_ecm)
        body = xsec[:-1] if last_is_overflow else xsec
        smeared = convolute_gauss(body, bes, peak_ecm=self.peak_ecm)
        if last_is_overflow:
            smeared = pd.concat([smeared, xsec[-1:]])
        return smeared

    def _smear_cross_sections(self):
        self.xsec_dict_smeared = {tag: self.smear(xsec) for tag, xsec in self.xsec_dict.items()}

    def template(self, tag="nominal"):
        return self.xsec_dict_smeared[tag]

    # ------------------------------------------------------------------
    # Morphing
    # ------------------------------------------------------------------
    def _morph_one(self, param):
        """Return DataFrame with the relative variation (xsec_var/xsec_nom - 1)."""
        xsec_nom = self.template()
        if param == "BEC":
            xsec_var = self.smear(self._bec_raw)
        elif param == "BES":
            xsec_var = self.smear(self.xsec_dict["nominal"], bes=self.beam_energy_res * (1 + self.input_var["BES"]))
        elif param == "sw2":
            xsec_var = self.smear(self._sw2_raw)
            self.xsec_dict_smeared["sw2_var"] = xsec_var
        else:
            xsec_var = self.template(f"{param}_var")
        return pd.DataFrame({"ecm": xsec_nom["ecm"], "xsec": xsec_var["xsec"] / xsec_nom["xsec"] - 1})

    def _morph_cross_sections(self):
        self.morph_dict = {p: self._morph_one(p) for p in self.param_names}
        if not self.read_scale_vars and self.mass_scheme != "1S" and not self.shift_scan:
            for extra in ("BEC", "BES", "sw2"):
                self.morph_dict[extra] = self._morph_one(extra)

    @staticmethod
    def _bec_var_dir(var):
        return f"scan_p{var:.0f}" if var >= 0 else f"scan_m{abs(var):.0f}"

    @staticmethod
    def _bec_dir_to_var(directory):
        s = directory.replace("scan_p", "").replace("scan_m", "")
        return float(s) * (-1 if directory.startswith("scan_m") else 1)

    # ------------------------------------------------------------------
    # Parameter <-> value conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _is_bin_nuisance(name):
        """True for BEC/BES per-bin or correlated parameters — those whose
        fit-space value is the parameter itself (no nominal+step rescaling)."""
        return name in ("BEC", "BES") or name.startswith(("BEC_bin", "BES_bin"))

    def value_from_param(self, par, name):
        if name == "sw2":
            return par * self.input_var["sw2"]
        if self._is_bin_nuisance(name):
            return par
        return self.d_params["nominal"][name] + par * self.parameters.step(name)

    def param_from_value(self, val, name):
        return (val - self.d_params["nominal"][name]) / self.parameters.step(name)

    # ------------------------------------------------------------------
    # Scenario
    # ------------------------------------------------------------------
    def init_scenario(self, scan_min, scan_max, scan_step, total_lumi, last_lumi,
                      add_last_ecm, same_evts=False):
        scan_list = [ecm_to_str(e) for e in np.arange(scan_min, scan_max + scan_step / 2, scan_step)]
        self.init_scenario_custom(scan_list, total_lumi, last_lumi, add_last_ecm, same_evts)

    def init_scenario_custom(self, scan_list, total_lumi, last_lumi, add_last_ecm, same_evts=False):
        if self.constrain_yukawa and add_last_ecm:
            raise ValueError(
                "Yukawa constraint + last-ecm point unsupported; "
                "pass constrain_yukawa=False (--fitYukawa) to float Yukawa, "
                "or set add_last_ecm=False (drop --lastecm) to skip the above-threshold point."
            )
        self.scenario_dict = {
            "scan_list": scan_list,
            "total_lumi": total_lumi,
            "last_lumi": last_lumi,
            "add_last_ecm": add_last_ecm,
            "same_evts": same_evts,
        }
        self.create_scenario(**self.scenario_dict)

    def create_scenario(self, scan_list, total_lumi, last_lumi, add_last_ecm,
                        same_evts, init_vars=True, pseudodata=None):
        if self.debug:
            print("Creating threshold scan scenario")

        scenario_dict = {k: total_lumi / len(scan_list) for k in scan_list}
        if add_last_ecm:
            scenario_dict[ecm_to_str(self.last_ecm)] = last_lumi
        for ecm in scenario_dict:
            if ecm not in self.l_ecm:
                raise ValueError(f"Scenario ecm {ecm} not in template ecm list")
        self.scenario = dict(sorted(scenario_dict.items(), key=lambda kv: float(kv[0])))

        self.xsec_scenario = self.slice_to_scenario(self.template())
        if init_vars:
            self.scale_var_scenario = np.ones(len(self.xsec_scenario["xsec"]))

        if pseudodata is None:
            pseudodata = self.template(self.pseudodata_tag)
        self.pseudo_data_scenario = self.slice_to_scenario(pseudodata)["xsec"]

        if same_evts:
            overall_factor = total_lumi / np.sum([1 / sigma for sigma in self.pseudo_data_scenario])
            self.scenario = {
                ecm: overall_factor / sigma
                for ecm, sigma in zip(self.scenario.keys(), self.pseudo_data_scenario)
            }

        unc_stat = (np.array(self.pseudo_data_scenario) / np.array(list(self.scenario.values()))) ** 0.5
        unc_stat *= self.card.SCENARIO["stat_inflation"]
        self.unc_pseudodata_scenario = unc_stat
        if not self.asimov:
            if self.legacy_pseudo_rng:
                # Historical (buggy) behaviour: re-seed the global MT19937
                # stream on every call, so every create_scenario draws the
                # same noise. See __init__ docstring.
                np.random.seed(42)
                self.pseudo_data_scenario = np.random.normal(
                    self.pseudo_data_scenario, self.unc_pseudodata_scenario)
            else:
                self.pseudo_data_scenario = self._rng.normal(
                    self.pseudo_data_scenario, self.unc_pseudodata_scenario)

        self.morph_scenario = {p: self.slice_to_scenario(self.morph_dict[p]) for p in self.param_names}
        for extra in ("BEC", "BES", "sw2"):
            if extra in self.morph_dict:
                self.morph_scenario[extra] = self.slice_to_scenario(self.morph_dict[extra])

    def slice_to_scenario(self, df):
        """Select the rows of ``df`` whose ECM is in ``self.scenario``.

        Comparison is on the canonical one-decimal string form
        (``ecm_to_str`` applied to both sides), not on raw float values —
        robust to changes in the format string and to slightly-shifted
        ECM templates (e.g. the BEC variations are loaded at
        340.01 / 340.11 / ... and snap to ``"340.0"`` / ``"340.1"`` / ...
        via the formatter)."""
        keep = set(self.scenario.keys())
        return df[df["ecm"].map(ecm_to_str).isin(keep)]

    # ------------------------------------------------------------------
    # Chi2 + Minuit
    # ------------------------------------------------------------------
    def chi2(self, params):
        """Chi2 evaluated by Minuit. Requires :meth:`init_minuit` to have
        run — it consumes pre-built caches (``_cov_factor``, ``_morph_matrix``,
        ``_xsec_base``, ``_idx``) that don't exist until init_minuit
        constructs them.

        **Morphing convention.** The model cross section is built as the
        nominal lineshape times a product of per-parameter shape factors:
        ``th_xsec = _xsec_base · Π_i (1 + p_i · morph_i)``. This assumes
        the per-parameter variations combine multiplicatively, which is
        exact only when cross-terms ``p_i · p_j · morph_i · morph_j`` are
        negligible — i.e. in the small-variation regime where each
        ``p_i · morph_i`` stays well below unity. Good for the WbWb /
        WW threshold fits where the floating params sit near zero, but
        the linearity assumption is implicit; deviations from it would
        show up as a non-quadratic chi² far from the minimum.

        The αₛ constraint (always on) and the Yukawa constraint (under
        ``constrain_yukawa=True``) are Gaussian penalties centred at
        ``self.alphas_center`` / ``self.yukawa_center``. By default these
        are the pseudodata "true" values — Asimov-self-consistent (fit, data,
        and constraint all sit at the pseudo point so no bias on the fit
        minimum). Set ``card.PRIORS["alphas"]["center"]`` /
        ``["yukawa"]["center"]`` to override (e.g. SM-centred = 0.1184 / 1.0
        for real-data analysis or bias studies). Read the resulting Asimov
        uncertainty as "achievable resolution with an external constraint
        of that width centred at the chosen value".
        """
        # physical_fit_params runs first because subclasses (e.g. WbWbFit
        # with SM_width) resolve cross-parameter relations on the dependent
        # entries (e.g. width). Pre-copy so the override can mutate without
        # touching Minuit's state array; the template uses the resolved
        # values, the prior terms below stay on the raw free params.
        resolved_params, prior_extra = self.physical_fit_params(params.copy())
        th_xsec = self._xsec_base * np.prod(1 + resolved_params[:, None] * self._morph_matrix, axis=0)

        res = self.pseudo_data_scenario - th_xsec
        chi2_val = float(res @ cho_solve(self._cov_factor, res))

        # 1-D Gaussian constraints (driven by card.CONSTRAINTS via
        # self._constraints). The yukawa entry is gated by
        # self.constrain_yukawa as long as the entry script wires
        # --fitYukawa that way (transitional; commit 7 generalises CLI
        # gating).
        for name, c in self._constraints.items():
            if not c["always_on"] and name == "yukawa" and not self.constrain_yukawa:
                continue
            sigma_fs = c["sigma"] / self.parameters.step(name)
            chi2_val += ((params[self._idx[name]]
                          - self.param_from_value(c["center"], name)) / sigma_fs) ** 2

        if self.bec_nuisances:
            chi2_val += self._nuisance_prior(params, "BEC")
        if self.bes_nuisances:
            chi2_val += self._nuisance_prior(params, "BES")
        if self.sw2_nuisance:
            chi2_val += (params[self._idx["sw2"]] / max(self.sw2_prior, _PRIOR_FLOOR)) ** 2

        return chi2_val + prior_extra

    def _nuisance_prior(self, params, kind):
        prior_u = max(getattr(self, f"{kind.lower()}_prior_uncorr"), _PRIOR_FLOOR)
        prior_c = max(getattr(self, f"{kind.lower()}_prior_corr"), _PRIOR_FLOOR)
        bin_idx = self._bec_bin_idx if kind == "BEC" else self._bes_bin_idx
        bin_params = params[bin_idx]
        corr_idx = self._idx[kind]
        return float(np.sum((bin_params / prior_u) ** 2) + (params[corr_idx] / prior_c) ** 2)

    def init_minuit(self):
        self._build_cov()
        self._build_chi2_caches()
        self.minuit = iminuit.Minuit(self.chi2, np.zeros(len(self.param_names)), name=self.param_names)
        self.minuit.errordef = 1

    def _build_cov(self):
        """Rebuild ``cov`` / ``_cov_factor`` / ``lumi_uncorr_ecm`` from the
        current ``lumi_uncorr`` / ``lumi_corr`` / scenario state.

        ``lumi_uncorr_ecm`` is the per-ECM uncorr lumi-uncertainty array.
        Threshold points all get the card-level ``lumi.uncorr`` figure (it
        is by convention a per-ECM-point fractional uncertainty). When
        ``add_last_ecm=True``, the above-threshold entry is divided by
        ``sqrt(factor_above)``, where ``factor_above`` is the ratio of the
        above-threshold-point lumi to the per-threshold lumi — so the
        larger lumi at that point gives a proportionally smaller per-point
        uncertainty.

        TODO: re-introduce a way to rescale ``lumi.uncorr`` when the
        scenario's total lumi / N differs from the calibration assumption.
        The previous ``scale_uncorr`` card flag did this by multiplying by
        sqrt(N_threshold) — it was removed because it was dead code (always
        False) and its semantics weren't pinned down. A proper rewrite
        should compute the per-point scaling from the actual per-point
        lumi (``self.scenario`` values) against a card-declared
        calibration reference, so it works for ``same_evts=True`` and
        custom scenarios too.

        Called by ``init_minuit`` but also directly by scans that mutate the
        lumi covariance without needing a fresh ``minuit`` (e.g.
        ``scan_lumi``)."""
        cov_stat = np.diag(self.unc_pseudodata_scenario ** 2)
        lumi_uncorr_ecm = np.full(len(self.pseudo_data_scenario), self.lumi_uncorr, dtype=float)
        if self.scenario_dict["add_last_ecm"]:
            factor_above = self.scenario[ecm_to_str(self.last_ecm)] / self.scenario[list(self.scenario.keys())[0]]
            lumi_uncorr_ecm[-1] = self.lumi_uncorr / factor_above ** 0.5
        self.lumi_uncorr_ecm = lumi_uncorr_ecm

        cov_lumi_uncorr = np.diag(self.pseudo_data_scenario * lumi_uncorr_ecm) ** 2
        cov_lumi_corr = np.outer(self.pseudo_data_scenario, self.pseudo_data_scenario) * self.lumi_corr ** 2
        self.cov = cov_lumi_uncorr + cov_lumi_corr + cov_stat
        # Pre-factor the (constant within migrad) covariance once; chi2 then
        # does a cheap triangular solve per call instead of a fresh LU.
        self._cov_factor = cho_factor(self.cov)

    def _build_chi2_caches(self):
        """Rebuild the chi2 hot-path caches (``_idx``, ``_bec_bin_idx``,
        ``_bes_bin_idx``, ``_xsec_base``, ``_morph_matrix``) from
        ``param_names`` / ``xsec_scenario`` / ``scale_var_scenario`` /
        ``morph_scenario``.

        Called by ``init_minuit`` and by scans that mutate one of those
        inputs (e.g. ``scan_scale_vars`` rebuilding ``_xsec_base``)."""
        # param_names is final by the time migrad starts (add_*_nuisances
        # mutate it; init_minuit always runs afterwards).
        self._idx = {name: i for i, name in enumerate(self.param_names)}
        self._bec_bin_idx = [i for i, (k, _) in self._per_bin_meta.items() if k == "BEC"]
        self._bes_bin_idx = [i for i, (k, _) in self._per_bin_meta.items() if k == "BES"]
        # Vectorised chi2 inputs: stack all morph templates into one ndarray
        # so the per-call loop becomes a single np.prod. Per-bin BEC/BES
        # rows are sparse (one-hot at the bin's own ECM) — synthesise from
        # _per_bin_meta + morph_scenario[kind] rather than holding N sparse
        # copies of the base morph in morph_scenario.
        self._xsec_base = np.asarray(self.xsec_scenario["xsec"]) * np.asarray(self.scale_var_scenario)
        n_ecm = len(self._xsec_base)
        rows = []
        for i, name in enumerate(self.param_names):
            meta = self._per_bin_meta.get(i)
            if meta is not None:
                kind, bin_idx = meta
                sparse = np.zeros(n_ecm)
                sparse[bin_idx] = np.asarray(self.morph_scenario[kind]["xsec"])[bin_idx]
                rows.append(sparse)
            else:
                rows.append(np.asarray(self.morph_scenario[name]["xsec"]))
        self._morph_matrix = np.stack(rows)

    def rebuild_chi2_state(self, *, init_vars=True, pseudodata=None):
        """Re-run ``create_scenario`` from the current ``scenario_dict``,
        then rebuild the cov and chi2-cache derivations. For scans that
        mutate ``scenario_dict`` in place — restores ``fit`` to a consistent
        state for the next chi² evaluation without touching ``fit.minuit``."""
        self.create_scenario(**self.scenario_dict, init_vars=init_vars, pseudodata=pseudodata)
        self._build_cov()
        self._build_chi2_caches()

    def results_from_minuit(self, minuit):
        """Compute physical (``value_from_param``-converted) fit-results from
        a Minuit instance.

        Scans that mutate ``fit`` in place and run a fresh local Minuit on
        ``fit.chi2`` (the same pattern as ``scan_chi2``) use this helper to
        read results without touching ``self.minuit`` — which keeps the
        per-scan state-isolation invariant satisfied.
        """
        _warn_if_invalid(minuit)
        vals = [minuit.values[p] for p in self.param_names]
        params_w_cov = list(unc.correlated_values(vals, minuit.covariance))
        resolved_params, _ = self.physical_fit_params(params_w_cov)
        return [self.value_from_param(p, name)
                for p, name in zip(resolved_params, self.param_names)]

    def fit_parameters(self, init_minuit=True):
        if init_minuit:
            self.init_minuit()
        self.minuit.migrad()

    def update(self, update_scenario=True, init_vars=False,
               pseudo_data=None, init_minuit=True):
        # init_minuit defaults to True because update() typically follows a
        # change to state that feeds the chi2 caches (smeared templates,
        # morph_scenario, scale_var_scenario, scenario tensors). Skipping
        # init_minuit would leave _xsec_base / _morph_matrix / _cov_factor
        # stale relative to the new state. Pass init_minuit=False only if
        # you know nothing chi2-relevant has changed.
        self._smear_cross_sections()
        self._morph_cross_sections()
        if update_scenario:
            self.create_scenario(**self.scenario_dict, init_vars=init_vars, pseudodata=pseudo_data)
        self.fit_parameters(init_minuit=init_minuit)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def fit_params_with_cov(self):
        _warn_if_invalid(self.minuit)
        vals = [self.minuit.values[p] for p in self.param_names]
        params_w_cov = list(unc.correlated_values(vals, self.minuit.covariance))
        resolved_params, _ = self.physical_fit_params(params_w_cov)
        return resolved_params

    def fit_results(self, printout=True):
        params_w_cov = self.fit_params_with_cov()
        for i, name in enumerate(self.param_names):
            params_w_cov[i] = self.value_from_param(params_w_cov[i], name)
        if printout:
            self._print_results(params_w_cov)
        return params_w_cov

    def _print_results(self, params_w_cov):
        for i, name in enumerate(self.param_names):
            val = params_w_cov[i]
            if name == "alphas":
                print(f"Fitted {name}: {val:.5f}")
            else:
                unit = " GeV" if name in ("mass", "width") else ""
                print(f"Fitted {name}: {val:.3f}{unit}")
                if name == "width" and self.sm_width:
                    print("including theory uncertainty in SM relation")
                    print(f"fitted theory parameter = {self.minuit.values[name]:.2f} +/- "
                          f"{self.minuit.errors[name]:.2f} (constrained to 1)")
                if name == "yukawa" and self.constrain_yukawa:
                    print(f"constrained with uncertainty {self.input_uncert_yukawa:.3f}")

            if name == "width" and self.sm_width:
                pull = unc.ufloat(self.minuit.values[name], self.minuit.errors[name])
            elif name == "sw2" or self._is_bin_nuisance(name):
                pull = val
            else:
                pull = val - self.d_params[self.pseudodata_tag][name]
            print(f"Pull {name}: {pull.n / pull.s:.3f}")
            if name == "mass":
                print(f"uncertainty in mass: {val.s * 1e3:.2f} MeV")
            print()
        print("Correlation matrix:")
        print(self.param_names[:4])
        corr = np.round(unc.correlation_matrix(params_w_cov), 2)
        print(corr[:4, :4])
        self.last_fit_results = params_w_cov

    # ------------------------------------------------------------------
    # Nuisance management
    # ------------------------------------------------------------------
    def add_bec_nuisances(self, prior_uncorr=None, prior_corr=None):
        self.bec_nuisances = True
        bec_p = self.card.PRIORS["BEC"]
        if prior_uncorr is None:
            prior_uncorr = bec_p["uncorr"]
        if prior_corr is None:
            prior_corr = bec_p["corr"]
        self.set_bec_priors(prior_uncorr=prior_uncorr, prior_corr=prior_corr)
        self._expand_per_bin_nuisance("BEC")

    def set_bec_priors(self, prior_uncorr, prior_corr):
        self.bec_prior_uncorr = prior_uncorr / self.input_var["BEC"]
        self.bec_prior_corr = prior_corr / self.input_var["BEC"]

    def add_bes_nuisances(self, uncert_uncorr=None, uncert_corr=None):
        self.bes_nuisances = True
        bes_p = self.card.PRIORS["BES"]
        if uncert_uncorr is None:
            uncert_uncorr = bes_p["uncorr"]
        if uncert_corr is None:
            uncert_corr = bes_p["corr"]
        self.set_bes_priors(uncert_uncorr=uncert_uncorr, uncert_corr=uncert_corr)
        self._expand_per_bin_nuisance("BES")

    def set_bes_priors(self, uncert_uncorr, uncert_corr):
        self.bes_prior_uncorr = uncert_uncorr / self.input_var["BES"]
        self.bes_prior_corr = uncert_corr / self.input_var["BES"]

    def add_sw2_nuisance(self, prior=None):
        if prior is None:
            prior = self.card.PRIORS["sw2"]["default"]
        self.sw2_nuisance = True
        self.set_sw2_prior(prior)
        self.param_names.append("sw2")

    def set_sw2_prior(self, prior):
        self.sw2_prior = prior / self.input_var["sw2"]

    def _expand_per_bin_nuisance(self, kind):
        """Register N + 1 nuisance parameter names for ``kind`` ∈ {BEC, BES}:
        one per-ECM-bin parameter plus a fully-correlated parameter. Stores
        the ``(kind, bin_idx)`` metadata for each per-bin entry so the
        sparse one-hot morph rows can be synthesised in
        ``_build_chi2_caches`` without re-parsing the names."""
        nbins = len(self.morph_scenario[kind])
        for i in range(nbins):
            self._per_bin_meta[len(self.param_names)] = (kind, i)
            self.param_names.append(f"{kind}_bin{i}")
        self.param_names.append(kind)

    # ------------------------------------------------------------------
    # Systematic-table support
    # ------------------------------------------------------------------
    def reinitialise_to_stat(self):
        self.input_uncert_alphas = _OFF
        self.input_uncert_yukawa = _OFF
        if self.bec_nuisances:
            self.bec_prior_corr = _OFF
            self.bec_prior_uncorr = _OFF
        if self.bes_nuisances:
            self.bes_prior_corr = _OFF
            self.bes_prior_uncorr = _OFF
        self.lumi_corr = _OFF
        self.lumi_uncorr = _OFF

    def reinitialise_to_nominal(self):
        self.input_uncert_alphas = self.card.CONSTRAINTS["alphas"]["sigma"]
        if "yukawa" in self.card.CONSTRAINTS:
            self.input_uncert_yukawa = self.card.CONSTRAINTS["yukawa"]["sigma"]
        if self.bec_nuisances:
            self.set_bec_priors(
                prior_uncorr=self.card.PRIORS["BEC"]["uncorr"],
                prior_corr=self.card.PRIORS["BEC"]["corr"],
            )
        if self.bes_nuisances:
            self.set_bes_priors(
                uncert_uncorr=self.card.PRIORS["BES"]["uncorr"],
                uncert_corr=self.card.PRIORS["BES"]["corr"],
            )
        self.lumi_corr = self.card.PRIORS["lumi"]["corr"]
        self.lumi_uncorr = self.card.PRIORS["lumi"]["uncorr"]
