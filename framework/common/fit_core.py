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

from framework.common.parameters import Parameters
from framework.common.smearing import convolute_gauss


def ecm_to_str(ecm):
    return f"{ecm:.1f}"


def bec_var_dir(var: float) -> str:
    """Signed-MeV BEC shift → BEC-variation subdir name (``scan_p10`` / ``scan_m30``)."""
    return f"scan_p{var:.0f}" if var >= 0 else f"scan_m{abs(var):.0f}"


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
#     ``systematics.py``) write ``OFF = 1e-10`` into the priors for the
#     syst-table flow. The floor brings them to 1e-6 — the chi2 penalty is
#     enormous either way and the parameter is pinned, but the Hessian is
#     better-conditioned at 1e-6.
#   - ``_scan_nuisance(BEC)`` at its leftmost grid point (v = 1e-6) yields
#     a BEC prior of 1e-7 after dividing by ``input_var["BEC"] = 10``.
#     Without the floor the BEC-bin Hessian rows reach ~1e14, close to the
#     conditioning threshold of the cov inversion. (BES doesn't hit this:
#     ``input_var["BES"] = 0.1`` keeps the prior at 1e-5 > floor.)
# alpha_s and Yukawa aren't floored: their OFF values are reached only in
# one-dimensional penalty terms where the cov stays well-conditioned.
# ---------------------------------------------------------------------------
_PRIOR_FLOOR = 1.0e-6
OFF = 1.0e-10


_KNOWN_SYST_TYPES = {"constraint", "binned", "global"}


def _split_systematics_by_type(card):
    """Group ``card.SYSTEMATICS`` entries by their ``type`` so FitCore can
    iterate "all binned nuisances" / "all constraints" without re-filtering
    on every call. Raises on unknown types so a typo doesn't silently
    disappear from the chi²."""
    out = {t: {} for t in _KNOWN_SYST_TYPES}
    for name, spec in card.SYSTEMATICS.items():
        t = spec["type"]
        if t not in _KNOWN_SYST_TYPES:
            raise ValueError(
                f"Unknown SYSTEMATICS type {t!r} for {name!r} (allowed: {sorted(_KNOWN_SYST_TYPES)})."
            )
        out[t][name] = spec
    return out


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
       ``xsec_dict``, ``xsec_dict_smeared``, ``morph_dict``,
       ``_nuisance_morph_raw``, plus card-derived scalars (scales, priors, BES, lumi,
       last_ecm). Nuisance active sets (``_active_binned_nuisances``,
       ``_active_global_nuisances``) start empty.

    2. ``init_scenario(...)`` — pick the ecm
       grid, total lumi, optional above-threshold point, and Asimov /
       pseudo-data flavour. Populates ``scenario_dict``, ``scenario``,
       ``xsec_scenario``, ``scale_var_scenario``, ``pseudo_data_scenario``,
       ``unc_pseudodata_scenario``, ``morph_scenario``. Between this step
       and step 3 the ``add_binned_nuisance(kind)`` /
       ``add_global_nuisance(kind)`` methods may extend ``param_names``
       and ``morph_scenario`` with nuisance bins.

    3. ``init_minuit(...)`` — invoked implicitly by ``fit_parameters`` and
       ``update``. Builds the chi2 hot-path caches (``cov``, ``_cov_factor``,
       ``lumi_uncorr_ecm``, ``_idx``, ``_per_kind_bin_idx``,
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
        asimov=True,
        read_scale_vars=False,
        mass_scheme=None,
        shift_scan=False,
        legacy_pseudo_rng=False,
        debug=False,
    ):
        self.card = card
        # Group SYSTEMATICS by type once; downstream code reads
        # self._systematics_meta["binned" / "global" / "constraint"]
        # instead of separate top-level card dicts.
        self._systematics_meta = _split_systematics_by_type(card)
        # Lumi-uncertainty mode: "cov" (default) keeps σ_lumi in the data
        # covariance; "nuisance" represents it as a binned nuisance with
        # a constant-shape morph (kind="flat"). The injected SYSTEMATICS
        # entry lives only on this instance — the card stays unchanged
        # so a parallel cov-mode fit can be constructed from the same card.
        self.lumi_mode = getattr(card, "LUMI_MODE", "cov")
        if self.lumi_mode not in ("cov", "nuisance"):
            raise ValueError(
                f"card.LUMI_MODE must be 'cov' or 'nuisance' (got {self.lumi_mode!r})")
        if self.lumi_mode == "nuisance":
            if "lumi" not in card.INPUT_VAR:
                raise ValueError(
                    "LUMI_MODE='nuisance' requires INPUT_VAR['lumi'] in the card "
                    "(the fit-parameter unit, e.g. 0.01 for 1%).")
            self._systematics_meta["binned"]["lumi"] = {
                "type": "binned", "source": {"kind": "flat"}}
        self.generator = generator
        self.debug = debug
        self.asimov = asimov
        self.read_scale_vars = read_scale_vars
        # shift_scan moves the scan list off the original ECM grid, so the
        # BEC/BES/sw2 templates would need interpolation — skip them instead.
        self.shift_scan = shift_scan
        # legacy_pseudo_rng=True restores the global np.random.seed(42)
        # pattern that gave every create_scenario call the *same* noise;
        # kept only for byte-reproducing pre-fix pseudo runs.
        self.legacy_pseudo_rng = legacy_pseudo_rng
        self._rng = None if legacy_pseudo_rng else np.random.default_rng(42)

        # Mass scheme & input/plot directories ------------------------------
        # Cards without a real alternate mass scheme (e.g. WW — OS is the
        # only m_W convention) may omit MASS_SCHEME entirely; default "OS".
        self.mass_scheme = (mass_scheme if mass_scheme is not None
                            else getattr(card, "MASS_SCHEME", "OS"))
        is_1s = self.mass_scheme == "1S"
        if input_dir is None:
            if is_1s:
                input_dir = card.INPUT_DIRS["scale_1S"] if read_scale_vars else card.INPUT_DIRS["nominal_1S"]
            else:
                input_dir = card.INPUT_DIRS["scale_vars"] if read_scale_vars else card.INPUT_DIRS["nominal"]
        self.input_dir = input_dir
        self.plot_dir = card.PLOT_DIR_1S if is_1s else card.PLOT_DIR

        # Renormalisation scales — fall back to a scale-less default for
        # processes (e.g. WW) where the chain has no μ-renormalisation
        # scale to vary. ``mass``/``width`` only enter template filenames
        # as scaffolding; ``vars`` empty means no scale-variation templates.
        scales = getattr(card, "RENORM_SCALES",
                         {"mass": 80.0, "width": 80.0, "vars": []})

        # Parameters --------------------------------------------------------
        card_params = card.PARAMETERS_1S if is_1s else card.PARAMETERS
        # CROSS_TERMS is WW-only at the moment — WbWb card doesn't define it,
        # so the getattr keeps the feature strictly opt-in.
        self.parameters = Parameters(
            card_params,
            scale_vars=scales["vars"] if read_scale_vars else [],
            cross_terms=getattr(card, "CROSS_TERMS", ()),
        )
        self.d_params = self.parameters.as_dict()
        self.param_names = list(self.parameters.names)
        # Pseudodata template defaults to ``pseudodata``; subclasses can
        # override _select_pseudodata_tag (WbWb uses ``mass_var`` under
        # SM_width to feed a shifted-true-value pseudodata).
        self.pseudodata_tag = self._select_pseudodata_tag()

        if read_scale_vars:
            self.mass_scale = scales["alt"]["mass"]
            self.width_scale = scales["alt"]["width"]
        else:
            self.mass_scale = scales["mass"]
            self.width_scale = scales["width"]
        self.scale_vars = self.parameters.scale_vars

        # Beam-energy spectrum ---------------------------------------------
        self.beam_energy_res = card.BEAM_ENERGY_RES
        self.peak_ecm = card.PEAK_ECM
        self.last_ecm = card.LAST_ECM

        # Priors -----------------------------------------------------------
        self.lumi_uncorr = card.PRIORS["lumi"]["uncorr"]
        self.lumi_corr = card.PRIORS["lumi"]["corr"]
        self.input_var = card.INPUT_VAR

        # 1-D Gaussian-constraint registry. Centre defaults to the
        # pseudodata "true" value; override per-entry via
        # ``SYSTEMATICS[name]["center"]``. Entries with ``always_on=False``
        # start inactive — process subclasses (or entry scripts) flip
        # ``_constraints[name]["active"]`` after construction.
        self._constraints = {}
        for name, spec in self._systematics_meta["constraint"].items():
            if name not in self.parameters.names:
                continue
            self._constraints[name] = {
                "sigma":  card.PRIORS[name],
                "center": spec.get("center", self.d_params[self.pseudodata_tag][name]),
                "active": spec["always_on"],
            }

        self._active_binned_nuisances = set()
        self._active_global_nuisances = set()
        self._nuisance_priors = {}
        # Per-bin rescale of an uncorrelated counting-measurement prior: a
        # counting measurement's point-to-point uncertainty scales as
        # 1/√(L_point). Set in create_scenario when card.LUMI_UNCORR_SCALES is
        # True; None disables the rescale (uniform card prior). See _nuisance_prior.
        self._uncorr_perbin_scale = None
        # Binned nuisances whose uncorrelated component is a counting
        # measurement (per-point precision ∝ 1/√L_point) and so share the same
        # per-point √(L_ref/L_i) rescale. lumi is always included; the card may
        # add more (e.g. BES, di-muon-monitored) via UNCORR_COUNTING_KINDS.
        self._counting_uncorr_kinds = {"lumi", *getattr(card, "UNCORR_COUNTING_KINDS", ())}
        # param_idx -> (kind, bin_idx_in_morph_scenario[kind]) — populated
        # by _expand_per_bin_nuisance. Avoids re-parsing "BEC_bin{i}" names.
        self._per_bin_meta = {}

        # chi2 hot-path caches — populated by init_minuit. Listed here so
        # that AttributeError-style failures from calling chi2 before
        # init_minuit show a clear "this attribute is None" instead.
        self._idx = None
        self._per_kind_bin_idx = None
        self._xsec_base = None
        self._morph_matrix = None
        # Bilinear POI×POI cross-term cache; populated by
        # ``_build_chi2_caches`` only when the card declares ``CROSS_TERMS``
        # and the corner templates exist on disk. ``_cross_idx`` is shape
        # ``(n_pairs, 2)`` of param indices into ``param_names``;
        # ``_cross_matrix`` is shape ``(n_pairs, n_ecm)`` of bilinear
        # residuals ``(1+m_corner)/((1+m_a)(1+m_b)) - 1`` that multiply
        # into ``th_xsec`` as ``Π (1 + p_a p_b · x_{ab})``.
        self._cross_idx = None
        self._cross_matrix = None
        self._cov_factor = None

        # Placeholder cross-section systematics (a fully-correlated and a
        # fully-uncorrelated component across √s, each sized as a fraction of the
        # per-point statistical uncertainty). OFF until a driver calls
        # ``set_xsec_systematics`` — mirrors the declare-in-card / activate-in-
        # driver pattern of the binned nuisances. Added to the data covariance in
        # ``_build_cov``; the ``*_nom`` values are restored by
        # ``reinitialise_to_nominal`` between syst-table switch-offs.
        self.xsec_syst_corr = self._xsec_syst_corr_nom = 0.0
        self.xsec_syst_uncorr = self._xsec_syst_uncorr_nom = 0.0
        # Global point-to-point statistical correlation imposed on the measured
        # cross sections (0 = independent counting). Swept by
        # ``scans.scan_stat_correlation``; consumed by ``_build_cov``.
        self._stat_corr = 0.0

        # Build templates --------------------------------------------------
        if debug:
            print(f"Input directory: {self.input_dir}")
            print(f"Parameters: {self.param_names}")
            print(f"Beam energy resolution: {self.beam_energy_res}")
            print(f"Asimov fit: {self.asimov}")

        self._read_cross_sections()
        self._read_aux_templates()
        self._smear_cross_sections()
        self._morph_cross_sections()
        if debug:
            print("Initialization done")

    def _read_aux_templates(self):
        """Pre-load raw nuisance template DataFrames once.

        Iterates the binned + global SYSTEMATICS specs and loads the ones
        whose ``source["kind"]`` is ``"template_dir"``. Cached so
        ``update()`` (called e.g. by scan_beam_resolution) doesn't re-read
        the filesystem; the on-disk content never changes.

        BEC-style sources may set:
        * ``var_subdir=True`` — the actual templates live in a subdir
          named ``scan_p{var}`` / ``scan_m{var}`` (the C++ scan emits one
          subdir per ±variation magnitude in ``INPUT_VAR[kind]``).
        * ``snap_to_grid=True`` — the loaded ECMs are shifted by
          ±input_var MeV; snap them back to the nominal 0.1-GeV grid so
          the morph computation divides per-row against the nominal
          template. Constraint: variation magnitude must stay ≤ 40 MeV,
          beyond which the one-decimal rounding would alias onto the next
          nominal ECM bin (banker's rounding at .05).
        """
        self._nuisance_morph_raw = {}
        if self.read_scale_vars or self.mass_scheme == "1S" or self.shift_scan:
            return
        nuisance_specs = {**self._systematics_meta["binned"],
                          **self._systematics_meta["global"]}
        for kind, spec in nuisance_specs.items():
            source = spec["source"]
            if source["kind"] != "template_dir":
                continue
            path = self.card.INPUT_DIRS[kind]
            if source.get("var_subdir"):
                path = os.path.join(path, bec_var_dir(self.input_var[kind]))
            raw = self._scan_for_tag("nominal", indir=path)
            if source.get("snap_to_grid"):
                raw["ecm"] = raw["ecm"].round(1)
            self._nuisance_morph_raw[kind] = raw

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

    def tracked_pois(self):
        """POIs from ``card.POI_DISPLAY`` that are free in this fit. Skips
        entries currently held as a constrained nuisance (e.g. yukawa
        under the WbWb default, where ``--fitYukawa`` is not set)."""
        return [poi for poi in self.card.POI_DISPLAY
                if poi in self.param_names
                and not (poi in self._constraints and self._constraints[poi]["active"])]

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

    def _validate_scenario(self, add_last_ecm):
        """Hook: process subclasses can reject scenario combinations that
        don't make physical sense (e.g. WbWb raises when the Yukawa
        constraint is on AND ``add_last_ecm`` is set). Default: accept."""
        pass

    def _select_pseudodata_tag(self):
        """Hook: which template tag to use as the pseudodata reference.
        Default: ``"pseudodata"``. WbWb subclass returns ``"mass_var"``
        under SM_width to feed a shifted-true-value pseudodata template."""
        return "pseudodata"

    def is_scannable_poi(self, name):
        """Hook: whether ``name`` is meaningful as a free POI for scan
        plotting. Default: True. WbWb returns False for ``"width"``
        under SM_width — the parameter becomes a constrained theory
        knob in that mode, not a free POI to profile."""
        return True

    def _print_param_extras(self, name, val):
        """Hook: process subclasses print extra annotation lines after the
        ``Fitted {name}`` line. Default: no extras."""
        pass

    def _pull_for(self, name, val):
        """Pull value to compare with its hesse uncertainty. Default
        formula is ``val - pseudodata`` for free POIs and the raw
        parameter value for nuisances. Subclasses can override for
        parameters whose pull semantics differ (e.g. WbWb's SM-width
        theory knob, constrained to 1 by convention)."""
        if name in self._systematics_meta["global"] or self._is_bin_nuisance(name):
            return val
        return val - self.d_params[self.pseudodata_tag][name]

    def stat_breakdown_default(self):
        """Whether ``print_syst_table`` should compute the per-POI stat
        breakdown. Default: ``True`` (do the breakdown). Subclasses can
        override — WbWb returns False when yukawa is constrained, since
        the per-POI breakdown is mostly noise in that regime."""
        return True

    # ------------------------------------------------------------------
    # File I/O & templates
    # ------------------------------------------------------------------
    def read_xsec(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cross-section template not found: {path}")
        # ``comment='#'`` lets templates carry a ``# key: value`` preamble
        # (see e.g. framework.process.ww.template_metadata) without breaking
        # the (ecm, xsec) parser.
        with open(path) as fh:
            return pd.read_csv(fh, header=None, names=["ecm", "xsec"], comment="#")

    def template_metadata(self) -> dict:
        """Return the ``# key: value`` preamble of the nominal template, if
        any. Cached on the first call. Empty dict for templates predating
        the metadata header. The preferred entry point for plot/fit code
        that wants to describe the *actual* templates being read rather
        than the live card configuration.
        """
        cached = getattr(self, "_template_metadata_cache", None)
        if cached is not None:
            return cached
        try:
            from framework.process.ww.template_metadata import read_header
        except ImportError:
            self._template_metadata_cache = {}
            return self._template_metadata_cache
        values = self.d_params["nominal"]
        path = self.generator.file_name(
            values, mass_scale=self.mass_scale, width_scale=self.width_scale,
            mass_scheme=self.mass_scheme, indir=self.input_dir,
        )
        try:
            self._template_metadata_cache = read_header(path)
        except OSError:
            self._template_metadata_cache = {}
        return self._template_metadata_cache

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
        if bes is None:
            bes = self.beam_energy_res
        if bes == 0:
            return xsec
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
        """Return a DataFrame with the relative variation
        (xsec_var/xsec_nom - 1) for ``param``.

        Dispatches on the parameter kind:
        * parameter-of-interest (mass / width / yukawa / ...): variation
          template lives in ``xsec_dict_smeared[f"{param}_var"]``;
        * nuisance kind (binned or global, found via ``SYSTEMATICS``):
          variation computed from ``source["kind"]`` — ``template_dir``
          loads a pre-cached smeared variation, ``smear_shift`` re-smears
          the nominal with a shifted beam-energy resolution.
        For global nuisances the smeared variation is also stashed into
        ``xsec_dict_smeared[f"{param}_var"]`` so ``plot_parameter_variations``
        (which calls ``template(name+"_var")``) can render it.
        """
        xsec_nom = self.template()
        if param in self.parameters.names:
            xsec_var = self.template(f"{param}_var")
        else:
            spec = (self._systematics_meta["binned"].get(param) or
                    self._systematics_meta["global"].get(param))
            if spec is None:
                raise ValueError(f"Unknown morph parameter: {param!r}")
            source = spec["source"]
            kind = source["kind"]
            if kind == "template_dir":
                xsec_var = self.smear(self._nuisance_morph_raw[param])
            elif kind == "smear_shift":
                xsec_var = self.smear(self.xsec_dict["nominal"],
                                      bes=self.beam_energy_res * (1 + self.input_var[param]))
            elif kind == "flat":
                # Constant relative shift (lumi nuisance): a +1·input_var
                # variation rescales σ uniformly across bins. Return
                # xsec_var = xsec_nom · (1 + input_var) directly — the
                # divide-then-subtract-1 below recovers +input_var per bin.
                xsec_var = pd.DataFrame({
                    "ecm":  xsec_nom["ecm"],
                    "xsec": xsec_nom["xsec"] * (1.0 + self.input_var[param]),
                })
            else:
                raise ValueError(f"Unknown nuisance source kind: {kind!r}")
            if param in self._systematics_meta["global"]:
                self.xsec_dict_smeared[f"{param}_var"] = xsec_var
        return pd.DataFrame({"ecm": xsec_nom["ecm"],
                             "xsec": xsec_var["xsec"] / xsec_nom["xsec"] - 1})

    def _morph_one_cross(self, name_a, name_b):
        """Relative shift at the (a, b) cross-term corner template:
        ``(σ_corner − σ_nom) / σ_nom`` per bin. The bilinear *residual*
        relative to the multiplicative linear factor is derived in
        ``_build_chi2_caches`` once all the linear rows are in hand."""
        tag = self.parameters.cross_tag(name_a, name_b)
        xsec_nom = self.template()
        xsec_var = self.template(tag)
        return pd.DataFrame({"ecm":  xsec_nom["ecm"],
                             "xsec": xsec_var["xsec"] / xsec_nom["xsec"] - 1})

    def _is_per_bin_name(self, name):
        """True if ``name`` is a per-bin child nuisance (``<kind>_bin{i}``) of a
        binned systematic. Unlike :meth:`_is_bin_nuisance`, the correlated parent
        ``kind`` itself does NOT match: per-bin children are skipped from the
        global morph (their rows are synthesised sparsely in
        ``_build_chi2_caches``) while the parent ``kind`` keeps its own row."""
        return any(name.startswith(f"{k}_bin")
                   for k in self._systematics_meta["binned"])

    def _morph_cross_sections(self):
        # Skip per-bin nuisance expansion names (e.g. ``BEC_bin0``,
        # ``lumi_bin3``) — their morph rows are synthesised sparsely from
        # ``morph_scenario[kind]`` in ``_build_chi2_caches``, never stored
        # individually here. (Without this filter, ``update()`` after
        # ``add_binned_nuisance(kind)`` would re-enter ``_morph_one`` with
        # a per-bin name and raise on the ``_systematics_meta`` lookup.)
        self.morph_dict = {p: self._morph_one(p)
                           for p in self.param_names
                           if not self._is_per_bin_name(p)}
        # POI cross-term corner rows (only when the card declares CROSS_TERMS).
        for (a, b) in self.parameters.cross_terms:
            self.morph_dict[self.parameters.cross_tag(a, b)] = self._morph_one_cross(a, b)
        if not self.read_scale_vars and self.mass_scheme != "1S" and not self.shift_scan:
            for kind in (*self._systematics_meta["binned"], *self._systematics_meta["global"]):
                self.morph_dict[kind] = self._morph_one(kind)


    # ------------------------------------------------------------------
    # Parameter <-> value conversion
    # ------------------------------------------------------------------
    def _is_bin_nuisance(self, name):
        """True for any binned-nuisance parameter (correlated ``{kind}`` or
        per-bin ``{kind}_bin{i}``) — those whose fit-space value is the
        parameter itself, no nominal+step rescaling."""
        for kind in self._systematics_meta["binned"]:
            if name == kind or name.startswith(f"{kind}_bin"):
                return True
        return False

    def value_from_param(self, par, name):
        if name in self._systematics_meta["global"]:
            return par * self.input_var[name]
        if self._is_bin_nuisance(name):
            return par
        return self.d_params["nominal"][name] + par * self.parameters.step(name)

    def param_from_value(self, val, name):
        return (val - self.d_params["nominal"][name]) / self.parameters.step(name)

    # ------------------------------------------------------------------
    # Scenario
    # ------------------------------------------------------------------
    def init_scenario(self, *, total_lumi, last_lumi, add_last_ecm=False,
                      same_evts=False, scan_list=None,
                      scan_min=None, scan_max=None, scan_step=None,
                      lumi_dict=None):
        """Initialise the scan scenario (ecm grid + lumi distribution).

        Either pass an explicit ``scan_list`` (already-formatted ecm-string
        list) or the ``(scan_min, scan_max, scan_step)`` triplet from which
        a uniform grid is built.

        ``lumi_dict`` (optional) gives an explicit per-ECM luminosity
        ``{ecm_str: lumi}`` overriding the default equal split of
        ``total_lumi`` across ``scan_list``. The keys must equal
        ``scan_list`` exactly (same ecm-string format). ``lumi_dict`` and
        ``same_evts`` are mutually exclusive — passing both raises (the
        equal-events reweighting would otherwise silently overwrite the
        explicit per-ECM lumi).

        Stores everything in ``self.scenario_dict`` and triggers the
        first ``create_scenario`` build."""
        if scan_list is None:
            if scan_min is None or scan_max is None or scan_step is None:
                raise ValueError(
                    "init_scenario needs either scan_list= or "
                    "scan_min=/scan_max=/scan_step=."
                )
            scan_list = [ecm_to_str(e)
                         for e in np.arange(scan_min, scan_max + scan_step / 2, scan_step)]
        elif any(x is not None for x in (scan_min, scan_max, scan_step)):
            raise ValueError("init_scenario: pass scan_list= XOR scan_min=/scan_max=/scan_step=.")
        if same_evts and lumi_dict is not None:
            raise ValueError(
                "init_scenario: same_evts and lumi_dict are mutually exclusive "
                "(equal-events reweighting would overwrite the explicit lumi_dict)."
            )
        self._validate_scenario(add_last_ecm)
        self.scenario_dict = {
            "scan_list": scan_list,
            "total_lumi": total_lumi,
            "last_lumi": last_lumi,
            "add_last_ecm": add_last_ecm,
            "same_evts": same_evts,
            "lumi_dict": lumi_dict,
        }
        self.create_scenario()
        # In nuisance mode the lumi binned nuisance is *mandatory* (otherwise
        # the lumi uncertainty is missing entirely — cov terms are skipped in
        # _build_cov). Auto-activate here so entry scripts don't need a
        # parallel BES/BEC-style flag for it.
        if self.lumi_mode == "nuisance":
            self.add_binned_nuisance("lumi")
        # Always-on global nuisances (``always_on=True`` in card.SYSTEMATICS,
        # e.g. the WW EW-coupling normalization nuisance ``aemEW``) auto-activate
        # here — mirroring the always-on constraint nuisances registered in
        # __init__ — so every entry script picks them up without an explicit
        # add_global_nuisance call. Opt-in globals (no always_on, e.g. WbWb
        # ``sw2``) are left for entry scripts. Idempotent.
        for kind, spec in self._systematics_meta["global"].items():
            if spec.get("always_on"):
                self.add_global_nuisance(kind)

    def create_scenario(self, *, init_vars=True, pseudodata=None):
        """Build the scenario tensors from ``self.scenario_dict``.

        ``init_vars`` controls whether ``scale_var_scenario`` is reset to
        ones (default True; set False to preserve a custom scale variation
        injected by ``scan_scale_vars``). ``pseudodata`` overrides the
        pseudodata cross section (default: smeared ``self.pseudodata_tag``
        template)."""
        sd = self.scenario_dict
        scan_list = sd["scan_list"]
        total_lumi = sd["total_lumi"]
        last_lumi = sd["last_lumi"]
        add_last_ecm = sd["add_last_ecm"]
        same_evts = sd["same_evts"]
        lumi_dict = sd.get("lumi_dict")

        if self.debug:
            print("Creating threshold scan scenario")

        if lumi_dict is not None:
            missing = set(scan_list) - set(lumi_dict)
            extra = set(lumi_dict) - set(scan_list)
            if missing or extra:
                raise ValueError(
                    f"lumi_dict keys must match scan_list exactly; "
                    f"missing={sorted(missing)} extra={sorted(extra)}")
            scenario_dict = dict(lumi_dict)
        else:
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
        # Selection+reconstruction efficiency ε: the measured (efficiency-
        # corrected) cross section is reconstructed from N_sel = ε·σ·L selected
        # events, so its per-point statistical uncertainty grows by 1/√ε. ε
        # defaults to 1 (no reconstruction modelled, e.g. WbWb).
        eff = self.card.SCENARIO.get("selection_efficiency", 1.0)
        if eff <= 0:
            raise ValueError(
                f"SCENARIO['selection_efficiency'] must be > 0 (got {eff!r})")
        unc_stat = unc_stat / eff ** 0.5
        # Optional per-fit statistical-only rescale (default 1.0). Used by the
        # channel extrapolation to represent a sample with a different effective
        # event yield (e.g. inclusive WW = μνqq̄ × 1/B → stat × √B) WITHOUT
        # touching the luminosity, so the lumi-measurement priors stay tied to
        # the real machine luminosity rather than the fictitious yield boost.
        unc_stat *= getattr(self, "_extra_stat_scale", 1.0)
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

        # Per-bin nuisance expansion names (``BEC_bin0`` etc.) aren't keys
        # in ``morph_dict`` — they share the parent kind's morph, applied
        # sparsely in ``_build_chi2_caches``. Skip them here for the same
        # reason as ``_morph_cross_sections``.
        self.morph_scenario = {p: self.slice_to_scenario(self.morph_dict[p])
                               for p in self.param_names
                               if not self._is_per_bin_name(p)}
        for kind in (*self._systematics_meta["binned"], *self._systematics_meta["global"]):
            if kind in self.morph_dict:
                self.morph_scenario[kind] = self.slice_to_scenario(self.morph_dict[kind])
        # Cross-term corner rows are scenario-aliased alongside the linear
        # rows so ``_build_chi2_caches`` can read them off ``morph_scenario``.
        for (a, b) in self.parameters.cross_terms:
            tag = self.parameters.cross_tag(a, b)
            if tag in self.morph_dict:
                self.morph_scenario[tag] = self.slice_to_scenario(self.morph_dict[tag])

        # Above-threshold rescaling for the lumi nuisance morph — the
        # FALLBACK used only when LUMI_UNCORR_SCALES is False. In cov mode the
        # last-bin uncorr lumi unc is divided by sqrt(factor_above) (see
        # ``_build_cov``); this mirrors it by scaling the lumi morph at the
        # above-threshold bin. When LUMI_UNCORR_SCALES is True, the general
        # per-point ``_uncorr_perbin_scale`` below already tightens the last
        # (higher-lumi) bin via sqrt(L_ref/L_i) — which equals 1/sqrt(factor_above)
        # there — so this block MUST be gated off to avoid double-counting the
        # last-bin shrink (the cov path is made mutually exclusive the same way).
        if (self.lumi_mode == "nuisance"
                and not getattr(self.card, "LUMI_UNCORR_SCALES", False)
                and self.scenario_dict["add_last_ecm"]
                and "lumi" in self.morph_scenario):
            factor_above = self.scenario[ecm_to_str(self.last_ecm)] / self.scenario[list(self.scenario.keys())[0]]
            morph = self.morph_scenario["lumi"].copy()
            morph.iloc[-1, morph.columns.get_loc("xsec")] = (
                morph.iloc[-1]["xsec"] / factor_above ** 0.5)
            self.morph_scenario["lumi"] = morph

        # Per-point rescale of an uncorrelated counting-measurement prior. The
        # luminosity (di-photon / large-angle Bhabha) and any other counting
        # measurement (e.g. BES, di-muon-monitored) have a point-to-point
        # uncertainty that scales as 1/√(L_point): a scan with fewer points
        # concentrates more luminosity per point → smaller per-point uncorr.
        # uncorr_i = uncorr_ref · √(L_ref / L_i), with (uncorr_ref, L_ref) the
        # card calibration. Applied to the per-bin nuisances of
        # _counting_uncorr_kinds in _nuisance_prior; the correlated
        # (common-normalisation) piece does not scale. The per-point lumi L_i is
        # taken in the (ecm-sorted) order that matches the nuisance bins.
        if getattr(self.card, "LUMI_UNCORR_SCALES", False):
            L_i = np.array(list(self.scenario.values()), dtype=float)
            L_ref = float(self.card.LUMI_UNCORR_CALIB_LUMI)
            self._uncorr_perbin_scale = np.sqrt(L_ref / L_i)
        else:
            self._uncorr_perbin_scale = None

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
        ``th_xsec = _xsec_base · Π_i (1 + p_i · morph_i)``. The
        multiplicative product already carries the per-pair factorisable
        cross-term ``p_i p_j m_i m_j``; the residual non-factorisable
        bilinear curvature is captured by an additional
        ``Π_{(a,b)} (1 + p_a p_b · x_{ab})`` factor when the card
        declares ``CROSS_TERMS`` (one corner template per pair, used to
        derive the bilinear residual
        ``x_{ab} = (1+m_corner)/((1+m_a)(1+m_b)) - 1``). At the corner
        the product collapses to ``σ_0·(1+m_corner)`` exactly, so the
        bilinear morph is closure-accurate on the calibration points;
        away from the corner it interpolates smoothly. WW activates
        this via ``CROSS_TERMS = [("mass", "width")]``; WbWb leaves the
        attribute absent and recovers the pure-linear behaviour.

        Active 1-D Gaussian constraints (entries in ``self._constraints``
        with ``active=True``) add ``((param - centre) / sigma)²`` penalty
        terms. Centres default to the pseudodata "true" values —
        Asimov-self-consistent (fit, data, and constraint all sit at the
        pseudo point so no bias on the fit minimum). Set
        ``card.SYSTEMATICS[name]["center"]`` to override (e.g. SM-centred
        for real-data analysis or bias studies).
        """
        # physical_fit_params runs first because subclasses (e.g. WbWbFit
        # with SM_width) resolve cross-parameter relations on the dependent
        # entries (e.g. width). Pre-copy so the override can mutate without
        # touching Minuit's state array; the template uses the resolved
        # values, the prior terms below stay on the raw free params.
        resolved_params, prior_extra = self.physical_fit_params(params.copy())
        th_xsec = self._xsec_base * np.prod(1 + resolved_params[:, None] * self._morph_matrix, axis=0)
        if self._cross_matrix is not None:
            p_a = resolved_params[self._cross_idx[:, 0]]
            p_b = resolved_params[self._cross_idx[:, 1]]
            th_xsec = th_xsec * np.prod(
                1 + (p_a * p_b)[:, None] * self._cross_matrix, axis=0)

        res = self.pseudo_data_scenario - th_xsec
        chi2_val = float(res @ cho_solve(self._cov_factor, res))

        # 1-D Gaussian constraints (driven by card.SYSTEMATICS via
        # self._constraints). The ``active`` flag is set at __init__
        # time so the hot path is name-agnostic.
        for name, c in self._constraints.items():
            if not c["active"]:
                continue
            sigma_fs = c["sigma"] / self.parameters.step(name)
            chi2_val += ((params[self._idx[name]]
                          - self.param_from_value(c["center"], name)) / sigma_fs) ** 2

        for kind in self._active_binned_nuisances:
            chi2_val += self._nuisance_prior(params, kind)
        for kind in self._active_global_nuisances:
            prior = max(self._nuisance_priors[kind]["prior"], _PRIOR_FLOOR)
            chi2_val += (params[self._idx[kind]] / prior) ** 2

        return chi2_val + prior_extra

    def _nuisance_prior(self, params, kind):
        priors = self._nuisance_priors[kind]
        prior_u = max(priors["uncorr"], _PRIOR_FLOOR)
        prior_c = max(priors["corr"], _PRIOR_FLOOR)
        bin_idx = self._per_kind_bin_idx[kind]
        bin_params = params[bin_idx]
        corr_idx = self._idx[kind]
        scale = self._uncorr_perbin_scale
        if (scale is not None and kind in self._counting_uncorr_kinds
                and len(scale) == len(bin_params)):
            # Counting-measurement scaling: per-point uncorr prior
            # uncorr_i = prior_u · √(L_ref/L_i) (more lumi/point → tighter).
            # Applies to lumi and to card.UNCORR_COUNTING_KINDS (e.g. BES,
            # whose spread is di-muon-monitored → precision ∝ 1/√L_point).
            prior_u_vec = np.maximum(prior_u * scale, _PRIOR_FLOOR)
            return float(np.sum((bin_params / prior_u_vec) ** 2)
                         + (params[corr_idx] / prior_c) ** 2)
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

        Under ``LUMI_MODE='nuisance'`` the lumi cov-matrix terms are
        skipped: the same priors are absorbed into the binned-nuisance
        chi² penalty (see ``add_binned_nuisance('lumi')`` auto-activation
        in ``init_scenario``). ``lumi_uncorr_ecm`` is still computed in
        that mode so downstream printouts (e.g. scan_lumi_yukawa_ratio)
        retain their reference figure.

        When ``card.LUMI_UNCORR_SCALES`` is set, the uncorr lumi figure is
        rescaled per point as ``uncorr_i = uncorr_ref·√(L_ref/L_i)`` from the
        actual per-point lumi (``self.scenario`` values) against the card
        calibration ``LUMI_UNCORR_CALIB_LUMI`` — the counting-measurement
        scaling, mirroring the nuisance-mode treatment in ``_nuisance_prior``.
        (Works for ``same_evts=True`` and custom scenarios too.)

        Called by ``init_minuit`` but also directly by scans that mutate the
        lumi covariance without needing a fresh ``minuit`` (e.g.
        ``scan_lumi``)."""
        stat = self.unc_pseudodata_scenario
        # Statistical covariance. Diagonal variances by default (independent
        # Poisson counting). When ``_stat_corr`` (ρ) is set — by the ρ-sweep
        # study ``scans.scan_stat_correlation`` — impose a global point-to-point
        # correlation on the measured cross sections: cov_ij = ρ·σ_i·σ_j (i≠j).
        # ρ=1 is exactly rank-1 (singular: cov = σσᵀ), so cap it just below 1 to
        # keep the Cholesky factorisation well-defined. As ρ→1 only the common
        # normalisation mode stays uncertain; the shape (m_W/Γ_W) is pinned ever
        # tighter, so σ_POI→0 (see scans.scan_stat_correlation).
        rho = getattr(self, "_stat_corr", 0.0)
        if rho:
            rho = min(rho, 1.0 - 1.0e-6)
            cov_stat = np.outer(stat, stat) * rho
            np.fill_diagonal(cov_stat, stat ** 2)
        else:
            cov_stat = np.diag(stat ** 2)
        # Placeholder cross-section systematics (activated via
        # ``set_xsec_systematics``): a fully-correlated and a fully-uncorrelated
        # component across √s, each sized as a fraction of the per-point
        # statistical uncertainty, added straight to the data covariance.
        # Toggled per-component by the syst table exactly like the cov-mode lumi
        # priors (see systematics._turn_off).
        cov_data = cov_stat
        if self.xsec_syst_uncorr:
            cov_data = cov_data + np.diag((self.xsec_syst_uncorr * stat) ** 2)
        if self.xsec_syst_corr:
            d = self.xsec_syst_corr * stat
            cov_data = cov_data + np.outer(d, d)
        if getattr(self.card, "LUMI_UNCORR_SCALES", False):
            L_i = np.array(list(self.scenario.values()), dtype=float)
            lumi_uncorr_ecm = self.lumi_uncorr * np.sqrt(
                float(self.card.LUMI_UNCORR_CALIB_LUMI) / L_i)
        else:
            lumi_uncorr_ecm = np.full(len(self.pseudo_data_scenario), self.lumi_uncorr, dtype=float)
            if self.scenario_dict["add_last_ecm"]:
                factor_above = self.scenario[ecm_to_str(self.last_ecm)] / self.scenario[list(self.scenario.keys())[0]]
                lumi_uncorr_ecm[-1] = self.lumi_uncorr / factor_above ** 0.5
        self.lumi_uncorr_ecm = lumi_uncorr_ecm

        if self.lumi_mode == "nuisance":
            self.cov = cov_data
        else:
            cov_lumi_uncorr = np.diag(self.pseudo_data_scenario * lumi_uncorr_ecm) ** 2
            cov_lumi_corr = np.outer(self.pseudo_data_scenario, self.pseudo_data_scenario) * self.lumi_corr ** 2
            self.cov = cov_lumi_uncorr + cov_lumi_corr + cov_data
        # Pre-factor the (constant within migrad) covariance once; chi2 then
        # does a cheap triangular solve per call instead of a fresh LU.
        self._cov_factor = cho_factor(self.cov)

    def _build_chi2_caches(self):
        """Rebuild the chi2 hot-path caches (``_idx``, ``_per_kind_bin_idx``,
        ``_xsec_base``, ``_morph_matrix``) from ``param_names`` /
        ``xsec_scenario`` / ``scale_var_scenario`` / ``morph_scenario``.

        Called by ``init_minuit`` and by scans that mutate one of those
        inputs (e.g. ``scan_scale_vars`` rebuilding ``_xsec_base``)."""
        # param_names is final by the time migrad starts (add_*_nuisances
        # mutate it; init_minuit always runs afterwards).
        self._idx = {name: i for i, name in enumerate(self.param_names)}
        self._per_kind_bin_idx = {}
        for i, (k, _) in self._per_bin_meta.items():
            self._per_kind_bin_idx.setdefault(k, []).append(i)
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

        # Bilinear cross-term: residual after the multiplicative linear
        # product (1+p_a m_a)(1+p_b m_b) already implied by the chi2's
        # ``np.prod`` line. Closure at the corner is exact: at p_a=p_b=1
        # the prediction becomes σ_0·(1+m_corner) = σ_corner. Pairs are
        # skipped if either POI is missing from this fit configuration
        # (e.g. fixed under SM_width) or the corner template wasn't loaded.
        cross_idx, cross_rows = [], []
        for (a, b) in self.parameters.cross_terms:
            if a not in self._idx or b not in self._idx:
                continue
            tag = self.parameters.cross_tag(a, b)
            if tag not in self.morph_scenario:
                continue
            m_a = np.asarray(self.morph_scenario[a]["xsec"])
            m_b = np.asarray(self.morph_scenario[b]["xsec"])
            m_corner = np.asarray(self.morph_scenario[tag]["xsec"])
            x_ab = (1.0 + m_corner) / ((1.0 + m_a) * (1.0 + m_b)) - 1.0
            cross_idx.append((self._idx[a], self._idx[b]))
            cross_rows.append(x_ab)
        if cross_rows:
            self._cross_idx = np.asarray(cross_idx, dtype=int)
            self._cross_matrix = np.stack(cross_rows)
        else:
            self._cross_idx = None
            self._cross_matrix = None

    def rebuild_chi2_state(self, *, init_vars=True, pseudodata=None):
        """Re-run ``create_scenario`` from the current ``scenario_dict``,
        then rebuild the cov and chi2-cache derivations. For scans that
        mutate ``scenario_dict`` in place — restores ``fit`` to a consistent
        state for the next chi² evaluation without touching ``fit.minuit``."""
        self.create_scenario(init_vars=init_vars, pseudodata=pseudodata)
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
            self.create_scenario(init_vars=init_vars, pseudodata=pseudo_data)
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
                self._print_param_extras(name, val)

            pull = self._pull_for(name, val)
            print(f"Pull {name}: {pull.n / pull.s:.3f}")
            if name == "mass":
                print(f"uncertainty in mass: {val.s * 1e3:.2f} MeV")
            print()
        # Print the POI/constraint block (mass, width + the PARAMETERS-key
        # constraint nuisances alphas/aem_isr) PLUS any active global nuisances
        # (e.g. the EW-coupling normalization nuisance aemEW) so their
        # correlation with the POIs is visible. self.parameters.names is the
        # POI/constraint slice (never mutated by add_*_nuisance). Per-bin
        # binned nuisances (lumi/BES/BEC bins) are omitted — they would swamp
        # the matrix.
        n_poi = len(self.parameters.names)
        extra = [i for i, k in enumerate(self.param_names)
                 if i >= n_poi and k in self._active_global_nuisances]
        sel = list(range(n_poi)) + extra
        corr = np.round(unc.correlation_matrix(params_w_cov), 2)
        print("Correlation matrix:")
        print([self.param_names[i] for i in sel])
        print(corr[np.ix_(sel, sel)])
        self.last_fit_results = params_w_cov

    # ------------------------------------------------------------------
    # Nuisance management — data-driven via card.SYSTEMATICS (binned /
    # global entries) + card.PRIORS. Entry scripts call
    # add_binned_nuisance("BEC") / add_global_nuisance("sw2") / etc.
    # ------------------------------------------------------------------
    def add_binned_nuisance(self, kind, *, prior_uncorr=None, prior_corr=None):
        """Activate the binned nuisance ``kind`` (must appear in
        ``card.SYSTEMATICS`` with ``type=binned``): adds N + 1 fit
        parameters (``{kind}_bin0`` … ``{kind}_binN`` plus the correlated
        ``{kind}``) and sets its Gaussian priors. Priors default to
        ``card.PRIORS[kind]``. Idempotent."""
        if kind in self._active_binned_nuisances:
            return
        if kind not in self._systematics_meta["binned"]:
            raise ValueError(
                f"{kind!r} is not declared as type=binned in card.SYSTEMATICS"
            )
        self._active_binned_nuisances.add(kind)
        card_priors = self.card.PRIORS[kind]
        if prior_uncorr is None:
            prior_uncorr = card_priors["uncorr"]
        if prior_corr is None:
            prior_corr = card_priors["corr"]
        self.set_binned_nuisance_priors(kind, uncorr=prior_uncorr, corr=prior_corr)
        self._expand_per_bin_nuisance(kind)

    def set_binned_nuisance_priors(self, kind, *, uncorr, corr):
        iv = self.input_var[kind]
        self._nuisance_priors.setdefault(kind, {})
        self._nuisance_priors[kind]["uncorr"] = uncorr / iv
        self._nuisance_priors[kind]["corr"] = corr / iv

    def add_global_nuisance(self, kind, *, prior=None):
        """Activate the global (non-binned) nuisance ``kind`` (must appear
        in ``card.SYSTEMATICS`` with ``type=global``): adds a single fit
        parameter named ``kind`` and sets its Gaussian prior. Prior
        defaults to ``card.PRIORS[kind]``. Idempotent."""
        if kind in self._active_global_nuisances:
            return
        if kind not in self._systematics_meta["global"]:
            raise ValueError(
                f"{kind!r} is not declared as type=global in card.SYSTEMATICS"
            )
        self._active_global_nuisances.add(kind)
        if prior is None:
            prior = self.card.PRIORS[kind]
        self.set_global_nuisance_prior(kind, prior=prior)
        self.param_names.append(kind)

    def set_global_nuisance_prior(self, kind, *, prior):
        self._nuisance_priors.setdefault(kind, {})["prior"] = prior / self.input_var[kind]

    def set_xsec_systematics(self, *, corr_frac=0.0, uncorr_frac=0.0):
        """Activate the placeholder cross-section systematics: a fully-correlated
        and a fully-uncorrelated component across √s, each sized as the given
        fraction of the per-point statistical uncertainty, added to the data
        covariance in :meth:`_build_cov`. The fractions are stored as the
        syst-table "nominal" so ``reinitialise_to_nominal`` restores them after a
        per-source switch-off. Call after ``init_scenario`` — the covariance is
        rebuilt on the next ``init_minuit`` / ``_build_cov``. Pass 0 for a
        component to leave it off."""
        self.xsec_syst_corr = self._xsec_syst_corr_nom = float(corr_frac)
        self.xsec_syst_uncorr = self._xsec_syst_uncorr_nom = float(uncorr_frac)

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
        for c in self._constraints.values():
            c["sigma"] = OFF
        for kind in self._active_binned_nuisances:
            self._nuisance_priors[kind]["uncorr"] = OFF
            self._nuisance_priors[kind]["corr"] = OFF
        for kind in self._active_global_nuisances:
            self._nuisance_priors[kind]["prior"] = OFF
        self.lumi_corr = OFF
        self.lumi_uncorr = OFF
        # Cov-based cross-section systematics: drop both components (0, not OFF —
        # they are covariance fractions, not penalised nuisances, so a clean
        # zero is exact and keeps the stat-only covariance well-conditioned).
        self.xsec_syst_corr = 0.0
        self.xsec_syst_uncorr = 0.0
        # Rebuild the covariance NOW so the cov-based systematics (xsec, and
        # cov-mode lumi) actually drop. Unlike the penalised nuisances — which
        # the chi² reads live from the priors — these live only in ``self.cov``,
        # so a syst-table refit with ``init_minuit=False`` (the parametric stat
        # branch) would otherwise keep the stale, syst-inflated covariance.
        self._build_cov()

    def reinitialise_to_nominal(self):
        for name, c in self._constraints.items():
            c["sigma"] = self.card.PRIORS[name]
        for kind in self._active_binned_nuisances:
            card_priors = self.card.PRIORS[kind]
            self.set_binned_nuisance_priors(kind, uncorr=card_priors["uncorr"], corr=card_priors["corr"])
        for kind in self._active_global_nuisances:
            self.set_global_nuisance_prior(kind, prior=self.card.PRIORS[kind])
        self.lumi_corr = self.card.PRIORS["lumi"]["corr"]
        self.lumi_uncorr = self.card.PRIORS["lumi"]["uncorr"]
        # Restore the cov-based cross-section systematics to the fractions the
        # driver activated via set_xsec_systematics (0 if never activated), and
        # rebuild the covariance so they take effect even for an ``init_minuit=
        # False`` refit (mirror of reinitialise_to_stat).
        self.xsec_syst_corr = self._xsec_syst_corr_nom
        self.xsec_syst_uncorr = self._xsec_syst_uncorr_nom
        self._build_cov()
