"""WbWb-specific fit subclass.

Adds the SM-width constraint hook used when ``--SMwidth`` is set: the top
width is rewritten from the floating mass via the N3LO QCD relation, and
the "theory" parameter is given a Gaussian prior centred at one with width
``input_uncert_SM_width`` (in MeV).

Also owns the Yukawa-as-nuisance toggle (``constrain_yukawa`` kwarg +
property): WbWb treats yukawa as either a free POI or a Gaussian-priored
nuisance depending on whether the above-threshold ECM point is included
(see ``_validate_scenario``). The mechanism is WbWb-specific — no other
process in the framework has this dual-use top yukawa parameter.
"""

import uncertainties as unc

from common.fit_core import _OFF, FitCore


class WbWbFit(FitCore):
    """FitCore + the SM-width / Yukawa pieces that only make sense for top."""

    def __init__(self, *args, sm_width=False, constrain_yukawa=True, **kwargs):
        # sm_width must be set BEFORE super().__init__() so the
        # _select_pseudodata_tag hook (called by FitCore.__init__) sees it.
        self.sm_width = sm_width
        super().__init__(*args, **kwargs)
        self.input_uncert_SM_width = self.card.PRIORS.get("SM_width", _OFF)
        if "yukawa" in self._constraints:
            self._constraints["yukawa"]["active"] = constrain_yukawa

    @property
    def constrain_yukawa(self):
        return self._constraints["yukawa"]["active"]

    @constrain_yukawa.setter
    def constrain_yukawa(self, v):
        self._constraints["yukawa"]["active"] = v

    def _validate_scenario(self, add_last_ecm):
        if self.constrain_yukawa and add_last_ecm:
            raise ValueError(
                "Yukawa constraint + last-ecm point unsupported; "
                "pass constrain_yukawa=False (--fitYukawa) to float Yukawa, "
                "or set add_last_ecm=False (drop --lastecm) to skip the above-threshold point."
            )

    def _select_pseudodata_tag(self):
        return "mass_var" if self.sm_width else "pseudodata"

    def is_scannable_poi(self, name):
        # Under SM_width "width" is a theory knob constrained to 1, not
        # a free POI — don't include it in scan profiles.
        if name == "width" and self.sm_width:
            return False
        return super().is_scannable_poi(name)

    def _print_param_extras(self, name, val):
        if name == "width" and self.sm_width:
            print("including theory uncertainty in SM relation")
            print(f"fitted theory parameter = {self.minuit.values[name]:.2f} +/- "
                  f"{self.minuit.errors[name]:.2f} (constrained to 1)")
        if name == "yukawa" and self.constrain_yukawa:
            print(f"constrained with uncertainty {self._constraints['yukawa']['sigma']:.3f}")

    def _pull_for(self, name, val):
        # Under SM_width the width fit parameter is a dimensionless theory
        # knob constrained to 1, not the physical Γ_t — pull from the raw
        # minuit value/error instead of value_from_param(val).
        if name == "width" and self.sm_width:
            return unc.ufloat(self.minuit.values[name], self.minuit.errors[name])
        return super()._pull_for(name, val)

    def stat_breakdown_default(self):
        # Skip the per-POI stat breakdown when yukawa is held constrained;
        # the breakdown is uninformative in that regime.
        return not self.constrain_yukawa

    def physical_fit_params(self, params):
        """Resolve the SM-width relation when ``--SMwidth`` is active.

        Under SM_width=True, the "width" fit parameter is reinterpreted as
        a dimensionless theory knob: the physical width is derived from
        mass + theory_knob via :meth:`_width_n3lo_local_linearisation`,
        and the knob itself gets a Gaussian-priored-at-zero-width-1
        penalty (its only constraint, since no data term references it
        directly).

        The caller pre-copies ``params`` (see ``FitCore.physical_fit_params``
        docstring), so we mutate it in place.
        """
        if not self.sm_width:
            return params, 0.0
        i_width = self._idx["width"]
        i_mass = self._idx["mass"]
        prior_width = params[i_width] ** 2
        width = self._width_n3lo_local_linearisation(
            mt_PS=self.value_from_param(params[i_mass], "mass"),
            theory_knob=params[i_width],
            th_uncert_mev=self.input_uncert_SM_width,
        )
        params[i_width] = self.param_from_value(width, "width")
        return params, prior_width

    # ------------------------------------------------------------------
    # SM Γ_t vs PS mass — two forms of the same N3LO prediction.
    # ``physical_fit_params`` calls the *local-linearisation* form; the
    # *pole-form* is the original parity-with-doFit version and is kept here
    # as a reference (no live call site).
    # ------------------------------------------------------------------
    def _width_n3lo_local_linearisation(self, mt_PS, theory_knob=0.0, th_uncert_mev=5.0):
        """N3LO top width vs PS mass, **linearised around the pseudodata
        anchor** ``(mt_pseudo, Γ_pseudo)`` taken from the ``mass_var`` tag in
        the card's parameter dict.

        Returns ``Γ_pseudo + 0.027 · (mt_PS − mt_pseudo) + knob · th_uncert``.

        Valid whenever the pseudodata mass sits within a few hundred MeV of
        the SM prediction so the linear slope (0.027 GeV/GeV) captures the
        whole relation. ``th_uncert_mev`` is the symmetric theory band on
        Γ_SM (default 5 MeV per arXiv:2309.01937) modulated by the floating
        ``theory_knob`` (= the fit's width parameter, Gaussian-priored at
        zero with width 1).
        """
        mt_ref = self.d_params["mass_var"]["mass"]
        return (self.d_params["mass_var"]["width"]
                + 0.027 * (mt_PS - mt_ref)
                + theory_knob * th_uncert_mev * 1e-3)

    def _width_n3lo_pole_form(self, mt_PS, theory_knob=0.0):
        """N3LO top width vs **pole** mass, anchored at the absolute SM
        prediction ``(m_pole, Γ) = (172.69 GeV, 1.3148 GeV)``.

        Returns ``1.3148 + 0.027 · (mt_pole − 172.69) + knob · 5 MeV``.

        Needs a PS→pole conversion (via ``utils_convert.scheme_conversion``)
        because the SM prediction is anchored in pole mass; therefore more
        expensive than the linearised form and requires the converter to be
        importable. Use when the linearisation anchor (``mass_var``) drifts
        away from the SM — re-wire ``physical_fit_params`` to call this.
        """
        import utils_convert.scheme_conversion as scheme_conversion  # type: ignore
        mt_ref = self.d_params["mass_var"]["mass"]
        mt_pole = mt_PS + scheme_conversion.calculate_mt_Pole(mt_ref, self.mass_scale) - mt_ref
        return 1.3148 + 0.027 * (mt_pole - 172.69) + theory_knob * 0.005
