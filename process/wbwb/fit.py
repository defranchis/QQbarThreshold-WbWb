"""WbWb-specific fit subclass.

Adds the SM-width constraint hook used when ``--SMwidth`` is set: the top
width is rewritten from the floating mass via the N3LO QCD relation, and
the "theory" parameter is given a Gaussian prior centred at one with width
``input_uncert_SM_width`` (in MeV).
"""

from common.fit_core import FitCore


class WbWbFit(FitCore):
    """FitCore + the SM-width / Yukawa pieces that only make sense for top."""

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
