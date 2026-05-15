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
        if not self.sm_width:
            return 0.0
        i_width = self._idx["width"]
        i_mass = self._idx["mass"]
        prior_width = params[i_width] ** 2
        width = self._width_n3lo(
            mt_PS=self.value_from_param(params[i_mass], "mass"),
            theory_knob=params[i_width],
            th_uncert_mev=self.input_uncert_SM_width,
        )
        params[i_width] = self.param_from_value(width, "width")
        return prior_width

    def _width_n3lo(self, mt_PS, theory_knob=0.0, th_uncert_mev=5.0, fix_to_nom=True):
        """N3LO top width vs PS mass.

        With ``fix_to_nom=True`` (the historical default), the relation is
        anchored to the pseudodata point and only the slope (0.027 GeV/GeV)
        and the theory shift (``th_uncert_mev`` MeV) survive.
        """
        mt_ref = self.d_params["mass_var"]["mass"]
        if fix_to_nom:
            return (self.d_params["mass_var"]["width"]
                    + 0.027 * (mt_PS - mt_ref)
                    + theory_knob * th_uncert_mev * 1e-3)
        # Full conversion path (unused by default, kept for parity with the
        # original code).
        import utils_convert.scheme_conversion as scheme_conversion  # type: ignore
        mt_pole = mt_PS + scheme_conversion.calculate_mt_Pole(mt_ref, self.mass_scale) - mt_ref
        return 1.3148 + 0.027 * (mt_pole - 172.69) + theory_knob * 0.005
