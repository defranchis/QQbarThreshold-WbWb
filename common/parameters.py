"""Parameter bookkeeping for the threshold fit.

The class takes a per-parameter dictionary (typically ``CARD.PARAMETERS``)
and builds the nested ``{tag: {param_name: value}}`` dictionary the rest of
the fit machinery consumes.

Tags produced:
  * ``nominal``     — central values, used for the reference template.
  * ``pseudodata``  — central + per-parameter ``pseudo`` offset (the "true"
                      values used to build the pseudodata cross section).
  * ``<param>_var`` — central + ``variation`` for ``<param>``, central for
                      every other parameter; used to build morphing templates.

Compared to the old ``xsec_calculator/parameter_def.py``, the parameter list
is no longer hardcoded — it follows the keys of the card dictionary, so
adding/removing parameters of interest is a card-level change only.
"""

import copy


class Parameters:
    def __init__(self, card_parameters, scale_vars=None):
        """
        Parameters
        ----------
        card_parameters : mapping ``{name: {nominal, pseudo, variation, round_dec}}``.
        scale_vars : list of renormalisation-scale variation values, or ``None``.
        """
        self._raw = copy.deepcopy(card_parameters)
        self.names = list(self._raw.keys())
        self.scale_vars = list(scale_vars) if scale_vars else []
        self._dict = self._build_dict()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _round(self, name, value):
        return round(value, self._raw[name]["round_dec"])

    def _build_dict(self):
        nominal = {n: self._round(n, self._raw[n]["nominal"]) for n in self.names}
        pseudo = {
            n: self._round(n, self._raw[n]["nominal"] + self._raw[n]["pseudo"])
            for n in self.names
        }
        out = {"nominal": nominal, "pseudodata": pseudo}
        for n in self.names:
            varied = dict(nominal)
            variation = self._raw[n]["variation"]
            varied[n] = self._round(n, self._raw[n]["nominal"] + variation)
            # Fail fast: a variation that rounds away (literal 0 or
            # < 10^-round_dec) gives step=0 → param_from_value silently
            # produces NaN/inf for that parameter downstream.
            if varied[n] == nominal[n]:
                raise ValueError(
                    f"Parameter '{n}' has step=0 after rounding "
                    f"(variation={variation}, round_dec={self._raw[n]['round_dec']}). "
                    f"Either increase variation or decrease round_dec in the card."
                )
            out[f"{n}_var"] = varied
        return out

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------
    @property
    def tags(self):
        """All variation tags ``['nominal', 'pseudodata', '<name>_var', ...]``."""
        return list(self._dict.keys())

    def values(self, tag):
        """Return the dict ``{param_name: value}`` for a given variation tag."""
        return self._dict[tag]

    def as_dict(self):
        """Return a *copy* of the full nested dictionary."""
        return copy.deepcopy(self._dict)

    def step(self, name):
        """Distance between the ``<name>_var`` value and the nominal value.

        Used by the chi2 to convert a fitted dimensionless parameter back into
        a physical shift (and vice versa).
        """
        return self._dict[f"{name}_var"][name] - self._dict["nominal"][name]
