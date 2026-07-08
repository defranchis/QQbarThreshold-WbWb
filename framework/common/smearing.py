"""Gaussian smearing of a cross-section lineshape by the beam-energy spectrum."""

import numpy as np
import pandas as pd
import scipy.ndimage


_PITCH_TOLERANCE = 6  # decimals


def find_peak(df_xsec, range_max_ecm=None, step=1.0):
    """Return ``(peak_xsec, peak_ecm)`` ignoring an out-of-range tail.

    The recursion mimics the original behaviour: if the maximum is found in
    the last ``step``-wide window of the provided range, the range is shrunk
    and the search repeated — this protects against picking up an isolated
    above-threshold point (e.g. the 365 GeV point) as the threshold peak.
    """
    if range_max_ecm is None:
        range_max_ecm = df_xsec["ecm"].max()
    in_range = df_xsec[df_xsec["ecm"] <= range_max_ecm]
    peak_xsec = in_range["xsec"].max()
    peak_ecm = in_range.loc[in_range["xsec"] == peak_xsec, "ecm"].values[0]
    if peak_ecm > range_max_ecm - step / 2:
        return find_peak(df_xsec, range_max_ecm - step, step)
    return peak_xsec, peak_ecm


def convolute_gauss(df_xsec, beam_energy_res, peak_ecm=None):
    """Convolve the lineshape with a Gaussian of width ``BES * peak / sqrt(2)``.

    Parameters
    ----------
    df_xsec : DataFrame with columns ``ecm`` (uniformly spaced) and ``xsec``.
    beam_energy_res : relative beam energy spread, in **percent**, per beam.
    peak_ecm : reference ecm used for the kernel width; if ``None`` it is
        looked up via :func:`find_peak`.
    """
    ecms = df_xsec["ecm"].to_numpy()
    if len(ecms) < 2:
        raise ValueError("convolute_gauss needs >=2 ecm points")
    pitches = np.round(np.diff(ecms), _PITCH_TOLERANCE)
    pitch = pitches[0]
    if not np.all(pitches == pitch):
        raise ValueError("convolute_gauss requires a uniform ecm pitch")

    if peak_ecm is None:
        _, peak_ecm = find_peak(df_xsec)

    sigma_bins = peak_ecm * beam_energy_res / np.sqrt(2) / 100 / pitch
    xsec_smeared = scipy.ndimage.gaussian_filter1d(df_xsec["xsec"], sigma_bins)
    return pd.DataFrame({"ecm": df_xsec["ecm"], "xsec": xsec_smeared})
