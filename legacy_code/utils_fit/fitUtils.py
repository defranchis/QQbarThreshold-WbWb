
import numpy as np
import pandas as pd # type: ignore
import scipy.ndimage

def getMaxXsec(df_xsec, range_max_ecm = None, step = 1):
    if range_max_ecm is None:
        range_max_ecm = df_xsec['ecm'].max()
    max_xsec = df_xsec[df_xsec['ecm'] <= range_max_ecm]['xsec'].max()
    max_ecm = df_xsec.loc[df_xsec['xsec'] == max_xsec, 'ecm'].values[0]
    if max_ecm > range_max_ecm - step/2:
        return getMaxXsec(df_xsec, range_max_ecm - step, step)
    return max_xsec, max_ecm

def convoluteXsecGauss(df_xsec, sqrt_res, peak_ecm = None):
    tolerance = 6 # number of decimal places to consider
    pitch = round(df_xsec['ecm'][1] - df_xsec['ecm'][0],tolerance)
    pitches = [round(df_xsec['ecm'][i+1] - df_xsec['ecm'][i],tolerance) for i in range(len(df_xsec['ecm'])-1)]
    if not (pitches == pitch).all():
        raise ValueError('Non-uniform pitch')
    if peak_ecm is None:
        _ , peak_ecm = getMaxXsec(df_xsec)
    sigma = peak_ecm *sqrt_res/(2**.5)/100/pitch
    xsec_smear = scipy.ndimage.gaussian_filter1d(df_xsec['xsec'], sigma)
    df_xsec_smear = pd.DataFrame({'ecm': df_xsec['ecm'], 'xsec': xsec_smear})
    return df_xsec_smear
