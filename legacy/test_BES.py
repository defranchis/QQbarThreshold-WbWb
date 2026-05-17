from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
from scipy.optimize import curve_fit


def convoluteXsecGauss(df_xsec, sqrt_res):
    tolerance = 6 # number of decimal places to consider
    pitch = round(df_xsec['ecm'][1] - df_xsec['ecm'][0],tolerance)
    pitches = [round(df_xsec['ecm'][i+1] - df_xsec['ecm'][i],tolerance) for i in range(len(df_xsec['ecm'])-1)]
    if not (pitches == pitch).all():
        raise ValueError('Non-uniform pitch')
    _ , max_ecm = getMaxXsec(df_xsec)
    sigma = max_ecm *sqrt_res*(2**.5)/100/pitch
    xsec_smear = scipy.ndimage.gaussian_filter1d(df_xsec['xsec'], sigma)
    df_xsec_smear = pd.DataFrame({'ecm': df_xsec['ecm'], 'xsec': xsec_smear})
    return df_xsec_smear

def getMaxXsec(df_xsec, range_max_ecm = None, step = 1):
    if range_max_ecm is None:
        range_max_ecm = df_xsec['ecm'].max()
    max_xsec = df_xsec[df_xsec['ecm'] <= range_max_ecm]['xsec'].max()
    max_ecm = df_xsec.loc[df_xsec['xsec'] == max_xsec, 'ecm'].values[0]
    if max_ecm > range_max_ecm - step/2:
        return getMaxXsec(df_xsec, range_max_ecm - step, step)
    return max_xsec, max_ecm

df_xsec = pd.DataFrame({'ecm': np.arange(340.0, 350.0, 0.001), 'xsec': np.zeros(10000)})
df_xsec.loc[round(df_xsec['ecm'],1) == 345, 'xsec'] = 10
df_xsec_smear = convoluteXsecGauss(df_xsec, 0.23)
plt.plot(df_xsec['ecm'], df_xsec['xsec'], label='N3LO')
plt.plot(df_xsec_smear['ecm'], df_xsec_smear['xsec'], label='N3LO+ISR')
plt.legend()
plt.xlabel('$\sqrt{s}$ [GeV]')
plt.ylabel('Cross section [pb]')
plt.title('N3LO vs N3LO+ISR')
plt.savefig('test.png')

def getFWHM(df_xsec_smear):
    max_xsec = df_xsec_smear['xsec'].max()
    half_max_xsec = max_xsec / 2
    left_index = df_xsec_smear[df_xsec_smear['xsec'] >= half_max_xsec].index[0]
    right_index = df_xsec_smear[df_xsec_smear['xsec'] >= half_max_xsec].index[-1]
    fwhm = df_xsec_smear['ecm'][right_index] - df_xsec_smear['ecm'][left_index]
    return fwhm

fwhm = getFWHM(df_xsec_smear)
std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))
print("Standard Deviation:", std_dev)
print('expected = ',345*0.23*(2**.5)/100)
print('ratio = ',std_dev/(345*0.23*(2**.5)/100))