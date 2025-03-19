from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
import mplhep as hep
plt.style.use(hep.style.CMS)

import scipy.ndimage

orders = ['NLO', 'NNLO', 'N3LO']
schemes = ['PS']

plotdir = 'plots/plots_ISR'

mass_scan = {
    'PS': ['170.5', '171.5', '172.5'],
    'MS': ['159.0', '160.0', '161.0']
}

central_mass = {
    'PS': '171.50',
    'MS': '163.00'
}

def get_Xsec_ISR(order,scheme, mass='', yukawa='', width = '', alphaS='Nominal'):
    if mass == '': mass = central_mass[scheme]
    if yukawa == '': yukawa = '1.0'
    if width == '': width = '1.33'
    print(order, scheme, mass, width, yukawa,alphaS)
    ecms = [round(ecm,1) for ecm in np.arange(335.0, 360.0, 0.1)]
    xsec = []
    for ecm in ecms:
        f = open('zz_old/output_ISR/no_grid/{}_scan_{}_ISR_ecm{:.1f}_mass{}_width{}_yukawa{}_as{}'.format(order,scheme,ecm,mass,width,yukawa,alphaS), 'r')
        xsec.append(float(f.readlines()[0].split(',')[-1]))
    df = pd.DataFrame({'ecm': ecms, 'xsec': xsec})
    return df

def get_Xsec(order,scheme, mass='', yukawa='', width = '', alphaS='Nominal'):
    if mass == '': mass = central_mass[scheme]
    if yukawa == '': yukawa = '1.0'
    if width == '': width = '1.33'
    print(order, scheme, mass, width, yukawa,alphaS)
    ecms = []

    xsec = []
    f = open('zz_old/output/{}_scan_{}_noISR_mass{}_width{}_yukawa{}_as{}.txt'.format(order,scheme,mass,width,yukawa,alphaS), 'r')
    for l in f.read().splitlines():
        ecm, x = l.split(',')
        ecms.append(float(ecm))
        xsec.append(float(x))
    df = pd.DataFrame({'ecm': ecms, 'xsec': xsec})
    return df

def getMaxXsec(df_xsec, range_max_ecm = None, step = 1):
    if range_max_ecm is None:
        range_max_ecm = df_xsec['ecm'].max()
    max_xsec = df_xsec[df_xsec['ecm'] <= range_max_ecm]['xsec'].max()
    max_ecm = df_xsec.loc[df_xsec['xsec'] == max_xsec, 'ecm'].values[0]
    if max_ecm > range_max_ecm - step/2:
        return getMaxXsec(df_xsec, range_max_ecm - step, step)
    return max_xsec, max_ecm

def convoluteXsecGauss(df_xsec, sqrt_res):
    tolerance = 6 # number of decimal places to consider
    pitch = round(df_xsec['ecm'][1] - df_xsec['ecm'][0],tolerance)
    pitches = [round(df_xsec['ecm'][i+1] - df_xsec['ecm'][i],tolerance) for i in range(len(df_xsec['ecm'])-1)]
    if not (pitches == pitch).all():
        raise ValueError('Non-uniform pitch')
    _ , max_ecm = getMaxXsec(df_xsec)
    sigma = max_ecm *sqrt_res/(2**.5)/100/pitch
    xsec_smear = scipy.ndimage.gaussian_filter1d(df_xsec['xsec'], sigma)
    df_xsec_smear = pd.DataFrame({'ecm': df_xsec['ecm'], 'xsec': xsec_smear})
    return df_xsec_smear


def main():
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    df = get_Xsec('N3LO','PS')
    df_ISR = get_Xsec_ISR('N3LO','PS')
    beam_energy_res = 0.23 # per beam, in percent
    df_ISR_conv = convoluteXsecGauss(df_ISR, beam_energy_res)
    fig, ax  = plt.subplots()
    plt.plot(df['ecm'], df['xsec'], label='N$^3$LO', linestyle=':', linewidth=4)
    plt.plot(df_ISR['ecm'], df_ISR['xsec'], label='N$^3$LO+ISR', linestyle='--', linewidth=3)
    plt.plot(df_ISR_conv['ecm'], df_ISR_conv['xsec'], label='N$^3$LO+ISR+BES', linestyle='-', linewidth=3)
    plt.legend(loc='lower right', fontsize=23)
    plt.xlabel('$\sqrt{s}$ [GeV]')
    plt.xlim(339, 352)
    plt.ylabel('WbWb total cross section [pb]')
    # plt.title('Preliminary', fontsize=23, loc='right', fontstyle='italic')

    plt.text(0.95, 0.37, 'QQbar_Threshold N$^3$LO', fontsize=23, transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.33, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.27, 'FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')

    plt.savefig('{}/ISR_comparison_N3LO.png'.format(plotdir))
    plt.savefig('{}/ISR_comparison_N3LO.pdf'.format(plotdir))

    
    # Ratio plot
    plt.clf()
    plt.plot(df['ecm'], df_ISR['xsec'] / df['xsec'], label='N3LO+ISR / N3LO')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.legend()
    plt.xlabel('$\sqrt{s}$ [GeV]')
    plt.ylabel('Ratio')
    plt.title('Ratio of N3LO+ISR to N3LO')
    plt.savefig('{}/ISR_ratio_N3LO.png'.format(plotdir))


    return

if __name__ == '__main__':
    main()







