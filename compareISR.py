from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

orders = ['NLO', 'NNLO', 'N3LO']
schemes = ['PS']

plotdir = 'plots_ISR'

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
        f = open('output_ISR/{}_scan_{}_ISR_ecm{:.1f}_mass{}_width{}_yukawa{}_as{}'.format(order,scheme,ecm,mass,width,yukawa,alphaS), 'r')
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
    f = open('output/{}_scan_{}_noISR_mass{}_width{}_yukawa{}_as{}.txt'.format(order,scheme,mass,width,yukawa,alphaS), 'r')
    for l in f.read().splitlines():
        ecm, x = l.split(',')
        ecms.append(float(ecm))
        xsec.append(float(x))
    df = pd.DataFrame({'ecm': ecms, 'xsec': xsec})
    return df


def main():
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    df_ISR = get_Xsec_ISR('N3LO','PS')
    df = get_Xsec('N3LO','PS')
    plt.plot(df['ecm'], df['xsec'], label='N3LO')
    plt.plot(df_ISR['ecm'], df_ISR['xsec'], label='N3LO+ISR')
    plt.legend()
    plt.xlabel('ECM [GeV]')
    plt.ylabel('Cross section [pb]')
    plt.title('N3LO vs N3LO+ISR')
    plt.savefig('{}/ISR_comparison_N3LO.png'.format(plotdir))

    # Ratio plot
    plt.clf()
    plt.plot(df['ecm'], df_ISR['xsec'] / df['xsec'], label='N3LO+ISR / N3LO')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.legend()
    plt.xlabel('ECM [GeV]')
    plt.ylabel('Ratio')
    plt.title('Ratio of N3LO+ISR to N3LO')
    plt.savefig('{}/ISR_ratio_N3LO.png'.format(plotdir))

    



    return

if __name__ == '__main__':
    main()







