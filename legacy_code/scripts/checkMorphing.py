from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from compareISR import convoluteXsecGauss

smearing = True
beam_energy_res = 0.23 # per beam, in percent

orders = ['N3LO']
schemes = ['PS']

plotdir = 'plots/plots_ISR_morphing'

central_mass = {
    'PS': '171.50',
    'MS': '163.00'
}

nominal_width = 1.33

def get_Xsec(order,scheme, mass='', yukawa='', width = '', alphaS='Nominal'):
    if mass == '': mass = central_mass[scheme]
    if yukawa == '': yukawa = '1.0'
    if width == '': width = str(nominal_width)
    print(order, scheme, mass, width, yukawa,alphaS)
    ecms = [round(ecm,1) for ecm in np.arange(340.0, 350.0, 0.1)]
    xsec = []
    for ecm in ecms:
        f = open('zz_old/output_ISR/small_grid_mass_width/{}_scan_{}_ISR_ecm{:.1f}_mass{}_width{}_yukawa{}_as{}.txt'.format(order,scheme,ecm,mass,width,yukawa,alphaS), 'r')
        xsec.append(float(f.readlines()[0].split(',')[-1]))
    df = pd.DataFrame({'ecm': ecms, 'xsec': xsec})

    return df if not smearing else convoluteXsecGauss(df,beam_energy_res)


def addTitles(order,ratio=False):
    plt.legend()
    plt.xlabel('$\sqrt{s}$ [GeV]')
    plt.ylabel('WbWb cross section ratio')
    plt.title('QQbar_Threshold {}'.format(order))
    return



def main():
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    order = 'N3LO'
    scheme = 'PS'

    masses = np.round(np.array([-0.03, 0.0, 0.03])+float(central_mass[scheme]), 2)
    widths = np.round(np.array([-0.05, 0.0, 0.05])+nominal_width, 2)

    for mass in masses:
        for width in widths:
            if mass == float(central_mass[scheme]) or width == nominal_width:
                continue
            df = get_Xsec(order,scheme, mass=mass, width=width)
            df_mass = get_Xsec(order,scheme, mass=mass)
            df_width = get_Xsec(order,scheme, width=width)
            df_nominal = get_Xsec(order,scheme)

            mass_morhp = df_mass['xsec']/df_nominal['xsec']
            width_morhp = df_width['xsec']/df_nominal['xsec']
            xsec_morhp = df_nominal['xsec']*mass_morhp*width_morhp
            df_morhp = pd.DataFrame({'ecm': df['ecm'], 'xsec': xsec_morhp})

            plt.plot(df['ecm'],df['xsec']/df_nominal['xsec'],label='varied')
            plt.plot(df['ecm'],df_morhp['xsec']/df_nominal['xsec'],label='nominal morphed', linestyle='--')
            plt.plot(df['ecm'],df_nominal['xsec']/df_nominal['xsec'],label='nominal')
            addTitles(order)
            plt.savefig('{}/{}_{}_mass{}_width{}.png'.format(plotdir,order,scheme,mass,width))
            plt.close()

            ratio = df_morhp['xsec'] / df['xsec']
            plt.plot(df['ecm'], ratio, label='ratio')
            addTitles(order, ratio=True)
            plt.savefig('{}/{}_{}_mass{}_width{}_ratio.png'.format(plotdir, order, scheme, mass, width))
            plt.close()

    df_mass_up = get_Xsec(order,scheme, mass=masses[-1])
    df_mass_intermediate = get_Xsec(order,scheme, mass=float(central_mass[scheme])+0.02)
    df_nominal = get_Xsec(order,scheme)
    df_mass_up = get_Xsec(order, scheme, mass=masses[-1])
    df_nominal = get_Xsec(order, scheme)
    ratio = df_mass_up['xsec'] / df_nominal['xsec']
    ratio_interm = df_mass_intermediate['xsec'] / df_nominal['xsec']
    ratio_interm_morph = (ratio-1)*2/3 + 1
    plt.plot(df_mass_up['ecm'], ratio, label='mass +30 MeV')
    plt.plot(df_mass_intermediate['ecm'], ratio_interm, label='mass +20 MeV')
    plt.plot(df_mass_up['ecm'], ratio_interm_morph, label='morphed', linestyle='--')
    addTitles(order, ratio=True)
    plt.savefig('{}/{}_{}_mass_up_ratio.png'.format(plotdir, order, scheme))
    plt.close()

    df_width_up = get_Xsec(order, scheme, width=widths[-1])
    df_width_intermediate = get_Xsec(order, scheme, width=nominal_width + 0.03)
    df_nominal = get_Xsec(order, scheme)
    df_width_up = get_Xsec(order, scheme, width=widths[-1])
    df_nominal = get_Xsec(order, scheme)
    ratio = df_width_up['xsec'] / df_nominal['xsec']
    ratio_interm = df_width_intermediate['xsec'] / df_nominal['xsec']
    ratio_interm_morph = (ratio - 1) * 3/5 + 1
    plt.plot(df_width_up['ecm'], ratio, label='width 50 MeV')
    plt.plot(df_width_intermediate['ecm'], ratio_interm, label='width 30 MeV')
    plt.plot(df_width_up['ecm'], ratio_interm_morph, label='morphed', linestyle='--')
    addTitles(order, ratio=True)
    plt.savefig('{}/{}_{}_width_up_ratio.png'.format(plotdir, order, scheme))
    plt.close()

    return

if __name__ == '__main__':
    main()







