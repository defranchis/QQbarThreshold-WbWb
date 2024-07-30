from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

orders = ['NLO', 'NNLO', 'N3LO']
#schemes = ['MS', 'PS']
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

def get_Xsec(order,scheme, mass='', yukawa='', width = '', alphaS='Nominal'):
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

def doPlotScheme(scheme):

    if scheme not in schemes:
        raise ValueError('Invalid scheme')

    # Get data for the reference order (the first order in the list)
    reference_order = orders[-1]
    df_reference = get_Xsec(reference_order, scheme)
    max_xsec = df_reference['xsec'].max()
    max_ecm = df_reference.loc[df_reference['xsec'] == max_xsec, 'ecm'].values[0]

    # Plot the cross section vs energy for different orders
    for order in orders:
        df = get_Xsec(order, scheme)
        plt.plot(df['ecm'], df['xsec'], label=order)
    plt.axvline(x=max_ecm, color='black', linestyle='dotted',label='peak position')
    plt.axvline(x=340, color='grey', linestyle='dotted',label='validity region')
    plt.axvline(x=345, color='grey', linestyle='dotted',label='')
    plt.legend()

    plt.xlabel('Energy')
    plt.ylabel('Cross Section')
    plt.title('Cross Section vs Energy')
    plt.savefig('{}/plot_{}.png'.format(plotdir,scheme))
    plt.clf()

    # Plot the ratio of cross sections with respect to the reference order
    for order in orders:  # Skip the reference order
        df = get_Xsec(order, scheme)
        ratio = df['xsec'] / df_reference['xsec']
        plt.plot(df['ecm'], ratio, label=f'{order} / {reference_order}')
    plt.axvline(x=max_ecm, color='black', linestyle='dotted',label='peak position')
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Ratio of Cross Section')
    plt.title(f'Ratio of Cross Section vs Energy (Relative to {reference_order})')
    plt.savefig('{}/plot_{}_ratio.png'.format(plotdir,scheme))
    plt.clf()
    return


def compareSchemes(order):
    if not order in orders:
        raise ValueError('Invalid order')
    for scheme in schemes:
        df = get_Xsec(order,scheme)
        plt.plot(df['ecm'], df['xsec'], label=scheme)
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Cross Section')
    plt.title('Cross Section vs Energy')
    plt.savefig('{}/plot_{}.png'.format(plotdir,order))
    plt.clf()

def compareMasses(order,scheme):
    if not order in orders:
        raise ValueError('Invalid order')
    if not scheme in schemes:
        raise ValueError('Invalid scheme')
    for mass in mass_scan[scheme]:
        df = get_Xsec(order,scheme,mass)
        plt.plot(df['ecm'], df['xsec'], label=mass)
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Cross Section')
    plt.title('Cross Section vs Energy')
    plt.savefig('{}/plot_{}_{}_mass.png'.format(plotdir,order,scheme))
    plt.clf()


def compareYukawa(order,scheme):
    if order not in orders:
        raise ValueError('Invalid order')
    if scheme not in schemes:
        raise ValueError('Invalid scheme')

    # Get data for Yukawa = 1.0
    df_y1 = get_Xsec(order, scheme, '', '1.0')

    # Plot the cross section vs energy for different Yukawa values
    for yt in ['0.9', '1.0', '1.1']:
        df = get_Xsec(order, scheme, '', yt)
        plt.plot(df['ecm'], df['xsec'], label=f'Yukawa = {yt}')
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Cross Section')
    plt.title('Cross Section vs Energy')
    plt.savefig('{}/plot_{}_{}_yukawa.png'.format(plotdir,order, scheme))
    plt.clf()

    # Plot the ratio of cross sections with respect to Yukawa = 1.0
    for yt in ['0.9', '1.1']:
        df = get_Xsec(order, scheme, '', yt)
        ratio = df['xsec'] / df_y1['xsec']
        plt.plot(df['ecm'], ratio, label=f'Yukawa = {yt} / Yukawa = 1.0')
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Ratio of Cross Section')
    plt.title('Ratio of Cross Section vs Energy')
    plt.savefig('{}/plot_{}_{}_yukawa_ratio.png'.format(plotdir,order, scheme))
    plt.clf()

def compareAlphaS(order,scheme):

    if order not in orders:
        raise ValueError('Invalid order')
    if scheme not in schemes:
        raise ValueError('Invalid scheme')

    # Get data for alphaS = 'Nominal'
    df_nominal = get_Xsec(order, scheme, width='', yukawa='', alphaS='Nominal')

    # Plot the cross section vs energy for different alphaS values
    for alphaS in ['Nominal', 'Up', 'Down']:
        df = get_Xsec(order, scheme, width='', yukawa='', alphaS=alphaS)
        plt.plot(df['ecm'], df['xsec'], label=alphaS)
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Cross Section')
    plt.title('Cross Section vs Energy')
    plt.savefig('{}/plot_{}_{}_alphaS.png'.format(plotdir,order, scheme))
    plt.clf()

    # Plot the ratio of cross sections with respect to alphaS = 'Nominal'
    for alphaS in ['Up', 'Down']:
        df = get_Xsec(order, scheme, width='', yukawa='', alphaS=alphaS)
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label=f'{alphaS} / Nominal')
    plt.legend()
    plt.xlabel('Energy')
    plt.ylabel('Ratio of Cross Section')
    plt.title('Ratio of Cross Section vs Energy')
    plt.savefig('{}/plot_{}_{}_alphaS_ratio.png'.format(plotdir,order, scheme))
    plt.clf()

def compareWidthYukawaAlphaS(order,scheme):
    if order not in orders:
        raise ValueError('Invalid order')
    if scheme not in schemes:
        raise ValueError('Invalid scheme')

    # Get data for Yukawa = 1.0
    df_nominal = get_Xsec(order, scheme, '', '1.0')
    energy_peak = float(central_mass[scheme])*2
    max_xsec = df_nominal[(df_nominal['ecm'] >= energy_peak-1) & (df_nominal['ecm'] <= energy_peak+1)]['xsec'].max()
    max_ecm = df_nominal.loc[df_nominal['xsec'] == max_xsec, 'ecm'].values[0]
    print("ecm value corresponding to max xsec:", max_ecm)
    df_nominal['ecm'] -= max_ecm

    # Plot the cross section vs energy for different Yukawa values
    for yt in ['0.9', '1.0', '1.1']:
        df = get_Xsec(order, scheme, '', yt)
        df['ecm'] -= max_ecm
        plt.plot(df['ecm'], df['xsec'], label=f'Yukawa = {yt}')
    addTitles(plt,order)
    plt.savefig('{}/plot_{}_{}_yukawa.png'.format(plotdir,order, scheme))
    plt.clf()

    # Plot the ratio of cross sections with respect to Yukawa = 1.0
    for yt in ['0.9', '1.1']:
        df = get_Xsec(order, scheme, '', yt)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label=f'Yukawa = {yt} / Yukawa = 1.0')
    addTitles(plt,order)
    plt.savefig('{}/plot_{}_{}_yukawa_ratio.png'.format(plotdir,order, scheme))
    plt.clf()

    for alphaS in ['Up', 'Down']:
        df = get_Xsec(order, scheme, width='', yukawa='', alphaS=alphaS)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label=f'{alphaS} / Nominal')
    addTitles(plt,order)
    plt.savefig('{}/plot_{}_{}_alphaS_ratio.png'.format(plotdir,order, scheme))
    plt.clf()

    for width in ['1.28', '1.33', '1.38']:
        df = get_Xsec(order, scheme, width=width)
        df['ecm'] -= max_ecm
        plt.plot(df['ecm'], df['xsec'], label=f'Width = {width}')
    addTitles(plt,order)
    plt.savefig('{}/plot_{}_{}_width.png'.format(plotdir,order, scheme))
    plt.clf()

    for yt in ['0.9', '1.1']:
        df = get_Xsec(order, scheme, '', yt)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label='Yukawa +{:.0f}%'.format((float(yt)-1)*100) if float(yt) > 1 else '', color = 'blue', linestyle='dashed' if float(yt) < 1 else 'solid')  
    for alphaS in ['Up', 'Down']:
        df = get_Xsec(order, scheme, width='', yukawa='', alphaS=alphaS)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label='aS + 0.0002 (from Z)' if alphaS=='Up' else '', color='orange', linestyle='dashed' if alphaS == 'Down' else 'solid')
    for width in ['1.28', '1.38']:
        df = get_Xsec(order, scheme, width=width)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label=f'Width + 50 MeV' if float(width) > 1.33 else '', color='red', linestyle='dashed' if float(width) < 1.33 else 'solid')
    for mass in ['{:.2f}'.format(float(central_mass[scheme])+0.03), '{:.2f}'.format(float(central_mass[scheme])-0.03)]:
        df = get_Xsec(order, scheme, mass=mass)
        df['ecm'] -= max_ecm
        ratio = df['xsec'] / df_nominal['xsec']
        plt.plot(df['ecm'], ratio, label='mt +{:.0f} MeV'.format((float(mass)-float(central_mass[scheme]))*1000) if float(mass) > float(central_mass[scheme]) else '', color='green', linestyle='dashed' if float(mass) < float(central_mass[scheme]) else 'solid')
        if float(mass) > float(central_mass[scheme]):
            min_ratio = ratio.min()
            min_ratio_ecm = df.loc[ratio == min_ratio, 'ecm'].values[0]
            print("min ratio:", min_ratio_ecm)
            #plt.axvline(x=max_ratio_ecm, color='red', linestyle='dotted', label='Max Ratio')

    plt.axvline(x=0, color='black', linestyle='dotted',label='$\sqrt{s}$'+'={:.1f} GeV (peak)'.format(max_ecm))
    plt.axvline(x=340-max_ecm, color='grey', linestyle='dotted',label='{} validity region'.format(order))
    plt.axvline(x=345-max_ecm, color='grey', linestyle='dotted',label='')

    
    plt.xticks(np.arange(int(df_nominal['ecm'].min())-2, int(df_nominal['ecm'].max())+3, 2))
    plt.legend()
    plt.ylim(0.95,1.05)
    addTitles(plt,order,ratio=True)
    plt.savefig('{}/plot_{}_{}_yukawa_alphaS_ratio.png'.format(plotdir,order, scheme))
    plt.clf()

def addTitles(plt,order,ratio=False):
    plt.legend()
    plt.xlabel('$\sqrt{s} - m_{res}$ [GeV]')
    plt.ylabel('WbWb cross section {}'.format('ratio' if ratio else '[pb]'))
    plt.title('QQbar_Threshold {}'.format(order))
    return

def main():
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    doPlotScheme('PS')
    for order in orders:
        compareWidthYukawaAlphaS(order,'PS')
    #compareWidthYukawaAlphaS('N3LO','MS')

    



    return

if __name__ == '__main__':
    main()







