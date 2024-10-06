# Run this script from work directory
# python scripts/plotScaleVars.py

import pandas as pd # type: ignore
import os, sys
import numpy as np
import matplotlib.pyplot as plt # type: ignore

import mplhep as hep # type: ignore
plt.style.use(hep.style.CMS)


sys.path.append(os.path.abspath(os.getcwd()))
from doFit import formFileTag
from xsec_calculator.parameter_def import parameters

input_dir = 'output_scale_vars'
outdir = 'plots/scale_vars'


d_params = parameters(do_scale_vars=True).getDict()

def formFileName(mass_scale, width_scale):
        param_names = list(d_params['nominal'].keys())
        tag = 'nominal'
        infile_tag = formFileTag(*[d_params[tag][p] for p in param_names])
        return 'N3LO_scan_PS_ISR_{}_scaleM{:.1f}_scaleW{:.1f}.txt'.format(infile_tag,mass_scale,width_scale)


def readScan(mass_scale=80.,width_scale=350.):
    filename = input_dir + '/' + formFileName(mass_scale, width_scale)
    if not os.path.exists(filename):
        raise ValueError('File {} not found'.format(filename))
    f = open(filename, 'r')
    df = pd.read_csv(f, header=None, names=['ecm','xsec'])
    f.close()
    return df 

def savePlot(plt, dir, filename, pdf=False):
    plt.xlabel('$\sqrt{s}$ [GeV]')
    plt.ylabel('Cross Section' if 'ratio' not in filename else 'Ratio to $\mu_{}$ = {} GeV'.format('m' if 'mass' in filename else '\Gamma', 80 if 'mass' in filename else 350))
    plt.legend(loc='best')
    plt.title('Preliminary', fontsize=23, loc='right', fontstyle='italic')

    plt.savefig(dir + '/' + filename)
    if pdf:
        plt.savefig(dir + '/' + filename.replace('.png','.pdf'))
    plt.clf()

def main():

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for mass_scale in np.linspace(50., 350., 11):
        df = readScan(mass_scale=mass_scale)
        plt.plot(df['ecm'], df['xsec'], label='$\mu_m$ = {:.0f} GeV'.format(mass_scale))
    savePlot(plt, outdir, 'scale_vars_mass.png')

    for width_scale in np.linspace(50., 350., 11):
        df = readScan(width_scale=width_scale)
        plt.plot(df['ecm'], df['xsec'], label=f"Width Scale: {width_scale}")   
    savePlot(plt, outdir, 'scale_vars_width.png')
    
    # Plot ratios
    for mass_scale in np.linspace(50., 350., 11):
        if mass_scale < 70.: continue #to remove
        df_nom = readScan()
        df = readScan(mass_scale=mass_scale)
        ecm_max = list(df_nom['ecm'])[-2]
        ecm_max = min(min(df['ecm'].max(), df_nom['ecm'].max()),ecm_max)    
        if ecm_max < 347 or np.isnan(ecm_max):
            print(f"Warning: ecm_max = {ecm_max} for mass scale {mass_scale}. Skipping.")
            continue 
        df_nom = df_nom[df_nom['ecm'] <= ecm_max]
        ratio = df['xsec'] / df_nom['xsec']
        plt.plot(df['ecm'], ratio, label='$\mu_m$ = {:.0f} GeV'.format(mass_scale))

    plt.text(.57, 0.9, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
    plt.text(.57, 0.86, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
    savePlot(plt, outdir, 'scale_vars_mass_ratio.png', pdf=True)




  
    ecm_max = 350
    for width_scale in np.linspace(50., 350., 11):
        if width_scale < 70.: continue #to remove
        df = readScan(width_scale=width_scale)
        df = df[df['ecm'] <= ecm_max]
        df_nom = df_nom[df_nom['ecm'] <= ecm_max]
        ratio = df['xsec'] / df_nom['xsec']
        plt.plot(df['ecm'], ratio, label='$\mu_{\Gamma}$ '+'= {:.0f} GeV'.format(width_scale))
    plt.text(.6, 0.1, 'QQbar_Threshold N3LO+ISR', fontsize=22, transform=plt.gca().transAxes, ha='right')
    plt.text(.6, 0.06, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
    savePlot(plt, outdir, 'scale_vars_width_ratio.png', pdf=True)
    

if __name__ == '__main__':
    main()