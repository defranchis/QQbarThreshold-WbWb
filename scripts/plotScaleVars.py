# Run this script from work directory
# python scripts/plotScaleVars.py

import pandas as pd # type: ignore
import os, sys
import numpy as np
import matplotlib.pyplot as plt # type: ignore

sys.path.append(os.path.abspath(os.getcwd()))
from doFit import formFileTag
from xsec_calculator.parameter_def import parameters

input_dir = 'output_scale_vars'
outdir = 'plots/scale_vars'

d_params = parameters().getDict()

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

def savePlot(plt, dir, filename):
    plt.xlabel('ecm [GeV]')
    plt.ylabel('Cross Section')
    plt.legend()
    plt.savefig(dir + '/' + filename)
    plt.clf()

def main():

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for mass_scale in np.linspace(50., 350., 11):
        df = readScan(mass_scale=mass_scale)
        plt.plot(df['ecm'], df['xsec'], label=f"Mass Scale: {mass_scale}")
    savePlot(plt, outdir, 'scale_vars_mass.png')

    for width_scale in np.linspace(50., 350., 11):
        df = readScan(width_scale=width_scale)
        plt.plot(df['ecm'], df['xsec'], label=f"Width Scale: {width_scale}")   
    savePlot(plt, outdir, 'scale_vars_width.png')
    
    # Plot ratios
    max_ecm = 346 # some ecm points are not calculated for higher mass scales. to be checked
    df_nom = readScan()
    df_nom = df_nom[df_nom['ecm'] < max_ecm]
    for mass_scale in np.linspace(50., 350., 11):
        df = readScan(mass_scale=mass_scale)
        df = df[df['ecm'] < max_ecm]
        ratio = df['xsec'] / df_nom['xsec']
        try:
            plt.plot(df['ecm'], ratio, label=f"Mass Scale: {mass_scale}")
        except:
            print('skipping mass scale', mass_scale)
    savePlot(plt, outdir, 'scale_vars_mass_ratio.png')

    for width_scale in np.linspace(50., 350., 11):
        df = readScan(width_scale=width_scale)
        ratio = df['xsec'] / df_nom['xsec']
        plt.plot(df['ecm'], ratio, label=f"Width Scale: {width_scale}")
    savePlot(plt, outdir, 'scale_vars_width_ratio.png')
    

if __name__ == '__main__':
    main()