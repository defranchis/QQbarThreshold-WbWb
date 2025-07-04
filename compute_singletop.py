import xsec_calculator.xsec_single_top as xsec_single_top # type: ignore
from xsec_calculator.parameter_def import parameters
import multiprocessing, argparse, time, sys, copy
import numpy as np

class XsecCalculator:
    def __init__(self, outdir):
        self.outdir = outdir
        self.params = parameters(False)
        self.d = self.params.getDict()


    def calculate_xsec(self, key, order = 2, mass_shift = 0):
        mass_scale = self.params.mass_scale
        width_scale = self.params.width_scale
        param_dict = self.d

        print(f"Calculating cross section for {key}...")
        xsec_single_top.do_scan(order=order, PS_mass=param_dict[key]['mass']+mass_shift, width=param_dict[key]['width'], mass_scale=mass_scale, width_scale=width_scale,
                          yukawa=param_dict[key]['yukawa'], as_var=param_dict[key]['alphas'], outdir=self.outdir)
                

    def run_calculations(self):
        self.calculate_xsec('nominal', order = 0)
        self.calculate_xsec('nominal', order = 0, mass_shift = 1)
        # self.calculate_xsec('nominal', order = 1)
        # self.calculate_xsec('nominal', order = 2)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='output_single_top')

    args = parser.parse_args()

    xsec_calculator = XsecCalculator(outdir=args.outdir)
    xsec_calculator.run_calculations()

if __name__ == '__main__':
    main()
