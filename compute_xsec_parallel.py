import xsec_calculator.xsec_calc as xsec_calc # type: ignore
from xsec_calculator.parameter_def import parameters
import multiprocessing, argparse, time, sys, copy
import numpy as np

class XsecCalculator:
    def __init__(self, doScaleVars=False, outdir='output', n_cores=6, nominal=False, pseudo_data=False, oneS_mass=False):
        self.doScaleVars = doScaleVars
        self.outdir = outdir
        self.n_cores = n_cores
        self.params = parameters(do_scale_vars=doScaleVars, oneS_mass=oneS_mass)
        self.d = self.params.getDict()
        self.nominal = nominal
        self.pseudo_data = pseudo_data
        self.oneS_mass = oneS_mass


    def calculate_xsec(self, key, mass_scale=None, width_scale=None, param_dict = None):
        if mass_scale is None:
            mass_scale = self.params.mass_scale
        if width_scale is None:
            width_scale = self.params.width_scale
        if param_dict is None:
            param_dict = self.d
        print(f"Calculating cross section for {key}...")
        xsec_calc.do_scan(order=self.params.order, PS_mass=param_dict[key]['mass'], width=param_dict[key]['width'], mass_scale=mass_scale, width_scale=width_scale,
                          yukawa=param_dict[key]['yukawa'], as_var=param_dict[key]['alphas'], outdir=self.outdir, oneS_mass=self.oneS_mass)
                

    def run_calculations(self):

        if self.nominal:
            self.calculate_xsec('nominal')
            return
        
        if self.pseudo_data:
            mass_vars = np.linspace(0, 0.5, 51)
            param_dict_l = []
            for mass in mass_vars:
                param_dict = copy.deepcopy(self.d)
                param_dict['nominal']['mass'] += mass
                param_dict_l.append(param_dict) 

            with multiprocessing.Pool(processes=self.n_cores) as pool:     
                processes = [pool.apply_async(self.calculate_xsec, args=(), kwds={'key':'nominal', 'param_dict': param_dict}) for param_dict in param_dict_l]
                pool.close()

                while processes:
                    time.sleep(1)

                    sys.stdout.write(f"\rRemaining processes: {len(processes)}")
                    sys.stdout.flush()

                    processes = [p for p in processes if not p.ready()]

            print("All pseudo data processes have finished.")
            return


        with multiprocessing.Pool(processes=self.n_cores) as pool:
            processes = [pool.apply_async(self.calculate_xsec, args=(key,)) for key in self.d]
            if self.doScaleVars:
                for mass_scale in self.params.scale_vars:
                    print(f"Calculating cross section for mass scale {mass_scale}...")
                    processes.append(pool.apply_async(self.calculate_xsec, args=(), kwds={'key':'nominal','mass_scale': mass_scale}))
                for width_scale in self.params.scale_vars:
                    print(f"Calculating cross section for width scale {width_scale}...")
                    processes.append(pool.apply_async(self.calculate_xsec, args=(), kwds={'key':'nominal','width_scale': width_scale}))
            pool.close()

            while processes:
                time.sleep(1)

                sys.stdout.write(f"\rRemaining processes: {len(processes)}\n")
                sys.stdout.flush()

                processes = [p for p in processes if not p.ready()]

            print("All processes have finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncores', type=int, default=6)
    parser.add_argument('--doScaleVars', action='store_true')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--nominal', action='store_true')
    parser.add_argument('--pseudodata', action='store_true')
    parser.add_argument('--oneS', action='store_true', help='1S mass definition')

    args = parser.parse_args()

    if args.nominal and args.doScaleVars:
        print("Warning: will not run scale variations when nominal is set to True.")

    if args.pseudodata and (args.nominal or args.doScaleVars):
        print("Warning: will not run pseudo data when nominal or scale variations are set to True.")

    xsec_calculator = XsecCalculator(doScaleVars=args.doScaleVars, outdir=args.outdir, n_cores=args.ncores, nominal=args.nominal, pseudo_data=args.pseudodata, oneS_mass=args.oneS)
    xsec_calculator.run_calculations()

if __name__ == '__main__':
    main()
