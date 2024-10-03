import xsec_calculator.xsec_calc as xsec_calc # type: ignore
from xsec_calculator.parameter_def import parameters
import multiprocessing, argparse, time, sys
import numpy as np

class XsecCalculator:
    def __init__(self, doScaleVars=False, outdir='output', n_cores=6):
        self.doScaleVars = doScaleVars
        self.outdir = outdir
        self.n_cores = n_cores
        self.params = parameters(doScaleVars)
        self.d = self.params.getDict()

    def calculate_xsec(self, key, mass_scale=None, width_scale=None):
        if mass_scale is None:
            mass_scale = self.params.mass_scale
        if width_scale is None:
            width_scale = self.params.width_scale
        print(f"Calculating cross section for {key}...")
        xsec_calc.do_scan(order=self.params.order, PS_mass=self.d[key]['mass'], width=self.d[key]['width'], mass_scale=mass_scale, width_scale=width_scale,
                          yukawa=self.d[key]['yukawa'], as_var=self.d[key]['alphas'], outdir=self.outdir)
                

    def run_calculations(self):
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

                sys.stdout.write(f"\rRemaining processes: {len(processes)}")
                sys.stdout.flush()

                processes = [p for p in processes if not p.ready()]

            print("All processes have finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncores', type=int, default=6)
    parser.add_argument('--doScaleVars', action='store_true')
    parser.add_argument('--outdir', type=str, default='output')

    args = parser.parse_args()
    xsec_calculator = XsecCalculator(doScaleVars=args.doScaleVars, outdir=args.outdir, n_cores=args.ncores)
    xsec_calculator.run_calculations()

if __name__ == '__main__':
    main()
