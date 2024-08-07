import xsec_calculator.xsec_calc as xsec_calc
from xsec_calculator.parameter_def import parameters
import multiprocessing

class XsecCalculator:
    def __init__(self):
        self.params = parameters()
        self.d = self.params.getDict()

    def calculate_xsec(self, key):
        print(f"Calculating cross section for {key}...")
        xsec_calc.do_scan(order=3, PS_mass=self.d[key]['mass'], width=self.d[key]['width'], mass_scale=80., width_scale=350.,
                          yukawa=self.d[key]['yukawa'], as_var=self.d[key]['alphas'], outdir='output')

    def run_calculations(self):
        processes = []
        for key in self.d:
            p = multiprocessing.Process(target=self.calculate_xsec, args=(key,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            print(f"Process {p.pid} has finished.")


def main():
    xsec_calculator = XsecCalculator()
    xsec_calculator.run_calculations()

if __name__ == '__main__':
    main()
