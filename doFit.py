import numpy as np
import pandas as pd # type: ignore
import iminuit, scipy # type: ignore
import os, argparse
import uncertainties as unc # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils_fit.fitUtils import convoluteXsecGauss # type: ignore
import utils_convert.scheme_conversion as scheme_conversion # type: ignore
from xsec_calculator.parameter_def import parameters # type: ignore


def getWidthN3LO(mt_PS):
    return 1.3148 + 0.0277*(scheme_conversion.calculate_mt_Pole(mt_PS)-172.69)

def formFileTag(mass, width, yukawa, alphas):
    return 'mass{:.2f}_width{:.2f}_yukawa{:.2f}_asVar{:.4f}'.format(mass,width,yukawa,alphas)

class fit:
    def __init__(self, beam_energy_res = 0.221, smearXsec = True, input_dir= 'output', debug = False, asimov = True) -> None:
        self.input_dir = input_dir
        self.d_params = parameters().getDict()
        self.param_names = list(self.d_params['nominal'].keys())
        self.l_tags = list(self.d_params.keys())
        self.beam_energy_res = beam_energy_res
        self.smearXsec = smearXsec
        self.asimov = asimov
        self.debug = debug
        if self.debug:
            print('Input directory: {}'.format(self.input_dir))
            print('Parameters: {}'.format(self.param_names))
            print('Beam energy resolution: {}'.format(self.beam_energy_res))
            print('Smear cross sections: {}'.format(self.smearXsec))
        
        if self.debug:
            print('Reading cross sections')
        self.readCrossSections()
        self.l_ecm = self.xsec_dict[self.l_tags[0]]['ecm'].astype(str).tolist()
        if self.debug:
            print('Smearing cross sections')
        self.smearCrossSections()
        if self.debug:
            print('Morphing cross sections')
        self.morphCrossSections()
        if self.debug:
            print('Initialization done')

    def formFileName(self, tag):
        infile_tag = formFileTag(*[self.d_params[tag][p] for p in self.param_names])
        return 'N3LO_scan_PS_ISR_{}_scaleM80.0_scaleW350.0.txt'.format(infile_tag)

    def readScanPerTag(self,tag):
        filename = self.input_dir + '/' + self.formFileName(tag)
        if not os.path.exists(filename):
            raise ValueError('File {} not found'.format(filename))
        f = open(filename, 'r')
        df = pd.read_csv(f, header=None, names=['ecm','xsec'])
        f.close()
        return df
          
    def readCrossSections(self):
        self.xsec_dict = {}
        for tag in self.l_tags:
            self.xsec_dict[tag] = self.readScanPerTag(tag)

    def smearCrossSection(self,xsec):
        if not self.smearXsec:
            return xsec
        xsec_to_smear = xsec[:-1] if self.l_ecm[-1] == '365.0' else xsec
        xsec_smeared = convoluteXsecGauss(xsec_to_smear,self.beam_energy_res)
        if self.l_ecm[-1] == '365.0':
            xsec_smeared = pd.concat([xsec_smeared, xsec[-1:]])
        return xsec_smeared

    def smearCrossSections(self):
        self.xsec_dict_smeared = {}
        for tag in self.l_tags:
            self.xsec_dict_smeared[tag] = self.smearCrossSection(self.xsec_dict[tag])

    def morphCrossSection(self,param):
        xsec_nom = self.getXsecTemplate()
        xsec_var = self.getXsecTemplate('{}_var'.format(param))
        df_morph = pd.DataFrame({'ecm': xsec_nom['ecm'], 'xsec': xsec_var['xsec']/xsec_nom['xsec'] -1})
        return df_morph      

    def morphCrossSections(self):
        self.morph_dict = {}
        for param in self.param_names:
            self.morph_dict[param] = self.morphCrossSection(param)

    def getXsecTemplate(self,tag='nominal'):
        return self.xsec_dict_smeared[tag]
    
    def getXsecParams(self):
        params = unc.correlated_values([self.minuit.values[i] for i in range(len(self.param_names))], self.minuit.covariance)
        th_xsec = np.array(self.getXsecTemplate()['xsec'])
        for i, param in enumerate(self.param_names):
            th_xsec = th_xsec * (1 + params[i]*np.array(self.morph_dict[param]['xsec']))
        df_th_xsec = pd.DataFrame({'ecm': self.l_ecm, 'xsec': [th.n for th in th_xsec], 'unc': [th.s for th in th_xsec]})
        return df_th_xsec
    
    def getXsecScenario(self,xsec):
        xsec_scenario = xsec[[float(ecm) in [float(e) for e in self.scenario.keys()] for ecm in xsec['ecm']]]
        return xsec_scenario
    
    def createScenario(self,scenario):
        if self.debug:
            print('Creating threshold scan scenario')
        for ecm in scenario.keys():
            if ecm not in self.l_ecm:
                raise ValueError('Invalid scenario key: {}'.format(ecm))
        self.scenario = dict(sorted(scenario.items(), key=lambda x: float(x[0])))
        self.xsec_scenario = self.getXsecScenario(self.getXsecTemplate()) # just nominal for now
        self.pseudo_data_scenario = np.array(self.getXsecScenario(self.getXsecTemplate('pseudodata'))['xsec'])
        self.unc_pseudodata_scenario = (np.array(self.pseudo_data_scenario)/np.array(list(self.scenario.values())))**.5
        if not self.asimov:
            self.pseudo_data_scenario = np.random.normal(self.pseudo_data_scenario, self.unc_pseudodata_scenario)
        self.morph_scenario = {param: self.getXsecScenario(self.morph_dict[param]) for param in self.param_names}

    def chi2(self, params):
        th_xsec = np.array(self.xsec_scenario['xsec'])
        for i, param in enumerate(self.param_names):
            th_xsec *= (1 + params[i]*np.array(self.morph_scenario[param]['xsec']))
        return np.sum(((self.pseudo_data_scenario - th_xsec)/self.unc_pseudodata_scenario)**2) + params[-1]**2


    def initMinuit(self):
        self.minuit = iminuit.Minuit(self.chi2, np.zeros(len(self.param_names)))
        self.minuit.errordef = 1


    def fitPatameters(self):
        if self.debug:
            print('Initializing Minuit')
        self.initMinuit()
        if self.debug:
            print('Fitting parameters')
        self.minuit.migrad()
        if self.debug:
            print('Fit done')

        nom_values = []
        for i, param in enumerate(self.param_names):
            param_w_unc = unc.ufloat(self.minuit.values[i],self.minuit.errors[i])
            if param != 'as':
                param_w_unc = self.d_params['nominal'][param] + param_w_unc*(self.d_params['{}_var'.format(param)][param] - self.d_params['nominal'][param])
            nom_values.append(param_w_unc.n)
            print('Fitted {}: {:.3f} {}'.format(param,param_w_unc,'GeV' if param in ['mass','width'] else ''))
            pull = (param_w_unc - self.d_params['pseudodata'][param])
            print('Pull {}: {:.1f}'.format(param, pull.n/pull.s))
            print()


        correlation_matrix = np.zeros((len(self.param_names),len(self.param_names)))
        for i in range(len(self.param_names)):
            for j in range(len(self.param_names)):
                correlation_matrix[i,j] = self.minuit.covariance[(i,j)]/(self.minuit.errors[i]*self.minuit.errors[j])

        print('Correlation matrix:')
        print(self.param_names)
        print(np.round(correlation_matrix, 2))

        covariance_matrix = np.zeros((len(self.param_names),len(self.param_names)))
        for i in range(len(self.param_names)):
            for j in range(len(self.param_names)):
                covariance_matrix[i,j] = correlation_matrix[(i,j)]*self.minuit.errors[i]*self.minuit.errors[j]

        params_w_cov = unc.correlated_values(nom_values, covariance_matrix)

        return params_w_cov
    
    def plotFitScenario(self):
        plt.figure()
        #plot pseudodata
        plt.errorbar(self.xsec_scenario['ecm'],self.pseudo_data_scenario,yerr=self.unc_pseudodata_scenario,fmt='.',label='Pseudo data')
        # plot nominal model
        xsec_nom = self.getXsecTemplate()
        xsec_pseudodata = self.getXsecTemplate('pseudodata')
        # plot fitted model
        xsec_fit = self.getXsecParams()
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec'],label='Fitted model')
        plt.plot(xsec_nom['ecm'],xsec_nom['xsec'],label='Nominal model', linestyle='--')
        plt.xlabel('Ecm [GeV]')
        plt.ylabel('Cross section [pb]')
        plt.legend()
        plt.savefig('fit_scenario.png')

        plt.clf()
        plt.errorbar(self.xsec_scenario['ecm'],self.pseudo_data_scenario/self.getXsecScenario(xsec_nom)['xsec'], yerr=self.unc_pseudodata_scenario/self.getXsecScenario(xsec_nom)['xsec'], fmt='.', label = 'Pseudo data')
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec']/xsec_nom['xsec'], label='fitted cross section')
        plt.fill_between(xsec_nom['ecm'], (xsec_fit['xsec']-xsec_fit['unc'])/xsec_nom['xsec'], (xsec_fit['xsec']+xsec_fit['unc'])/xsec_nom['xsec'], alpha=0.5, label='uncertainty')
        plt.plot(xsec_pseudodata['ecm'], xsec_pseudodata['xsec']/xsec_nom['xsec'], label='pseudodata cross section', linestyle='--')
        plt.axhline(1, color='black', linestyle='--', label='nominal xsec')
        plt.xlabel('Ecm [GeV]')
        plt.ylabel('Ratio to nominal')
        plt.legend(loc='lower right')
        plt.savefig('residuals.png')
        


def formatScenario(scenario):
    return ['{:.1f}'.format(float(e)) for e in scenario]

def main():
    parser = argparse.ArgumentParser(description='Specify options')
    parser.add_argument('--asimov', action='store_true', help='Asimov data')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    f = fit(debug=args.debug, asimov=args.asimov)
    n_IP_4 = True
    total_lumi = 0.36 * 1E06 #pb^-1
    scan_min = 340
    scan_max = 346
    scan_step = 1
    scenario = formatScenario(np.arange(scan_min,scan_max+scan_step/2,scan_step))
    scenario_dict = {k: total_lumi/len(scenario) for k in scenario}
    scenario_dict['365.0'] = 0.58*4 * 1E06 
    if not n_IP_4:
        for k in scenario_dict.keys():
            scenario_dict[k] = scenario_dict[k]/1.8
    f.createScenario(scenario_dict)
    f.fitPatameters()
    f.plotFitScenario()


if __name__ == '__main__':
    main()