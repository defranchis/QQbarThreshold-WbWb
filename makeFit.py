import numpy as np
import pandas as pd
import iminuit, scipy
import os, argparse
import uncertainties as unc
import matplotlib.pyplot as plt
from fitUtils import convoluteXsecGauss

indir = 'output_ISR/for_fit'
parameters = ['mass','width','yukawa','as']

def formTag(mass, width, yukawa, alphas):
    return 'mass{:.2f}_width{:.2f}_yukawa{:.1f}_as{}'.format(mass,width,yukawa,alphas)

class fit:
    def __init__(self, input_dir, parameters, beam_energy_res = 0.221, smearXsec = True, debug = False) -> None:
        self.input_dir = input_dir
        self.params = parameters
        self.beam_energy_res = beam_energy_res
        self.smearXsec = smearXsec
        self.debug = debug
        if self.debug:
            print('Input directory: {}'.format(self.input_dir))
            print('Parameters: {}'.format(self.params))
            print('Beam energy resolution: {}'.format(self.beam_energy_res))
            print('Smear cross sections: {}'.format(self.smearXsec))
        
        if self.debug:
            print('Fetching inputs')
        self.fetchInputs()
        if self.debug:
            print('Checking parameter names')
        self.checkParameterNames()
        if self.debug:
            print('Getting parameter values')
        self.getParameterValues()
        if self.debug:
            print('Reading cross sections')
        self.readCrossSections()
        if self.debug:
            print('Smearing cross sections')
        self.smearCrossSections()
        if self.debug:
            print('Morphing cross sections')
        self.morphCrossSections()
        if self.debug:
            print('Initialization done')

    def fetchInputs(self):

        l_file = os.listdir(self.input_dir)
        l_file = [f for f in l_file if '_scan_' in f and f.endswith('.txt')] #just in case
        l_tag = list(set(['_'.join(f.split('N3LO_scan_PS_ISR_')[1].split('.txt')[0].split('_')[1:]) for f in l_file]))
        l_ecm = list(set([f.split('_')[4].split('ecm')[1] for f in l_file]))
        l_ecm.sort()

        self.l_tag = l_tag
        self.l_file = l_file
        self.l_ecm = l_ecm


    def checkParameterNames(self):
        for param in self.params:
            for tag in self.l_tag:
                if param not in tag:
                    raise ValueError('Parameter {} not found'.format(param))
        for i, param in enumerate(self.params):
            if '_'+parameters[i+1] not in self.l_tag[0].split(param)[1]:
                raise ValueError('Wrong order of parameters, please check input files')
            if i == len(parameters)-2:
                break
                

    def getParameterValues(self):
        self.param_dict = {}
        for param in parameters:
            l = [tag.split('_')[parameters.index(param)].replace(param,'') for tag in self.l_tag]
            if param != 'as':
                l = [float(i) for i in l]
            self.param_dict['{}_nom'.format(param)] = max(set(l), key=l.count)
            self.param_dict['{}_var'.format(param)] = min(set(l), key=l.count)


    def readScanPerTag(self,tag):
        xsec = []
        for ecm in self.l_ecm:
            fl = [f for f in self.l_file if ecm in f and tag in f]
            if len(fl) > 1:
                raise ValueError('More than one file found for tag {} and ecm {}'.format(tag,ecm))
            elif len(fl) == 0:
                raise ValueError('No file found for tag {} and ecm {}'.format(tag,ecm))
            f = open('{}/{}'.format(self.input_dir,fl[0]), 'r')
            xsec.append(float(f.readlines()[0].split(',')[-1]))
        df_xsec = pd.DataFrame({'ecm': [float(ecm) for ecm in self.l_ecm], 'xsec': xsec})
        self.xsec_dict[tag] = df_xsec
            
    def readCrossSections(self):
        self.xsec_dict = {}
        for tag in self.l_tag:
            self.readScanPerTag(tag)

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
        for tag in self.l_tag:
            self.xsec_dict_smeared[tag] = self.smearCrossSection(self.xsec_dict[tag])

    def morphCrossSection(self,param):
        xsec_nom = self.getXsecTemplate()
        xsec_var = self.getXsecTemplate([param])
        df_morph = pd.DataFrame({'ecm': xsec_nom['ecm'], 'xsec': xsec_var['xsec']/xsec_nom['xsec'] -1})
        return df_morph      

    def morphCrossSections(self):
        self.morph_dict = {}
        for param in self.params:
            self.morph_dict[param] = self.morphCrossSection(param)

    def getXsecTemplate(self,var_param = []):
        tag_input = [self.param_dict['{}_nom'.format(p) if p not in var_param else '{}_var'.format(p)] for p in self.params]
        return self.xsec_dict_smeared[formTag(*tag_input)]
    
    def getXsecScenario(self,xsec,scenario):
        xsec_scenario = xsec[[ecm in [float(e) for e in scenario.keys()] for ecm in xsec['ecm']]]
        return xsec_scenario
    
    def createScenario(self,scenario):
        if self.debug:
            print('Creating threshold scan scenario')
        for ecm in scenario.keys():
            if ecm not in self.l_ecm:
                raise ValueError('Invalid scenario key: {}'.format(ecm))
        self.scenario = dict(sorted(scenario.items(), key=lambda x: float(x[0])))
        self.xsec_scenario = self.getXsecScenario(self.getXsecTemplate(), self.scenario) # just nominal for now
        self.unc_xsec = (np.array(self.xsec_scenario['xsec'])/np.array(list(self.scenario.values())))**.5
        #self.pseudo_data = np.array(self.xsec_scenario['xsec']) + np.random.normal(0,self.unc_xsec)
        self.pseudo_data = self.getXsecScenario(self.getXsecTemplate([]), self.scenario)['xsec'] #mass up
        self.morph_scenario = {param: self.getXsecScenario(self.morph_dict[param], self.scenario) for param in self.params}

    def chi2(self, params):
        th_xsec = np.array(self.xsec_scenario['xsec'])
        for i, param in enumerate(self.params):
            th_xsec *= (1 + params[i]*np.array(self.morph_scenario[param]['xsec']))
        return np.sum(((self.pseudo_data - th_xsec)/self.unc_xsec)**2) + params[-1]**2
        
            


    def initMinuit(self):
        self.fit_params = np.zeros(len(self.params))
        self.minuit = iminuit.Minuit(self.chi2, self.fit_params)
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
        for i, param in enumerate(self.params):
            param_w_unc = unc.ufloat(self.minuit.values[i],self.minuit.errors[i])
            if param != 'as':
                param_w_unc = float(self.param_dict['{}_nom'.format(param)]) + param_w_unc*(float(self.param_dict['{}_var'.format(param)]) - float(self.param_dict['{}_nom'.format(param)]))
            nom_values.append(param_w_unc.n)
            print('Fitted {}: {:.3f} {}'.format(param,param_w_unc,'GeV' if param in ['mass','width'] else ''))

        correlation_matrix = np.zeros((len(self.params),len(self.params)))
        for i in range(len(self.params)):
            for j in range(len(self.params)):
                correlation_matrix[i,j] = self.minuit.covariance[(i,j)]/(self.minuit.errors[i]*self.minuit.errors[j])

        covariance_matrix = np.zeros((len(self.params),len(self.params)))
        for i in range(len(self.params)):
            for j in range(len(self.params)):
                covariance_matrix[i,j] = correlation_matrix[(i,j)]*self.minuit.errors[i]*self.minuit.errors[j]

        params_w_cov = unc.correlated_values(nom_values, covariance_matrix)

        return params_w_cov


def formatScenario(scenario):
    return ['{:.1f}'.format(float(e)) for e in scenario]

def main():
    parser = argparse.ArgumentParser(description='Specify options')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    f = fit(indir,parameters,debug=args.debug)
    n_IP_4 = True
    total_lumi = 0.36 * 1E06 #pb^-1
    scan_min = 340
    scan_max = 350
    scan_step = 1
    scenario = formatScenario(np.arange(scan_min,scan_max+scan_step/2,scan_step))
    scenario_dict = {k: total_lumi/len(scenario) for k in scenario}
    scenario_dict['365.0'] = 0.58*4 * 1E06 
    if not n_IP_4:
        for k in scenario_dict.keys():
            scenario_dict[k] = scenario_dict[k]/1.8
    f.createScenario(scenario_dict)
    f.fitPatameters()


if __name__ == '__main__':
    main()