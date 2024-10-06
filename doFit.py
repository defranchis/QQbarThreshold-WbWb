import numpy as np
import pandas as pd # type: ignore
import iminuit, scipy # type: ignore
import os, argparse
import copy
import uncertainties as unc # type: ignore
import matplotlib.pyplot as plt # type: ignore
from utils_fit.fitUtils import convoluteXsecGauss # type: ignore
import utils_convert.scheme_conversion as scheme_conversion # type: ignore
from xsec_calculator.parameter_def import parameters # type: ignore
import mplhep as hep
plt.style.use(hep.style.CMS)

plot_dir = 'plots/fit'

uncert_yukawa_default = 0.03 # only when parameter is constrained. hardcoded for now

def formFileTag(mass, width, yukawa, alphas):
    return 'mass{:.2f}_width{:.2f}_yukawa{:.2f}_asVar{:.4f}'.format(mass,width,yukawa,alphas)

def ecmToString(ecm):
    return '{:.1f}'.format(ecm)

class fit:
    def __init__(self, beam_energy_res = 0.23, smearXsec = True, SM_width = False, input_dir= None, debug = False, asimov = True, constrain_Yukawa = False, read_scale_vars = False) -> None:
        if input_dir is None:
            self.input_dir = 'output_full' if not read_scale_vars else 'output_alternative'
        else: self.input_dir = input_dir
        self.parameters = parameters(read_scale_vars)
        self.d_params = self.parameters.getDict()
        self.param_names = list(self.d_params['nominal'].keys())
        self.SM_width = SM_width
        self.pseudodata_tag = 'pseudodata' if not self.SM_width else 'mass_var'
        self.constrain_Yukawa = constrain_Yukawa
        self.asimov = asimov
        self.l_tags = list(self.d_params.keys())
        self.beam_energy_res = beam_energy_res
        self.smearXsec = smearXsec
        self.debug = debug
        self.read_scale_vars = read_scale_vars
        self.last_ecm = 365.0 #hardcoded

        if self.debug:
            print('Input directory: {}'.format(self.input_dir))
            print('Parameters: {}'.format(self.param_names))
            print('Beam energy resolution: {}'.format(self.beam_energy_res))
            print('Smear cross sections: {}'.format(self.smearXsec))
            print('Constrain width to SM value: {}'.format(self.SM_width))
            print('Constrain Yukawa coupling: {}'.format(self.constrain_Yukawa))
            if self.constrain_Yukawa:
                print('\tYukawa coupling uncertainty: {}'.format(uncert_yukawa_default))
            print('Asimov fit: {}'.format(self.asimov))
        
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

    def update(self):
        if self.debug:
            print('Updating object')
        self.smearCrossSections()
        self.morphCrossSections()
        self.createScenario(**self.scenario_dict, init_vars = False)
        self.initMinuit()
        self.fitParameters()

    def getWidthN2LO(self,mt_PS, mu = None): # TODO: not yet used
        if mu is None:
            mu = self.parameters.mass_scale
        return scheme_conversion.calculate_width(mt_PS, mu)

    def getWidthN3LO(self, mt_PS, mu = None, fit_param = 0.):
        if mu is None:
            mu = self.parameters.mass_scale
        mt_ref = self.d_params['nominal']['mass']
        mt_pole = mt_PS + scheme_conversion.calculate_mt_Pole(mt_ref, mu) - mt_ref # constant
        return 1.3148 + 0.0277*(mt_pole-172.69) + fit_param*0.005

    def formFileName(self, tag, scaleM, scaleW):
        infile_tag = formFileTag(*[self.d_params[tag][p] for p in self.param_names])
        return 'N3LO_scan_PS_ISR_{}_scaleM{:.1f}_scaleW{:.1f}.txt'.format(infile_tag, scaleM, scaleW)

    def readScanPerTag(self, tag, scaleM = None, scaleW = None):
        if scaleM is None:
            scaleM = self.parameters.mass_scale
        if scaleW is None:
            scaleW = self.parameters.width_scale
        filename = self.input_dir + '/' + self.formFileName(tag, scaleM, scaleW)
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

        self.l_ecm = self.xsec_dict[self.l_tags[0]]['ecm'].astype(str).tolist()

        if not self.read_scale_vars:
            return

        for scaleM in self.parameters.scale_vars:
            xsec = self.readScanPerTag(tag='nominal', scaleM=scaleM)
            if len(xsec) < len(self.l_ecm)-1:
                print('Warning: length of xsec for scaleM = {} is not equal to nominal. Skipping'.format(scaleM))
            else:
                self.xsec_dict['scaleM_{:.1f}'.format(scaleM)] = xsec

        for scaleW in self.parameters.scale_vars:
            xsec = self.readScanPerTag(tag='nominal', scaleW=scaleW)
            if len(xsec) < len(self.l_ecm)-1:
                print('Warning: length of xsec for scaleW = {} is not equal to nominal. Skipping'.format(scaleW))
            else:
                self.xsec_dict['scaleW_{:.1f}'.format(scaleW)] = xsec

    def smearCrossSection(self,xsec):
        if not self.smearXsec:
            return xsec
        last_ecm_xsec = ecmToString(float(xsec['ecm'].iloc[-1]))
        xsec_to_smear = xsec[:-1] if last_ecm_xsec == ecmToString(self.last_ecm) else xsec
        xsec_smeared = convoluteXsecGauss(xsec_to_smear,self.beam_energy_res)
        if last_ecm_xsec == ecmToString(self.last_ecm):
            xsec_smeared = pd.concat([xsec_smeared, xsec[-1:]])
        return xsec_smeared

    def smearCrossSections(self):
        self.xsec_dict_smeared = {}
        for tag in self.xsec_dict.keys():
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
    
    def getValueFromParameter(self,par,par_name):
        return self.d_params['nominal'][par_name] + par * (self.d_params['{}_var'.format(par_name)][par_name] - self.d_params['nominal'][par_name])
    
    def getParameterFromValue(self,val,par_name):
        return (val - self.d_params['nominal'][par_name])/(self.d_params['{}_var'.format(par_name)][par_name] - self.d_params['nominal'][par_name])
    
    def getXsecParams(self):
        params = self.getParamsWithCovarianceMinuit()
        th_xsec = np.array(self.getXsecTemplate()['xsec'])
        for param_name in self.param_names:
            param = params[self.param_names.index(param_name)]
            th_xsec = th_xsec * (1 + param*np.array(self.morph_dict[param_name]['xsec']))
        df_th_xsec = pd.DataFrame({'ecm': self.l_ecm, 'xsec': [th.n for th in th_xsec], 'unc': [th.s for th in th_xsec]})
        return df_th_xsec
    
    def getXsecScenario(self,xsec):
        xsec_scenario = xsec[[float(ecm) in [float(e) for e in self.scenario.keys()] for ecm in xsec['ecm']]]
        return xsec_scenario
    
    def initScenario(self,n_IPs=4, scan_min=340, scan_max=346, scan_step=1, total_lumi=0.36 * 1E06, last_lumi = 0.58*4 * 1E06, add_last_ecm = True, same_evts = False, create_scenario = True):
        if self.constrain_Yukawa and add_last_ecm:
            print('\nWarning: constraining Yukawa coupling and adding last ecm is not supported. Setting add_last_ecm to False\n')
            add_last_ecm = False
        self.scenario_dict = {'n_IPs': n_IPs, 'scan_min': scan_min, 'scan_max': scan_max, 'scan_step': scan_step, 'total_lumi': total_lumi,
                             'last_lumi': last_lumi, 'add_last_ecm': add_last_ecm, 'same_evts': same_evts}
        if create_scenario:
            self.createScenario(**self.scenario_dict)
    

    def createScenario(self,n_IPs=4, scan_min=340, scan_max=346, scan_step=1, total_lumi=0.36 * 1E06, last_lumi = 0.58*4 * 1E06, add_last_ecm = True, same_evts = False, init_vars = True):
        if self.debug:
            print('Creating threshold scan scenario')
        if not n_IPs in [2,4]:
            raise ValueError('Invalid number of IPs')

        scenario = [ecmToString(e) for e in np.arange(scan_min,scan_max+scan_step/2,scan_step)]
        scenario_dict = {k: total_lumi/len(scenario) for k in scenario}
        if add_last_ecm:
            scenario_dict[ecmToString(self.last_ecm)] = last_lumi
        #if n_IPs == 2:
        #    for k in scenario_dict.keys():
        #        scenario_dict[k] = scenario_dict[k]/1.8
        for ecm in scenario_dict.keys():
            if ecm not in self.l_ecm:
                raise ValueError('Invalid scenario key: {}'.format(ecm))
        self.scenario = dict(sorted(scenario_dict.items(), key=lambda x: float(x[0])))
        self.xsec_scenario = self.getXsecScenario(self.getXsecTemplate()) # just nominal for now
        if init_vars:
            self.scale_var_scenario = np.ones(len(self.xsec_scenario['xsec']))
        self.pseudo_data_scenario = np.array(self.getXsecScenario(self.getXsecTemplate(self.pseudodata_tag))['xsec'])
        if same_evts:
            overall_factor = total_lumi / np.sum(np.array([1/sigma for sigma in self.pseudo_data_scenario]))
            self.scenario = {ecm: overall_factor/sigma for ecm, sigma in zip(self.scenario.keys(), self.pseudo_data_scenario)}
        self.unc_pseudodata_scenario = (np.array(self.pseudo_data_scenario)/np.array(list(self.scenario.values())))**.5
        if not self.asimov:
            np.random.seed(42)
            self.pseudo_data_scenario = np.random.normal(self.pseudo_data_scenario, self.unc_pseudodata_scenario)
        self.morph_scenario = {param: self.getXsecScenario(self.morph_dict[param]) for param in self.param_names}


    def getPhysicalFitParams(self,params):
        if not self.SM_width:
            return 0
        prior_width = params[self.param_names.index('width')]**2
        width = self.getWidthN3LO(self.getValueFromParameter(params[self.param_names.index('mass')], 'mass'),params[self.param_names.index('width')])
        params[self.param_names.index('width')] = self.getParameterFromValue(width, 'width')
        return prior_width

    def chi2(self, params):
        th_xsec = np.array(self.xsec_scenario['xsec'])
        th_xsec *= self.scale_var_scenario
        prior_width = self.getPhysicalFitParams(params) # can be zero
        for i, param in enumerate(self.param_names):
            th_xsec *= (1 + params[i]*np.array(self.morph_scenario[param]['xsec']))
        chi2 = np.sum(((self.pseudo_data_scenario - th_xsec)/self.unc_pseudodata_scenario)**2)
        chi2 += (params[self.param_names.index('alphas')] - self.getParameterFromValue(self.d_params[self.pseudodata_tag]['alphas'],'alphas'))**2
        if self.constrain_Yukawa:
            uncert_yukawa = uncert_yukawa_default / (self.d_params['yukawa_var']['yukawa'] - self.d_params['nominal']['yukawa'])
            chi2 += ((params[self.param_names.index('yukawa')] - self.getParameterFromValue(self.d_params[self.pseudodata_tag]['yukawa'], 'yukawa'))/uncert_yukawa)**2
        return chi2 + prior_width


    def initMinuit(self):
        self.minuit = iminuit.Minuit(self.chi2, np.zeros(len(self.param_names)), name = self.param_names)
        self.minuit.errordef = 1

    def fitParameters(self):
        if self.debug:
            print('Initializing Minuit')
        self.initMinuit()
        if self.debug:
            print('Fitting parameters')
        self.minuit.migrad()
        if self.debug:
            print('Fit done')

    def getParamsWithCovarianceMinuit(self):
        params_w_cov = list(unc.correlated_values([self.minuit.values[param] for param in self.param_names], self.minuit.covariance))
        self.getPhysicalFitParams(params_w_cov)
        return params_w_cov

    def getFitResults(self, printout = True):
        params_w_cov = self.getParamsWithCovarianceMinuit()
        for i, param in enumerate(self.param_names):
            params_w_cov[i] = self.getValueFromParameter(params_w_cov[i], param)
            if printout:
                if param == 'alphas':
                    print('Fitted {}: {:.5f}'.format(param, params_w_cov[i]))
                else:
                    print('Fitted {}: {:.3f} {}'.format(param, params_w_cov[i], 'GeV' if param in ['mass','width'] else ''))
                    if param == 'width' and self.SM_width: 
                            print('including theory uncertainty in SM relation')
                            print('fitted theory parameter = {:.2f} +/- {:.2f} (constrained to 1)'.format(self.minuit.values[param], self.minuit.errors[param]))
                    if param == 'yukawa' and self.constrain_Yukawa: print('constrained with uncertainty {:.3f}'.format(uncert_yukawa_default))
                if param == 'width' and self.SM_width:
                    pull = unc.ufloat(self.minuit.values[param], self.minuit.errors[param])
                else:
                    pull = (params_w_cov[i] - self.d_params[self.pseudodata_tag][param])
                print('Pull {}: {:.3f}\n'.format(param, pull.n/pull.s))

        if printout:
            print('Correlation matrix:')
            print(self.param_names)
            print(np.round(unc.correlation_matrix(params_w_cov), 2))
        if printout:
            self.fit_results = params_w_cov
        return params_w_cov
    
    
    def plotFitScenario(self):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.figure()
        plt.errorbar(self.xsec_scenario['ecm'],self.pseudo_data_scenario,yerr=self.unc_pseudodata_scenario,fmt='.',label='Pseudo data' if not self.asimov else 'Asimov data')
        xsec_nom = self.getXsecTemplate()
        xsec_pseudodata = self.getXsecTemplate(self.pseudodata_tag)
        xsec_fit = self.getXsecParams()
        if not self.scenario_dict['add_last_ecm']:
            xsec_nom = xsec_nom[:-1]
            xsec_fit = xsec_fit[:-1]
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec'],label='Fitted model', linewidth=2)
        plt.plot(xsec_nom['ecm'],xsec_nom['xsec'],label='Nominal model', linestyle='--', linewidth=2)
        plt.xlabel('Ecm [GeV]')
        plt.ylabel('Cross section [pb]')
        plt.legend()
        plt.savefig(plot_dir + '/fit_scenario_{}.png'.format('pseudo' if not self.asimov else 'asimov'))

        plt.clf()
        plt.errorbar(self.xsec_scenario['ecm'],self.pseudo_data_scenario/self.getXsecScenario(xsec_nom)['xsec'], yerr=self.unc_pseudodata_scenario/self.getXsecScenario(xsec_nom)['xsec'], 
                    fmt='.', label = 'Pseudodata (stat)' if not self.asimov else 'Asimov data (stat)', linewidth=2)
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec']/xsec_nom['xsec'], label='Fitted cross section', linewidth=2)
        plt.fill_between(xsec_nom['ecm'], (xsec_fit['xsec']-xsec_fit['unc'])/xsec_nom['xsec'], (xsec_fit['xsec']+xsec_fit['unc'])/xsec_nom['xsec'], alpha=0.5, label='Parametric uncertainty (stat)')
        plt.plot(xsec_pseudodata['ecm'], xsec_pseudodata['xsec']/xsec_nom['xsec'], label='Pseudodata cross section' if not self.asimov else 'Asimov cross section', linestyle='--', linewidth=2)
        plt.axhline(1, color='grey', linestyle='--', label='Reference cross section', linewidth=2)
        plt.xlabel('$\sqrt{s}$ [GeV]')
        plt.ylabel('WbWb total cross section ratio')
        lumi = self.scenario_dict['total_lumi']/1E03
        plt.title('Preliminary ({:.0f}'.format(lumi)+' fb$^{-1}$)', loc='right', fontsize=20, style='italic')
        plt.legend(loc='lower right', fontsize=20)
        if not self.scenario_dict['add_last_ecm']:
        #if True:
            plt.xlim(339.7, 347)
        plt.text(.95, 0.47, 'QQbar_Threshold N3LO', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.95, 0.42, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.95, 0.37, 'FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.text(.35, 0.9, '$m_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('mass')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.35, 0.85, '$\Gamma_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('width')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.savefig(plot_dir + '/fit_scenario_ratio_{}.png'.format('pseudo' if not self.asimov else 'asimov'))
        plt.savefig(plot_dir + '/fit_scenario_ratio_{}.pdf'.format('pseudo' if not self.asimov else 'asimov'))
        plt.clf()

    def doLSscan (self, min, max, step):
        if min == 0:
            min = 1E-6
        l_beam_energy_res = np.arange(min,max+step/2,step)
        d = {var : [] for var in self.param_names}
        params_to_scan = [param for param in self.param_names if param != 'alphas']
        if self.SM_width:
            params_to_scan.remove('width')
        if self.constrain_Yukawa:
            params_to_scan.remove('yukawa')
        print('\nScanning parameters: {}\n'.format(params_to_scan))
        
        for res in l_beam_energy_res:
            f = copy.deepcopy(self)
            f.beam_energy_res = res
            f.update()
            fit_results = f.getFitResults(printout=False)
            for i, param in enumerate(params_to_scan):
                d[param].append(fit_results[i].s)

        for param in params_to_scan:
            plt.plot(l_beam_energy_res, d[param], 'b-', label='Stat uncertainty in top {}'.format(param))
            plt.plot(self.beam_energy_res, self.getFitResults(printout=False)[self.param_names.index(param)].s, 'ro', label='nominal (resol/beam = {:.3f}%)'.format(self.beam_energy_res))
            plt.legend()
            plt.title('top {}'.format(param))
            plt.xlabel('Beam energy resolution per beam [%]')
            plt.ylabel('Stat uncertainty in top {} [MeV]'.format(param))
            plt.savefig(plot_dir + '/uncert_{}_vs_beam_energy_res.png'.format(param))
            plt.clf()

        return 
    
    def doScaleVarsParam(self, scale_param):
        if scale_param not in ['mass','width']:
            raise ValueError('Invalid parameter for scale variations')
        tag_par = 'scaleM' if scale_param == 'mass' else 'scaleW'
        nominal_scenario = self.getXsecScenario(self.getXsecTemplate())
        l_vars = []
        l_mass = []
        l_width = []
        for var in self.parameters.scale_vars:
            tag = tag_par + '_{:.1f}'.format(var)
            if not tag in self.xsec_dict.keys():
                print('Skipping {}'.format(tag))
                continue
            l_vars.append(var)
            variation_scenario = self.getXsecScenario(self.getXsecTemplate(tag))
            f = copy.deepcopy(self)
            f.scale_var_scenario = np.array(variation_scenario['xsec'])/np.array(nominal_scenario['xsec'])
            f.update()
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].n-self.d_params[self.pseudodata_tag]['mass'])
            l_width.append(fit_results[self.param_names.index('width')].n-self.d_params[self.pseudodata_tag]['width'])


        for param in ['mass','width']:
            l_scan = l_mass if param == 'mass' else l_width
            plt.plot(l_vars, np.array(l_scan)*1E03, 'b-', label='Shift in fitted top {} [MeV]'.format(param))
            central_scale = self.parameters.mass_scale if scale_param == 'mass' else self.parameters.width_scale
            nominal_fit_results = self.getFitResults(printout=False)
            nominal_shift = (nominal_fit_results[self.param_names.index(param)].n - self.d_params[self.pseudodata_tag][param])*1E03
            plt.plot(central_scale, nominal_shift, 'ro', label='nominal ({} scale = {:.1f})'.format(scale_param, central_scale))
            plt.legend()
            plt.title('top {} vs {} scale'.format(param, scale_param))
            plt.xlabel('{} scale [GeV]'.format(scale_param))
            plt.ylabel('Shift in fitted top {} [MeV]'.format(param))
            plt.savefig(plot_dir + '/uncert_{}_vs_{}_scale.png'.format(param,scale_param))
            plt.clf()
                
    def doScaleVars(self):
        for param in ['mass','width']:
            self.doScaleVarsParam(param)



def main():
    parser = argparse.ArgumentParser(description='Specify options')
    parser.add_argument('--pseudo', action='store_true', help='Pseudodata')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--LSscan', action='store_true', help='Do beam energy resolution scan')
    parser.add_argument('--scaleVars', action='store_true', help='Do scale variations')
    parser.add_argument('--SMwidth', action='store_true', help='Constrain width to SM value')
    parser.add_argument('--constrainYukawa', action='store_true', help='Constrain Yukawa coupling in fit')
    parser.add_argument('--lastecm', action='store_true', help='Add last ecm to scenario')
    parser.add_argument('--sameNevts', action='store_true', help='Same number of events in each ecm')
    args = parser.parse_args()
    
    f = fit(debug=args.debug, asimov=not args.pseudo, SM_width=args.SMwidth, constrain_Yukawa=args.constrainYukawa, read_scale_vars = args.scaleVars)
    f.initScenario(n_IPs=4, scan_min=340, scan_max=344.5, scan_step=.5, total_lumi=0.7 * 1E06, last_lumi = 3 * 1E06, add_last_ecm = args.lastecm, same_evts = args.sameNevts,  create_scenario = True)
    #f.initScenario(n_IPs=4, scan_min=342, scan_max=345, scan_step=2, total_lumi=0.36 * 1E06/10, last_lumi = 0.58*4 * 1E06, add_last_ecm = False, create_scenario = True)
    #f.initScenario(n_IPs=4, scan_min=342, scan_max=344, scan_step=2, total_lumi=0.36 * 1E06/10, last_lumi = 0.58*4 * 1E06, add_last_ecm = False, create_scenario = True)
    
    
    f.fitParameters()
    f.getFitResults()
    f.plotFitScenario()
    if args.LSscan:
        f.doLSscan(0,0.5,0.01)
    if args.scaleVars:
        f.doScaleVars()


if __name__ == '__main__':
    main()
