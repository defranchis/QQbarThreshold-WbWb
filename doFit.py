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
import mplhep as hep # type: ignore
plt.style.use(hep.style.CMS)


plot_dir = 'plots/fit'
indir_BEC = 'BEC_variations' # hardcoded

BEC_input_var = 10 # 10 MeV, for morphing, hardcoded
BES_input_var = 0.1 # 10%, for morphing, hardcoded

uncert_yukawa_default = 0.03 # only when parameter is constrained. hardcoded for now
uncert_alphas_default =  0.0001 # hardcoded for now

uncert_lumi_default_uncorr = 1E-3 # hardcoded for now
uncert_BES_default_uncorr = 0.01 # hardcoded for now
uncert_BEC_default_uncorr = 5 # hardcoded for now

scale_lumi_uncorr = False

uncert_lumi_default_corr = uncert_lumi_default_uncorr / 2 # hardcoded for now
uncert_BES_default_corr = uncert_BES_default_uncorr / 2 # hardcoded for now
uncert_BEC_default_corr = uncert_BEC_default_uncorr / 2 # hardcoded for now

label_d = {'mass': '$m_t$ [GeV]', 'width': r'$\Gamma_t$'+' [GeV]', 'yukawa': 'y_t', 'alphas': r'\alpha_S'}


def formFileTag(mass, width, yukawa, alphas):
    return 'mass{:.2f}_width{:.2f}_yukawa{:.2f}_asVar{:.4f}'.format(mass,width,yukawa,alphas)

def ecmToString(ecm):
    return '{:.1f}'.format(ecm)

class fit:
    def __init__(self, beam_energy_res = 0.23, smearXsec = True, SM_width = False, input_dir= None, debug = False, asimov = True, 
                constrain_Yukawa = False, read_scale_vars = False) -> None:
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
        self.lumi_uncorr = uncert_lumi_default_uncorr # estimate for full lumi! i.e. 410/fb
        self.lumi_corr = uncert_lumi_default_corr # estimate for theory cross section uncertainty
        self.input_uncert_Yukawa = uncert_yukawa_default
        self.input_uncert_alphas = uncert_alphas_default
        self.BEC_nuisances = False
        self.BES_nuisances = False

        if self.debug:
            print('Input directory: {}'.format(self.input_dir))
            print('Parameters: {}'.format(self.param_names))
            print('Beam energy resolution: {}'.format(self.beam_energy_res))
            print('Smear cross sections: {}'.format(self.smearXsec))
            print('Constrain width to SM value: {}'.format(self.SM_width))
            print('Constrain Yukawa coupling: {}'.format(self.constrain_Yukawa))
            if self.constrain_Yukawa:
                print('\tYukawa coupling uncertainty: {}'.format(self.input_uncert_Yukawa))
            print('\talpha_s uncertainty: {}'.format(self.input_uncert_alphas))
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

    def update(self, update_scenario = True, exclude_stat = False):
        if self.debug:
            print('Updating object')
        self.smearCrossSections()
        self.morphCrossSections()
        if update_scenario:
            self.createScenario(**self.scenario_dict, init_vars = False)
        #self.initMinuit(exclude_stat = exclude_stat)
        self.fitParameters(exclude_stat = exclude_stat)

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

    def readScanPerTag(self, tag, scaleM = None, scaleW = None, indir = None):
        if scaleM is None:
            scaleM = self.parameters.mass_scale
        if scaleW is None:
            scaleW = self.parameters.width_scale
        if indir is None:
            indir = self.input_dir
        filename = os.path.join(indir, self.formFileName(tag, scaleM, scaleW))
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

    def smearCrossSection(self,xsec, BES=None):
        if not self.smearXsec:
            return xsec
        if BES is None: BES = self.beam_energy_res
        last_ecm_xsec = ecmToString(float(xsec['ecm'].iloc[-1]))
        xsec_to_smear = xsec[:-1] if last_ecm_xsec == ecmToString(self.last_ecm) else xsec
        xsec_smeared = convoluteXsecGauss(xsec_to_smear,BES)
        if last_ecm_xsec == ecmToString(self.last_ecm):
            xsec_smeared = pd.concat([xsec_smeared, xsec[-1:]])
        return xsec_smeared

    def smearCrossSections(self):
        self.xsec_dict_smeared = {}
        for tag in self.xsec_dict.keys():
            self.xsec_dict_smeared[tag] = self.smearCrossSection(self.xsec_dict[tag])

    def morphCrossSection(self,param):
        xsec_nom = self.getXsecTemplate()
        if param == 'BEC':
            tmp = self.readScanPerTag('nominal', indir=os.path.join(indir_BEC,self.varToDir(BEC_input_var)))
            tmp ['ecm'] = tmp['ecm'].round(1)
            xsec_var = self.smearCrossSection(tmp)
        elif param == 'BES':
            xsec_var = self.smearCrossSection(self.xsec_dict['nominal'], BES=self.beam_energy_res*(1+BES_input_var))
        else:
            xsec_var = self.getXsecTemplate('{}_var'.format(param))
        df_morph = pd.DataFrame({'ecm': xsec_nom['ecm'], 'xsec': xsec_var['xsec']/xsec_nom['xsec'] -1})
        return df_morph      

    def morphCrossSections(self):
        self.morph_dict = {}
        for param in self.param_names:
            self.morph_dict[param] = self.morphCrossSection(param)
        if not self.read_scale_vars:
            self.morph_dict['BEC'] = self.morphCrossSection('BEC')
            self.morph_dict['BES'] = self.morphCrossSection('BES')

    def getXsecTemplate(self,tag='nominal'):
        return self.xsec_dict_smeared[tag]
    
    def getValueFromParameter(self,par,par_name):
        if not 'BEC_bin' in par_name and not 'BES_bin' in par_name and par_name != 'BES' and par_name != 'BEC':
            return self.d_params['nominal'][par_name] + par * (self.d_params['{}_var'.format(par_name)][par_name] - self.d_params['nominal'][par_name])
        return par
    
    def getParameterFromValue(self,val,par_name):
        return (val - self.d_params['nominal'][par_name])/(self.d_params['{}_var'.format(par_name)][par_name] - self.d_params['nominal'][par_name])
    
    def getXsecParams(self):
        params = self.getParamsWithCovarianceMinuit()
        th_xsec = np.array(self.getXsecTemplate()['xsec'])
        for param_name in self.param_names:
            if 'BEC_bin' in param_name or 'BES_bin' in param_name or param_name == 'BES' or param_name == 'BEC':
                continue
            param = params[self.param_names.index(param_name)]
            th_xsec = th_xsec * (1 + param*np.array(self.morph_dict[param_name]['xsec']))
        df_th_xsec = pd.DataFrame({'ecm': self.l_ecm, 'xsec': [th.n for th in th_xsec], 'unc': [th.s for th in th_xsec]})
        return df_th_xsec
    
    def getXsecScenario(self,xsec):
        xsec_scenario = xsec[[float(ecm) in [float(e) for e in self.scenario.keys()] for ecm in xsec['ecm']]]
        return xsec_scenario
    
    def initScenario(self, scan_min, scan_max, scan_step, total_lumi, last_lumi, add_last_ecm, same_evts = False):
        scan_list = [ecmToString(e) for e in np.arange(scan_min,scan_max+scan_step/2,scan_step)]
        self.initScenarioCustom(scan_list, total_lumi, last_lumi, add_last_ecm, same_evts)
        
    def initScenarioCustom(self, scan_list, total_lumi, last_lumi, add_last_ecm, same_evts = False):
        if self.constrain_Yukawa and add_last_ecm:
            print('\nWarning: constraining Yukawa coupling and adding last ecm is not supported. Setting add_last_ecm to False\n')
            add_last_ecm = False
        self.scenario_dict = {'scan_list': scan_list, 'total_lumi': total_lumi, 'last_lumi': last_lumi, 'add_last_ecm': add_last_ecm, 'same_evts': same_evts}
        self.createScenario(**self.scenario_dict)
    

    def createScenario(self, scan_list , total_lumi, last_lumi, add_last_ecm, same_evts, init_vars = True):
        if self.debug:
            print('Creating threshold scan scenario')

        scenario_dict = {k: total_lumi/len(scan_list) for k in scan_list}
        if add_last_ecm:
            scenario_dict[ecmToString(self.last_ecm)] = last_lumi

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
        if not self.read_scale_vars:
            self.morph_scenario['BEC'] = self.getXsecScenario(self.morph_dict['BEC'])
            self.morph_scenario['BES'] = self.getXsecScenario(self.morph_dict['BES'])

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
        res = self.pseudo_data_scenario - th_xsec
        chi2 = np.dot(res, np.linalg.solve(self.cov, res))
        uncert_alphas = self.input_uncert_alphas / (self.d_params['alphas_var']['alphas'] - self.d_params['nominal']['alphas'])
        chi2 += ((params[self.param_names.index('alphas')] - self.getParameterFromValue(self.d_params[self.pseudodata_tag]['alphas'],'alphas'))/uncert_alphas)**2
        if self.constrain_Yukawa:
            uncert_yukawa = self.input_uncert_Yukawa / (self.d_params['yukawa_var']['yukawa'] - self.d_params['nominal']['yukawa'])
            chi2 += ((params[self.param_names.index('yukawa')] - self.getParameterFromValue(self.d_params[self.pseudodata_tag]['yukawa'], 'yukawa'))/uncert_yukawa)**2
        if self.BEC_nuisances:
            BEC_indices = [i for i, param in enumerate(self.param_names) if 'BEC_bin' in param]
            BEC_params = params[BEC_indices]
            if self.BEC_prior_uncorr < 1E-6:
                self.BEC_prior_uncorr = 1E-6 # avoid division by zero
            chi2 += sum((BEC_params/self.BEC_prior_uncorr)**2)
            if self.BEC_prior_corr < 1E-6:
                self.BEC_prior_corr = 1E-6
            BEC_index = self.param_names.index('BEC')
            chi2 += (params[BEC_index]/self.BEC_prior_corr)**2
        if self.BES_nuisances:
            if self.BES_prior_uncorr < 1E-6:
                self.BES_prior_uncorr = 1E-6
            BES_indices = [i for i, param in enumerate(self.param_names) if 'BES_bin' in param]
            BES_params = params[BES_indices]
            chi2 += sum((BES_params/self.BES_prior_uncorr)**2)
            if self.BES_prior_corr < 1E-6:
                self.BES_prior_corr = 1E-6
            BES_index = self.param_names.index('BES')
            chi2 += (params[BES_index]/self.BES_prior_corr)**2
        return chi2 + prior_width
    

    def initMinuit(self, exclude_stat = False):
        cov_stat = np.diag(self.unc_pseudodata_scenario**2)
        factor_thresh = len(self.scenario) if not self.scenario_dict['add_last_ecm'] else len(self.scenario) - 1
        if not scale_lumi_uncorr:
            factor_thresh = 1
        self.lumi_uncorr_ecm = np.array([self.lumi_uncorr * factor_thresh**.5 for _ in self.pseudo_data_scenario])
        if self.scenario_dict['add_last_ecm']:
            factor_above_thresh = self.scenario[str(self.last_ecm)]/self.scenario[list(self.scenario.keys())[0]]
            self.lumi_uncorr_ecm[-1] = self.lumi_uncorr / (factor_above_thresh*factor_thresh)**.5
        cov_lumi_uncorr = np.diag(self.pseudo_data_scenario*self.lumi_uncorr_ecm)**2
        cov_lumi_corr = np.outer(self.pseudo_data_scenario, self.pseudo_data_scenario)*self.lumi_corr**2
        self.cov = cov_stat + cov_lumi_uncorr + cov_lumi_corr if not exclude_stat else cov_lumi_uncorr + cov_lumi_corr
        self.minuit = iminuit.Minuit(self.chi2, np.zeros(len(self.param_names)), name = self.param_names)
        self.minuit.errordef = 1

    def fitParameters(self, exclude_stat = False, initMinuit = True):
        if initMinuit:
            if self.debug:
                print('Initializing Minuit')
            self.initMinuit(exclude_stat = exclude_stat)
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
                    if param == 'yukawa' and self.constrain_Yukawa: print('constrained with uncertainty {:.3f}'.format(self.input_uncert_Yukawa))
                if param == 'width' and self.SM_width:
                    pull = unc.ufloat(self.minuit.values[param], self.minuit.errors[param])
                elif 'BEC_bin' in param or 'BES_bin' in param or param == 'BES' or param == 'BEC':
                    pull = (params_w_cov[i])
                else:
                    pull = (params_w_cov[i] - self.d_params[self.pseudodata_tag][param])
                print('Pull {}: {:.3f}\n'.format(param, pull.n/pull.s))

        if printout:
            print('Correlation matrix:')
            print(self.param_names[:4])
            corr_matrix = np.round(unc.correlation_matrix(params_w_cov), 2)
            print(corr_matrix[:4, :4])
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
                     fmt='.', label = 'Pseudodata (stat)' if not self.asimov else 'Asimov data (stat)', linewidth=2, markersize=10)
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec']/xsec_nom['xsec'], label='Fitted cross section', linewidth=2)
        plt.fill_between(xsec_nom['ecm'], (xsec_fit['xsec']-xsec_fit['unc'])/xsec_nom['xsec'], (xsec_fit['xsec']+xsec_fit['unc'])/xsec_nom['xsec'], alpha=0.5, label='Parametric uncertainty (stat)')
        plt.plot(xsec_pseudodata['ecm'], xsec_pseudodata['xsec']/xsec_nom['xsec'], label='Pseudodata cross section' if not self.asimov else 'Asimov cross section', linestyle='--', linewidth=2)
        plt.axhline(1, color='grey', linestyle='--', label='Reference cross section', linewidth=2)
        plt.xlabel('$\sqrt{s}$ [GeV]')
        plt.ylabel('WbWb total cross section ratio')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.legend(loc='lower right', fontsize=20)
        if not self.scenario_dict['add_last_ecm']:
            plt.xlim(339.7, 347)
        plt.text(.96, 0.45, 'QQbar_Threshold $N^{3}LO$+ISR', fontsize=22, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.41, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.36, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.text(.35, 0.935, '$m_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('mass')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.35, 0.885, '$\Gamma_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('width')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.savefig(plot_dir + '/fit_scenario_ratio_{}.png'.format('pseudo' if not self.asimov else 'asimov'))
        if not self.asimov:
            plt.savefig(plot_dir + '/fit_scenario_ratio_{}.pdf'.format('pseudo' if not self.asimov else 'asimov'))
        plt.clf()

    def doLSscan (self, min = 0, max = 0.5, step = 0.01):
        if min == 0:
            min = 1E-6
        l_beam_energy_res = np.arange(min,max+step/2,step)
        d = {var : [] for var in self.param_names}
        params_to_scan = [param for param in self.param_names if param != 'alphas' and not 'BEC' in param and not 'BES' in param]
        if self.SM_width:
            params_to_scan.remove('width')
        if self.constrain_Yukawa:
            params_to_scan.remove('yukawa')
        
        print('\nScanning parameters: {}\n'.format(params_to_scan))
        
        f = copy.deepcopy(self)
        f.reinitialiseToStat()
        f.param_names = [param for param in self.param_names if not 'BEC' in param and not 'BES' in param]
        f.BEC_nuisances = False
        f.BES_nuisances = False
        for res in l_beam_energy_res:
            f.beam_energy_res = res
            f.update()
            fit_results = f.getFitResults(printout=False)
            for i, param in enumerate(params_to_scan):
                d[param].append(fit_results[i].s)

        f.beam_energy_res = self.beam_energy_res
        f.update()
        f.fit_results = f.getFitResults(printout=False)
        mass_nom = f.fit_results[f.param_names.index('mass')].s
        width_nom = f.fit_results[f.param_names.index('width')].s

        plt.figure()
        plt.plot(l_beam_energy_res, np.array(d['mass'])*1E03, 'b-', label='Stat. uncert. in $m_t$', linewidth=2)
        plt.plot(l_beam_energy_res, np.array(d['width'])*1E03, 'g--', label='Stat. uncert. in $\Gamma_t$',linewidth=2)
        plt.plot(self.beam_energy_res, mass_nom*1E03, 'ro', label='Baseilne $m_t$'.format(self.beam_energy_res), markersize=8)
        plt.plot(self.beam_energy_res, width_nom*1E03, 's', color = 'orange', label='Baseline $\Gamma_t$'.format(self.beam_energy_res), markersize=7)
        plt.legend()
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Beam energy spread [%]')
        plt.ylabel('Statistical uncertainty [MeV]')

        offset = 0.2
        plt.text(.95, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.95, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='right')

        plt.savefig(plot_dir + '/uncert_mass_width_vs_BER.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BER.pdf')

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
                
    def doScaleVars(self): # thanks Copilot
        l_vars = []
        l_mass = []
        l_width = []
        for var in self.parameters.scale_vars:
            if var < 70: continue
            tag_mass = 'scaleM_{:.1f}'.format(var)
            tag_width = 'scaleM_{:.1f}'.format(var)
            if tag_mass not in self.xsec_dict.keys() or tag_width not in self.xsec_dict.keys():
                print('Skipping {}'.format(var))
                continue
            l_vars.append(var)
            nominal_scenario = self.getXsecScenario(self.getXsecTemplate())
            variation_scenario_mass = self.getXsecScenario(self.getXsecTemplate(tag_mass))
            variation_scenario_width = self.getXsecScenario(self.getXsecTemplate(tag_width))
            
            f_mass = copy.deepcopy(self)
            f_mass.scale_var_scenario = np.array(variation_scenario_mass['xsec']) / np.array(nominal_scenario['xsec'])
            f_mass.update()
            fit_results_mass = f_mass.getFitResults(printout=False)
            l_mass.append(fit_results_mass[self.param_names.index('mass')].n - self.fit_results[self.param_names.index('mass')].n)
            
            f_width = copy.deepcopy(self)
            f_width.scale_var_scenario = np.array(variation_scenario_width['xsec']) / np.array(nominal_scenario['xsec'])
            f_width.update()
            fit_results_width = f_width.getFitResults(printout=False)
            l_width.append(fit_results_width[self.param_names.index('width')].n - self.fit_results[self.param_names.index('width')].n)

        plt.plot(l_vars, np.array(l_mass) * 1E03, 'b-', label='Shift in fitted $m_t$', linewidth=2)
        plt.plot(l_vars, np.array(l_width) * 1E03, 'g--', label='Shift in fitted $\Gamma_t$', linewidth=2)
        plt.plot(self.parameters.mass_scale, 0, 'ro', label='Starting point', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Renormalisation scale $\mu$ [GeV]')
        plt.ylabel('Shift in fitted parameter [MeV]')
        plt.text(.6, 0.17, 'QQbar_Threshold $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.13, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.08, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_scale.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_scale.pdf')
        plt.clf()

    def plotScaleVars(self): #to be implemented
        pass

    def dirToVar(self,dir):
        return float(dir.replace('scan_p','').replace('scan_m','')) * (-1 if dir.startswith('scan_m') else 1)
    def varToDir(self,var):
        return 'scan_p{:.0f}'.format(var) if var >= 0 else 'scan_m{:.0f}'.format(abs(var))
    

    # Disclaimer: BEC functionalities are a bit ad hoc, and strongly depend on the directory naming
    # The cross sections with BEC variations must be produced with the same parameters as "nominal"
    # Unfortunately this cannot be easily automated, as the C++ code relies on constexpr variables for the centre-of-mass energy
    # As it stands, the C++ code must be recompiled for each BEC variation
    # This also prevents having a large number of points in the BEC scan

    def fitBECvar(self,var,indir_BEC):
        if abs(var) > 40:
            raise ValueError('Warning: BEC variation too large')
        if abs(var) > 1E-6: # different from zero
            df = self.readScanPerTag('nominal', indir=os.path.join(indir_BEC,self.varToDir(var)))
        else: df = self.readScanPerTag('nominal')

        df['ecm'] = df['ecm'].round(1) # so that it's identical to nominal. This is why variations must be 40 MeV at most
        df_smeared = self.smearCrossSection(df)

        f_bec = copy.deepcopy(self)
        f_bec.pseudo_data_scenario = np.array(self.getXsecScenario(df_smeared)['xsec'])
        f_bec.update(update_scenario=False)
        fit_results = f_bec.getFitResults(printout=False)
        return [res.n for res in fit_results[:2]]
    
        
    
    def addBECnuisances(self,prior_uncorr = None, prior_corr = None):
        self.BEC_nuisances = True
        if prior_uncorr is None:
            prior_uncorr = uncert_BEC_default_uncorr
        if prior_corr is None:
            prior_corr = uncert_BEC_default_corr
        self.setBECpriors(prior_uncorr=prior_uncorr, prior_corr=prior_corr)
        for i in range(0,len(self.morph_scenario['BEC'])):
            self.morph_scenario['BEC_bin{}'.format(i)] = self.morph_scenario['BEC'].copy()
            for j in range(0,len(self.morph_scenario['BEC'])):
                if i != j: self.morph_scenario['BEC_bin{}'.format(i)]['xsec'].iloc[j] = 0
            self.param_names.append('BEC_bin{}'.format(i))
        self.param_names.append('BEC')
        

    def setBECpriors(self, prior_uncorr, prior_corr): # prior in MeV
        self.BEC_prior_uncorr = prior_uncorr / BEC_input_var
        self.BEC_prior_corr = prior_corr / BEC_input_var
    
    def addBESnuisances(self, uncert_uncorr = None, uncert_corr = None):
        self.BES_nuisances = True
        if uncert_uncorr is None:
            uncert_uncorr = uncert_BES_default_uncorr
        if uncert_corr is None:
            uncert_corr = uncert_BES_default_corr
        self.setBESpriors(uncert_corr=uncert_corr, uncert_uncorr=uncert_uncorr)
        for i in range(0,len(self.morph_scenario['BES'])):
            self.morph_scenario['BES_bin{}'.format(i)] = self.morph_scenario['BES'].copy()
            for j in range(0,len(self.morph_scenario['BES'])):
                if i != j: self.morph_scenario['BES_bin{}'.format(i)]['xsec'].iloc[j] = 0
            self.param_names.append('BES_bin{}'.format(i))
        self.param_names.append('BES')
    
    def setBESpriors(self, uncert_uncorr, uncert_corr):
        self.BES_prior_corr = uncert_corr / BES_input_var
        self.BES_prior_uncorr = uncert_uncorr / BES_input_var

    def doBECscan(self, vars, type):
        if not type in ['uncorr', 'corr']:
            raise ValueError('Invalid BEC scan type')
        l_mass = []
        l_width = []

        f = copy.deepcopy(self)
        do_init = not 'BEC' in self.param_names
        if do_init:
            f.addBECnuisances(prior_uncorr=1E-6, prior_corr=1E-6)
        for var in vars:
            if var < 1E-6:
                var = 1E-6
            var_uncorr = var if type == 'uncorr' else 1E-6
            var_corr = var if type == 'corr' else 1E-6
            f.setBECpriors(prior_uncorr=var_uncorr, prior_corr=var_corr)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)
        return l_mass, l_width

    def doBECscans(self, min=0, max=10, step=0.5):
        variations = np.arange(min, max + step / 2, step)
        
        l_mass_uncorr, l_width_uncorr = self.doBECscan(variations, 'uncorr')
        l_mass_corr, l_width_corr = self.doBECscan(variations, 'corr')

        l_mass_uncorr = self.getImpactFromUncert(l_mass_uncorr)
        l_width_uncorr = self.getImpactFromUncert(l_width_uncorr)
        l_mass_corr = self.getImpactFromUncert(l_mass_corr)
        l_width_corr = self.getImpactFromUncert(l_width_corr)

        l_mass_uncorr_nominal, l_width_uncorr_nominal = self.doBECscan([0,uncert_BEC_default_uncorr], 'uncorr')
        l_mass_uncorr_nominal = self.getImpactFromUncert(l_mass_uncorr_nominal)
        l_width_uncorr_nominal = self.getImpactFromUncert(l_width_uncorr_nominal)

        plt.plot(variations, l_mass_uncorr * 1E03, 'b-', label='Impact on $m_t$ (uncorr.)', linewidth=2)
        plt.plot(variations, l_width_uncorr * 1E03, 'g-', label='Impact on $\Gamma_t$ (uncorr.)', linewidth=2)
        plt.plot(variations, l_mass_corr * 1E03, 'b--', label='Impact on $m_t$ (corr.)', linewidth=2)
        plt.plot(variations, l_width_corr * 1E03, 'g--', label='Impact on $\Gamma_t$ (corr.)', linewidth=2)
        
        plt.plot(uncert_BEC_default_uncorr, l_mass_uncorr_nominal[-1] * 1E03, 'ro', label='Baseline $m_t$ (uncorr.)', markersize=8)
        plt.plot(uncert_BEC_default_uncorr, l_width_uncorr_nominal[-1] * 1E03, 's', color='orange', label='Baseline $\Gamma_t$ (uncorr.)', markersize=7)

        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi'] / 1E03), loc='right', fontsize=20)
        plt.xlabel('Uncertainty in $\sqrt{s}$ [MeV]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = 0.35
        plt.text(.05, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='left')
        plt.text(.05, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='left')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC.pdf')
        plt.clf()

    def doBESscan(self,vars,type):
        if not type in ['uncorr','corr']:
            raise ValueError('Invalid BES scan type')
        l_mass = []
        l_width = []

        f = copy.deepcopy(self)
        do_init = not 'BES' in self.param_names
        if do_init:
            f.addBESnuisances(uncert_corr=1E-6, uncert_uncorr=1E-6)
        for var in vars:
            if var < 1E-6:
                var = 1E-6
            var_uncorr = var if type == 'uncorr' else 1E-6
            var_corr = var if type == 'corr' else 1E-6
            f.setBESpriors(uncert_corr=var_corr, uncert_uncorr=var_uncorr)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)
        return l_mass, l_width

    def getImpactFromUncert(self, l_uncert):
        l_uncert = np.array(l_uncert)
        return (l_uncert**2 - l_uncert[0]**2)**.5


    def doBESscans(self, min=0, max=0.03, step=0.001):
        variations = np.arange(min, max + step / 2, step)
        
        l_mass_uncorr, l_width_uncorr = self.doBESscan(variations,'uncorr')
        l_mass_corr, l_width_corr = self.doBESscan(variations,'corr')

        nominal_mass_uncorr, nominal_width_uncorr = self.doBESscan([0,uncert_BES_default_uncorr],'uncorr')
        nominal_mass_uncorr = self.getImpactFromUncert(nominal_mass_uncorr)
        nominal_width_uncorr = self.getImpactFromUncert(nominal_width_uncorr)

        l_mass_uncorr = self.getImpactFromUncert(l_mass_uncorr)
        l_width_uncorr = self.getImpactFromUncert(l_width_uncorr)
        l_mass_corr = self.getImpactFromUncert(l_mass_corr)
        l_width_corr = self.getImpactFromUncert(l_width_corr)

        plt.plot(variations * 100, l_mass_uncorr * 1E03, 'b-', label='Impact on $m_t$ (uncorr.)', linewidth=2)
        plt.plot(variations * 100, l_width_uncorr * 1E03, 'g-', label='Impact on $\Gamma_t$ (uncorr.)', linewidth=2)
        plt.plot(variations * 100, l_mass_corr * 1E03, 'b--', label='Impact on $m_t$ (corr.)', linewidth=2)
        plt.plot(variations * 100, l_width_corr * 1E03, 'g--', label='Impact on $\Gamma_t$ (corr.)', linewidth=2)
        plt.plot(uncert_BES_default_uncorr * 100, nominal_mass_uncorr[-1] * 1E03, 'ro', label='Baseline $m_t$ (uncorr.)', markersize=8)
        plt.plot(uncert_BES_default_uncorr * 100, nominal_width_uncorr[-1] * 1E03, 's', color='orange', label='Baseline $\Gamma_t$ (uncorr.)', markersize=7)

        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi'] / 1E03), loc='right', fontsize=20)
        plt.xlabel('BES uncertainty [%]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = 0.35
        plt.text(.05, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='left')
        plt.text(.05, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='left')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES.pdf')
        plt.clf()
    
    
    def doLumiScan(self,type,l_lumi):
        if not type in ['uncorr','corr']:
            raise ValueError('Invalid lumi scan type')
        l_mass = np.array([])
        l_width = np.array([])
        l_yukawa = np.array([])
        for lumi in l_lumi:
            f_lumi = copy.deepcopy(self)
            if type == 'uncorr':
                f_lumi.lumi_uncorr = lumi
                f_lumi.lumi_corr = 0
            else:
                f_lumi.lumi_corr = lumi
                f_lumi.lumi_uncorr = 0

            f_lumi.fitParameters(initMinuit=True)
            fit_results = f_lumi.getFitResults(printout=False)
            l_mass = np.append(l_mass, fit_results[self.param_names.index('mass')].s*1000)
            l_width = np.append(l_width, fit_results[self.param_names.index('width')].s*1000)
            l_yukawa = np.append(l_yukawa, fit_results[self.param_names.index('yukawa')].s*100)

        l_mass = self.getImpactFromUncert(l_mass)
        l_width = self.getImpactFromUncert(l_width)
        l_yukawa = self.getImpactFromUncert(l_yukawa)
        
        return l_mass, l_width, l_yukawa        

    def doLumiScans(self, min=0, max=3, points=11):
        dict_res = dict()
        l_lumi = np.linspace(min,max,points) * uncert_lumi_default_uncorr
        for type in ['uncorr','corr']:
            l_mass, l_width, l_yukawa = self.doLumiScan(type,l_lumi)
            dict_res[type] = [l_mass, l_width, l_yukawa]

        l_lumi *= 100
        plt.plot(l_lumi, dict_res['uncorr'][0], 'b-', label='Impact on $m_t$ (uncorr.)', linewidth=2)
        plt.plot(l_lumi, dict_res['uncorr'][1], 'g', label='Impact on $\Gamma_t$ (uncorr.)', linewidth=2)
        plt.plot(l_lumi, dict_res['corr'][0], 'b--', label='Impact on $m_t$ (corr.)', linewidth=2)
        plt.plot(l_lumi, dict_res['corr'][1], 'g--', label='Impact on $\Gamma_t$ (corr.)', linewidth=2)

        mass_nominal, width_nominal, _ = self.doLumiScan('uncorr',[0,uncert_lumi_default_uncorr])

        plt.plot(uncert_lumi_default_uncorr*100, mass_nominal[-1], 'ro', label='Baseline $m_t$ (uncorr.)', markersize=8)
        plt.plot(uncert_lumi_default_uncorr*100, width_nominal[-1], 's', color='orange', label='Baseline $\Gamma_t$ (uncorr.)', markersize=7)

        plt.legend()
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Integrated luminosity uncertainty [%]')
        plt.ylabel('Impact on fitted parameter [MeV]')    
        offset = -0.03
        x_pos = 0.92
        plt.text(x_pos, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(x_pos, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='right')


        plt.savefig(plot_dir + '/uncert_mass_width_vs_lumi.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_lumi.pdf')
        plt.clf()

        if not self.scenario_dict['add_last_ecm']:
            return
        
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['uncorr'][2], 'r-', label='Impact on $y_t$ (uncorr)', linewidth=2)
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['corr'][2], 'r--', label='Impact on $y_t$ (corr)', linewidth=2)
        plt.plot(1, dict_res['uncorr'][2][list(np.linspace(0.5, 1.5, 11)).index(1)], 'ro', label='Nominal value', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Projection}}$ ({:.2f} ab$^{{-1}}$)'.format(self.scenario_dict['last_lumi']/1E06), loc='right', fontsize=20)
        plt.xlabel('Luminosity uncert. / nominal value')
        plt.ylabel('Luminosity uncert. on fitted $y_t$ [%]')
        offset = 0.1
        plt.text(.96, 0.17 + offset, 'QQbar_Threshold $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.07, 'nominal uncorr (corr) uncert. = {:.3f} ({:.2f}) %'.format(self.lumi_uncorr_ecm[-1]*100,self.lumi_corr*100), fontsize=21, transform=plt.gca().transAxes, ha='right',)
        plt.savefig(plot_dir + '/uncert_yukawa_vs_lumi.png')
        plt.savefig(plot_dir + '/uncert_yukawa_vs_lumi.pdf')
        plt.clf()

    def doAlphaSscans(self, max_uncert = 3E-4, step = 1E-5):
        l_alphas_uncert = np.arange(1E-10, max_uncert + step / 2, step)
        l_mass = []
        l_width = []
        for alphas_u in l_alphas_uncert:
            f_alphas = copy.deepcopy(self)
            f_alphas.input_uncert_alphas = alphas_u
            f_alphas.fitParameters()
            fit_results = f_alphas.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s * 1000)
            l_width.append(fit_results[self.param_names.index('width')].s * 1000)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        nominal_mass = self.fit_results[self.param_names.index('mass')].s * 1000
        nominal_width = self.fit_results[self.param_names.index('width')].s * 1000
        nominal_mass = (nominal_mass**2 - l_mass[0]**2)**.5
        nominal_width = (nominal_width**2 - l_width[0]**2)**.5

        l_mass = self.getImpactFromUncert(l_mass)
        l_width = self.getImpactFromUncert(l_width)


        plt.plot(l_alphas_uncert * 1E03, l_mass, 'b-', label='Impact on $m_t$', linewidth=2)
        plt.plot(l_alphas_uncert * 1E03, l_width, 'g--', label='Impact on $\Gamma_t$', linewidth=2)
        plt.plot(uncert_alphas_default * 1E03, nominal_mass, 'ro', label='Baseline $m_t$', markersize=8)
        plt.plot(uncert_alphas_default * 1E03, nominal_width, 's', color='orange', label='Baseline $\Gamma_t$', markersize=7)

        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi'] / 1E03), loc='right', fontsize=20)
        plt.xlabel(r'Uncertainty in $\alpha_S$ [x$10^3$]')
        plt.ylabel(r'Impact on fitted parameter [MeV]')
        offset = 0
        x_pos = .92
        plt.text(x_pos, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(x_pos, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_alphas.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_alphas.pdf')
        plt.clf()
            

    def doYukawaScan(self, max_uncert = 0.05, step = 0.001):
        l_yukawa_uncert = np.arange(1E-10, max_uncert+step/2, step)
        l_mass = []
        l_width = []
        for yukawa_u in l_yukawa_uncert:
            f_yukawa = copy.deepcopy(self)
            f_yukawa.input_uncert_Yukawa = yukawa_u
            f_yukawa.constrain_Yukawa = True
            f_yukawa.fitParameters()
            fit_results = f_yukawa.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s*1000)
            l_width.append(fit_results[self.param_names.index('width')].s*1000)
        
        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        nominal_mass = self.fit_results[self.param_names.index('mass')].s*1000
        nominal_width = self.fit_results[self.param_names.index('width')].s*1000
        nominal_mass = (nominal_mass**2 - l_mass[0]**2)**.5
        nominal_width = (nominal_width**2 - l_width[0]**2)**.5

        l_mass = self.getImpactFromUncert(l_mass)
        l_width = self.getImpactFromUncert(l_width)

        plt.plot(l_yukawa_uncert*100, l_mass, 'b-', label='Impact on $m_t$', linewidth=2)
        plt.plot(l_yukawa_uncert*100, l_width, 'g--', label='Impact on $\Gamma_t$', linewidth=2)
        plt.plot(uncert_yukawa_default*100, nominal_mass, 'ro', label='Baseline $m_t$', markersize=8)
        plt.plot(uncert_yukawa_default*100, nominal_width, 's', color='orange', label='Baseline $\Gamma_t$', markersize=7)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncertainty in $y_t$ [%]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = 0
        plt.text(.92, 0.17 + offset, 'WbWb at $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.92, 0.12 + offset, '+ FCC-ee BES', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_yukawa.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_yukawa.pdf')
        plt.clf()

    def doChi2Scans(self):
        for i_param, param in enumerate(self.param_names):
            if param == 'yukawa' and self.constrain_Yukawa: continue
            elif param == 'width' and self.SM_width: continue
            elif param == 'alphas': continue
            elif 'BEC' in param: continue
            elif 'BES' in param: continue
            l_param = np.linspace(self.minuit.values[i_param]-3*self.minuit.errors[i_param], self.minuit.values[param]+3*self.minuit.errors[i_param], 101)
            l_chi2 = []
            for val in l_param:
                f = copy.deepcopy(self)
                f.minuit.fixed[i_param] = True
                f.minuit.values[i_param] = val
                f.fitParameters(initMinuit=False)
                l_chi2.append(f.minuit.fval)
            plt.plot(self.getValueFromParameter(l_param, param), l_chi2, label=param)
            plt.xlabel(label_d[param])
            plt.ylabel(r'$\chi^2$')
            plt.legend()
            plt.savefig(plot_dir + '/chi2_scan_{}.png'.format(param))
            plt.clf()
        # 2D contour plots
        f = copy.deepcopy(self)
        for i_param, param in enumerate(self.param_names):
            if param == 'yukawa' and self.constrain_Yukawa: continue
            elif param == 'width' and self.SM_width: continue
            elif param == 'alphas': continue
            elif 'BEC' in param: continue
            elif 'BES' in param: continue
            for j_param, param2 in enumerate(self.param_names):
                if param2 == 'yukawa' and self.constrain_Yukawa: continue
                elif param2 == 'width' and self.SM_width: continue
                elif param2 == 'alphas': continue
                elif 'BEC' in param2: continue
                elif 'BES' in param2: continue
                if j_param <= i_param: continue
                l_param = np.linspace(self.minuit.values[i_param]-3*self.minuit.errors[i_param], self.minuit.values[param]+3*self.minuit.errors[i_param], 51)
                l_param2 = np.linspace(self.minuit.values[j_param]-3*self.minuit.errors[j_param], self.minuit.values[param2]+3*self.minuit.errors[j_param], 51)
                l_chi2 = np.zeros((len(l_param),len(l_param2)))
                for i, val in enumerate(l_param):
                    for j, val2 in enumerate(l_param2):
                        f = copy.deepcopy(self)
                        for k in range(0,len(self.param_names)):
                            f.minuit.fixed[k] = False if k != i_param and k != j_param else True
                        f.minuit.values[i_param] = val
                        f.minuit.values[j_param] = val2
                        f.fitParameters(initMinuit=False)
                        l_chi2[i][j] = f.minuit.fval
                plt.contour(self.getValueFromParameter(l_param, param), self.getValueFromParameter(l_param2, param2), l_chi2, levels=[self.minuit.fval + 1, self.minuit.fval + 4], colors=['#377eb8', '#4daf4a'], linewidths=2)
                plt.plot(self.getValueFromParameter(self.minuit.values[i_param], param), self.getValueFromParameter(self.minuit.values[j_param], param2), 'k*', label='Best fit value (input)', markersize=10)
                plt.plot([], [], color='#377eb8', label='68% C.L.')
                plt.plot([], [], color='#4daf4a', label='95% C.L.')
                plt.xlabel(label_d[param])
                plt.ylabel(label_d[param2])
                plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
                plt.text(.95, 0.15, 'QQbar_Threshold $N^{3}LO$+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
                plt.text(.95, 0.11, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
                plt.text(.95, 0.06, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
                plt.legend(loc='upper left')
                y_offset = -0.003 if param2 == 'width' and param == 'mass' else 0
                plt.ylim(self.getValueFromParameter(l_param2[0], param2)+y_offset, self.getValueFromParameter(l_param2[-1], param2)+y_offset)
                plt.xticks(np.round(np.linspace(self.getValueFromParameter(l_param[0], param), self.getValueFromParameter(l_param[-1], param), 5), 2))
                plt.savefig(plot_dir + '/chi2_scan_{}_{}.png'.format(param,param2))
                plt.savefig(plot_dir + '/chi2_scan_{}_{}.pdf'.format(param,param2))
                plt.clf()

    def addSystToTable(self,syst_name):
        fit_results = self.getFitResults(printout=False)
        self.syst_mass[syst_name] = fit_results[self.param_names.index('mass')].s*1000
        self.syst_width[syst_name] = fit_results[self.param_names.index('width')].s*1000
        if not self.constrain_Yukawa:
            self.syst_yukawa[syst_name] = fit_results[self.param_names.index('yukawa')].s*100
        if syst_name != 'stat' and syst_name != 'total':
            self.syst_mass[syst_name] = (self.syst_mass['total']**2 - self.syst_mass[syst_name]**2)**.5
            self.syst_width[syst_name] = (self.syst_width['total']**2 - self.syst_width[syst_name]**2)**.5
            if not self.constrain_Yukawa:
                self.syst_yukawa[syst_name] = (self.syst_yukawa['total']**2 - self.syst_yukawa[syst_name]**2)**.5

    def estimateSyst(self,syst_name):
        if syst_name == 'stat':
            self.reinitialiseToStat()
        elif syst_name == 'alphaS':
            self.input_uncert_alphas = 1E-10
        elif syst_name == 'Yukawa':
            self.input_uncert_Yukawa = 1E-10
        elif syst_name == 'BES_corr':
            self.BES_prior_corr = 1E-10
        elif syst_name == 'BES_uncorr':
            self.BES_prior_uncorr = 1E-10
        elif syst_name == 'BEC_corr':
            self.BEC_prior_corr = 1E-10
        elif syst_name == 'BEC_uncorr':
            self.BEC_prior_uncorr = 1E-10
        elif syst_name == 'lumi_corr':
            self.lumi_corr = 1E-10
        elif syst_name == 'lumi_uncorr':
            self.lumi_uncorr = 1E-10
        self.fitParameters()
        self.addSystToTable(syst_name)
        self.reinitialiseToNominal()


    def reinitialiseToStat(self):
        self.input_uncert_alphas = 1E-10
        self.input_uncert_Yukawa = 1E-10
        #self.constrain_Yukawa = True
        self.BEC_prior_corr = 1E-10
        self.BEC_prior_uncorr = 1E-10
        self.BES_prior_corr = 1E-10
        self.BES_prior_uncorr = 1E-10
        self.lumi_corr = 1E-10
        self.lumi_uncorr = 1E-10

    def reinitialiseToNominal(self):
        self.input_uncert_alphas = uncert_alphas_default
        self.input_uncert_Yukawa = uncert_yukawa_default
        #self.constrain_Yukawa = True
        self.setBECpriors(prior_corr=uncert_BEC_default_corr, prior_uncorr=uncert_BEC_default_uncorr)
        self.setBESpriors(uncert_corr=uncert_BES_default_corr, uncert_uncorr=uncert_BES_default_uncorr)
        self.lumi_corr = uncert_lumi_default_corr
        self.lumi_uncorr = uncert_lumi_default_uncorr


    def printSystTable(self):
        f = copy.deepcopy(self)
        f.syst_mass = dict()
        f.syst_width = dict()
        if not self.constrain_Yukawa:
            f.syst_yukawa = dict()
        #TODO: automatic list of systematics when nuisances are added
        syst_list = ['stat','total','alphaS','Yukawa','BES_uncorr','BES_corr','BEC_uncorr','BEC_corr','lumi_uncorr','lumi_corr']
        if not self.constrain_Yukawa:
            syst_list.remove('Yukawa')
        for syst in syst_list:
            f.estimateSyst(syst) 
        
        total_mass_unc = f.syst_mass.pop('total')
        total_width_unc = f.syst_width.pop('total')
        if not self.constrain_Yukawa:
            total_yukawa_unc = f.syst_yukawa.pop('total')

        if self.constrain_Yukawa:
            print()
            print(f"{'Systematic':<12} {'Mass [MeV]':<12} {'Width [MeV]':<12}")
            print("-" * 36)
            for syst, mass_unc in f.syst_mass.items():
                width_unc = f.syst_width[syst]
                print(f"{syst:<12} {mass_unc:<12.1f} {width_unc:<12.1f}")
            print("-" * 36)
            print(f"{'total exp':<12} {total_mass_unc:<12.1f} {total_width_unc:<12.1f}")

            theory_mass_unc = 35 #hardcoded
            theory_width_unc = 25 #hardcoded
            print(f"{'theory':<12} {theory_mass_unc:<12.0f} {theory_width_unc:<12.0f}")

        else:
            print()
            print(f"{'Systematic':<12} {'Mass [MeV]':<12} {'Width [MeV]':<12} {'Yukawa [%]':<12}")
            print("-" * 48)
            for syst, mass_unc in f.syst_mass.items():
                width_unc = f.syst_width[syst]
                yukawa_unc = f.syst_yukawa[syst]
                print(f"{syst:<12} {mass_unc:<12.1f} {width_unc:<12.1f} {yukawa_unc:<12.1f}")
            print("-" * 48)
            print(f"{'total exp':<12} {total_mass_unc:<12.1f} {total_width_unc:<12.1f} {total_yukawa_unc:<12.1f}")
            theory_mass_unc = 35 #hardcoded
            theory_width_unc = 25
            theory_yukawa_unc = 99
            print(f"{'theory':<12} {theory_mass_unc:<12.0f} {theory_width_unc:<12.0f} {theory_yukawa_unc:<12.0f}")

        latex_table = r"""
        \begin{table}[h!]
        \centering
        \begin{tabular}{|l|r|r|}
        \hline
        Systematic & Mass Uncertainty (MeV) & Width Uncertainty (MeV) \\
        \hline
        """

        for syst, mass_unc in f.syst_mass.items():
            width_unc = f.syst_width[syst]
            latex_table += f"{syst} & {mass_unc:.1f} & {width_unc:.1f} \\\\\n"

        latex_table += f"\hline\n"
        latex_table += f"total & {total_mass_unc:.1f} & {total_width_unc:.1f} \\\\\n"
        latex_table += f"theory & {theory_mass_unc:.0f} & {theory_width_unc:.0f} \\\\\n\\hline\n"

        latex_table += r"""
        \end{tabular}
        \caption{Systematic uncertainties on mass and width.}
        \label{tab:syst_unc}
        \end{table}
        """

        with open("systematics_table.tex", "w") as f:
            f.write(latex_table)



def main():
    parser = argparse.ArgumentParser(description='Specify options')
    parser.add_argument('--pseudo', action='store_true', help='Pseudodata')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--LSscan', action='store_true', help='Do beam energy resolution scan')
    parser.add_argument('--scaleVars', action='store_true', help='Do scale variations')
    parser.add_argument('--SMwidth', action='store_true', help='Constrain width to SM value')
    parser.add_argument('--fitYukawa', action='store_true', help='Constrain Yukawa coupling in fit')
    parser.add_argument('--lastecm', action='store_true', help='Add last ecm to scenario')
    parser.add_argument('--sameNevts', action='store_true', help='Same number of events in each ecm')
    parser.add_argument('--BECscans', action='store_true', help='Do beam energy calibration scans')
    parser.add_argument('--BESscans', action='store_true', help='Do beam energy spread scans')
    parser.add_argument('--lumiscans', action='store_true', help='Do luminosity scans')
    parser.add_argument('--alphaSscan', action='store_true', help='Do alphaS scan')
    parser.add_argument('--chi2scans', action='store_true', help='Do chi2 scans')
    parser.add_argument('--BECnuisances', action='store_true', help='add BEC nuisances')
    parser.add_argument('--BESnuisances' , action='store_true', help='add BES nuisances')
    parser.add_argument('--systTable', action='store_true', help='Produce systematic table')
    args = parser.parse_args()

    if (args.BECscans or args.BESscans or args.BECnuisances or args.BESnuisances) and args.scaleVars:
        raise ValueError('BEC scan currently incompatible with scale variations')
    if args.alphaSscan and args.lastecm:
        raise ValueError('AlphaS scan currently incompatible with last ecm')
    if args.SMwidth:
        raise ValueError('SM width assumption currently not supported') # to be fixed
    
    
    threshold_lumi = 0.41 * 1E06 # hardcoded
    above_threshold_lumi = 2.65 * 1E06 # hardcoded

    f = fit(debug=args.debug, asimov=not args.pseudo, SM_width=args.SMwidth, constrain_Yukawa= not args.fitYukawa, read_scale_vars = args.scaleVars)
    f.initScenario(scan_min=340.5, scan_max=345, scan_step=.5, total_lumi=threshold_lumi, last_lumi=above_threshold_lumi, add_last_ecm = args.lastecm, same_evts = args.sameNevts)
    
    if args.BECnuisances or args.systTable:
        f.addBECnuisances()
    if args.BESnuisances or args.systTable:
        f.addBESnuisances()
    f.fitParameters()
    f.getFitResults()
    f.plotFitScenario()
    if args.LSscan:
        f.doLSscan()
    if args.scaleVars:
        f.doScaleVars()
        f.plotScaleVars() # to be implemented
    if args.BECscans:
        f.doBECscans()
    if args.BESscans:
        f.doBESscans()
    if args.lumiscans:
        f.doLumiScans()
    if args.alphaSscan:
        f.doAlphaSscans()
        if not args.fitYukawa:
            f.doYukawaScan() # by default
    if args.chi2scans:
        f.doChi2Scans()
    if args.systTable:
        f.printSystTable()
    


if __name__ == '__main__':
    main()
