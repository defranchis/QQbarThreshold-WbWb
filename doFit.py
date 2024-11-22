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

uncert_yukawa_default = 0.03 # only when parameter is constrained. hardcoded for now
uncert_alphas_default =  0.0003 

label_d = {'mass': '$m_t$ [GeV]', 'width': r'$\Gamma_t$'+' [GeV]', 'yukawa': 'y_t', 'alphas': r'\alpha_S'}


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
        self.lumi_uncorr = 1E-3 # hardcoded: estimate for full lumi! i.e. 410/fb
        self.lumi_corr = 1E-4 # hardcoded: estimate for theory cross section uncertainty
        self.input_uncert_Yukawa = uncert_yukawa_default
        self.input_uncert_alphas = uncert_alphas_default

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
        if param == 'BEC' and not self.read_scale_vars:
            tmp = self.readScanPerTag('nominal', indir=os.path.join(indir_BEC,self.varToDir(BEC_input_var)))
            tmp ['ecm'] = tmp['ecm'].round(1)
            xsec_var = self.smearCrossSection(tmp)
        else:
            xsec_var = self.getXsecTemplate('{}_var'.format(param))
        df_morph = pd.DataFrame({'ecm': xsec_nom['ecm'], 'xsec': xsec_var['xsec']/xsec_nom['xsec'] -1})
        return df_morph      

    def morphCrossSections(self):
        self.morph_dict = {}
        for param in self.param_names:
            self.morph_dict[param] = self.morphCrossSection(param)
        self.morph_dict['BEC'] = self.morphCrossSection('BEC')

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
            self.BEC_var_scenario = np.ones(len(self.xsec_scenario['xsec']))
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
        th_xsec *= self.BEC_var_scenario
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
        return chi2 + prior_width


    def initMinuit(self, exclude_stat = False):
        cov_stat = np.diag(self.unc_pseudodata_scenario**2)
        factor_thresh = len(self.scenario) if not self.scenario_dict['add_last_ecm'] else len(self.scenario) - 1
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
                     fmt='.', label = 'Pseudodata (stat)' if not self.asimov else 'Asimov data (stat)', linewidth=2, markersize=10)
        plt.plot(xsec_nom['ecm'],xsec_fit['xsec']/xsec_nom['xsec'], label='Fitted cross section', linewidth=2)
        plt.fill_between(xsec_nom['ecm'], (xsec_fit['xsec']-xsec_fit['unc'])/xsec_nom['xsec'], (xsec_fit['xsec']+xsec_fit['unc'])/xsec_nom['xsec'], alpha=0.5, label='Parametric uncertainty (stat)')
        plt.plot(xsec_pseudodata['ecm'], xsec_pseudodata['xsec']/xsec_nom['xsec'], label='Pseudodata cross section' if not self.asimov else 'Asimov cross section', linestyle='--', linewidth=2)
        plt.axhline(1, color='grey', linestyle='--', label='Reference cross section', linewidth=2)
        plt.xlabel('$\sqrt{s}$ [GeV]')
        plt.ylabel('WbWb total cross section ratio')
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.legend(loc='lower right', fontsize=20)
        if not self.scenario_dict['add_last_ecm']:
        #if True:
            plt.xlim(339.7, 347)
        plt.text(.96, 0.45, 'QQbar_Threshold N3LO+ISR', fontsize=22, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.41, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.36, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.text(.35, 0.935, '$m_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('mass')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.35, 0.885, '$\Gamma_t$ (stat) = {:.0f} MeV'.format(self.fit_results[self.param_names.index('width')].s*1E03), fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.savefig(plot_dir + '/fit_scenario_ratio_{}.png'.format('pseudo' if not self.asimov else 'asimov'))
        if not self.asimov:
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
        
        plt.figure()
        plt.plot(l_beam_energy_res, np.array(d['mass'])*1E03, 'b-', label='$m_t$ statistical uncertainty', linewidth=2)
        plt.plot(l_beam_energy_res, np.array(d['width'])*1E03, 'g--', label='$\Gamma_t$ statistical uncertainty',linewidth=2)
        plt.plot(self.beam_energy_res, self.fit_results[self.param_names.index('mass')].s*1E03, 'ro', label='Nominal $m_t$'.format(self.beam_energy_res), markersize=8)
        plt.plot(self.beam_energy_res, self.fit_results[self.param_names.index('width')].s*1E03, 's', color = 'orange', label='Nominal $\Gamma_t$'.format(self.beam_energy_res), markersize=7)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Beam energy spread [%]')
        plt.ylabel('Statistical uncertainty [MeV]')

        plt.text(.6, 0.65, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.61, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.56, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')

        plt.savefig(plot_dir + '/uncert_mass_width_vs_beam_energy_res.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_beam_energy_res.pdf')

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
        plt.plot(self.parameters.mass_scale, 0, 'ro', label='Nominal fit', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Renormalisation scale $\mu$ [GeV]')
        plt.ylabel('Shift in fitted parameter [MeV]')
        plt.text(.6, 0.17, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
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
    

    def doBECscanCorr(self):
        if not os.path.exists(indir_BEC):
            raise ValueError('Directory {} not found'.format(indir_BEC))
        bec_dirs = [d for d in os.listdir(indir_BEC) if os.path.isdir(os.path.join(indir_BEC, d))]
        variations = [self.dirToVar(bec_dir) for bec_dir in bec_dirs]
        variations.append(0)
        variations.sort()

        l_mass = []
        l_width = []
        for var in variations:
            mass, width = self.fitBECvar(var, indir_BEC)
            l_mass.append(mass)
            l_width.append(width)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        l_mass -= l_mass[variations.index(0)]
        l_width -= l_width[variations.index(0)]

        plt.plot(variations, l_mass*1E03, 'b-', label='Shift in fitted $m_t$', linewidth=2)
        plt.plot(variations, l_width*1E03, 'g--', label='Shift in fitted $\Gamma_t$', linewidth=2)
        plt.plot(0, 0, 'ro', label='Nominal fit', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel(r'Shift in $\sqrt{s}$ [MeV]')
        plt.ylabel('Shift in fitted parameter [MeV]')
        plt.text(.6, 0.17, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.13, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.6, 0.08, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC.pdf')
        plt.clf()


    def doBECscanCorrMorph(self, min = -30, max = 30, points = 11):

        variations = np.linspace(min, max, points)
        l_mass = []
        l_width = []
        for var in variations:
            mass, width = self.fitBECvarMorph(var/BEC_input_var)
            l_mass.append(mass)
            l_width.append(width)

        nominal_mass, nominal_width = self.fitBECvarMorph(0)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        l_mass -= nominal_mass
        l_width -= nominal_width

        plt.plot(variations, l_mass*1E03, 'b-', label='Shift in fitted $m_t$', linewidth=2)
        plt.plot(variations, l_width*1E03, 'g--', label='Shift in fitted $\Gamma_t$', linewidth=2)
        plt.plot(0, 0, 'ro', label='Nominal fit', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel(r'Shift in $\sqrt{s}$ [MeV]')
        plt.ylabel('Shift in fitted parameter [MeV]')
        plt.text(.9, 0.17, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_morph.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_morph.pdf')
        plt.clf()

    def fitBECvarMorph(self,var):
        f = copy.deepcopy(self)
        f.BEC_var_scenario *= (1 + var*np.array(self.morph_scenario['BEC']['xsec']))
        f.fitParameters(initMinuit=False)
        return [res.n for res in f.getFitResults(printout=False)[:2]]

    
    def doLumiScan(self,type):
        if not type in ['uncorr','corr']:
            raise ValueError('Invalid lumi scan type')
        l_lumi = np.linspace(0.5, 1.5, 11)*self.lumi_uncorr if type == 'uncorr' else np.linspace(0.5, 1.5, 11)*self.lumi_corr
        l_mass = []
        l_width = []
        l_yukawa = []
        for lumi in l_lumi:
            f_lumi = copy.deepcopy(self)
            f_lumi.lumi_uncorr = lumi if type == 'uncorr' else self.lumi_uncorr
            f_lumi.lumi_corr = lumi if type == 'corr' else self.lumi_corr
            f_lumi.update(exclude_stat=True)
            fit_results = f_lumi.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s*1000)
            l_width.append(fit_results[self.param_names.index('width')].s*1000)
            l_yukawa.append(fit_results[self.param_names.index('yukawa')].s*100)
        return l_mass, l_width, l_yukawa        

    def doLumiScans(self):
        dict_res = dict()
        for type in ['uncorr','corr']:
            l_mass, l_width, l_yukawa = self.doLumiScan(type)
            dict_res[type] = [l_mass, l_width, l_yukawa]

        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['uncorr'][0], 'b-', label='Impact on $m_t$ (uncorr)', linewidth=2)
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['uncorr'][1], 'g', label='Impact on $\Gamma_t$ (uncorr)', linewidth=2)
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['corr'][0], 'b--', label='Impact on $m_t$ (corr)', linewidth=2)
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['corr'][1], 'g--', label='Impact on $\Gamma_t$ (corr)', linewidth=2)

        plt.plot(1, dict_res['uncorr'][0][list(np.linspace(0.5, 1.5, 11)).index(1)], 'ro', label='Nominal values', markersize=8)
        plt.plot(1, dict_res['uncorr'][1][list(np.linspace(0.5, 1.5, 11)).index(1)], 'ro', label=None, markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Luminosity uncert. / nominal value')
        plt.ylabel('Luminosity uncert. on fitted parameter [MeV]')    
        offset = 0.1   
        plt.text(.96, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.07, 'nominal uncorr (corr) uncert. = {:.1f} ({:.2f}) %'.format(self.lumi_uncorr*100,self.lumi_corr*100), fontsize=21, transform=plt.gca().transAxes, ha='right',)
        plt.savefig(plot_dir + '/uncert_mass_width_vs_lumi.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_lumi.pdf')
        plt.clf()

        if not self.scenario_dict['add_last_ecm']:
            return
        
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['uncorr'][2], 'r-', label='Impact on $y_t$ (uncorr)', linewidth=2)
        plt.plot(np.linspace(0.5, 1.5, 11), dict_res['corr'][2], 'r--', label='Impact on $y_t$ (corr)', linewidth=2)
        plt.plot(1, dict_res['uncorr'][2][list(np.linspace(0.5, 1.5, 11)).index(1)], 'ro', label='Nominal value', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Preliminary}}$ ({:.2f} ab$^{{-1}}$)'.format(self.scenario_dict['last_lumi']/1E06), loc='right', fontsize=20)
        plt.xlabel('Luminosity uncert. / nominal value')
        plt.ylabel('Luminosity uncert. on fitted $y_t$ [%]')
        offset = 0.1
        plt.text(.96, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.text(.96, 0.07, 'nominal uncorr (corr) uncert. = {:.3f} ({:.2f}) %'.format(self.lumi_uncorr_ecm[-1]*100,self.lumi_corr*100), fontsize=21, transform=plt.gca().transAxes, ha='right',)
        plt.savefig(plot_dir + '/uncert_yukawa_vs_lumi.png')
        plt.savefig(plot_dir + '/uncert_yukawa_vs_lumi.pdf')
        plt.clf()

    def doAlphaSscanFixYukawa(self, max_uncert, step, noYukawa, yukawa_uncert):
        if noYukawa and yukawa_uncert:
            raise ValueError('Cannot have no Yukawa and Yukawa uncertainty')
        l_alphas_uncert = np.arange(1E-10, max_uncert+step/2, step)
        l_mass = []
        l_width = []
        yukawa_uncert = yukawa_uncert if not yukawa_uncert is None else self.input_uncert_Yukawa
        if noYukawa: yukawa_uncert = 1E-10
        print('Yukawa uncertainty: {}'.format(yukawa_uncert))
        for alphas_u in l_alphas_uncert:
            f_alphas = copy.deepcopy(self)
            f_alphas.input_uncert_alphas = alphas_u
            f_alphas.input_uncert_Yukawa = yukawa_uncert
            f_alphas.constrain_Yukawa = True
            f_alphas.update()
            fit_results = f_alphas.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s*1000)
            l_width.append(fit_results[self.param_names.index('width')].s*1000)
        return l_alphas_uncert, l_mass, l_width, yukawa_uncert
    
    def doAlphaSscans(self, max_uncert = 7E-4, step = 1E-5, noYukawa = False, yukawa_uncert = None):
        l_as, l_m, l_w, yt_unc = self.doAlphaSscanFixYukawa(max_uncert, step, noYukawa = noYukawa, yukawa_uncert = yukawa_uncert)
        plt.plot(l_as *1E03, l_m, 'b-', label='Impact on $m_t$', linewidth=2)
        plt.plot(l_as *1E03, l_w, 'g--', label='Impact on $\Gamma_t$', linewidth=2)
        label_nom = '$y_t$ uncert. {}%'.format(yt_unc*100) if not noYukawa else 'w/o $y_t$'
        plt.plot(uncert_alphas_default *1E03, l_m[min(range(len(l_as)), key=lambda i: abs(l_as[i] - uncert_alphas_default))], 'ro', label='Nominal fit \n{}'.format(label_nom), markersize=8)
        plt.plot(uncert_alphas_default *1E03, l_w[min(range(len(l_as)), key=lambda i: abs(l_as[i] - uncert_alphas_default))], 'ro', label='', markersize=8)

        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel(r'Uncertainty in $\alpha_S$ [x$10^3$]')
        plt.ylabel(r'Uncertainty on fitted parameter [MeV]')
        plt.text(.9, 0.17, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.ylim(min(l_m)-1, max(l_w)+2)
        outname = 'uncert_mass_width_vs_alphas'
        if noYukawa: outname += '_noYukawa'
        if yt_unc < self.input_uncert_Yukawa: outname += '_optimisticYukawa'
        plt.savefig(plot_dir + '/{}.png'.format(outname))
        plt.savefig(plot_dir + '/{}.pdf'.format(outname))
        plt.clf()

    def doYukawaScan(self, max_uncert = 0.1, step = 0.005):
        l_yukawa_uncert = np.arange(0.005, max_uncert+step/2, step)
        l_mass = []
        l_width = []
        for yukawa_u in l_yukawa_uncert:
            f_yukawa = copy.deepcopy(self)
            f_yukawa.input_uncert_Yukawa = yukawa_u
            f_yukawa.constrain_Yukawa = True
            f_yukawa.update()
            fit_results = f_yukawa.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s*1000)
            l_width.append(fit_results[self.param_names.index('width')].s*1000)
        plt.plot(l_yukawa_uncert, l_mass, 'b-', label='Impact on $m_t$', linewidth=2)
        plt.plot(l_yukawa_uncert, l_width, 'g--', label='Impact on $\Gamma_t$', linewidth=2)
        label_nom = r'$\alpha_S$ uncert. {}'.format(uncert_alphas_default)
        plt.plot(self.input_uncert_Yukawa, l_mass[min(range(len(l_yukawa_uncert)), key=lambda i: abs(l_yukawa_uncert[i] - self.input_uncert_Yukawa))], 'ro', label='Nominal fit \n{}'.format(label_nom), markersize=8)
        plt.plot(self.input_uncert_Yukawa, l_width[min(range(len(l_yukawa_uncert)), key=lambda i: abs(l_yukawa_uncert[i] - self.input_uncert_Yukawa))], 'ro', label='', markersize=8)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncertainty in $y_t$')
        plt.ylabel('Uncertainty on fitted parameter [MeV]')
        offset = .3
        plt.text(.9, 0.17+offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13+offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08+offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.ylim(min(l_mass)-1, max(l_width)+2)
        plt.savefig(plot_dir + '/uncert_mass_width_vs_yukawa.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_yukawa.pdf')
        plt.clf()

    def doChi2Scans(self):
        for i_param, param in enumerate(self.param_names):
            if param == 'yukawa' and self.constrain_Yukawa: continue
            elif param == 'width' and self.SM_width: continue
            elif param == 'alphas': continue
            l_param = np.linspace(self.minuit.values[i_param]-3*self.minuit.errors[i_param], self.minuit.values[param]+3*self.minuit.errors[i_param], 101)
            l_chi2 = []
            for val in l_param:
                f = copy.deepcopy(self)
                f.minuit.fixed[i_param] = True
                f.minuit.values[i_param] = val
                #f.update()
                f.fitParameters(initMinuit=False)
                l_chi2.append(f.minuit.fval)
            #plt.plot(l_param, l_chi2, label=param)
            plt.plot(self.getValueFromParameter(l_param, param), l_chi2, label=param)
            plt.xlabel(label_d[param])
            plt.ylabel(r'$\chi^2$')
            plt.legend()
            plt.savefig(plot_dir + '/chi2_scan_{}.png'.format(param))
            plt.clf()
        # 2D contour plots
        for i_param, param in enumerate(self.param_names):
            if param == 'yukawa' and self.constrain_Yukawa: continue
            elif param == 'width' and self.SM_width: continue
            elif param == 'alphas': continue
            for j_param, param2 in enumerate(self.param_names):
                if param2 == 'yukawa' and self.constrain_Yukawa: continue
                elif param2 == 'width' and self.SM_width: continue
                elif param2 == 'alphas': continue
                if j_param <= i_param: continue
                l_param = np.linspace(self.minuit.values[i_param]-3*self.minuit.errors[i_param], self.minuit.values[param]+3*self.minuit.errors[i_param], 51)
                l_param2 = np.linspace(self.minuit.values[j_param]-3*self.minuit.errors[j_param], self.minuit.values[param2]+3*self.minuit.errors[j_param], 51)
                l_chi2 = np.zeros((len(l_param),len(l_param2)))
                for i, val in enumerate(l_param):
                    for j, val2 in enumerate(l_param2):
                        f = copy.deepcopy(self)
                        f.minuit.fixed[i_param] = True
                        f.minuit.fixed[j_param] = True
                        f.minuit.values[i_param] = val
                        f.minuit.values[j_param] = val2
                        f.fitParameters(initMinuit=False)
                        l_chi2[i][j] = f.minuit.fval
                plt.contour(self.getValueFromParameter(l_param, param), self.getValueFromParameter(l_param2, param2), l_chi2, levels=[self.minuit.fval + 1, self.minuit.fval + 4], colors=['#377eb8', '#4daf4a'], linewidths=2)
                plt.plot(self.getValueFromParameter(self.minuit.values[i_param], param), self.getValueFromParameter(self.minuit.values[j_param], param2), 'k*', label='Best fit value', markersize=10)
                plt.plot([], [], color='#377eb8', label='68% CL')
                plt.plot([], [], color='#4daf4a', label='95% CL')
                plt.xlabel(label_d[param])
                plt.ylabel(label_d[param2])
                plt.title(r'$\mathit{{Preliminary}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
                plt.text(.95, 0.15, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
                plt.text(.95, 0.11, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
                plt.text(.95, 0.06, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
                plt.legend(loc='upper left')
                y_offset = -0.003 if param2 == 'width' and param == 'mass' else 0
                plt.ylim(self.getValueFromParameter(l_param2[0], param2)+y_offset, self.getValueFromParameter(l_param2[-1], param2)+y_offset)
                plt.savefig(plot_dir + '/chi2_scan_{}_{}.png'.format(param,param2))
                plt.savefig(plot_dir + '/chi2_scan_{}_{}.pdf'.format(param,param2))
                plt.clf()


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
    parser.add_argument('--BECscan', action='store_true', help='Do beam energy calibration scan')
    parser.add_argument('--lumiscan', action='store_true', help='Do luminosity scan')
    parser.add_argument('--alphaSscan', action='store_true', help='Do alphaS scan')
    parser.add_argument('--chi2Scan', action='store_true', help='Do chi2 scans')
    args = parser.parse_args()

    if args.BECscan and args.scaleVars:
        raise ValueError('BEC scan currently incompatible with scale variations')
    if args.alphaSscan and args.lastecm:
        raise ValueError('AlphaS scan currently incompatible with last ecm')
    if args.SMwidth:
        raise ValueError('SM width assumption currently not supported') # to be fixed
    
    
    threshold_lumi = 0.41 * 1E06 # hardcoded
    above_threshold_lumi = 2.65 * 1E06 # hardcoded

    f = fit(debug=args.debug, asimov=not args.pseudo, SM_width=args.SMwidth, constrain_Yukawa=args.constrainYukawa, read_scale_vars = args.scaleVars)
    f.initScenario(scan_min=340.5, scan_max=345, scan_step=.5, total_lumi=threshold_lumi, last_lumi=above_threshold_lumi, add_last_ecm = args.lastecm, same_evts = args.sameNevts)
    
    
    f.fitParameters()
    f.getFitResults()
    f.plotFitScenario()
    if args.LSscan:
        f.doLSscan(0,0.5,0.01)
    if args.scaleVars:
        f.doScaleVars()
        f.plotScaleVars() # to be implemented
    if args.BECscan:
        f.doBECscanCorr()
        f.doBECscanCorrMorph()
        #f.doBECscanUncorr() # to be implemented
    if args.lumiscan:
        f.doLumiScans()
    if args.alphaSscan:
        f.doAlphaSscans()
        f.doAlphaSscans(noYukawa=True)
        f.doAlphaSscans(yukawa_uncert=0.01)
        f.doYukawaScan() # by default
    if args.chi2Scan:
        f.doChi2Scans()
    


if __name__ == '__main__':
    main()
