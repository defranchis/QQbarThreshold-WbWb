'''
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

    def doBECscanCorr(self, min = 0, max = 15, step = 1):
        variations = np.arange(min,max+step/2,step)
        l_mass = []
        l_width = []
        f = copy.deepcopy(self)
        f.addBECnuisances(0,0)
        for var in variations:
            f.setBECpriors(prior_corr=var, prior_uncorr=0)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        f.setBECpriors(prior_corr=uncert_BEC_default, prior_uncorr=0)
        f.fitParameters(initMinuit=True)
        fit_results = f.getFitResults(printout=False)
        nominal_mass = fit_results[self.param_names.index('mass')].s
        nominal_width = fit_results[self.param_names.index('width')].s

        nominal_mass = (nominal_mass**2 - l_mass[0]**2)**.5
        nominal_width = (nominal_width**2 - l_width[0]**2)**.5

        l_mass = (l_mass**2 - l_mass[0]**2)**.5
        l_width = (l_width**2 - l_width[0]**2)**.5


        plt.plot(variations, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.plot(uncert_BEC_default, (nominal_mass**2 - l_mass[0]**2)**.5*1E03, 'ro', label='Nominal fit', markersize=8)
        plt.plot(uncert_BEC_default, (nominal_width**2 - l_width[0]**2)**.5*1E03, 'ro', label='', markersize=8)
        plt.legend()
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel(r'Uncertainty in $\sqrt{s}$ [MeV]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = 0.1
        plt.text(.9, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_corr.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_corr.pdf')
        plt.clf()

            def doBECscanUncorr(self, min = 0, max = 30, step = 1):
        variations = np.arange(min,max+step/2,step)
        l_mass = []
        l_width = []
        f = copy.deepcopy(self)
        f.addBECnuisances(0,0)
        for var in variations:
            f.setBECpriors(prior_uncorr=var, prior_corr=0)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        plt.plot(variations, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncorrelated BEC uncertainty [MeV]')
        plt.ylabel('Total uncertainty on fitted parameter [MeV]')
        offset = .3
        plt.text(.9, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_uncorr_total.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_uncorr_total.pdf')
        plt.clf()

        l_mass = (l_mass**2 - l_mass[0]**2)**.5
        l_width = (l_width**2 - l_width[0]**2)**.5

        plt.plot(variations, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncorrelated BEC uncertainty [MeV]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = 0
        plt.text(.92, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.92, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.92, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_uncorr.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BEC_uncorr.pdf')
        plt.clf()

    def doBESscanCorr(self, min = 0, max = 0.03, step = 0.001):
        variations = np.arange(min,max+step/2,step)
        l_mass = []
        l_width = []
        f = copy.deepcopy(self)
        f.addBESnuisances(uncert_corr=0, uncert_uncorr=0)
        for var in variations:
            if var < 1E-6:
                var = 1E-10
            f.setBESpriors(uncert_corr=var,uncert_uncorr=0)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        plt.plot(variations*100, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations*100, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('BES uncertainty [%]')
        plt.ylabel('Total uncertainty on fitted parameter [MeV]')
        offset = .1
        plt.text(.9, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncert_total.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncert_total.pdf')
        plt.clf()

        l_mass = (l_mass**2 - np.min(l_mass)**2)**.5
        l_width = (l_width**2 - l_width[0]**2)**.5

        plt.plot(variations*100, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations*100, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('BES uncertainty [%]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = .1
        plt.text(.92, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.92, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.92, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncert.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncert.pdf')
        plt.clf()

    def doBESscanUncorr(self, min = 0, max = 0.03, step = 0.001):
        variations = np.arange(min,max+step/2,step)
        l_mass = []
        l_width = []
        f = copy.deepcopy(self)
        f.addBESnuisances(uncert_corr=0, uncert_uncorr=0)
        for var in variations:
            if var < 1E-6:
                var = 1E-10
            f.setBESpriors(uncert_corr=0,uncert_uncorr=var)
            f.fitParameters(initMinuit=True)
            fit_results = f.getFitResults(printout=False)
            l_mass.append(fit_results[self.param_names.index('mass')].s)
            l_width.append(fit_results[self.param_names.index('width')].s)

        l_mass = np.array(l_mass)
        l_width = np.array(l_width)

        plt.plot(variations*100, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations*100, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncorrelated BES uncertainty [%]')
        plt.ylabel('Total uncertainty on fitted parameter [MeV]')
        offset = .1
        plt.text(.9, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.9, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncorr_total.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncorr_total.pdf')
        plt.clf()

        l_mass = (l_mass**2 - np.min(l_mass)**2)**.5
        l_width = (l_width**2 - l_width[0]**2)**.5

        plt.plot(variations*100, l_mass*1E03, 'b-', label='Uncertainty in $m_t$', linewidth=2)
        plt.plot(variations*100, l_width*1E03, 'g--', label='Uncertainty in $\Gamma_t$', linewidth=2)
        plt.legend(loc='upper left')
        plt.title(r'$\mathit{{Projection}}$ ({:.0f} fb$^{{-1}}$)'.format(self.scenario_dict['total_lumi']/1E03), loc='right', fontsize=20)
        plt.xlabel('Uncorrelated BES uncertainty [%]')
        plt.ylabel('Impact on fitted parameter [MeV]')
        offset = -0.03
        plt.text(.95, 0.17 + offset, 'QQbar_Threshold N3LO+ISR', fontsize=23, transform=plt.gca().transAxes, ha='right')
        plt.text(.95, 0.13 + offset, '[JHEP 02 (2018) 125]', fontsize=18, transform=plt.gca().transAxes, ha='right')
        plt.text(.95, 0.08 + offset, '+ FCC-ee BES', fontsize=21, transform=plt.gca().transAxes, ha='right')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncorr.png')
        plt.savefig(plot_dir + '/uncert_mass_width_vs_BES_uncorr.pdf')
        plt.clf()

'''