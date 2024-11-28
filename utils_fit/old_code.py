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
    '''

    '''
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
    '''
    '''
    def fitBECvarMorph(self,var):
        f = copy.deepcopy(self)
        f.BEC_var_scenario *= (1 + var*np.array(self.morph_scenario['BEC']['xsec']))
        f.fitParameters(initMinuit=False)
        return [res.n for res in f.getFitResults(printout=False)[:2]]
    '''
