import numpy as np

class parameters:
    def __init__(self):
        self.order = 3
        self.mass = 171.5
        self.width = 1.33
        self.yukawa = 1.0
        self.alphas = 0.
        self.mass_scale = 170
        self.width_scale = 170
        self.mass_var = 0.03
        self.width_var = 0.05
        self.yukawa_var = 0.1
        self.alphas_var = 0.0003
        self.mass_pseudo = 0.01
        self.width_pseudo = -0.02
        self.yukawa_pseudo = 0.05
        self.alphas_pseudo = -0.0001
        self.params = ['mass','width','yukawa','alphas']
        self.scale_vars = [round(scale,1) for scale in np.arange(50., 351., 10.)]
        self.create_dict()

    def formDict(self, var):
        self.parameters_dict[var] = dict()
        for par in ['mass','width','yukawa','alphas']:
            if var == 'nominal':
                self.parameters_dict[var][par] = getattr(self, par)
            elif var == 'pseudodata':
                self.parameters_dict[var][par] = getattr(self, par) + getattr(self, par+'_pseudo')
            else:
                self.parameters_dict[var][par] = getattr(self, par) + getattr(self, par+'_var') if var == par+'_var' else getattr(self, par)
            self.parameters_dict[var][par] = round(self.parameters_dict[var][par], 2 if par != 'alphas' else 4)


    def create_dict(self):
        self.parameters_dict = dict()
        vars = ['nominal','pseudodata']
        for param in self.params:
            vars.append(param+'_var')
        for var in vars:
            self.formDict(var)

    def getDict(self):
        return self.parameters_dict




