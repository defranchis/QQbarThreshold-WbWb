import numpy as np
import pandas as pd
import iminuit, scipy
import os, argparse
import matplotlib.pyplot as plt
from fitUtils import convoluteXsecGauss

indir = 'output_ISR/for_fit'
parameters = ['mass','width','yukawa','as']

def formTag(mass, width, yukawa, alphas):
    return 'mass{:.2f}_width{:.2f}_yukawa{:.1f}_as{}'.format(mass,width,yukawa,alphas)

class fit:
    def __init__(self, input_dir, parameters, beam_energy_res = 0.221, smearXsec = True) -> None:
        self.input_dir = input_dir
        self.params = parameters
        self.beam_energy_res = beam_energy_res
        self.smearXsec = smearXsec
        self.fetchInputs()
        self.checkParameterNames()
        self.getParameterValues()
        self.readCrossSections()
        self.smearCrossSections()
        self.morphCrossSections()

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
        nominal = [self.param_dict['{}_nom'.format(p)] for p in self.params]
        varied = [self.param_dict['{}_nom'.format(p) if p != param else '{}_var'.format(p)] for p in self.params]
        xsec_nom = self.xsec_dict_smeared[formTag(*nominal)]
        xsec_var = self.xsec_dict_smeared[formTag(*varied)]
        df_morph = pd.DataFrame({'ecm': xsec_nom['ecm'], 'xsec': xsec_var['xsec']/xsec_nom['xsec'] -1})
        return df_morph      

    def morphCrossSections(self):
        self.morph_dict = {}
        for param in self.params:
            self.morph_dict[param] = self.morphCrossSection(param)    


def main():
    parser = argparse.ArgumentParser(description='Specify options')
    parser.add_argument('--debug', action='store_true', help='Debug mode') #does not do anything yet
    
    f = fit(indir,parameters)


if __name__ == '__main__':
    main()