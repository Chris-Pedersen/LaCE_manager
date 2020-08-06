import numpy as np
import matplotlib.pyplot as plt
import likelihood_parameter

# likelihood functions that we need to override
# like.free_params
# like.get_free_parameter_list()
# like.minus_log_prob()

class ToyLikelihood(object):
    """Toy likelihood class to test statistics"""

    def __init__(self,free_params,prior_rms=0.1):

        print('input free_parameter')
        for par in free_params:
            print(par.info_str())
        self.free_params=free_params
        # our whole likelihood is a Gaussian prior with input rms
        self.prior_rms=0.1

        return


    def get_free_parameter_list(self):

        names=[]
        for par in self.free_params:
            names.append(par.name)
        return names


    def minus_log_prob(self,values):
        return -1.0*self.log_prob(values)

    def log_prob(self,values):
        # we ignore prior, since our toy likelihood is already a prior...
        log_like=self.get_log_like(values)
        return log_like 

    def get_log_like(self,values):
        # compute chi2 (ignore covariance determinant)
        chi2=self.get_chi2(values)
        return -0.5*chi2

    def get_chi2(self,values):

        Np=len(values)
        assert Np==len(self.free_params),'length mismatch'
        # truth values are 0.5 (center of cube)
        return np.sum(((np.array(values)-0.5)/self.prior_rms)**2)

