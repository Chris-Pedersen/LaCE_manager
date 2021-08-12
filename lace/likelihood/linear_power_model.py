import numpy as np
import os
from lace.likelihood import likelihood_parameter
from lace.cosmo import fit_linP

class LinearPowerModel(object):
    """Store parameters describing the linear power in velocity units.
        It can work in two modes:
            - given CAMB object, parameterize cosmology and store parameters
            - construct with set of parameters, and store them."""

    def __init__(self,params=None,cosmo=None,camb_results=None,
                z_star=3.0,kp_kms=0.009,use_camb_fz=True,
                fit_kmin_kp=0.5,fit_kmax_kp=2.0):
        """Setup model, specifying redshift and pivot point"""

        # store pivot point
        self.z_star=z_star
        self.kp_kms=kp_kms

        # store (or compute) parameters and / or cosmology
        if params:
            assert cosmo is None, 'can not pass both cosmo and params'
            self._setup_from_parameters(params)
        else:
            # parameterize cosmology and store parameters
            self._setup_from_cosmology(cosmo,camb_results,use_camb_fz,
                    fit_kmin_kp=fit_kmin_kp,fit_kmax_kp=fit_kmax_kp)


    def _setup_from_parameters(self,params):
        """Setup object from dictionary with parameters.
            NOTE: these are not likelihood_parameters, should clarify. """

        # SHOULD WE CHECK HERE THAT INPUT PARAMETERS HAVE SAME KP / Z_STAR ?

        # copy input dictionary
        self.linP_params=params.copy()

        # will add polynomial describing the log power, around kp_kms
        linP_kms_2=0.5*params['alpha_star']
        linP_kms_1=params['n_star']
        A_star=(2*np.pi**2)*params['Delta2_star']/self.kp_kms**3
        linP_kms_0=np.log(A_star)
        linP_kms = np.poly1d([linP_kms_2,linP_kms_1,linP_kms_0])
        # why are we storing this poly1d object? When do we actually use it?
        self.linP_params['linP_kms']=linP_kms


    def _setup_from_cosmology(self,cosmo,camb_results,use_camb_fz,
                fit_kmin_kp,fit_kmax_kp):
        """Compute and store parameters describing the linear power."""

        self.linP_params=fit_linP.parameterize_cosmology_kms(cosmo,camb_results,
                self.z_star,self.kp_kms,use_camb_fz=use_camb_fz,
                fit_kmin_kp=fit_kmin_kp,fit_kmax_kp=fit_kmax_kp)


    def get_params(self):
        """Return dictionary with parameters (not likelihood parameters). """

        params={'f_star':self.get_f_star(), 'g_star':self.get_g_star(), 
                'Delta2_star':self.get_Delta2_star(), 
                'n_star':self.get_n_star(), 'alpha_star':self.get_alpha_star()}

        return params


    def get_likelihood_parameters(self):
        """Tell likelihood about the linear power likelihood parameters"""

        params=[]
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='g_star',min_value=0.95,max_value=0.99,
                        value=self.linP_params['g_star']))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='f_star',min_value=0.95,max_value=0.99,
                        value=self.linP_params['f_star']))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='Delta2_star',min_value=0.25,max_value=0.4,
                        value=self.linP_params['Delta2_star']))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='n_star',min_value=-2.35,max_value=-2.25,
                        value=self.linP_params['n_star']))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='alpha_star',min_value=-0.27,max_value=-0.16,
                        value=self.linP_params['alpha_star']))

        return params


    def update_parameters(self,like_params):
        """Update linear power parameters, if present in input list"""

        # get current dictionary with parameters, update and setup again
        params=self.get_params()

        for par in like_params:
            if par.name in params:
                params[par.name]=par.value

        self._setup_from_parameters(params)
        return


    def get_f_star(self):
        return self.linP_params['f_star']

    def get_g_star(self):
        return self.linP_params['g_star']

    def get_Delta2_star(self):
        return self.linP_params['Delta2_star']

    def get_n_star(self):
        return self.linP_params['n_star']

    def get_alpha_star(self):
        return self.linP_params['alpha_star']

