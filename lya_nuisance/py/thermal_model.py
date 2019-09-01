import numpy as np
import copy
import likelihood_parameter

class ThermalModel(object):
    """Use a handful of parameters to model the temperature-density relation
        as a function of redshift.
        For now, we use two polynomials to describe log(T_0) and log(gamma)."""

    def __init__(self,z_T=3.6,ln_T0_coeff=None,ln_gamma_coeff=None):
        """Construct model for T0 and gamma evolution.
        For T0, we use a broken power law at a central redshift.
        For gamma, we use a power law with running. """
        self.z_T=z_T
        if not ln_T0_coeff:
            ln_T0_coeff=[0.44,  9.37, -1.33]
        if not ln_gamma_coeff:
            ln_gamma_coeff=[-0.2,np.log(1.4)]
        self.ln_T0_coeff=ln_T0_coeff
        self.ln_gamma_coeff=ln_gamma_coeff
        # store list of likelihood parameters (might be fixed or free)
        self.set_T0_parameters()
        self.set_gamma_parameters()


    def get_T0(self,z):
        ''' Return T0(z) for a given model '''
        lnz=np.log(z/self.z_T)
        if z<self.z_T:
            log_poly=np.poly1d([self.ln_T0_coeff[0],self.ln_T0_coeff[1]])
            ln_T0=log_poly(lnz)
        else:
            log_poly=np.poly1d([self.ln_T0_coeff[2],self.ln_T0_coeff[1]])
            ln_T0=log_poly(lnz)
        return np.exp(ln_T0)

    def get_sigT_kms(self,z):
        """Thermal broadening at the input redshift, in km/s"""
        T0=self.get_T0(z)
        return thermal_broadening_kms(T0)

    def get_gamma(self,z):
        """gamma at the input redshift"""
        xz=np.log((1+z)/(1+self.z_T))
        ln_gamma_poly=np.poly1d(self.ln_gamma_coeff)
        ln_gamma=ln_gamma_poly(xz)
        return np.exp(ln_gamma)

    def get_T0_Nparam(self):
        """Number of parameters in the model of T_0"""
        assert len(self.ln_T0_coeff)==len(self.T0_params),"size mismatch"
        return len(self.ln_T0_coeff)

    def get_gamma_Nparam(self):
        """Number of parameters in the model of gamma"""
        assert len(self.ln_gamma_coeff)==len(self.gamma_params),"size mismatch"
        return len(self.ln_gamma_coeff)

    def set_T0_parameters(self):
        """Setup T0 likelihood parameters for thermal model"""

        self.T0_params=[]
        for i in range(3):
            name='T0_'+str(i+1)
            if i==0:
                xmin=0
                xmax=1
            elif i==1:
                xmin=7
                xmax=11
            elif i==2:
                xmin=-1.5
                xmax=-0.5      # note non-trivial order in coefficients
            value=self.ln_T0_coeff[i]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.T0_params.append(par)
        return


    def set_gamma_parameters(self):
        """Setup gamma likelihood parameters for thermal model"""

        self.gamma_params=[]
        Npar=len(self.ln_gamma_coeff)
        for i in range(Npar):
            name='ln_gamma_'+str(i)
            if i==0:
                xmin=np.log(1.1)
                xmax=np.log(2.0)
            else:
                xmin=-2.0
                xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_gamma_coeff[Npar-i-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.gamma_params.append(par)

        return


    def get_T0_parameters(self):
        """Return T0 likelihood parameters from the thermal model"""
        return self.T0_params


    def get_gamma_parameters(self):
        """Return gamma likelihood parameters from the thermal model"""
        return self.gamma_params


    def update_parameters(self,like_params):
        """Look for thermal parameters in list of parameters"""

        Npar_T0=self.get_T0_Nparam()
        Npar_gamma=self.get_gamma_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if 'T0' in like_par.name:
                # make sure you find the parameter
                found=False
                # loop over T0 parameters in thermal model
                for ip in range(len(self.T0_params)):
                    if self.T0_params[ip].is_same_parameter(like_par):
                        assert found==False,'can not update parameter twice'
                        self.ln_T0_coeff[Npar_T0-ip-1]=like_par.value
                        found=True
                assert found==True,'could not update parameter '+like_par.name
            elif 'ln_gamma' in like_par.name:
                # make sure you find the parameter
                found=False
                # loop over gamma parameters in thermal model
                for ip in range(len(self.gamma_params)):
                    if self.gamma_params[ip].is_same_parameter(like_par):
                        assert found==False,'can not update parameter twice'
                        self.ln_gamma_coeff[Npar_gamma-ip-1]=like_par.value
                        found=True
                assert found==True,'could not update parameter '+like_par.name

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        T = ThermalModel(z_T=self.z_T,
                            ln_T0_coeff=copy.deepcopy(self.ln_T0_coeff),
                            ln_gamma_coeff=copy.deepcopy(self.ln_gamma_coeff))
        T.update_parameters(parameters)
        return T


def thermal_broadening_kms(T_0):
    """Thermal broadening RMS in velocity units, given T_0"""

    sigma_T_kms=9.1 * np.sqrt(T_0/1.e4)
    return sigma_T_kms

