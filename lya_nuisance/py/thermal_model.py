import numpy as np
import likelihood_parameter


class ThermalModel(object):
    """Use a handful of parameters to model the temperature-density relation
        as a function of redshift.
        For now, we use two polynomials to describe log(T_0) and log(gamma)."""

    def __init__(self,z_T=3.5,ln_T0_coeff=None,ln_gamma_coeff=None):
        """Construct model with central redshift and (x2,x1,x0) polynomials."""
        self.z_T=z_T
        if not ln_T0_coeff:
            ln_T0_coeff=[0.0,np.log(1.e4)]
        if not ln_gamma_coeff:
            ln_gamma_coeff=[0.0,np.log(1.4)]
        self.ln_T0_coeff=ln_T0_coeff
        self.ln_gamma_coeff=ln_gamma_coeff


    def get_T0(self,z):
        """T_0 at the input redshift"""
        xz=np.log((1+z)/(1+self.z_T))
        ln_T0_poly=np.poly1d(self.ln_T0_coeff)
        ln_T0=ln_T0_poly(xz)
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
        return len(self.ln_T0_coeff)

    def get_gamma_Nparam(self):
        """Number of parameters in the model of gamma"""
        return len(self.ln_gamma_coeff)

    def get_T0_parameters(self):
        """Tell likelihood about T_0 parameters in the thermal model"""

        Npar=self.get_T0_Nparam()
        params=[]
        if Npar > 0:
            name='ln_T0_0'
            xmin=np.log(5e3)
            xmax=np.log(5e4)
            # note non-trivial order in coefficients
            value=self.ln_T0_coeff[Npar-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        if Npar > 1:
            name='ln_T0_1'
            xmin=-2.0
            xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_T0_coeff[Npar-2]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        return params


    def get_gamma_parameters(self):
        """Tell likelihood about gamma parameters in the thermal model"""

        Npar=self.get_gamma_Nparam()
        params=[]
        if Npar > 0:
            name='ln_gamma_0'
            xmin=np.log(1.1)
            xmax=np.log(2.0)
            # note non-trivial order in coefficients
            value=self.ln_gamma_coeff[Npar-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        if Npar > 1:
            name='ln_gamma_1'
            xmin=-2.0
            xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_gamma_coeff[Npar-2]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        return params


    def update_parameters(self,parameters):
        """Look for mean flux parameters in list of parameters"""

        Npar_T0=self.get_T0_Nparam()
        Npar_gamma=self.get_gamma_Nparam()

        # report how many parameters were updated
        counts=0

        for par in parameters:
            if par.name=='ln_T0_0':
                # note non-trivial order in coefficients
                self.ln_T0_coeff[Npar_T0-1] = par.value
                counts+=1
            if par.name=='ln_T0_1':
                # note non-trivial order in coefficients
                self.ln_T0_coeff[Npar_T0-2] = par.value
                counts+=1
            if par.name=='ln_gamma_0':
                # note non-trivial order in coefficients
                self.ln_gamma_coeff[Npar_gamma-1] = par.value
                counts+=1
            if par.name=='ln_gamma_1':
                # note non-trivial order in coefficients
                self.ln_gamma_coeff[Npar_gamma-2] = par.value
                counts+=1

        return counts


def thermal_broadening_kms(T_0):
    """Thermal broadening RMS in velocity units, given T_0"""

    sigma_T_kms=9.1 * np.sqrt(T_0/1.e4)
    return sigma_T_kms

