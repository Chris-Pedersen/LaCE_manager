import numpy as np
import likelihood_parameter


def mean_flux_Kamble2019(z):
    """Mean transmitted flux fraction from eBOSS data (Kamble et al. 2019)"""
    # unpublished work by Kamble et al., soon to appear on the arXiv
    tau = 0.0055*(1+z)**3.18
    return np.exp(-tau)


class MeanFluxModel(object):
    """Use a handful of parameters to model the mean transmitted flux fraction
        (or mean flux) as a function of redshift. 
         For now, we use a polynomial to describe log(tau_eff) around z_tau."""

    def __init__(self,z_tau=3.0,ln_tau_coeff=None):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""
        self.z_tau=z_tau
        if not ln_tau_coeff:
            mf_z=0.6365
            tau_0=-np.log(mf_z)
            ln_tau_coeff=[3.18,np.log(tau_0)]
        self.ln_tau_coeff=ln_tau_coeff


    def get_Nparam(self):
        """Number of parameters in the model"""
        return len(self.ln_tau_coeff)


    def get_tau_eff(self,z):
        """Effective optical depth at the input redshift"""
        xz=np.log((1+z)/(1+self.z_tau))
        ln_tau_poly=np.poly1d(self.ln_tau_coeff)
        ln_tau=ln_tau_poly(xz)
        return np.exp(ln_tau)


    def get_mean_flux(self,z):
        """Mean transmitted flux fraction at the input redshift"""
        tau=self.get_tau_eff(z)
        return np.exp(-tau)


    def get_parameters(self):
        """Tell likelihood about the parameters in the mean flux model"""

        Npar=self.get_Nparam()
        assert Npar==2, 'update get_parameters in mean_flux_model'
        params=[]
        if Npar > 0:
            name='ln_tau_0'
            xmin=-1.5
            xmax=-0.4
            # note non-trivial order in coefficients
            value=self.ln_tau_coeff[1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        if Npar > 1:
            name='ln_tau_1'
            xmin=3.0
            xmax=5.0
            # note non-trivial order in coefficients
            value=self.ln_tau_coeff[0]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        return params


    def update_parameters(self,parameters):
        """Look for mean flux parameters in list of parameters"""

        Npar=self.get_Nparam()
        assert Npar==2, 'update update_parameters in mean_flux_model'

        # report how many parameters were updated
        counts=0
        for par in parameters:
            if par.name=='ln_tau_0':
                # note non-trivial order in coefficients
                self.ln_tau_coeff[1] = par.value
                counts+=1
            if par.name=='ln_tau_1':
                # note non-trivial order in coefficients
                self.ln_tau_coeff[0] = par.value
                counts+=1

        return counts

