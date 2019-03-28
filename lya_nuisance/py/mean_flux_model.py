import numpy as np


def mean_flux_Kamble2019(z):
    """Mean transmitted flux fraction from eBOSS data (Kamble et al. 2019)"""
    # unpublished work by Kamble et al., soon to appear on the arXiv
    tau = 0.0055*(1+z)**3.18
    return np.exp(-tau)


class MeanFluxModel(object):
    """Use a handful of parameters to model the mean transmitted flux fraction
        (or mean flux) as a function of redshift. 
         For now, we use a polynomial to describe log(tau_eff) around z_tau."""

    def __init__(self,z_tau=3.0,ln_tau_coeff=[3.18,-0.7946]):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""
        self.z_tau=z_tau
        self.ln_tau_poly=np.poly1d(ln_tau_coeff)

    def get_Nparam(self):
        """Number of parameters in the model"""
        return 1+self.ln_tau_poly.order

    def get_tau_eff(self,z):
        """Effective optical depth at the input redshift"""
        xz=np.log((1+z)/(1+self.z_tau))
        ln_tau=self.ln_tau_poly(xz)
        return np.exp(ln_tau)

    def get_mean_flux(self,z):
        """Mean transmitted flux fraction at the input redshift"""
        tau=self.get_tau_eff(z)
        return np.exp(-tau)


