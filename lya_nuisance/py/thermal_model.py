import numpy as np


class ThermalModel(object):
    """Use a handful of parameters to model the temperature-density relation
        as a function of redshift.
        For now, we use two polynomials to describe log(T_0) and log(gamma)."""

    def __init__(self,z_T=3.5,ln_T0_coeff=[0.0,np.log(1e4)],
                    ln_gamma_coeff=[-0.2,np.log(1.45)]):
        """Construct model with central redshift and (x2,x1,x0) polynomials."""
        self.z_T=z_T
        self.ln_T0_poly=np.poly1d(ln_T0_coeff)
        self.ln_gamma_poly=np.poly1d(ln_gamma_coeff)

    def get_T0(self,z):
        """T_0 at the input redshift"""
        xz=np.log((1+z)/(1+self.z_T))
        ln_T0=self.ln_T0_poly(xz)
        return np.exp(ln_T0)

    def get_sigT_kms(self,z):
        """Thermal broadening at the input redshift, in km/s"""
        T0=self.get_T0(z)
        return thermal_broadening_kms(T_0)

    def get_gamma(self,z):
        """gamma at the input redshift"""
        xz=np.log((1+z)/(1+self.z_T))
        ln_gamma=self.ln_gamma_poly(xz)
        return np.exp(ln_gamma)


def thermal_broadening_kms(T_0):
    """Thermal broadening RMS in velocity units, given T_0"""

    sigma_T_kms=9.1 * np.sqrt(T_0/1.e4)
    return sigma_T_kms
