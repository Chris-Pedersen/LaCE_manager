import numpy as np

# lambda_F ~ 80 kpc ~ 0.08 Mpc ~ 0.055 Mpc/h ~ 5.5 km/s (Onorbe et al. 2016)
# k_F = 2 pi / lambda_F ~ 80 1/Mpc  110 h/Mpc ~ 1.1 s/km 

class PressureModel(object):
    """Use a handful of parameters to model the pressure smoothing length,
        in velocity units (km/s), as a function of redshift. 
        For now, we use a polynomial to describe log(k_F) around z_F."""

    def __init__(self,z_kF=3.5,ln_kF_coeff=[0,np.log(1.1)]):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""
        self.z_kF=z_kF
        self.ln_kF_poly=np.poly1d(ln_kF_coeff)

    def get_kF_kms(self,z):
        """Filtering length at the input redshift (in s/km)"""
        xz=np.log((1+z)/(1+self.z_kF))
        ln_kF=self.ln_kF_poly(xz)
        return np.exp(ln_kF)


