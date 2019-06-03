import numpy as np
import likelihood_parameter

# lambda_F ~ 80 kpc ~ 0.08 Mpc ~ 0.055 Mpc/h ~ 5.5 km/s (Onorbe et al. 2016)
# k_F = 1 / lambda_F ~ 12.5 1/Mpc ~ 18.2 h/Mpc ~ 0.182 s/km 

class PressureModel(object):
    """Use a handful of parameters to model the pressure smoothing length,
        in velocity units (km/s), as a function of redshift. 
        For now, we use a polynomial to describe log(k_F) around z_F."""

    def __init__(self,z_kF=3.5,ln_kF_coeff=None):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""
        self.z_kF=z_kF
        if not ln_kF_coeff:
            ln_kF_coeff=[0,np.log(0.182)]
        self.ln_kF_coeff=ln_kF_coeff

    def get_kF_kms(self,z):
        """Filtering length at the input redshift (in s/km)"""
        xz=np.log((1+z)/(1+self.z_kF))
        ln_kF_poly=np.poly1d(self.ln_kF_coeff)
        ln_kF=ln_kF_poly(xz)
        return np.exp(ln_kF)

    def get_Nparam(self):
        """Number of parameters in the model"""
        return len(self.ln_kF_coeff)

    def get_parameters(self):
        """Tell likelihood about parameters in the pressure model"""

        Npar=self.get_Nparam()
        params=[]
        if Npar > 0:
            name='ln_kF_0'
            xmin=np.log(0.05)
            xmax=np.log(0.5)
            # note non-trivial order in coefficients
            value=self.ln_kF_coeff[Npar-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        if Npar > 1:
            name='ln_kF_1'
            xmin=-2.0
            xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_kF_coeff[Npar-2]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            params.append(par)
        return params


    def update_parameters(self,parameters):
        """Look for pressure parameters in list of parameters"""

        Npar=self.get_Nparam()
        # report how many parameters were updated
        counts=0

        for par in parameters:
            if par.name=='ln_kF_0':
                # note non-trivial order in coefficients
                self.ln_kF_coeff[Npar-1] = par.value
                counts+=1
            if par.name=='ln_kF_1':
                # note non-trivial order in coefficients
                self.ln_kF_coeff[Npar-2] = par.value
                counts+=1

        return counts


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        kF = PressureModel(z_kF=self.z_kF, ln_kF_coeff=self.ln_kF_coeff)
        kF.update_parameters(parameters)
        return kF
