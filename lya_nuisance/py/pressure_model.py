import numpy as np
import copy
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
            ln_kF_coeff=[0.3,np.log(0.17)]
        self.ln_kF_coeff=ln_kF_coeff
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()


    def get_kF_kms(self,z):
        """Filtering length at the input redshift (in s/km)"""
        xz=np.log((1+z)/(1+self.z_kF))
        ln_kF_poly=np.poly1d(self.ln_kF_coeff)
        ln_kF=ln_kF_poly(xz)
        return np.exp(ln_kF)


    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_kF_coeff)==len(self.params),"size mismatch"
        return len(self.ln_kF_coeff)


    def set_parameters(self):
        """Setup likelihood parameters in the pressure model"""

        self.params=[]
        Npar=len(self.ln_kF_coeff)
        for i in range(Npar):
            name='ln_kF_'+str(i)
            if i==0:
                xmin=np.log(0.05)
                xmax=np.log(0.5)
            elif i==1:
                xmin=-1.0
                xmax=3.0
            else:
                xmin=-2.0
                xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_kF_coeff[Npar-i-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.params.append(par)

        return


    def get_parameters(self):
        """Return likelihood parameters for the pressure model"""

        return self.params


    def update_parameters(self,like_params):
        """Look for pressure parameters in list of parameters"""

        Npar=self.get_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if 'ln_kF' not in like_par.name:
                continue
            # make sure you find the parameter
            found=False
            # loop over parameters in pressure model
            for ip in range(len(self.params)):
                if self.params[ip].is_same_parameter(like_par):
                    assert found==False,'can not update parameter twice'
                    self.ln_kF_coeff[Npar-ip-1]=like_par.value
                    found=True
            assert found==True,'could not update parameter '+like_par.name

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        kF = PressureModel(z_kF=self.z_kF,
                            ln_kF_coeff=copy.deepcopy(self.ln_kF_coeff))
        kF.update_parameters(parameters)
        return kF
