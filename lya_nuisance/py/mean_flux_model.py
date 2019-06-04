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
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()


    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_tau_coeff)==len(self.params),"size mismatch"
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


    def set_parameters(self):
        """Setup likelihood parameters in the mean flux model"""

        self.params=[]
        Npar=len(self.ln_tau_coeff)
        for i in range(Npar):
            name='ln_tau_'+str(i)
            if i==0:
                xmin=-1.5
                xmax=-0.4
            elif i==1:
                xmin=3.0
                xmax=5.0
            else:
                xmin=-2.0
                xmax=2.0
            # note non-trivial order in coefficients
            value=self.ln_tau_coeff[Npar-i-1]
            par = likelihood_parameter.LikelihoodParameter(name=name,
                                value=value,min_value=xmin,max_value=xmax)
            self.params.append(par)

        return
 

    def get_parameters(self):
        """Return likelihood parameters for the mean flux model"""
        return self.params


    def update_parameters(self,like_params):
        """Update mean flux values using input list of likelihood parameters"""

        Npar=self.get_Nparam()

        # loop over likelihood parameters
        for like_par in like_params:
            if 'ln_tau' not in like_par.name:
                continue
            # make sure you find the parameter
            found=False
            # loop over parameters in mean flux model
            for ip in range(len(self.params)):
                if self.params[ip].is_same_parameter(like_par):
                    assert found==False,'can not update parameter twice'
                    self.ln_tau_coeff[Npar-ip-1]=like_par.value
                    found=True
            assert found==True,'could not update parameter '+like_par.name

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        mf = MeanFluxModel(z_tau=self.z_tau, ln_tau_coeff=self.ln_tau_coeff)
        mf.update_parameters(parameters)
        return mf

