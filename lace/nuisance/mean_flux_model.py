import numpy as np
import copy
from scipy.interpolate import interp1d
from lace.likelihood import likelihood_parameter
import os


def mean_flux_Kamble2019(z):
    """Mean transmitted flux fraction from eBOSS data (Kamble et al. 2019)"""
    # unpublished work by Kamble et al., soon to appear on the archive
    tau = 0.0055*(1+z)**3.18
    return np.exp(-tau)


class MeanFluxModel(object):
    """Use a handful of parameters to model the mean transmitted flux fraction
        (or mean flux) as a function of redshift. 
         For now, we use a polynomial to describe log(tau_eff) around z_tau.
         """

    def __init__(self,z_tau=3.0,ln_tau_coeff=None,
                    basedir="/lace/emulator/sim_suites/Australia20/"):
        """Construct model as a rescaling around a fiducial mean flux evolution"""

        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']

        ## Load fiducial model
        fiducial=np.loadtxt(repo+basedir+"fiducial_igm_evolution.txt")
        self.fid_z=fiducial[0]
        self.fid_tau_eff=fiducial[1] ## tau_eff(z)
        self.fid_tau_interp=interp1d(self.fid_z,self.fid_tau_eff,kind="cubic")

        self.z_tau=z_tau
        if not ln_tau_coeff:
            ln_tau_coeff=[0.0,0.0]
        self.ln_tau_coeff=ln_tau_coeff
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()


    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_tau_coeff)==len(self.params),"size mismatch"
        return len(self.ln_tau_coeff)


    def power_law_scaling(self,z):
        """ Power law rescaling around z_tau """
        xz=np.log((1+z)/(1+self.z_tau))
        ln_poly=np.poly1d(self.ln_tau_coeff)
        ln_out=ln_poly(xz)
        return np.exp(ln_out)


    def get_tau_eff(self,z):
        """Effective optical depth at the input redshift"""
        tau_eff=self.power_law_scaling(z)*self.fid_tau_interp(z)
        return tau_eff


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
                xmin=-0.2
                xmax=0.2
            else:
                xmin=-0.2
                xmax=0.2
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
                if self.params[ip].name == like_par.name:
                    assert found==False,'can not update parameter twice'
                    self.ln_tau_coeff[Npar-ip-1]=like_par.value
                    found=True
            assert found==True,'could not update parameter '+like_par.name

        return


    def get_new_model(self,parameters=[]):
        """Return copy of model, updating values from list of parameters"""

        mf = MeanFluxModel(z_tau=self.z_tau,
                            ln_tau_coeff=copy.deepcopy(self.ln_tau_coeff))
        mf.update_parameters(parameters)
        return mf

