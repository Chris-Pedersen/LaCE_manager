import numpy as np
import copy
from scipy.interpolate import interp1d
from lace_manager.likelihood import likelihood_parameter
import os

# lambda_F ~ 80 kpc ~ 0.08 Mpc ~ 0.055 Mpc/h ~ 5.5 km/s (Onorbe et al. 2016)
# k_F = 1 / lambda_F ~ 12.5 1/Mpc ~ 18.2 h/Mpc ~ 0.182 s/km 

class PressureModel(object):
    """ Model the redshift evolution of the pressure smoothing length.
        We use a power law rescaling around a fiducial simulation at the centre
        of the initial Latin hypercube in simulation space."""

    def __init__(self,z_kF=3.0,ln_kF_coeff=None,
                    basedir="/lace/emulator/sim_suites/Australia20/"):
        """Construct model with central redshift and (x2,x1,x0) polynomial."""

        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']

        ## Load fiducial model
        fiducial=np.loadtxt(repo+basedir+"fiducial_igm_evolution.txt")
        self.fid_z=fiducial[0]
        self.fid_kF=fiducial[4] ## kF_kms(z)
        self.fid_kF_interp=interp1d(self.fid_z,self.fid_kF,kind="cubic")


        self.z_kF=z_kF
        if not ln_kF_coeff:
            ln_kF_coeff=[0.0,0.0]
        self.ln_kF_coeff=ln_kF_coeff
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()


    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_kF_coeff)==len(self.params),"size mismatch"
        return len(self.ln_kF_coeff)


    def power_law_scaling(self,z):
        """ Power law rescaling around z_tau """
        xz=np.log((1+z)/(1+self.z_kF))
        ln_poly=np.poly1d(self.ln_kF_coeff)
        ln_out=ln_poly(xz)
        return np.exp(ln_out)


    def get_kF_kms(self,z):
        """kF_kms at the input redshift"""
        kF_kms=self.power_law_scaling(z)*self.fid_kF_interp(z)
        return kF_kms


    def set_parameters(self):
        """Setup likelihood parameters in the pressure model"""

        self.params=[]
        Npar=len(self.ln_kF_coeff)
        for i in range(Npar):
            name='ln_kF_'+str(i)
            if i==0:
                xmin=-0.2
                xmax=0.2
            else:
                xmin=-0.2
                xmax=0.2
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
                if self.params[ip].name == like_par.name:
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
