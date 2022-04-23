import numpy as np
import copy
import camb
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace_manager.likelihood import likelihood_parameter

class CAMBModel(object):
    """ Interface between CAMB object and FullTheory """

    def __init__(self,zs,cosmo=None,pivot_scalar=0.05,theta_MC=True):
        """Setup from CAMB object and list of redshifts.
          - theta_MC will determine whether we use 100theta_MC
            as a likelihood parameter, or H0 in the case that
            this flag is False """

        # list of redshifts at which we evaluate linear power
        self.zs=zs
        
        # setup CAMB cosmology object
        if cosmo is None:
            self.cosmo=camb_cosmo.get_cosmology(pivot_scalar=pivot_scalar)
        else:
            self.cosmo=cosmo

        # will cache CAMB results when computed
        self.cached_camb_results=None
        # will cache wavenumbers and linear power (at zs) when computed
        self.cached_linP_Mpc=None
        self.theta_MC=theta_MC


    def get_likelihood_parameters(self):
        """ Return a list of likelihood parameters """

        ## It apperas to me that the min max values here
        ## aren't actually used, which is what allows us to set
        ## custom prior volumes using Likelihood.free_param_limits
        ## So I will leave these for now, but this should be
        ## cleared up eventually.

        ## This is a tad confusing, and means that the only time
        ## this method is used is initialising the likelihood parameters,
        ## and the prior range is instantly overwritten. 

        ## If we want to be able to sample either H0 or theta_MC
        ## we might have to reconsider how this is done

        params=[]
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='ombh2',min_value=0.019,max_value=0.025,
                        value=self.cosmo.ombh2))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='omch2',min_value=0.10,max_value=0.15,
                        value=self.cosmo.omch2))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='As',min_value=1.0e-09,max_value=3.0e-09,
                        value=self.cosmo.InitPower.As))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='ns',min_value=0.90,max_value=1.05,
                        value=self.cosmo.InitPower.ns))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='mnu',min_value=0.0,max_value=1.0,
                        value=camb_cosmo.get_mnu(self.cosmo)))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='nrun',min_value=-0.8,max_value=0.8,
                        value=self.cosmo.InitPower.nrun))
        ## Check if we are using thetaMC or H0
        if self.theta_MC==True:
            if self.cached_camb_results is None:
                camb_results = self.get_camb_results()
            theta_MC=self.cached_camb_results.cosmomc_theta()
            params.append(likelihood_parameter.LikelihoodParameter(
                        name='cosmomc_theta',min_value=0.0140,max_value=0.0142,
                        value=theta_MC))
        else:
            params.append(likelihood_parameter.LikelihoodParameter(
                        name='H0',min_value=63,max_value=77,
                        value=self.cosmo.H0))

        return params


    def get_camb_results(self):
        """ Check if we have called CAMB.get_results yet, to save time.
            It returns a CAMB.results object."""

        if self.cached_camb_results is None:
            self.cached_camb_results = camb_cosmo.get_camb_results(self.cosmo,
                    zs=self.zs,fast_camb=True)

        return self.cached_camb_results


    def get_linP_Mpc(self):
        """ Check if we have already computed linP_Mpc, to save time.
            It returns (k_Mpc, zs, linP_Mpc)."""

        if self.cached_linP_Mpc is None:
            camb_results = self.get_camb_results()
            self.cached_linP_Mpc = camb_cosmo.get_linP_Mpc(pars=self.cosmo,
                    zs=self.zs,camb_results=camb_results)

        return self.cached_linP_Mpc


    def get_linP_Mpc_params(self,kp_Mpc):
        """ Get linear power parameters to call emulator, at each z.
            Amplitude, slope and running around pivot point kp_Mpc."""

        ## Get the P(k) at each z
        k_Mpc,z,pk_Mpc=self.get_linP_Mpc()

        # specify wavenumber range to fit
        kmin_Mpc = 0.5*kp_Mpc
        kmax_Mpc = 2.0*kp_Mpc

        linP_params=[]
        ## Fit the emulator call params
        for pk_z in pk_Mpc:
            linP_Mpc = fit_linP.fit_polynomial(kmin_Mpc/kp_Mpc,
                        kmax_Mpc/kp_Mpc, k_Mpc/kp_Mpc, pk_z,deg=2)
            # translate the polynomial to our parameters
            ln_A_p = linP_Mpc[0]
            Delta2_p = np.exp(ln_A_p)*kp_Mpc**3/(2*np.pi**2)
            n_p = linP_Mpc[1]
            # note that the curvature is alpha/2
            alpha_p = 2.0*linP_Mpc[2]
            linP_z={'Delta2_p':Delta2_p, 'n_p':n_p, 'alpha_p':alpha_p}
            linP_params.append(linP_z)

        return linP_params


    def get_M_of_zs(self):
        """ Return M(z)=H(z)/(1+z) for each z """

        # get CAMB results objects (might be cached already)
        camb_results=self.get_camb_results()
        
        M_of_zs=[]
        for z in self.zs:
            H_z=camb_results.hubble_parameter(z)
            M_of_zs.append(H_z/(1+z))
    
        return M_of_zs
    

    def get_new_model(self,like_params):
        """ For an arbitrary list of like_params, return a new CAMBModel """

        # store a dictionary with parameters set to input values
        camb_param_dict={}

        # loop over list of likelihood parameters own by this object
        for mypar in self.get_likelihood_parameters():
            # loop over input parameters
            for inpar in like_params:
                if inpar.name==mypar.name:
                    camb_param_dict[inpar.name]=inpar.value
                    continue
                
        # set cosmology object (use fiducial for parameters not provided)
        new_cosmo = camb_cosmo.get_cosmology_from_dictionary(camb_param_dict,
                cosmo_fid=self.cosmo)

        return CAMBModel(zs=copy.deepcopy(self.zs),cosmo=new_cosmo)
