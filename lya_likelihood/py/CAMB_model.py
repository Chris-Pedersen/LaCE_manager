import numpy as np
import copy
import camb
import camb_cosmo
import likelihood_parameter
import fit_linP

class CAMBModel(object):
    """ Interface between CAMB object and FullTheory """

    def __init__(self,zs,cosmo=None):
        """Setup from CAMB object and list of redshifts."""

        # list of redshifts at which we evaluate linear power
        self.zs=zs
        
        # setup CAMB cosmology object
        if cosmo is None:
            self.cosmo=camb_cosmo.get_cosmology()
        else:
            self.cosmo=cosmo

        # setup CAMB matter power spectrum calculation
        self.cosmo.set_matter_power(redshifts=self.zs,nonlinear=False,
                kmax=2.0*camb_cosmo.camb_kmax_Mpc,silent=True)

        # will cache CAMB results when computed
        self.cached_camb_results=None
        # will cache wavenumbers and linear power (at zs) when computed
        self.cached_linP_Mpc=None


    def get_likelihood_parameters(self):
        """ Return a list of likelihood parameters """

        params=[]
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='ombh2',min_value=0.019,max_value=0.025,
                        value=self.cosmo.ombh2))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='omch2',min_value=0.10,max_value=0.15,
                        value=self.cosmo.omch2))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='As',min_value=2.0e-09,max_value=2.2e-09,
                        value=self.cosmo.InitPower.As))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='ns',min_value=0.95,max_value=0.98,
                        value=self.cosmo.InitPower.ns))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='H0',min_value=63,max_value=77,
                        value=self.cosmo.H0))
        params.append(likelihood_parameter.LikelihoodParameter(
                        name='mnu',min_value=0.0,max_value=1.0,
                        value=camb_cosmo.get_mnu(self.cosmo)))

        return params


    def get_results(self):
        """ Check if we have called CAMB get_results yet, to save time.
            It returns a CAMB.results object."""

        if self.cached_camb_results is None:
            self.cached_camb_results = camb.get_results(self.cosmo)

        return self.cached_camb_results


    def get_linP_Mpc(self):
        """ Check if we have already computed linP_Mpc, to save time.
            It returns (k_Mpc, zs, linP_Mpc)."""

        if self.cached_linP_Mpc is None:
            results = self.get_results()
            self.cached_linP_Mpc = camb_cosmo.get_linP_Mpc(pars=self.cosmo,
                    zs=self.zs,camb_results=results)

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
        results=self.get_results()
        
        M_of_zs=[]
        for z in self.zs:
            H_z=results.hubble_parameter(z)
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
