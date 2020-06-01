import camb_cosmo
import likelihood_parameter
import copy
import fit_linP
import camb
import numpy as np

class CAMBModel(object):
    """ Container for a set of likelihood parameters describing a CAMB
    cosmology """

    def __init__(self,zs,cosmo=None,like_params=None,kp_Mpc=0.69,verbose=False):

        self.zs=zs
        self.kp_Mpc=kp_Mpc
        self.camb_params=["ombh2","omch2","As","ns"] ## List of parameters we'll use
                                                     ## to define a cosmology

        #assert (cosmo is None and like_params is None), "Cannot provide cosmology and like_params"

        
        if cosmo==None and like_params==None:
            self.cosmo=camb_cosmo.get_cosmology()
        elif cosmo is not None:
            self.cosmo=cosmo
        else:
            self._setup_from_like_params(like_params)
                            

    def _setup_from_like_params(self,like_params):
        """ Set up a CAMB object from an arbitrary list of likelihood parameters """

        camb_param_dict={}
        for par in like_params:
            if par.name in self.camb_params:
                camb_param_dict[par.name]=par.value

        self.cosmo=camb_cosmo.get_cosmology(params=camb_param_dict)

        return
        
    
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
        ### Imagine this is all we want to vary for now

        return params


    def get_linP_Mpc_params(self):
        """ For a given set of likelihood parameters, get emulator calls at each 
        z """

        ## Get the P(k) at each z

        k_Mpc,z,pk_Mpc=camb_cosmo.get_linP_Mpc(self.cosmo,zs=self.zs)

        # specify wavenumber range to fit
        kmin_Mpc = 0.5*self.kp_Mpc
        kmax_Mpc = 2.0*self.kp_Mpc

        linP_params=[]
        ## Fit the emulator call params
        for pk_z in pk_Mpc:
            linP_Mpc = fit_linP.fit_polynomial(kmin_Mpc/self.kp_Mpc,
                        kmax_Mpc/self.kp_Mpc,k_Mpc/self.kp_Mpc,
                        pk_z,deg=2)
            # translate the polynomial to our parameters
            ln_A_p = linP_Mpc[0]
            Delta2_p = np.exp(ln_A_p)*self.kp_Mpc**3/(2*np.pi**2)
            n_p = linP_Mpc[1]
            # note that the curvature is alpha/2
            alpha_p = 2.0*linP_Mpc[2]
            linP_z={'Delta2_p':Delta2_p,
                    'n_p':n_p, 'alpha_p':alpha_p}
            linP_params.append(linP_z)

        return linP_params

    def get_M_of_zs(self):
        """ Return M(z)=H(z)/(1+z) for each z """
        results=camb.get_results(self.cosmo)
        
        M_of_zs=[]
        for z in self.zs:
            H_z=results.hubble_parameter(z)
            M_of_zs.append(H_z/(1+z))
    
        return M_of_zs
    

    def get_new_model(self,like_params):
        """ For an arbitrary list of like_params, return a new model """

        new_model=CAMBModel(zs=copy.deepcopy(self.zs),
                        like_params=like_params,kp_Mpc=copy.deepcopy(self.kp_Mpc))

        return new_model