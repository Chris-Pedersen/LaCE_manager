import numpy as np
import matplotlib.pyplot as plt
import data_PD2013
import lya_theory
import likelihood_parameter
import camb_cosmo
import sim_params_cosmo
import sim_params_space
import fit_linP
import full_theory
import read_genic
import CAMB_model
import os
from scipy.optimize import minimize

class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(self,data=None,theory=None,emulator=None,
                    free_parameters=None,
                    free_param_limits=None,
                    verbose=False,
                    prior_Gauss_rms=0.2,
                    min_kp_kms=None,emu_cov_factor=1,
                    use_sim_cosmo=True):
        """Setup likelihood from theory and data. Options:
            - if prior_Gauss_rms is None it will use uniform priors
            - ignore k-bins with k > min_kp_kms
            - emu_cov_factor adjusts the contribution from emulator covariance
            set between 0 and 1.
            - use_sim_cosmo will extract the cosmological likelihood
              parameters from the fiducial simulation, and use these
              as a fiducial model"""


        self.verbose=verbose
        self.prior_Gauss_rms=prior_Gauss_rms
        self.emu_cov_factor=emu_cov_factor
        self.simpleLike=False

        if data:
            self.data=data
        else:
            if self.verbose: print('use default data')
            self.data=data_PD2013.P1D_PD2013(blind_data=True)

        # (optionally) get rid of low-k data points
        self.data._cull_data(min_kp_kms)
        self.free_parameters=free_parameters ## Just a list of the names
        self.free_param_limits=free_param_limits

        if theory:
            self.theory=theory
        else:
            ## Use the free_param_names to determine whether to use
            ## a LyaTheory or FullTheory object
            compressed=bool(set(self.free_parameters) & set(["Delta2_star",
                                            "n_star",
                                            "alpha_star",
                                            "f_star",
                                            "g_star"]))
            full=bool(set(self.free_parameters) & set(["H0",
                                            "As",
                                            "ns"]))
            assert (compressed and full)==False, "Cannot vary both compressed and full likelihood parameters"

            if use_sim_cosmo: ## Use the simulation cosmology as fiducial?
                repo=os.environ['LYA_EMU_REPO']
                ## Get dictionary from paramfile.genic
                sim_cosmo_dict=read_genic.camb_from_genic(repo+self.data.basedir+"sim_pair_"+str(self.data.sim_number)+"/sim_plus/paramfile.genic")
                ## Create CAMB object from dictionary
                sim_cosmo=camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)
            else:
                sim_cosmo=None
            
            if compressed:
                ## Set up a compressed LyaTheory object
                self.theory=lya_theory.LyaTheory(self.data.z,emulator=emulator,
                                            cosmo_fid=sim_cosmo)
            elif full:
                ## Set up a full theory object
                camb_model_sim=CAMB_model.CAMBModel(zs=self.data.z,cosmo=sim_cosmo)
                self.theory=full_theory.FullTheory(zs=data.z,emulator=emulator,
                                            camb_model_fid=camb_model_sim)
            else:
                print("Cannot only vary IGM!")
                quit()

        # setup parameters

        if not free_parameters:
            free_parameters=['ln_tau_0']
        self.set_free_parameters(free_parameters,self.free_param_limits)

        if self.verbose: print(len(self.free_params),'free parameters')

        return


    def set_free_parameters(self,free_parameter_names,free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params=[]

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        # select parameters using input list of names
        for par in params:
            if par.name in free_parameter_names:
                if free_param_limits is not None:
                    ## Set min and max of each parameter if
                    ## a list is given. otherwise leave as default
                    par.min_value=free_param_limits[self.free_parameters.index(par.name)][0]
                    par.max_value=free_param_limits[self.free_parameters.index(par.name)][1]
                self.free_params.append(par)

        Nfree=len(self.free_params)
        Nin=len(free_parameter_names)

        assert (Nfree==Nin), 'could not setup free parameters'

        if self.verbose:
            print('likelihood setup with {} free parameters'.format(Nfree))

        return


    def parameters_from_sampling_point(self,values):
        """Translate input array of values (in cube) to likelihood parameters"""
        
        if values is None:
            return []

        assert len(values)==len(self.free_params),'size mismatch'
        Npar=len(values)
        like_params=[]
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params


    def get_p1d_kms(self,k_kms=None,values=None,return_covar=False):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms=self.data.k

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]

        return self.theory.get_p1d_kms(k_kms,like_params=like_params,
                                            return_covar=return_covar)


    def get_chi2(self,values=None):
        """Compute chi2 using data and theory, without adding
            emulator covariance"""

        log_like=self.get_log_like(values,ignore_log_det_cov=True)
        if log_like is None:
            return None
        else:
            return -2.0*log_like


    def get_covmats(self,values=None):
        """ Return the data and emulator covmats for a given
        set of likelihood parameters. Will return a list of the
        covmats at each z """

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        data_covar=[]

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            data_covar.append(self.data.get_cov_iz(iz))

        return data_covar, emu_covar


    def get_log_like(self,values=None,ignore_log_det_cov=True):
        """Compute log(likelihood), including determinant of covariance
            unless you are setting ignore_log_det_cov=True."""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        log_like=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            p1d=self.data.get_Pk_iz(iz)
            data_cov=self.data.get_cov_iz(iz)
            # add covariance from emulator
            cov = data_cov + self.emu_cov_factor*emu_covar[iz]

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(cov)
            diff = (p1d-emu_p1d[iz])
            chi2_z = np.dot(np.dot(icov,diff),diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_z = -0.5*chi2_z
            else:
                (_, log_det_cov) = np.linalg.slogdet(cov)
                log_like_z = -0.5*(chi2_z + log_det_cov)
            log_like += log_like_z
            if self.verbose: print('added {} to log_like'.format(log_like_z))

        return log_like


    def log_prob(self,values):
        """Return log likelihood plus log priors"""

        # compute log_prior
        log_prior=self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        log_like=self.get_log_like(values,ignore_log_det_cov=False)

        if log_like is None:
            if self.verbose: print('was not able to emulate at least on P1D')
            return -np.inf

        return log_like + log_prior


    def get_log_prior(self,values):
        """Compute logarithm of prior"""

        assert len(values)==len(self.free_params),'size mismatch'

        # Always force parameter to be within range (for now)
        if max(values) > 1.0:
            return -np.inf
        if min(values) < 0.0:
            return -np.inf

        # Add Gaussian prior around fiducial parameter values
        if self.prior_Gauss_rms is None:
            return 0
        else:
            rms=self.prior_Gauss_rms
            fid_values=[p.value_in_cube() for p in self.free_params]
            log_prior=-np.sum((np.array(fid_values)-values)**2/(2*rms**2))
            return log_prior


    def maximise_acquisition(self,alpha,verbose=False,tolerance=0.1,cube=False):
        """ Maximise lnprob+alpha*sigma, where sigma is the exploration
        term as defined in Rogers et al (2019) """

        x0=np.ones(len(self.free_parameters))*0.5

        result = minimize(self.acquisition, x0,args=(alpha,verbose),
                method='nelder-mead',
                options={'xatol': tolerance, 'disp': True})

        if cube:
            return result.x
        else:
            ## Map to physical params
            theta_physical=np.empty(len(result.x))
            for aa, theta in enumerate(result.x):
                theta_physical[aa]=self.free_params[aa].value_from_cube(theta)
            return theta_physical


    def acquisition(self,theta,alpha,verbose):
        """ Acquisition function """
        logprob=self.log_prob(theta)
        explo=self.exploration(theta)
        theta_param=theta
        if verbose:
                print("\n theta=", theta_param)
                print("log prob = ", logprob)
                print("exploration = ", alpha*explo)
                print("acquisition function = ", logprob+alpha*explo)
        return -1.0*(logprob+alpha*explo)


    def exploration(self,values):
        """ Return exploration term for acquisition function """
        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d,emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        explor=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute exploration term for z={}'.format(z))
            # get data cov
            data_cov=self.data.get_cov_iz(iz)

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(data_cov)
            explor += np.dot(np.dot(icov,np.sqrt(np.diag(emu_covar[iz]))),np.sqrt(np.diag(emu_covar[iz])))

        return explor
        

    def go_silent(self):
        """Turn off verbosity on all objects in likelihood object"""

        self.verbose=False
        self.theory.verbose=False
        self.theory.cosmo.verbose=False
        self.theory.emulator.verbose=False
        self.theory.emulator.arxiv.verbose=False


    def go_loud(self):
        """Turn on verbosity on all objects in likelihood object"""

        self.verbose=True
        self.theory.verbose=True
        self.theory.cosmo.verbose=True
        self.theory.emulator.verbose=True
        self.theory.emulator.arxiv.verbose=True


    def fit_cosmology_params(self):
        """ Fit Delta2_star and n_star to each simulation
        in the training set """

        cube=self.theory.emulator.arxiv.cube_data
        cosmo_fid = camb_cosmo.get_cosmology()
        linP_model_fid=fit_linP.LinearPowerModel(cosmo=cosmo_fid,k_units='Mpc')

        Delta2_stars=[]
        n_stars=[]

        for aa in range(cube["nsamples"]-1):
            if aa==self.theory.emulator.arxiv.drop_sim_number:
                ## Don't include mock sim
                continue
            else:
                cosmo_sim=sim_params_cosmo.cosmo_from_sim_params(cube["param_space"],
                                cube["samples"][str(aa)],
                                linP_model_fid,verbose=False)
                sim_linP_model=fit_linP.LinearPowerModel(cosmo=cosmo_sim)
                Delta2_stars.append(sim_linP_model.get_Delta2_star())
                n_stars.append(sim_linP_model.get_n_star())
        
        return Delta2_stars, n_stars


    def plot_p1d(self,values=None,plot_every_iz=1):
        """Plot P1D in theory vs data. If plot_every_iz >1,
            plot only few redshift bins"""

        # get measured bins from data
        k_kms=self.data.k
        k_emu_kms=np.logspace(np.log10(min(k_kms)),np.log10(max(k_kms)),500)
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_cov = self.get_p1d_kms(k_emu_kms,values,return_covar=True)

        emu_calls=self.theory.get_emulator_calls(self.parameters_from_sampling_point(values))
        distances=[]
        for aa,call in enumerate(emu_calls):
            distances.append(self.theory.emulator.get_nearest_distance(call,z=self.data.z[aa]))

        if self.verbose: print('got P1D from emulator')

        # plot only few redshifts for clarity
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=zs[iz]
            p1d_data=self.data.get_Pk_iz(iz)
            p1d_cov=self.data.get_cov_iz(iz)
            p1d_theory=emu_p1d[iz]
            cov_theory=emu_cov[iz]
            
            if p1d_theory is None:
                if self.verbose: print(z,'emulator did not provide P1D')
                continue
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.errorbar(k_kms,p1d_data*k_kms/np.pi,color=col,
                    yerr=np.sqrt(np.diag(p1d_cov))*k_kms/np.pi,
                    label="z=%.1f, distance = %.3f" % (z,distances[iz]))
            #plt.plot(k_kms,p1d_theory*k_kms/np.pi,color=col,
            #        linestyle="--")
            plt.plot(k_emu_kms,(p1d_theory*k_emu_kms)/np.pi,color=col,linestyle="dashed")
            plt.fill_between(k_emu_kms,((p1d_theory+np.sqrt(np.diag(cov_theory)))*k_emu_kms)/np.pi,

            ((p1d_theory-np.sqrt(np.diag(cov_theory)))*k_emu_kms)/np.pi,alpha=0.35,color=col)
                
        plt.plot(-10,-10,linestyle="-",label="Data",color="k")
        plt.plot(-10,-10,linestyle=":",label="Fit",color="k")
        plt.yscale('log')
        plt.legend()
        plt.xlabel('k [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.xlim(min(k_kms)-0.001,max(k_kms)+0.001)
        plt.tight_layout()
        plt.show()

        return


    def overplot_emulator_calls(self,param_1,param_2,values=None,
                                tau_scalings=True,temp_scalings=True):
        """For parameter pair (param1,param2), overplot emulator calls
            with values stored in arxiv, color coded by redshift"""

        # mask post-process scalings (optional)
        emu_data=self.theory.emulator.arxiv.data
        Nemu=len(emu_data)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in emu_data]
        else:
            mask_tau=[True]*Nemu
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) 
                        & (x['scale_gamma']==1.0) for x in emu_data]
        else:
            mask_temp=[True]*Nemu

        # figure out values of param_1,param_2 in arxiv
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]
        emu_calls=self.theory.get_emulator_calls(like_params=like_params)
        # figure out values of param_1,param_2 called
        call_1=[emu_call[param_1] for emu_call in emu_calls]
        call_2=[emu_call[param_2] for emu_call in emu_calls]

        # overplot
        zs=self.data.z
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        zmin=min(min(emu_z),min(zs))
        zmax=max(max(emu_z),max(zs))
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        plt.scatter(call_1,call_2,c=zs,s=50,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return

class simpleLikelihood(object):
    """ Simpler Likelihood class to work with a different set of Likelihood
    parameters to what we will ultimately be using. These parameters will just be
    a simple rescaling of each emulator parameter at all redshifts """

    def __init__(self,data=None,theory=None,emulator=None,
                    free_parameters=None,verbose=False,
                    prior_Gauss_rms=0.2,
                    min_kp_kms=None,emu_cov_factor=1):
        """Setup likelihood from theory and data. Options:
            - if prior_Gauss_rms is None it will use uniform priors
            - ignore k-bins with k > min_kp_kms
            - emu_cov_factor adjusts the contribution from emulator covariance
            set between 0 and 1. """

        self.verbose=verbose
        self.prior_Gauss_rms=prior_Gauss_rms
        self.emu_cov_factor=emu_cov_factor
        self.emulator=emulator
        self.simpleLike=True

        if data:
            self.data=data
        else:
            print("Pass data object please")
            quit()
        
        if theory:
            self.theory=theory
        else:
            zs=self.data.z
            if self.verbose: print('use default theory')
            self.theory=lya_theory.LyaTheory(zs,emulator=emulator)
        
        self.free_parameters=free_parameters ## Just a list of the names
        ## An actual list of the objects is below, named self.free_params
        self.set_free_parameters(self.free_parameters)
        
    def set_free_parameters(self,free_parameters,limits=None):
        """ Set the parameters we want to vary - these will have the same
        names as the emulator parameters """

        self.free_params=[]
        for aa,name in enumerate(free_parameters):
            if limits==None:
                min=0.5
                max=1.5
            else:
                min=limits[aa][0]
                max=limits[aa][1]
            ## Set up parameter object
            like_param=likelihood_parameter.LikelihoodParameter(name=name,
                                        value=1.,
                                        min_value=min,max_value=max)
            self.free_params.append(like_param)

        return

    def go_silent(self):
        self.verbose=False
    
    def log_prob(self,values):
        """Return log likelihood plus log priors"""

        # compute log_prior
        log_prior=self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        log_like=self.get_log_like(values,ignore_log_det_cov=False)

        if log_like is None:
            if self.verbose: print('was not able to emulate at least on P1D')
            return -np.inf

        return log_like + log_prior

    def exploration(self,values):
        """ Return exploration term for acquisition function """
        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d,emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        explor=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute exploration term for z={}'.format(z))
            # get data cov
            data_cov=self.data.get_cov_iz(iz)

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(data_cov)
            explor += np.dot(np.dot(icov,np.sqrt(np.diag(emu_covar[iz]))),np.sqrt(np.diag(emu_covar[iz])))

        return explor

    def get_log_prior(self,values):
        """Compute logarithm of prior"""

        assert len(values)==len(self.free_params),'size mismatch'

        # Always force parameter to be within range (for now)
        if max(values) > 1.0:
            return -np.inf
        if min(values) < 0.0:
            return -np.inf

        # Add Gaussian prior around fiducial parameter values
        if self.prior_Gauss_rms is None:
            return 0
        else:
            rms=self.prior_Gauss_rms
            fid_values=[p.value_in_cube() for p in self.free_params]
            log_prior=-np.sum((np.array(fid_values)-values)**2/(2*rms**2))
            return log_prior

    def get_chi2(self,values=None):
        """Compute chi2 using data and theory, without adding
            emulator covariance"""

        log_like=self.get_log_like(values,ignore_log_det_cov=True)
        if log_like is None:
            return None
        else:
            return -2.0*log_like

    def get_covmats(self,values=None):
        """ Return the data and emulator covmats for a given
        set of likelihood parameters. Will return a list of the
        covmats at each z """

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        log_like=0

        data_covar=[]

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            data_covar.append(self.data.get_cov_iz(iz))

        return data_covar, emu_covar

    def get_log_like(self,values=None,ignore_log_det_cov=True):
        """Compute log(likelihood), including determinant of covariance
            unless you are setting ignore_log_det_cov=True.."""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d,emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        log_like=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            p1d=self.data.get_Pk_iz(iz)
            data_cov=self.data.get_cov_iz(iz)

            # add covariance from emulator
            cov = data_cov + self.emu_cov_factor*emu_covar[iz]

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(cov)
            diff = (p1d-emu_p1d[iz])
            chi2_z = np.dot(np.dot(icov,diff),diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_z = -0.5*chi2_z
            else:
                (_, log_det_cov) = np.linalg.slogdet(cov)
                log_like_z = -0.5*(chi2_z + log_det_cov)
            log_like += log_like_z
            if self.verbose: print('added {} to log_like'.format(log_like_z))

        return log_like

    def get_p1d_kms(self,k_kms=None,values=None,return_covar=False):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms=self.data.k

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]

        ## Use these likelihood parameters to get
        ## emulator calls
        emu_calls=self.get_emulator_calls(like_params)

        ## Now turn these emulator calls into p1ds
        # loop over redshifts and compute P1D
        p1d_kms=[]
        if return_covar:
            covars=[]
        for iz,z in enumerate(self.data.z):
            # will call emulator for this model
            model=emu_calls[iz]
            # emulate p1d
            dkms_dMpc=self.theory.cosmo.reconstruct_Hubble_iz(iz,self.theory.cosmo.linP_model_fid)/(1+z)
            k_Mpc = k_kms * dkms_dMpc
            if return_covar:
                p1d_Mpc, cov_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=True,
                                                        z=z)
            else:
                p1d_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=False,
                                                        z=z)
            if p1d_Mpc is None:
                if self.verbose: print('emulator did not provide P1D')
                p1d_kms.append(None)
                if return_covar:
                    covars.append(None)
            else:
                p1d_kms.append(p1d_Mpc * dkms_dMpc)
                if return_covar:
                    if cov_Mpc is None:
                        covars.append(None)
                    else:
                        covars.append(cov_Mpc * dkms_dMpc**2)

        if return_covar:
            return p1d_kms,covars
        else:
            return p1d_kms

    def parameters_from_sampling_point(self,values):
        """Translate input array of values (in cube) to likelihood parameters"""
        
        if values is None:
            return []

        assert len(values)==len(self.free_params),'size mismatch'
        Npar=len(values)
        like_params=[]
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params

    def get_emulator_calls(self,like_params):
        """ Get emulator calls for a given set of likelihood parameters """

        emu_calls=[]

        for iz,z in enumerate(self.data.z):
            model={}
            ## Set up a model with the true values
            for par_name in self.emulator.paramList:
                model[par_name]=self.data.truth[par_name][iz]
            ## For each parameter we are varying, modify the
            ## emulator call
            for like_param in like_params:
                model[like_param.name]*=like_param.value
            emu_calls.append(model)
            
        return emu_calls


    def plot_p1d(self,values=None,plot_every_iz=1):
        """Plot P1D in theory vs data. If plot_every_iz >1,
            plot only few redshift bins"""

        # get measured bins from data
        k_kms=self.data.k
        k_emu_kms=np.logspace(np.log10(min(k_kms)),np.log10(max(k_kms)),500)
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_cov = self.get_p1d_kms(k_emu_kms,values,return_covar=True)

        if self.verbose: print('got P1D from emulator')

        # plot only few redshifts for clarity
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=zs[iz]
            p1d_data=self.data.get_Pk_iz(iz)
            p1d_cov=self.data.get_cov_iz(iz)
            p1d_theory=emu_p1d[iz]
            cov_theory=emu_cov[iz]
            
            if p1d_theory is None:
                if self.verbose: print(z,'emulator did not provide P1D')
                continue
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.errorbar(k_kms,p1d_data*k_kms/np.pi,color=col,marker="o",ls="none",
                    yerr=np.sqrt(np.diag(p1d_cov))*k_kms/np.pi,ms=4.5,
                    label="z=%.1f" % (z))
            plt.plot(k_emu_kms,(p1d_theory*k_emu_kms)/np.pi,color=col,linestyle="dashed")
            plt.fill_between(k_emu_kms,((p1d_theory+np.sqrt(np.diag(cov_theory)))*k_emu_kms)/np.pi,
            ((p1d_theory-np.sqrt(np.diag(cov_theory)))*k_emu_kms)/np.pi,alpha=0.5,color=col)
                
        plt.plot(-10,-10,marker="o",linestyle="none",label="Simulation",color="k")
        plt.plot(-10,-10,linestyle="--",label="Emulator prediction",color="k")
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'$k_\parallel$ [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.xlim(min(k_kms)-0.001,max(k_kms)+0.001)
        plt.tight_layout()

        return

