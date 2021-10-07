import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import minimize
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.data import data_PD2013
from lace.likelihood import lya_theory
from lace.likelihood import likelihood_parameter
from lace.likelihood import linear_power_model
from lace.likelihood import full_theory
from lace.likelihood import CAMB_model
from lace.likelihood import marg_p1d_like
from lace.setup_simulations import read_genic
from lace.setup_simulations import sim_params_cosmo
from lace.setup_simulations import sim_params_space
from lace.likelihood import cmb_like

class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(self,data=None,theory=None,emulator=None,
                    free_param_names=None,
                    free_param_limits=None,
                    verbose=False,
                    prior_Gauss_rms=0.2,
                    kmin_kms=None,emu_cov_factor=1,
                    use_sim_cosmo=False,
                    pivot_scalar=0.05,
                    include_CMB=False,
                    use_compression=0,
                    reduced_IGM=False):
        """Setup likelihood from theory and data. Options:
            - free_param_names is a list of param names, in any order
            - free_param_limits list of tuples, same order than free_param_names
            - if prior_Gauss_rms is None it will use uniform priors
            - ignore k-bins with k > kmin_kms
            - emu_cov_factor adjusts the contribution from emulator covariance
            set between 0 and 1.
            - use_sim_cosmo will extract the cosmological likelihood
              parameters from the fiducial simulation, and use these
              as a fiducial model
            - pivot_scalar sets the pivot scale used for the primordial
              power spectrum in the case of using a full_theory object
            - include_CMB will use the CMB Gaussian likelihood approximation
              from Planck as a prior on each cosmological parameter
            - use_compression: 0 for no compression
                               1 to compress into 4 parameters
                               2 to compress into just 2
                               3 will use marginalised
                                 constraints on Delta2_star
                                 and n_star
            - reduced_IGM: temporary flag to determine in the case of
                           use_compression=3, we want to use the
                           covariance of a full IGM marginalisation
                           or only ln_tau_0 """


        self.verbose=verbose
        self.prior_Gauss_rms=prior_Gauss_rms
        self.emu_cov_factor=emu_cov_factor
        self.include_CMB=include_CMB
        self.use_compression=use_compression
        self.reduced_IGM=reduced_IGM

        if data:
            self.data=data
        else:
            if self.verbose: print('use default data')
            self.data=data_PD2013.P1D_PD2013()

        # (optionally) get rid of low-k data points
        self.data._cull_data(kmin_kms)

        if theory:
            self.theory=theory
        else:
            ## Use the free_param_names to determine whether to use
            ## a LyaTheory or FullTheory object
            compressed=bool(set(free_param_names) & set(["Delta2_star",
                                "n_star","alpha_star","f_star","g_star"]))

            full=bool(set(free_param_names) & set(["H0","mnu","As","ns"]))

            if self.verbose:
                if compressed:
                    print('using compressed theory')
                elif full:
                    print('using full theory')
                    assert ("H0" in free_param_names and "cosmomc_theta" in free_param_names)==False, "Cannot vary both H0 and theta_MC"
                else:
                    print('not varying cosmo params',free_param_names)

            assert (compressed and full)==False, "Cannot vary both compressed and full likelihood parameters"

            if use_sim_cosmo:
                # Use the simulation cosmology as fiducial, for mock data
                sim_cosmo=self.data.mock_sim.sim_cosmo
                if self.verbose:
                    print('use_sim_cosmo',camb_cosmo.print_info(sim_cosmo))
            else:
                sim_cosmo=None
            
            if compressed:
                ## Set up a compressed LyaTheory object
                self.theory=lya_theory.LyaTheory(self.data.z,emulator=emulator,
                        cosmo_fid=sim_cosmo,verbose=self.verbose)
            else:
                ## Set up a FullTheory object
                camb_model_sim=CAMB_model.CAMBModel(zs=self.data.z,
                        cosmo=sim_cosmo,pivot_scalar=pivot_scalar,theta_MC=("cosmomc_theta" in free_param_names))
                self.theory=full_theory.FullTheory(zs=data.z,emulator=emulator,
                        true_camb_model=camb_model_sim,verbose=self.verbose,
                        pivot_scalar=pivot_scalar,
                        theta_MC=("cosmomc_theta" in free_param_names),
                        use_compression=use_compression)
                assert self.data.mock_sim.sim_cosmo.InitPower.pivot_scalar == self.theory.true_camb_model.cosmo.InitPower.pivot_scalar

                if not full:
                    print("No cosmology parameters are varied")


        # setup parameters
        self.set_free_parameters(free_param_names,free_param_limits)

        if self.include_CMB==True:
            ## Set up a CMB likelihood object, using the simulation mock
            ## cosmology as the central values
            ## Check if neutrino mass is a free parameter
            nu_mass=False
            for par in self.free_params:
                if par.name=="mnu":
                    nu_mass=True
            self.cmb_like=cmb_like.CMBLikelihood(self.data.mock_sim.sim_cosmo,
                                    nu_mass=nu_mass)

        ## Set up a marginalised p1d likelihood if
        ## we are using compression mode 3
        if self.use_compression==3:
            ## Check if IGM parameters are free
            igm=False
            for par in self.free_params:
                if "ln_" in par.name:
                    igm=True
            assert igm==False, "Cannot run marginalised P1D with free IGM parameters"

            ## Set up marginalised p1d likelihood object
            self.marg_p1d=marg_p1d_like.MargP1DLike(self.data.sim_label,self.reduced_IGM)

        if self.verbose: print(len(self.free_params),'free parameters')

        return


    def set_free_parameters(self,free_param_names,free_param_limits):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params=[]

        if free_param_limits is not None:
            assert len(free_param_limits)==len(free_param_names), "wrong number of parameter limits"

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        ## select free parameters, make sure ordering
        ## in self.free_params is same as
        ## in self.free_parameters
        for par_name in free_param_names:
            for par in params:
                if par.name == par_name:
                    if free_param_limits is not None:
                        ## Set min and max of each parameter if
                        ## a list is given. otherwise leave as default
                        par.min_value=free_param_limits[free_param_names.index(par.name)][0]
                        par.max_value=free_param_limits[free_param_names.index(par.name)][1]
                    self.free_params.append(par)

        Nfree=len(self.free_params)
        Nin=len(free_param_names)

        assert (Nfree==Nin), 'could not setup free parameters'

        if self.verbose:
            print('likelihood setup with {} free parameters'.format(Nfree))

        return


    def get_free_parameter_list(self):
        """ Return a list of the names of all free parameters
        for this object """

        param_list=[]
        for par in self.free_params:
            param_list.append(par.name)

        return param_list


    def default_sampling_point(self):
        """Use default likelihood parameters to get array of values (in cube)"""

        return self.sampling_point_from_parameters(self.free_params)


    def sampling_point_from_parameters(self,like_params):
        """Get parameter values in cube for free parameters in input array.
            Note: input list could be longer than list of free parameters,
            and in different order."""

        # collect list of values of parameters in cube
        values=[]
        # loop over free parameters in likelihood, this sets the order
        for par in self.free_params:
            found=False
            # loop over input likelihood parameters and find match
            for inpar in like_params:
                if par.name == inpar.name:
                    assert found==False,'parameter found twice'
                    values.append(inpar.value_in_cube())
                    found=True
            if not found:
                print('free parameter not in input list',par.info_str())

        assert len(self.free_params)==len(values),'size mismatch'

        return values


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


    def cosmology_params_from_sampling_point(self,values):
        """ For a given point in sampling space, return a list of "CMB"
        cosmology params """

        like_params=self.parameters_from_sampling_point(values)

        ## Dictionary of cosmology parameters
        cosmo_dict={}

        for like_param in like_params:
            if like_param.name=="ombh2":
                cosmo_dict["ombh2"]=like_param.value
            elif like_param.name=="omch2":
                cosmo_dict["omch2"]=like_param.value
            elif like_param.name=="cosmomc_theta":
                cosmo_dict["cosmomc_theta"]=like_param.value
            elif like_param.name=="As":
                cosmo_dict["As"]=like_param.value
            elif like_param.name=="ns":
                cosmo_dict["ns"]=like_param.value

        assert len(cosmo_dict)>0, "No CMB cosmology parameters found in sampling space"

        return cosmo_dict


    def get_cmb_like(self,values):
        """ For a given point in sampling space, return the CMB likelihood """

        # get cosmology parameters from sampling points
        cosmo_dic=self.cosmology_params_from_sampling_point(values)

        # (pretty ugly way to) get true cosmology from full_theory object
        true_cosmo=self.theory.true_camb_model.cosmo

        # compute CMB likelihood by comparing both cosmologies
        cmb_like=self.cmb_like.get_cmb_like(cosmo_dic,true_cosmo)

        return cmb_like


    def get_p1d_kms(self,k_kms=None,values=None,return_covar=False,
                    camb_evaluation=None,return_blob=False):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms=self.data.k

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]

        return self.theory.get_p1d_kms(k_kms,like_params=like_params,
                                            return_covar=return_covar,
                                            camb_evaluation=camb_evaluation,
                                            return_blob=return_blob)


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


    def get_marg_lya_like(self,values=None,return_blob=False):
        """ Get log likelihood from P1D using pre-marginalised
            constraints on Delta2_star and n_star """

        ## Need to get Delta2_star and f_star from a set of values
        like_params=self.parameters_from_sampling_point(values)

        camb_model=self.theory.true_camb_model.get_new_model(like_params)
        linP_model=linear_power_model.LinearPowerModel(
                        cosmo=camb_model.cosmo,
                        camb_results=camb_model.get_camb_results(),
                        use_camb_fz=self.theory.use_camb_fz)

        log_like=self.marg_p1d.return_lya_like(np.array([linP_model.linP_params["Delta2_star"],
                                        linP_model.linP_params["n_star"]]))

        ## Check if we want blobs
        if return_blob:
            blob=self.theory.get_blob(camb_model)
            return log_like,blob
        else:
            return log_like


    def get_log_like(self,values=None,ignore_log_det_cov=True,
            camb_evaluation=None,return_blob=False):
        """Compute log(likelihood), including determinant of covariance
            unless you are setting ignore_log_det_cov=True."""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        if self.use_compression==3:
            if return_blob==True:
                log_like,blob=self.get_marg_lya_like(values=values,
                                            return_blob=True)
                return log_like,blob
            else:
                log_like=self.get_marg_lya_like(values=values,
                                            return_blob=False)
                return log_like

        # ask emulator prediction for P1D in each bin
        if return_blob:
            emu_p1d,emu_covar,blob=self.get_p1d_kms(k_kms,values,
                            return_covar=True,camb_evaluation=camb_evaluation,
                            return_blob=True)
        else:
            emu_p1d,emu_covar=self.get_p1d_kms(k_kms,values,
                            return_covar=True,camb_evaluation=camb_evaluation,
                            return_blob=False)

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


        if return_blob:
            return log_like,blob
        else:
            return log_like


    def compute_log_prob(self,values,return_blob=False):
        """Compute log likelihood plus log priors for input values
            - if return_blob==True, it will return also extra information"""

        # Always force parameter to be within range (for now)
        if (max(values) > 1.0) or (min(values) < 0.0):
            if return_blob:
                dummy_blob=self.theory.get_blob()
                return -np.inf, dummy_blob
            else:
                return -np.inf

        # compute log_prior
        log_prior=self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        if return_blob:
            log_like,blob=self.get_log_like(values,ignore_log_det_cov=False,
                                            return_blob=True)
        else:
            log_like=self.get_log_like(values,ignore_log_det_cov=False,
                                            return_blob=False)

        if log_like == None or math.isnan(log_like)==True:
            if self.verbose: print('was not able to emulate at least on P1D')
            if return_blob:
                dummy_blob=self.theory.get_blob()
                return -np.inf, dummy_blob
            else:
                return -np.inf

        if return_blob:
            return log_like + log_prior, blob
        else:
            return log_like + log_prior


    def log_prob(self,values):
        """Return log likelihood plus log priors"""

        return self.compute_log_prob(values,return_blob=False)


    def log_prob_and_blobs(self,values):
        """Function used by emcee to get both log_prob and extra information"""

        lnprob,blob=self.compute_log_prob(values,return_blob=True)
        # unpack tuple
        out=lnprob,*blob
        return out


    def get_log_prior(self,values):
        """Compute logarithm of prior"""

        assert len(values)==len(self.free_params),'size mismatch'

        # Always force parameter to be within range (for now)
        if max(values) > 1.0:
            return -np.inf
        if min(values) < 0.0:
            return -np.inf

        ## If we are including CMB information, use this as the prior
        if self.include_CMB==True:
            return self.get_cmb_like(values)
        ## Otherwise add Gaussian prior around fiducial parameter values
        elif self.prior_Gauss_rms is None:
            return 0
        else:
            rms=self.prior_Gauss_rms
            fid_values=[p.value_in_cube() for p in self.free_params]
            log_prior=-np.sum((np.array(fid_values)-values)**2/(2*rms**2))
            return log_prior


    def minus_log_prob(self,values):
        """Return minus log_prob (needed to maximise posterior)"""

        return -1.0*self.log_prob(values)


    def maximise_posterior(self,initial_values=None,method='nelder-mead',tol=1e-4):
        """Run scipy minimizer to find maximum of posterior"""

        return minimize(self.minus_log_prob, x0=initial_values,method=method,tol=tol)


    def maximise_acquisition(self,alpha,verbose=False,tolerance=0.1,cube=False):
        """ Maximise lnprob+alpha*sigma, where sigma is the exploration
        term as defined in Rogers et al (2019) """

        x0=np.ones(len(self.free_params))*0.5

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
        try: ## Only lya_theory object has a theory.cosmo object
            self.theory.cosmo.verbose=False
        except:
            pass
        self.theory.emulator.verbose=False
        self.theory.emulator.archive.verbose=False


    def go_loud(self):
        """Turn on verbosity on all objects in likelihood object"""

        self.verbose=True
        self.theory.verbose=True
        try:
            self.theory.cosmo.verbose=True
        except:
            pass
        self.theory.emulator.verbose=True
        self.theory.emulator.archive.verbose=True


    def get_simulation_suite_linP_params(self):
        """ Compute Delta2_star and n_star for each simulation
        in the training set of the emulator"""

        # simulation cube used in emulator
        cube_data=self.theory.emulator.archive.cube_data

        # collect linP params for each simulation
        Delta2_stars=[]
        n_stars=[]
        for sim_num in range(cube_data["nsamples"]):
            # Don't include simulation used to generate mock data
            if sim_num==self.theory.emulator.archive.drop_sim_number:
                continue
            else:
                sim_linP_params=self.get_simulation_linP_params(sim_num)
                Delta2_stars.append(sim_linP_params['Delta2_star'])
                n_stars.append(sim_linP_params['n_star'])
        
        return Delta2_stars, n_stars


    def get_simulation_linP_params(self,sim_num):
        """ Compute Delta2_star and n_star for a given simulation in suite"""

        # this function should only be called when using compressed parameters
        z_star = self.theory.cosmo.z_star
        kp_kms = self.theory.cosmo.kp_kms

        # setup cosmology from GenIC file
        sim_cosmo=self.theory.emulator.archive.get_simulation_cosmology(sim_num)

        # fit linear power parameters for simulation cosmology
        sim_linP_params=fit_linP.parameterize_cosmology_kms(
                cosmo=sim_cosmo,camb_results=None,
                z_star=z_star,kp_kms=kp_kms)

        return sim_linP_params


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
                    yerr=np.sqrt(np.diag(p1d_cov))*k_kms/np.pi,fmt="o",ms="4",
                    label="z=%.1f" % z)
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
            with values stored in archive, color coded by redshift"""

        # mask post-process scalings (optional)
        emu_data=self.theory.emulator.archive.data
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

        # figure out values of param_1,param_2 in archive
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
