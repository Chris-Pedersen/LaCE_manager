import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import emcee
import time
import scipy.stats
import multiprocessing as mg
from multiprocessing import Pool
from multiprocessing import Process
from chainconsumer import ChainConsumer
# our own modules
from lace.cosmo import fit_linP
from lace.cosmo import camb_cosmo
from lace.data import data_MPGADGET
from lace.data import data_PD2013
from lace.emulator import p1d_archive
from lace.emulator import gp_emulator
from lace.emulator import z_emulator
from lace.likelihood import lya_theory
from lace.likelihood import likelihood
from lace.likelihood import likelihood_parameter

class EmceeSampler(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(self,like=None,emulator=None,free_param_names=None,
                        nwalkers=None,read_chain_file=None,verbose=False,
                        subfolder=None,rootdir=None,
                        save_chain=True,progress=False):
        """Setup sampler from likelihood, or use default.
            If read_chain_file is provided, read pre-computed chain.
            rootdir allows user to search for saved chains in a different location
            to the code itself """

        # WE SHOULD DOCUMENT BETTER THE OPTIONAL INPUTS
        # WHEN WOULD SOMEONE PASS A LIKELIHOOD AND A LIST OF FREE PARAMETERS?
        # WOULDN'T like.free_params ALREADY CONTAIN THAT?

        # WHEN WOULD YOU LIKE TO HAVE A SAMPLER WITHOUT AN EMULATOR?

        self.verbose=verbose
        self.progress=progress

        if read_chain_file:
            if self.verbose: print('will read chain from file',read_chain_file)
            self.read_chain_from_file(read_chain_file,rootdir,subfolder)
            self.p0=None
            self.burnin_pos=None
        else: 
            if like:
                if self.verbose: print('use input likelihood')
                self.like=like
            else:
                if self.verbose: print('use default likelihood')
                data=data_PD2013.P1D_PD2013()
                zs=data.z
                theory=lya_theory.LyaTheory(zs,emulator=emulator)
                self.like=likelihood.Likelihood(data=data,theory=theory,
                                free_param_names=free_param_names,verbose=False)
            # number of free parameters to sample
            self.ndim=len(self.like.free_params)

            self.save_directory=None
            if save_chain:
                self._setup_chain_folder(rootdir,subfolder)
                backend_string=self.save_directory+"/backend.h5"
                self.backend=emcee.backends.HDFBackend(backend_string)
            else:
                self.backend=None

            # number of walkers
            if nwalkers:
                self.nwalkers=nwalkers
            else:
                self.nwalkers=10*self.ndim
            if self.verbose: print('setup with',self.nwalkers,'walkers')
            # setup sampler
            # setup walkers
            self.p0=self.get_initial_walkers()

        ## Set up list of parameter names in tex format for plotting
        self.paramstrings=[]
        for param in self.like.free_params:
            self.paramstrings.append(param_dict[param.name])

        self.set_truth()

        # Figure out what extra information will be provided as blobs
        self.blobs_dtype = self.like.theory.get_blobs_dtype()


    def set_truth(self):
        """ Set up dictionary with true values of cosmological
        likelihood parameters for plotting purposes """

        test_sim_cosmo=self.like.data.mock_sim.sim_cosmo
        test_results=camb_cosmo.get_camb_results(test_sim_cosmo)
        self.truth={}

        linP_truth=fit_linP.parameterize_cosmology_kms(
                        cosmo=test_sim_cosmo,
                        camb_results=test_results,
                        z_star=3.0, ## Hardcoding for now!!!
                        kp_kms=0.009)

        ## Are we using full theory or compressed theory
        if hasattr(self.like.theory,"emu_kp_Mpc"):
            ## Get all possible likelihood params
            all_truth={}
            all_truth["ombh2"]=test_sim_cosmo.ombh2
            all_truth["omch2"]=test_sim_cosmo.omch2
            all_truth["As"]=test_sim_cosmo.InitPower.As
            all_truth["ns"]=test_sim_cosmo.InitPower.ns
            all_truth["H0"]=test_sim_cosmo.H0
            all_truth["mnu"]=camb_cosmo.get_mnu(test_sim_cosmo)
            all_truth["cosmomc_theta"]=test_results.cosmomc_theta()
            ## Store truth for compressed parameters in case we want to
            ## plot them as derived parameters
            all_truth["Delta2_star"]=linP_truth["Delta2_star"]
            all_truth["n_star"]=linP_truth["n_star"]
            all_truth["alpha_star"]=linP_truth["alpha_star"]
            all_truth["f_star"]=linP_truth["f_star"]
            all_truth["g_star"]=linP_truth["g_star"]
            ## Store truth for all parameters, whether free or not
            ## in the full_theory case
            for param in cosmo_params:
                param_string=param_dict[param]
                self.truth[param_string]=all_truth[param]
        else:
            ## True compressed parameters
            all_truth=linP_truth
            ## Take only free parameters, and store values
            ## along with LaTeX strings
            for param in self.like.free_params:
                if param.name in cosmo_params:
                    param_string=param_dict[param.name]
                    self.truth[param_string]=all_truth[param.name]

        return


    def run_sampler(self,burn_in,max_steps,log_func=None,
                parallel=False,force_steps=False,timeout=None):
        """ Set up sampler, run burn in, run chains,
        return chains
            - force_steps will force the sampler to run
              until max_steps is reached regardless of
              convergence
            - timeout is the time in hours to run the
              sampler for """


        self.burnin_nsteps=burn_in
        # We'll track how the average autocorrelation time estimate changes
        self.autocorr = np.array([])
        # This will be useful to testing convergence
        old_tau = np.inf

        if parallel==False:
            ## Get initial walkers
            p0=self.get_initial_walkers()
            if log_func is None:
                log_func=self.like.log_prob_and_blobs
            sampler=emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                log_func,
                                                backend=self.backend,
                                                blobs_dtype=self.blobs_dtype)
            for sample in sampler.sample(p0, iterations=burn_in+max_steps,           
                                    progress=self.progress,):
                # Only check convergence every 100 steps
                if sampler.iteration % 100 or sampler.iteration < burn_in+1:
                    continue

                if self.progress==False:
                    print("Step %d out of %d " % (sampler.iteration, burn_in+max_steps))

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0,discard=burn_in)
                self.autocorr= np.append(self.autocorr,np.mean(tau))

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if force_steps == False:
                    if converged:
                        break
                old_tau = tau

        else:
            p0=self.get_initial_walkers()
            with Pool() as pool:
                sampler=emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                log_func,
                                                pool=pool,
                                                backend=self.backend,
                                                blobs_dtype=self.blobs_dtype)
                if timeout:
                    time_end=time.time() + 3600*timeout
                for sample in sampler.sample(p0, iterations=burn_in+max_steps,           
                                        progress=self.progress):
                    # Only check convergence every 100 steps
                    if sampler.iteration % 100 or sampler.iteration < burn_in+1:
                        continue

                    if self.progress==False:
                        print("Step %d out of %d " % (sampler.iteration, burn_in+max_steps))

                    # Compute the autocorrelation time so far
                    # Using tol=0 means that we'll always get an estimate even
                    # if it isn't trustworthy
                    tau = sampler.get_autocorr_time(tol=0,discard=burn_in)
                    self.autocorr= np.append(self.autocorr,np.mean(tau))

                    # Check convergence
                    converged = np.all(tau * 100 < sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                    if force_steps == False:
                        if converged:
                            print("Chains have converged")
                            break
                        if timeout:
                            if time.time()>time_end:
                                print("Timed out")
                                break
                    old_tau = tau

        ## Save chains
        self.chain=sampler.get_chain(flat=True,discard=self.burnin_nsteps)
        self.lnprob=sampler.get_log_prob(flat=True,discard=self.burnin_nsteps)
        self.blobs=sampler.get_blobs(flat=True,discard=self.burnin_nsteps)

        return 


    def resume_sampler(self,max_steps,log_func=None,timeout=None,force_timeout=False):
        """ Use the emcee backend to restart a chain from the last iteration
            - max_steps is the maximum number of steps for this run
            - log_func should be sampler.like.log_prob_and_blobs
              with pool apparently
            - timeout is the amount of time to run in hours before wrapping
              the job up. This is used to make sure timeouts on compute nodes
              don't corrupt the backend 
            - force_timeout will force chains to run for the time duration set
              instead of cutting at autocorrelation time convergence """

        ## Make sure we have a backend
        assert self.backend is not None, "No backend found, cannot run sampler"
        old_tau = np.inf
        with Pool() as pool:
            sampler=emcee.EnsembleSampler(self.backend.shape[0],
                                         self.backend.shape[1],
                                        log_func,
                                        pool=pool,
                                        backend=self.backend,
                                        blobs_dtype=self.blobs_dtype)
            if timeout:
                time_end=time.time() + 3600*timeout
            start_step=self.backend.iteration
            for sample in sampler.sample(self.backend.get_last_sample(), iterations=max_steps,           
                                    progress=self.progress):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                if self.progress==False:
                    print("Step %d out of %d " % (self.backend.iteration, start_step+max_steps))

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                self.autocorr= np.append(self.autocorr,np.mean(tau))

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if force_timeout==False:
                    if converged:
                        print("Chains have converged")
                        break
                if timeout:
                    if time.time()>time_end:
                        print("Timed out")
                        break
                old_tau = tau

        self.chain=sampler.get_chain(flat=True,discard=self.burnin_nsteps)
        self.lnprob=sampler.get_log_prob(flat=True,discard=self.burnin_nsteps)
        self.blobs=sampler.get_blobs(flat=True,discard=self.burnin_nsteps)
        
        return


    def get_initial_walkers(self):
        """Setup initial states of walkers in sensible points"""


        ndim=self.ndim
        nwalkers=self.nwalkers

        if self.verbose: 
            print('set %d walkers with %d dimensions'%(nwalkers,ndim))

        if self.like.prior_Gauss_rms is None:
            p0=np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))
        else:
            rms=self.like.prior_Gauss_rms
            p0=np.ndarray([nwalkers,ndim])
            for i in range(ndim):
                p=self.like.free_params[i]
                fid_value=p.value_in_cube()
                values=self.get_trunc_norm(fid_value,nwalkers)
                assert np.all(values >= 0.0)
                assert np.all(values <= 1.0)
                p0[:,i]=values

        return p0


    def get_trunc_norm(self,mean,n_samples):
        """ Wrapper for scipys truncated normal distribution
        Runs in the range [0,1] with a rms specified on initialisation """

        rms=self.like.prior_Gauss_rms
        values=scipy.stats.truncnorm.rvs((0.0-mean)/rms,
                            (1.0-mean)/rms, scale=rms,
                            loc=mean, size=n_samples)

        return values


    def go_silent(self):
        self.verbose=False
        self.like.go_silent()


    def get_chain(self,cube=True):
        """Figure out whether chain has been read from file, or computed"""

        chain=self.chain#.get_chain(flat=True,discard=self.burnin_nsteps)
        lnprob=self.lnprob#.get_log_prob(flat=True,discard=self.burnin_nsteps)
        blobs=self.blobs

        if cube == False:
            cube_values=chain
            list_values=[self.like.free_params[ip].value_from_cube(
                                cube_values[:,ip]) for ip in range(self.ndim)]
            chain=np.array(list_values).transpose()

        return chain,lnprob,blobs


    def plot_autocorrelation_time(self):
        """ Plot autocorrelation time as a function of
        sample numer """
        
        plt.figure()

        n = 100 * np.arange(1, len(self.autocorr)+1)
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, self.autocorr)
        plt.xlim(0, n.max())
        plt.ylim(0, self.autocorr.max() + 0.1 * (self.autocorr.max() - self.autocorr.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")
        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/autocorr_time.pdf")
        else:
            plt.show()

        return


    def get_all_params(self):
        """ Get a merged array of both sampled and derived parameters
            returns a 2D array of all parameters, and an ordered list of
            the LaTeX strings for each """
        
        chain,lnprob,blobs=self.get_chain(cube=False)

        if blobs==None:
             ## Old chains will have no blobs
             all_params=chain
             all_strings=self.paramstrings
        elif len(blobs[0])<6:
            ## For now this will represent a lya_theory chain
            ## so we ignore derived parameters for the time being
             all_params=chain
             all_strings=self.paramstrings
        elif len(blobs[0])==6:
            ## If blobs are length 6, we are using a full_theory chain.
            ## Build an array of chain + blobs, as chainconsumer
            ## doesn't know about the difference between sampled and derived
            ## parameters

            ## Array for blobs:
            blobs_full=np.hstack((np.vstack(blobs["Delta2_star"]),
                        np.vstack(blobs["n_star"]),
                        np.vstack(blobs["f_star"]),
                        np.vstack(blobs["g_star"]),
                        np.vstack(blobs["alpha_star"]),
                        np.vstack(blobs["H0"])))

            ## Array for all parameters
            all_params=np.hstack((chain,blobs_full))

            ## Ordered strings for all parameters
            all_strings=self.paramstrings+blob_strings
        else:
            print("Unkown blob configuration, just returning sampled params")
            all_params=chain
            all_strings=self.paramstrings

        return all_params, all_strings


    def read_chain_from_file(self,chain_number,rootdir,subfolder):
        """Read chain from file, and check parameters"""
        
        if rootdir:
            chain_location=rootdir
        else:
            assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
            chain_location=os.environ['LYA_EMU_REPO']+"/lace/sampler/chains/"
        if subfolder:
            self.save_directory=chain_location+"/"+subfolder+"/chain_"+str(chain_number)
        else:
            self.save_directory=chain_location+"/chain_"+str(chain_number)

        with open(self.save_directory+"/config.json") as json_file:  
            config = json.load(json_file)

        if self.verbose: print("Building archive")
        try:
            kp=config["kp_Mpc"]
        except:
            kp=None
        ## Set up the archive
        archive=p1d_archive.archiveP1D(basedir=config["basedir"],
                            drop_tau_rescalings=config["drop_tau_rescalings"],
                            drop_temp_rescalings=config["drop_temp_rescalings"],
                            nearest_tau=config["nearest_tau"],
                            z_max=config["z_max"],
                            drop_sim_number=config["data_sim_number"],
                            p1d_label=config["p1d_label"],                            
                            skewers_label=config["skewers_label"],
                            undersample_cube=config["undersample_cube"],
                            kp_Mpc=kp)

        if self.verbose: print("Setting up emulator")
        try:
            reduce_var=config["reduce_var"]
        except:
            reduce_var=False
        ## Set up the emulators
        if config["z_emulator"]:
            emulator=z_emulator.ZEmulator(paramList=config["paramList"],
                                train=True,
                                emu_type=config["emu_type"],
                                kmax_Mpc=config["kmax_Mpc"],
                                reduce_var_mf=reduce_var,
                                passarchive=archive,verbose=self.verbose)
        else:
            emulator=gp_emulator.GPEmulator(paramList=config["paramList"],
                                train=True,
                                emu_type=config["emu_type"],
                                kmax_Mpc=config["kmax_Mpc"],
                                asymmetric_kernel=config["asym_kernel"],
                                rbf_only=config["asym_kernel"],
                                reduce_var_mf=reduce_var,
                                passarchive=archive,verbose=self.verbose)

        ## Try/excepts are for backwards compatibility
        ## as old config files don't have these entries
        try:
            data_cov=config["data_cov_factor"]
        except:
            data_cov=1.
        try:
            data_year=config["data_year"]
        except:
            # if datacov_label wasn't recorded, it was PD2013
            data_year="PD2013"

        ## Old chains won't have pivot_scalar saved
        if "pivot_scalar" in config.keys():
            pivot_scalar=config["pivot_scalar"]
        else:
            pivot_scalar=0.05

        ## Set up mock data
        data=data_MPGADGET.P1D_MPGADGET(sim_label=config["data_sim_number"],
                                    basedir=config["basedir"],
                                    skewers_label=config["skewers_label"],
                                    z_list=np.asarray(config["z_list"]),
                                    data_cov_factor=data_cov,
                                    data_cov_label=data_year,
                                    pivot_scalar=pivot_scalar)

        if self.verbose: print("Setting up likelihood")
        ## Set up likelihood
        free_param_names=[]
        for item in config["free_params"]:
            free_param_names.append(item[0])

        ## Not all saved chains will have this flag
        try:
            free_param_limits=config["free_param_limits"]
        except:
            free_param_limits=None

        try:
            include_CMB=config["include_CMB"]
        except:
            include_CMB=False

        try:
            reduced_IGM=config["reduced_IGM"]
        except:
            reduced_IGM=False

        self.like=likelihood.Likelihood(data=data,emulator=emulator,
                            free_param_names=free_param_names,
                            free_param_limits=free_param_limits,
                            verbose=False,
                            prior_Gauss_rms=config["prior_Gauss_rms"],
                            emu_cov_factor=config["emu_cov_factor"],
                            pivot_scalar=pivot_scalar,
                            include_CMB=include_CMB,
                            reduced_IGM=reduced_IGM)

        if self.verbose: print("Load sampler data")

        ## Verify we have a backend, and load it
        assert os.path.isfile(self.save_directory+"/backend.h5"), "Backend not found, can't load chains"
        self.backend=emcee.backends.HDFBackend(self.save_directory+"/backend.h5")

        ## Load chains - build a sampler object to access the backend
        sampler=emcee.EnsembleSampler(self.backend.shape[0],
                                        self.backend.shape[1],
                                        self.like.log_prob_and_blobs,
                                        backend=self.backend)

        self.burnin_nsteps=config["burn_in"]

        self.chain=sampler.get_chain(flat=True,discard=self.burnin_nsteps)
        self.lnprob=sampler.get_log_prob(flat=True,discard=self.burnin_nsteps)
        self.blobs=sampler.get_blobs(flat=True,discard=self.burnin_nsteps)

        self.ndim=len(self.like.free_params)
        self.nwalkers=config["nwalkers"]
        self.burnin_nsteps=config["burn_in"]
        self.autocorr=np.asarray(config["autocorr"])

        return


    def _setup_chain_folder(self,rootdir=None,subfolder=None):
        """ Set up a directory to save files for this
        sampler run """

        if rootdir:
            chain_location=rootdir
        else:
            assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
            chain_location=os.environ['LYA_EMU_REPO']+"/lace/sampler/chains/"
        if subfolder:
            ## If there is one, check if it exists
            ## if not, make it
            if not os.path.isdir(chain_location+subfolder):
                os.mkdir(chain_location+"/"+subfolder)
            base_string=chain_location+"/"+subfolder+"/chain_"
        else:
            base_string=chain_location+"/chain_"
        
        ## Create a new folder for this chain
        chain_count=1
        sampler_directory=base_string+str(chain_count)
        while os.path.isdir(sampler_directory):
            chain_count+=1
            sampler_directory=base_string+str(chain_count)

        os.mkdir(sampler_directory)
        print("Made directory: ", sampler_directory)
        self.save_directory=sampler_directory

        return 


    def _write_dict_to_text(self,saveDict):
        """ Write the settings for this chain
        to a more easily readable .txt file """
        
        ## What keys don't we want to include in the info file
        dontPrint=["lnprob","flatchain","blobs","autocorr"]

        with open(self.save_directory+'/info.txt', 'w') as f:
            for item in saveDict.keys():
                if item not in dontPrint:
                    f.write("%s: %s\n" % (item,str(saveDict[item])))

        return


    def write_chain_to_file(self):
        """Write flat chain to file"""

        saveDict={}

        ## archive settings
        saveDict["basedir"]=self.like.theory.emulator.archive.basedir
        saveDict["skewers_label"]=self.like.theory.emulator.archive.skewers_label
        saveDict["p1d_label"]=self.like.theory.emulator.archive.p1d_label
        saveDict["drop_tau_rescalings"]=self.like.theory.emulator.archive.drop_tau_rescalings
        saveDict["drop_temp_rescalings"]=self.like.theory.emulator.archive.drop_temp_rescalings
        saveDict["nearest_tau"]=self.like.theory.emulator.archive.nearest_tau
        saveDict["z_max"]=self.like.theory.emulator.archive.z_max
        saveDict["undersample_cube"]=self.like.theory.emulator.archive.undersample_cube
        saveDict["kp_Mpc"]=self.like.theory.emulator.archive.kp_Mpc

        ## Emulator settings
        saveDict["paramList"]=self.like.theory.emulator.paramList
        saveDict["kmax_Mpc"]=self.like.theory.emulator.kmax_Mpc

        ## Do we train a GP on each z?
        if self.like.theory.emulator.emulators:
            z_emulator=True
            emu_hyperparams=[]
            for emu in self.like.theory.emulator.emulators:
                emu_hyperparams.append(emu.gp.param_array.tolist())
        else:
            z_emulator=False
            emu_hyperparams=self.like.theory.emulator.gp.param_array.tolist()
        saveDict["z_emulator"]=z_emulator

        ## Is this an asymmetric, rbf-only emulator?
        if self.like.theory.emulator.asymmetric_kernel and self.like.theory.emulator.rbf_only:
            saveDict["asym_kernel"]=True
        else:
            saveDict["asym_kernel"]=False

        saveDict["emu_hyperparameters"]=emu_hyperparams
        saveDict["emu_type"]=self.like.theory.emulator.emu_type
        saveDict["reduce_var"]=self.like.theory.emulator.reduce_var_mf

        ## Likelihood & data settings
        saveDict["prior_Gauss_rms"]=self.like.prior_Gauss_rms
        saveDict["z_list"]=self.like.theory.zs.tolist()
        saveDict["emu_cov_factor"]=self.like.emu_cov_factor
        saveDict["data_basedir"]=self.like.data.basedir
        saveDict["data_sim_number"]=self.like.data.sim_label
        saveDict["data_cov_factor"]=self.like.data.data_cov_factor
        saveDict["data_year"]=self.like.data.data_cov_label
        saveDict["include_CMB"]=self.like.include_CMB
        saveDict["use_compression"]=self.like.use_compression
        saveDict["reduced_IGM"]=self.like.reduced_IGM

        ## If we are sampling primordial power, save the pivot scale
        ## used to define As, ns
        if hasattr(self.like.theory,"true_camb_model"):
            pivot_scalar=self.like.theory.true_camb_model.cosmo.InitPower.pivot_scalar
        else:
            pivot_scalar=0.05
        saveDict["pivot_scalar"]=pivot_scalar

        free_params_save=[]
        free_param_limits=[]
        for par in self.like.free_params:
            ## The parameter limits are saved twice but for the sake
            ## of backwards compatibility I'm going to leave this
            free_params_save.append([par.name,par.min_value,par.max_value])
            free_param_limits.append([par.min_value,par.max_value])
        saveDict["free_params"]=free_params_save
        saveDict["free_param_limits"]=free_param_limits

        ## Sampler stuff
        saveDict["burn_in"]=self.burnin_nsteps
        saveDict["nwalkers"]=self.nwalkers
        saveDict["autocorr"]=self.autocorr.tolist()

        ## Save dictionary to json file in the
        ## appropriate directory
        if self.save_directory is None:
            self._setup_chain_folder()
        with open(self.save_directory+"/config.json", "w") as json_file:
            json.dump(saveDict,json_file)

        self._write_dict_to_text(saveDict)

        ## Save plots
        ## Using try as have latex issues when running on compute
        ## nodes on some clusters
        try:
            self.plot_best_fit()
        except:
            print("Can't plot best fit")
        try:
            self.plot_prediction()
        except:
            print("Can't plot prediction")
        try:
            self.plot_autocorrelation_time()
        except:
            print("Can't plot autocorrelation time")
        try:
            self.plot_corner()
        except:
            print("Can't plot corner")

        return


    def plot_histograms(self,cube=False):
        """Make histograms for all dimensions, using re-normalized values if
            cube=True"""

        # get chain (from sampler or from file)
        chain,lnprob,blobs=self.get_chain()
        plt.figure()

        for ip in range(self.ndim):
            param=self.like.free_params[ip]
            if cube:
                values=chain[:,ip]
                title=param.name+' in cube'
            else:
                cube_values=chain[:,ip]
                values=param.value_from_cube(cube_values)
                title=param.name

            plt.hist(values, 100, color="k", histtype="step")
            plt.title(title)
            plt.show()

        return


    def plot_corner(self,plot_params=None,cmb_prior=False):
        """ Make corner plot in ChainConsumer
         - plot_params: Pass a list of parameters to plot (in LaTeX form),
                        or leave as None to
                        plot all (including derived) """

        c=ChainConsumer()

        params_plot, strings_plot=self.get_all_params()

        if cmb_prior==True:
            mean_cmb = self.like.cmb_like.true_values
            data_cmb = self.like.cmb_like.return_CMB_only()
            c.add_chain(data_cmb,parameters=self.like.cmb_like.param_list,
                                name="CMB Likelihood")
        
        c.add_chain(params_plot,parameters=strings_plot,name="Chains")

        c.configure(diagonal_tick_labels=False, tick_font_size=10,
                    label_font_size=25, max_ticks=4)

        ## Decide which parameters to plot
        if plot_params==None:
            ## Plot all parameters
            params_to_plot=strings_plot
        else:
            ## Plot params passed as argument
            params_to_plot=plot_params


        if cmb_prior==True:
            ## Only plot the parameters that are varied in the chain
            ## not priors for parameters that aren't being varied
            plot_param_strings=[]
            for par in self.like.get_free_parameter_list():
                plot_param_strings.append(param_dict[par])
            fig = c.plotter.plot(figsize=(12,12),
                    parameters=plot_param_strings,truth=self.truth)
        else:
            fig = c.plotter.plot(figsize=(12,12),
                    parameters=params_to_plot,truth=self.truth)

        if self.save_directory is not None:
            fig.savefig(self.save_directory+"/corner.pdf")
        
        else:
            fig.show()

        return


    def plot_best_fit(self):

        """ Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        chain,lnprob,blobs=self.get_chain()
        plt.figure()
        mean_value=[]
        for parameter_distribution in np.swapaxes(chain,0,1):
            mean_value.append(np.mean(parameter_distribution))
        print("Mean values:", mean_value)
        plt.title("MCMC best fit")
        self.like.plot_p1d(values=mean_value)

        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/best_fit.pdf")
        else:
            plt.show()

        return


    def plot_prediction(self):

        """ Plot the P1D of the data and the emulator prediction
        for the fiducial model """

        plt.figure()
        plt.title("Fiducial model")
        self.like.plot_p1d(values=None)
        
        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/fiducial.pdf")
        else:
            plt.show()

        return


## Dictionary to convert likelihood parameters into latex strings
param_dict={
            "Delta2_p":"$\Delta^2_p$",
            "mF":"$F$",
            "gamma":"$\gamma$",
            "sigT_Mpc":"$\sigma_T$",
            "kF_Mpc":"$k_F$",
            "n_p":"$n_p$",
            "Delta2_star":"$\Delta^2_\star$",
            "n_star":"$n_\star$",
            "alpha_star":"$\\alpha_\star$",
            "g_star":"$g_\star$",
            "f_star":"$f_\star$",
            "ln_tau_0":"$\mathrm{ln}\,\\tau_0$",
            "ln_tau_1":"$\mathrm{ln}\,\\tau_1$",
            "ln_sigT_kms_0":"$\mathrm{ln}\,\sigma^T_0$",
            "ln_sigT_kms_1":"$\mathrm{ln}\,\sigma^T_1$",
            "ln_gamma_0":"$\mathrm{ln}\,\gamma_0$",
            "ln_gamma_1":"$\mathrm{ln}\,\gamma_1$",
            "ln_kF_0":"$\mathrm{ln}\,k^F_0$",
            "ln_kF_1":"$\mathrm{ln}\,k^F_1$",
            "H0":"$H_0$",
            "mnu":"$\Sigma m_\\nu$",
            "As":"$A_s$",
            "ns":"$n_s$",
            "ombh2":"$\omega_b$",
            "omch2":"$\omega_c$",
            "cosmomc_theta":"$\\theta_{MC}$"
            }


## List of all possibly free cosmology params for the truth array
## for chainconsumer plots
cosmo_params=["Delta2_star","n_star","alpha_star",
                "f_star","g_star","cosmomc_theta",
                "H0","mnu","As","ns","ombh2","omch2"]

## list of strings for blobs
blob_strings=["$\Delta^2_\star$","$n_\star$","$f_\star$","$g_\star$","$\\alpha_\star$","$H_0$"]

def compare_corners(chain_files,labels,plot_params=None,save_string=None,
                    rootdir=None,subfolder=None):
    """ Function to take a list of chain files and overplot the chains
    Pass a list of chain files (ints) and a list of labels (strings)
     - plot_params: list of parameters (in code variables, not latex form)
                    to plot if only a subset is desired
     - save_string: to save the plot. Must include
                    file extension (i.e. .pdf, .png etc) """
    
    assert len(chain_files)==len(labels)
    
    truth_dict={}
    c=ChainConsumer()
    
    ## Add each chain we want to plot
    for aa,chain_file in enumerate(chain_files):
        sampler=EmceeSampler(read_chain_file=chain_file,
                                subfolder=subfolder,rootdir=rootdir)
        params,strings=sampler.get_all_params()
        c.add_chain(params,parameters=strings,name=labels[aa])
        
        ## Do not check whether truth results are the same for now
        ## Take the longest truth dictionary for disjoint chains
        if len(sampler.truth)>len(truth_dict):
            truth_dict=sampler.truth
    
    c.configure(diagonal_tick_labels=False, tick_font_size=15,
                label_font_size=25, max_ticks=4)
    if plot_params==None:
        fig = c.plotter.plot(figsize=(15,15),truth=truth_dict)
    else:
        ## From plot_param list, build list of parameter
        ## strings to plot
        plot_param_strings=[]
        for par in plot_params:
            plot_param_strings.append(param_dict[par])
        fig = c.plotter.plot(figsize=(10,10),
                parameters=plot_param_strings,truth=truth_dict)
    if save_string:
        fig.savefig("%s" % save_string)
    fig.show()

    return
