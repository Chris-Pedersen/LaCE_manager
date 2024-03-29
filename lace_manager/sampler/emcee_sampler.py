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
from lace_manager.data import data_MPGADGET
from lace_manager.data import data_PD2013
from lace.emulator import p1d_archive
from lace.emulator import gp_emulator
from lace_manager.emulator import p1d_archive_Nyx
from lace_manager.emulator import z_emulator
from lace_manager.likelihood import lya_theory
from lace_manager.likelihood import likelihood
from lace_manager.likelihood import marg_p1d_like


class EmceeSampler(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(self,like=None,
                        nwalkers=None,read_chain_file=None,verbose=False,
                        subfolder=None,rootdir=None,
                        save_chain=True,progress=False,
                        train_when_reading=True,
                        ignore_grid_when_reading=False):
        """Setup sampler from likelihood, or use default.
            If read_chain_file is provided, read pre-computed chain.
            rootdir allows user to search for saved chains in a different
            location to the code itself.
            If not train_when_reading, emulator can not be used when reading.
            Use ignore_grid_when_reading for plotting marginalised chains."""

        self.verbose=verbose
        self.progress=progress

        if read_chain_file:
            if self.verbose: print('will read chain from file',read_chain_file)
            assert not like, "likelihood specified but reading chain from file"
            self.read_chain_from_file(read_chain_file,rootdir,subfolder,
                        train_when_reading,ignore_grid_when_reading)
            self.burnin_pos=None
        else: 
            self.like=like
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

        ## Set up list of parameter names in tex format for plotting
        self.paramstrings=[]
        for param in self.like.free_params:
            self.paramstrings.append(param_dict[param.name])

        # when running on simulated data, we can store true cosmo values
        self.set_truth()

        # Figure out what extra information will be provided as blobs
        self.blobs_dtype = self.like.theory.get_blobs_dtype()


    def set_truth(self):
        """ Set up dictionary with true values of cosmological
        likelihood parameters for plotting purposes """

        # likelihood contains true parameters, but not in latex names
        like_truth=self.like.truth

        # store truth for all parameters, with LaTeX keywords
        self.truth={}
        for param in cosmo_params:
            param_string=param_dict[param]
            self.truth[param_string]=like_truth[param]

        return


    def run_sampler(self,burn_in,max_steps,log_func=None,
                parallel=False,timeout=None,force_timeout=False):
        """ Set up sampler, run burn in, run chains,
        return chains
            - timeout is the time in hours to run the
              sampler for
            - force_timeout will continue to run the chains
              until timeout, regardless of convergence """

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

                    ## Check if we are over time limit
                    if time.time()>time_end:
                        print("Timed out")
                        break
                    ## If not, only halt on convergence criterion if
                    ## force_timeout is false
                    if (force_timeout == False) and (converged==True):   
                        print("Chains have converged")
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


    def get_initial_walkers(self,initial=0.1):
        """Setup initial states of walkers in sensible points
           -- initial will set a range within unit volume around the
              fiducial values to initialise walkers (set to 0.5 to
              distribute across full prior volume) """


        ndim=self.ndim
        nwalkers=self.nwalkers

        if self.verbose: 
            print('set %d walkers with %d dimensions'%(nwalkers,ndim))

        if self.like.prior_Gauss_rms is None:
            p0=np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))
            p0=p0*initial+0.5
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


    def get_chain(self,cube=True,delta_lnprob_cut=None):
        """Figure out whether chain has been read from file, or computed.
            - if cube=True, return values in range [0,1]
            - if delta_lnprob_cut is set, use it to remove low-prob islands"""

        chain=self.chain#.get_chain(flat=True,discard=self.burnin_nsteps)
        lnprob=self.lnprob#.get_log_prob(flat=True,discard=self.burnin_nsteps)
        blobs=self.blobs

        if delta_lnprob_cut:
            max_lnprob=np.max(lnprob)
            cut_lnprob=max_lnprob-delta_lnprob_cut
            mask=lnprob>cut_lnprob
            # total number and masked points in chain
            nt=len(lnprob)
            nm=sum(mask)
            if self.verbose:
                print('will keep {} \ {} points from chain'.format(nm,nt))
            chain=chain[mask]
            lnprob=lnprob[mask]
            blobs=blobs[mask]

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


    def get_all_params(self,delta_lnprob_cut=None):
        """ Get a merged array of both sampled and derived parameters
            returns a 2D array of all parameters, and an ordered list of
            the LaTeX strings for each.
                - if delta_lnprob_cut is set, keep only high-prob points"""
        
        chain,lnprob,blobs=self.get_chain(cube=False,
                    delta_lnprob_cut=delta_lnprob_cut)

        if blobs is None:
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


    def read_chain_from_file(self,chain_number,rootdir,subfolder,
                train_when_reading,ignore_grid_when_reading):
        """Read chain from file, check parameters and setup likelihood"""
        
        if rootdir:
            chain_location=rootdir
        else:
            assert ('LACE_MANAGER_REPO' in os.environ),'export LACE_MANAGER_REPO'
            chain_location=os.environ['LACE_MANAGER_REPO']+"/lace_manager/sampler/chains/"
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

        if "nyx_fname" in config:
            nyx_fname=config["nyx_fname"]
            archive=p1d_archive_Nyx.archiveP1D_Nyx(fname=nyx_fname,kp_Mpc=kp)
        else:
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
                                train=train_when_reading,
                                emu_type=config["emu_type"],
                                kmax_Mpc=config["kmax_Mpc"],
                                reduce_var_mf=reduce_var,
                                passarchive=archive,verbose=self.verbose)
        else:
            emulator=gp_emulator.GPEmulator(paramList=config["paramList"],
                                train=train_when_reading,
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

        # Some chains might have pivot_scalar saved
        if "pivot_scalar" in config.keys():
            assert config["pivot_scalar"]==0.05,"non-standard pivot_scalar"

        ## Set up mock data
        data=data_MPGADGET.P1D_MPGADGET(sim_label=config["data_sim_number"],
                                    basedir=config["basedir"],
                                    skewers_label=config["skewers_label"],
                                    z_list=np.asarray(config["z_list"]),
                                    data_cov_factor=data_cov,
                                    data_cov_label=data_year,
                                    polyfit=(config["emu_type"]=="polyfit"))

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

        # read what type of fiducial cosmology we used
        try:
            cosmo_fid_label=config["cosmo_fid_label"]
        except:
            cosmo_fid_label='default'

        # figure out compression and marginalised P1D (if needed)
        try:
            use_compression=config["use_compression"]
        except:
            use_compression=0
        if use_compression==3:
            if "grid_fname" in config and not ignore_grid_when_reading:
                fname=config["grid_fname"]
                marg_p1d=marg_p1d_like.MargP1DLike(grid_fname=fname)
            else:
                try:
                    reduced_IGM=config["reduced_IGM"]
                except:
                    reduced_IGM=False
                marg_p1d=marg_p1d_like.MargP1DLike(sim_label=data.sim_label,
                                reduced_IGM=reduced_IGM,polyfit=data.polyfit)
        else:
            marg_p1d=None

        # figure out whether to set extra P1D data (from HIRES)
        if "extra_p1d_label" in config:
            extra_p1d_data=data_MPGADGET.P1D_MPGADGET(basedir=config["basedir"],
                                    sim_label=config["data_sim_number"],
                                    skewers_label=config["skewers_label"],
                                    zmax=config["extra_p1d_zmax"],
                                    data_cov_factor=1.0,
                                    data_cov_label=config["extra_p1d_label"],
                                    polyfit=(config["emu_type"]=="polyfit"))
        else:
            extra_p1d_data=None

        # figure out emulator covariance
        if "old_emu_cov" in config:
            old_emu_cov=config["old_emu_cov"]
        else:
            # old chains used old emulator covariance
            old_emu_cov=True

        # special likelihood settings
        if "prior_only" in config:
            prior_only=config["prior_only"]
        else:
            prior_only=False
        if "ignore_chi2" in config:
            ignore_chi2=config["ignore_chi2"]
        else:
            ignore_chi2=False

        # set up likelihood
        self.like=likelihood.Likelihood(data=data,emulator=emulator,
                            free_param_names=free_param_names,
                            free_param_limits=free_param_limits,
                            verbose=False,
                            prior_Gauss_rms=config["prior_Gauss_rms"],
                            emu_cov_factor=config["emu_cov_factor"],
                            old_emu_cov=old_emu_cov,
                            include_CMB=include_CMB,
                            use_compression=use_compression,
                            marg_p1d=marg_p1d,
                            cosmo_fid_label=cosmo_fid_label,
                            prior_only=prior_only,
                            ignore_chi2=ignore_chi2,
                            extra_p1d_data=extra_p1d_data)

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
        self.autocorr=np.asarray(config["autocorr"])

        return


    def _setup_chain_folder(self,rootdir=None,subfolder=None):
        """ Set up a directory to save files for this
        sampler run """

        if rootdir:
            chain_location=rootdir
        else:
            assert ('LACE_MANAGER_REPO' in os.environ),'export LACE_MANAGER_REPO'
            chain_location=os.environ['LACE_MANAGER_REPO']+"/lace_manager/sampler/chains/"
        if subfolder:
            ## If there is one, check if it exists
            ## if not, make it
            if not os.path.isdir(chain_location+"/"+subfolder):
                os.mkdir(chain_location+"/"+subfolder)
            base_string=chain_location+"/"+subfolder+"/chain_"
        else:
            base_string=chain_location+"/chain_"
        
        ## Create a new folder for this chain
        chain_count=1
        while True:
            sampler_directory=base_string+str(chain_count)
            if os.path.isdir(sampler_directory):
                chain_count+=1
                continue
            else:
                try:
                    os.mkdir(sampler_directory)
                    print('Created directory:',sampler_directory)
                    break
                except FileExistsError:
                    print('Race condition for:',sampler_directory)
                    # try again after one mili-second
                    time.sleep(0.001)
                    chain_count+=1
                    continue
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


    def write_kde(self,delta_lnprob_cut=50,N=40):
        """Compute KDE for (Delta2_star,n_star) and save to file.
            - delta_lnprob_cut: reject low-probability points
            - N: number of points in 2D grid for KDE. """

        fname='{}/kde.npz'.format(self.save_directory)
        if self.verbose: print('will print KDE to',fname)

        # get chain points and probabilities
        chain,lnprob,blobs=self.get_chain(cube=False,
                delta_lnprob_cut=delta_lnprob_cut)
        # read compressed parameters and set range
        x=blobs['Delta2_star']
        y=blobs['n_star']
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        if self.verbose:
            print('{:.3f} < Delta2_star < {:.3f}'.format(xmin,xmax))
            print('{:.3f} < n_star < {:.3f}'.format(ymin,ymax))
        if self.verbose:
            # maximum likelihood points
            max_lnprob=np.max(lnprob)
            imax=np.where(lnprob==max_lnprob)
            max_like_D2_star=x[imax][0]
            max_like_n_star=y[imax][0]
            print('Delta2_star (max like) = {:.3f}'.format(max_like_D2_star))
            print('n_star (max like) = {:.3f}'.format(max_like_n_star))
        # setup regular 2D grid for KDE
        Nj=complex(0,N)
        X, Y = np.mgrid[xmin:xmax:Nj, ymin:ymax:Nj]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = scipy.stats.gaussian_kde(values,bw_method=None)
        Z = np.reshape(kernel(positions).T, X.shape)
        # store to file
        np.savez(fname,Delta2_star=np.unique(X),n_star=np.unique(Y),density=Z)

        return


    def get_best_fit(self,delta_lnprob_cut=None):
        """ Return an array of best fit values (mean) from the MCMC chain,
            in unit likelihood space.
                - if delta_lnprob_cut is set, use only high-prob points"""

        chain,lnprob,blobs=self.get_chain(delta_lnprob_cut=delta_lnprob_cut)
        mean_values=[]
        for parameter_distribution in np.swapaxes(chain,0,1):
            mean_values.append(np.mean(parameter_distribution))

        return mean_values


    def write_chain_to_file(self,residuals=False,plot_nersc=False,
                plot_delta_lnprob_cut=None):
        """Write flat chain to file"""

        saveDict={}

        # identify Nyx archives
        if hasattr(self.like.theory.emulator.archive,"fname"):
            saveDict["nyx_fname"]=self.like.theory.emulator.archive.fname
        else:
            saveDict["basedir"]=self.like.theory.emulator.archive.basedir
            saveDict["skewers_label"]=self.like.theory.emulator.archive.skewers_label
            saveDict["p1d_label"]=self.like.theory.emulator.archive.p1d_label
            saveDict["drop_tau_rescalings"]=self.like.theory.emulator.archive.drop_tau_rescalings
            saveDict["drop_temp_rescalings"]=self.like.theory.emulator.archive.drop_temp_rescalings
            saveDict["nearest_tau"]=self.like.theory.emulator.archive.nearest_tau
            saveDict["z_max"]=self.like.theory.emulator.archive.z_max
            saveDict["undersample_cube"]=self.like.theory.emulator.archive.undersample_cube

        # Emulator settings
        saveDict["kp_Mpc"]=self.like.theory.emulator.archive.kp_Mpc
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
        saveDict["old_emu_cov"]=self.like.old_emu_cov
        saveDict["data_basedir"]=self.like.data.basedir
        saveDict["data_sim_number"]=self.like.data.sim_label
        saveDict["data_cov_factor"]=self.like.data.data_cov_factor
        saveDict["data_year"]=self.like.data.data_cov_label
        saveDict["include_CMB"]=self.like.include_CMB
        saveDict["prior_only"]=self.like.prior_only
        saveDict["ignore_chi2"]=self.like.ignore_chi2
        saveDict["cosmo_fid_label"]=self.like.cosmo_fid_label
        saveDict["use_compression"]=self.like.use_compression
        if self.like.use_compression==3:
            if self.like.marg_p1d.grid_fname:
                saveDict["grid_fname"]=self.like.marg_p1d.grid_fname
            else:
                saveDict["reduced_IGM"]=self.like.marg_p1d.reduced_IGM

        # Add information about the extra-p1d data (high-resolution P1D)
        if self.like.extra_p1d_like:
            extra_p1d_data=self.like.extra_p1d_like.data
            saveDict["extra_p1d_label"]=extra_p1d_data.data_cov_label
            saveDict["extra_p1d_zmax"]=max(extra_p1d_data.z)
        else:
            print("did not have extra P1D likelihood")

        # Make sure (As,ns,nrun) were defined in standard pivot_scalar
        if hasattr(self.like.theory,"cosmo_model_fid"):
            cosmo_fid=self.like.theory.cosmo_model_fid.cosmo
            pivot_scalar=cosmo_fid.InitPower.pivot_scalar
            assert pivot_scalar==0.05,"non-standard pivot_scalar"

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

        # save config info in plain text as well
        self._write_dict_to_text(saveDict)

        # save KDE for (Delta2_star,n_star), to use as marginalised P1D
        self.write_kde()

        ## Save plots
        ## Using try as have latex issues when running on compute
        ## nodes on some clusters
        try:
            self.plot_best_fit(residuals=residuals,
                    delta_lnprob_cut=plot_delta_lnprob_cut)
        except:
            print("Can't plot best fit")
        try:
            self.plot_prediction(residuals=residuals)
        except:
            print("Can't plot prediction")
        try:
            self.plot_autocorrelation_time()
        except:
            print("Can't plot autocorrelation time")
        try:
            self.plot_corner(usetex=(not plot_nersc),serif=(not plot_nersc),
                    delta_lnprob_cut=plot_delta_lnprob_cut)
        except:
            print("Can't plot corner")

        return


    def plot_histograms(self,cube=False,delta_lnprob_cut=None):
        """Make histograms for all dimensions, using re-normalized values if
            cube=True
            - if delta_lnprob_cut is set, use only high-prob points"""

        # get chain (from sampler or from file)
        chain,lnprob,blobs=self.get_chain(delta_lnprob_cut=delta_lnprob_cut)
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


    def plot_corner(self,plot_params=None,cmb_prior=False,
                delta_lnprob_cut=None,usetex=True,serif=True):
        """ Make corner plot in ChainConsumer
            - plot_params: Pass a list of parameters to plot (in LaTeX form),
                        or leave as None to
                        plot all (including derived)
            - if delta_lnprob_cut is set, keep only high-prob points"""

        c=ChainConsumer()

        params_plot, strings_plot=self.get_all_params(
                                            delta_lnprob_cut=delta_lnprob_cut)

        if cmb_prior==True:
            mean_cmb = self.like.cmb_like.true_values
            data_cmb = self.like.cmb_like.return_CMB_only()
            c.add_chain(data_cmb,parameters=self.like.cmb_like.param_list,
                                name="CMB Likelihood")
        
        c.add_chain(params_plot,parameters=strings_plot,name="Chains")

        c.configure(diagonal_tick_labels=False, tick_font_size=10,
                    label_font_size=25, max_ticks=4,
                    usetex=usetex, serif=serif)

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


    def plot_best_fit(self,figsize=(8,6),plot_every_iz=1,
                residuals=False,delta_lnprob_cut=None):

        """ Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        mean_value=self.get_best_fit(delta_lnprob_cut=delta_lnprob_cut)
        print("Mean values:", mean_value)
        
        plt.figure(figsize=figsize)
        plt.title("MCMC best fit")
        self.like.plot_p1d(values=mean_value,
                plot_every_iz=plot_every_iz,residuals=residuals)

        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/best_fit.pdf")
        else:
            plt.show()

        return


    def plot_prediction(self,figsize=(8,6),values=None,plot_every_iz=1,
                residuals=False):

        """ Plot the P1D of the data and the emulator prediction
        for the fiducial model """

        plt.figure(figsize=figsize)
        if values == None:
            plt.title("Fiducial model")
        else:
            plt.title("P1D at %s" % values )
        self.like.plot_p1d(values=values,
                plot_every_iz=plot_every_iz,residuals=residuals)
        
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
            "nrun":"$n_\mathrm{run}$",
            "ombh2":"$\omega_b$",
            "omch2":"$\omega_c$",
            "cosmomc_theta":"$\\theta_{MC}$"
            }


## List of all possibly free cosmology params for the truth array
## for chainconsumer plots
cosmo_params=["Delta2_star","n_star","alpha_star",
                "f_star","g_star","cosmomc_theta",
                "H0","mnu","As","ns","nrun","ombh2","omch2"]

## list of strings for blobs
blob_strings=["$\Delta^2_\star$","$n_\star$","$f_\star$","$g_\star$","$\\alpha_\star$","$H_0$"]


def compare_corners(chain_files,labels,plot_params=None,save_string=None,
                    rootdir=None,subfolder=None,delta_lnprob_cut=None,
                    usetex=True,serif=True):
    """ Function to take a list of chain files and overplot the chains
    Pass a list of chain files (ints) and a list of labels (strings)
     - plot_params: list of parameters (in code variables, not latex form)
                    to plot if only a subset is desired
     - save_string: to save the plot. Must include
                    file extension (i.e. .pdf, .png etc)
     - if delta_lnprob_cut is set, keep only high-prob points"""
    
    assert len(chain_files)==len(labels)
    
    truth_dict={}
    c=ChainConsumer()
    
    ## Add each chain we want to plot
    for aa,chain_file in enumerate(chain_files):
        sampler=EmceeSampler(read_chain_file=chain_file,
                                subfolder=subfolder,rootdir=rootdir,
                                train_when_reading=False)
        params,strings=sampler.get_all_params(delta_lnprob_cut=delta_lnprob_cut)
        c.add_chain(params,parameters=strings,name=labels[aa])
        
        ## Do not check whether truth results are the same for now
        ## Take the longest truth dictionary for disjoint chains
        if len(sampler.truth)>len(truth_dict):
            truth_dict=sampler.truth
    
    c.configure(diagonal_tick_labels=False, tick_font_size=15,
                label_font_size=25, max_ticks=4,
                usetex=usetex, serif=serif)

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
