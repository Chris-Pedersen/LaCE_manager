import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import emcee
import corner
# our own modules
import simplest_emulator
import linear_emulator
import gp_emulator
import z_emulator
import data_PD2013
import mean_flux_model
import thermal_model
import lya_theory
import likelihood
import likelihood_parameter
import scipy.stats
import p1d_arxiv
import data_MPGADGET
import multiprocessing as mg
import itertools
from multiprocessing import Pool
from multiprocessing import Process


class EmceeSampler(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(self,like=None,emulator=None,free_parameters=None,
                        nwalkers=None,read_chain_file=None,verbose=False,
                        progress=False):
        """Setup sampler from likelihood, or use default.
            If read_chain_file is provided, read pre-computed chain."""

        # WE SHOULD DOCUMENT BETTER THE OPTIONAL INPUTS
        # WHEN WOULD SOMEONE PASS A LIKELIHOOD AND A LIST OF FREE PARAMETERS?
        # WOULDN'T like.free_params ALREADY CONTAIN THAT?

        # WHEN WOULD YOU LIKE TO HAVE A SAMPLER WITHOUT AN EMULATOR?

        self.verbose=verbose
        self.store_distances=False
        self.progress=progress

        if read_chain_file:
            if self.verbose: print('will read chain from file',read_chain_file)
            self.read_chain_from_file(read_chain_file)
            self.p0=None
            self.burnin_pos=None
        else: 
            if like:
                if self.verbose: print('use input likelihood')
                self.like=like
                if free_parameters:
                    self.like.set_free_parameters(free_parameters,like.free_param_limits)
            else:
                if self.verbose: print('use default likelihood')
                data=data_PD2013.P1D_PD2013(blind_data=True)
                zs=data.z
                theory=lya_theory.LyaTheory(zs,emulator=emulator)
                self.like=likelihood.Likelihood(data=data,theory=theory,
                                free_parameters=free_parameters,verbose=False)
            # number of free parameters to sample
            self.ndim=len(self.like.free_params)
            self.chain_from_file=None

            self.save_directory=None
            self._setup_chain_folder()

            # number of walkers
            if nwalkers:
                self.nwalkers=nwalkers
            else:
                self.nwalkers=10*self.ndim
            if self.verbose: print('setup with',self.nwalkers,'walkers')
            # setup sampler
            # setup walkers
            self.p0=self.get_initial_walkers()


        ## Dictionary to convert likelihood parameters into latex strings
        self.param_dict={
                        "Delta2_p":"$\Delta^2_p$",
                        "mF":"$F$",
                        "gamma":"$\gamma$",
                        "sigT_Mpc":"$\sigma_T$",
                        "kF_Mpc":"$k_F$",
                        "n_p":"$n_p$",
                        "Delta2_star":"$\Delta^2_\star$",
                        "n_star":"$n_\star$",
                        "g_star":"g_\star",
                        "f_star":"f_\star",
                        "ln_tau_0":"$ln(\tau_0)$",
                        "ln_tau_1":"$ln(\tau_1)$",
                        "ln_sigT_kms_0":"$ln(\sigma^T_0)$",
                        "ln_sigT_kms_1":"$ln(\sigma^T_1)$",
                        "ln_gamma_0":"$ln(\gamma_0)$",
                        "ln_gamma_1":"$ln(\gamma_1)$",
                        "ln_kF_0":"$ln(kF_0)$",
                        "ln_kF_1":"$ln(kF_1)$",
                        "H0":"$H_0$",
                        "As":"$A_s$",
                        "ns":"$n_s$",
                        "ombh2":"$\omega_b$",
                        "omch2":"$\omega_c$"
                        }

        ## Set up list of parameter names in tex format for plotting
        self.paramstrings=[]
        self.truth=[] ## Truth value for chainconsumer plots
        for param in self.like.free_parameters:
            self.paramstrings.append(self.param_dict[param])
        for param in self.like.free_params:
            self.truth.append(param.value)

        self.distances=[]
        for aa in range(len(self.like.data.z)):
            self.distances.append([])


    def run_sampler(self,burn_in,max_steps,log_func,parallel=False,force_steps=False):
        """ Set up sampler, run burn in, run chains,
        return chains """


        self.burnin_nsteps=burn_in
        # We'll track how the average autocorrelation time estimate changes
        self.autocorr = np.array([])
        # This will be useful to testing convergence
        old_tau = np.inf

        if parallel==False:
            ## Get initial walkers
            p0=self.get_initial_walkers()
            sampler=emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                    log_func)
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
                        break
                old_tau = tau

        else:
            p0=self.get_initial_walkers()
            with Pool() as pool:
                sampler=emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                        log_func,
                                                        pool=pool)
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
                            break
                    old_tau = tau

        ## Save chains
        self.chain=sampler.get_chain(flat=True,discard=self.burnin_nsteps)
        self.lnprob=sampler.get_log_prob(flat=True,discard=self.burnin_nsteps)

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


    def log_prob(self,values):
        """Function that will actually be called by emcee"""

        test_log_prob=self.like.log_prob(values=values)
        if np.isnan(test_log_prob):
            if self.verbose:
                print('parameter values outside hull',values)
                return -np.inf
        if self.store_distances:
            self.add_euclidean_distances(values)

        return test_log_prob        


    def add_euclidean_distances(self,values):
        """ For a given set of likelihood parameters
        find the Euclidean distances to the nearest
        training point for each emulator call """

        emu_calls=self.like.theory.get_emulator_calls(self.like.parameters_from_sampling_point(values))
        for aa,call in enumerate(emu_calls):
            self.distances[aa].append(self.like.theory.emulator.get_nearest_distance(call,z=self.like.data.z[aa]))

        return 


    def go_silent(self):
        self.verbose=False
        self.like.go_silent()


    def get_chain(self,cube=True):
        """Figure out whether chain has been read from file, or computed"""

        if not self.chain_from_file is None:
            chain=self.chain_from_file['chain']
            lnprob=self.chain_from_file['lnprob']
        else:
            chain=self.chain#.get_chain(flat=True,discard=self.burnin_nsteps)
            lnprob=self.lnprob#.get_log_prob(flat=True,discard=self.burnin_nsteps)

        if cube == False:
            cube_values=chain
            list_values=[self.like.free_params[ip].value_from_cube(
                                cube_values[:,ip]) for ip in range(self.ndim)]
            chain=np.array(list_values).transpose()

        return chain,lnprob


    def plot_autocorrelation_time(self):
        """ Plot autocorrelation time as a function of
        sample numer """
        
        plt.figure()

        n = 100 * np.arange(self.burnin_nsteps, len(self.autocorr)+1)
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


    def read_chain_from_file(self,chain_number):
        """Read chain from file, and check parameters"""
        
        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']
        self.save_directory=repo+"/lya_sampler/chains/chain_"+str(chain_number)

        with open(self.save_directory+"/config.json") as json_file:  
            config = json.load(json_file)

        if self.verbose: print("Building arxiv")
        ## Set up the arxiv
        archive=p1d_arxiv.ArxivP1D(basedir=config["basedir"],
                            drop_tau_rescalings=config["drop_tau_rescalings"],
                            drop_temp_rescalings=config["drop_temp_rescalings"],
                            nearest_tau=config["nearest_tau"],
                            z_max=config["z_max"],
                            drop_sim_number=config["data_sim_number"],
                            p1d_label=config["p1d_label"],                            
                            skewers_label=config["skewers_label"],
                            undersample_cube=config["undersample_cube"])

        if self.verbose: print("Setting up emulator")
        try:
            reduce_var=config["reduce_var"]
        except:
            reduce_var=False
        ## Set up the emulators
        if config["z_emulator"]:
            emulator=z_emulator.ZEmulator(paramList=config["paramList"],
                                train=False,
                                emu_type=config["emu_type"],
                                kmax_Mpc=config["kmax_Mpc"],
                                reduce_var_mf=reduce_var,
                                passArxiv=archive,verbose=self.verbose)
            ## Now loop over emulators, passing the saved hyperparameters
            for aa,emu in enumerate(emulator.emulators):
                ## Load emulator hyperparams..
                emu.load_hyperparams(np.asarray(config["emu_hyperparameters"][aa]))
        else:
            emulator=gp_emulator.GPEmulator(paramList=config["paramList"],
                                train=False,
                                emu_type=config["emu_type"],
                                kmax_Mpc=config["kmax_Mpc"],
                                reduce_var_mf=reduce_var,
                                passArxiv=archive,verbose=self.verbose)
            emulator.load_hyperparams(np.asarray(config["emu_hyperparameters"]))

        try:
            data_cov=config["data_cov_factor"]
        except:
            data_cov=1.


        ## Set up mock data
        data=data_MPGADGET.P1D_MPGADGET(sim_number=config["data_sim_number"],
                                    basedir=config["basedir"],
                                    skewers_label=config["skewers_label"],
                                    z_list=np.asarray(config["z_list"]),
                                    data_cov_factor=data_cov)

        if self.verbose: print("Setting up likelihood")
        ## Set up likelihood
        free_param_list=[]
        limits_list=[]
        for item in config["free_params"]:
            free_param_list.append(item[0])
            limits_list.append([item[1],item[2]])

        ## Not all saved chains will have this flag
        try:
            free_param_limits=config["free_param_limits"]
        except:
            free_param_limits=None
    
        self.like=likelihood.Likelihood(data=data,emulator=emulator,
                            free_parameters=free_param_list,
                            free_param_limits=free_param_limits,
                            verbose=False,
                            prior_Gauss_rms=config["prior_Gauss_rms"],
                            emu_cov_factor=config["emu_cov_factor"])

        if self.verbose: print("Load sampler data")
        ## Load chains
        self.chain_from_file={}
        self.chain_from_file["chain"]=np.asarray(config["flatchain"])
        self.chain_from_file["lnprob"]=np.asarray(config["lnprob"])


        print("Chain shape is ", np.shape(self.chain_from_file["chain"]))

        self.ndim=len(self.like.free_params)
        self.nwalkers=config["nwalkers"]
        self.burnin_nsteps=config["burn_in"]
        self.autocorr=np.asarray(config["autocorr"])

        return


    def _setup_chain_folder(self):
        """ Set up a directory to save files for this
        sampler run """

        repo=os.environ['LYA_EMU_REPO']
        base_string=repo+"/lya_sampler/chains/chain_"
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
        dontPrint=["lnprob","flatchain","autocorr"]

        with open(self.save_directory+'/info.txt', 'w') as f:
            for item in saveDict.keys():
                if item not in dontPrint:
                    f.write("%s: %s\n" % (item,str(saveDict[item])))

        return


    def write_chain_to_file(self):
        """Write flat chain to file"""

        saveDict={}

        ## Arxiv settings
        saveDict["basedir"]=self.like.theory.emulator.arxiv.basedir
        saveDict["skewers_label"]=self.like.theory.emulator.arxiv.skewers_label
        saveDict["p1d_label"]=self.like.theory.emulator.arxiv.p1d_label
        saveDict["drop_tau_rescalings"]=self.like.theory.emulator.arxiv.drop_tau_rescalings
        saveDict["drop_temp_rescalings"]=self.like.theory.emulator.arxiv.drop_temp_rescalings
        saveDict["nearest_tau"]=self.like.theory.emulator.arxiv.nearest_tau
        saveDict["z_max"]=self.like.theory.emulator.arxiv.z_max
        saveDict["undersample_cube"]=self.like.theory.emulator.arxiv.undersample_cube

        ## Emulator settings
        saveDict["paramList"]=self.like.theory.emulator.paramList
        saveDict["kmax_Mpc"]=self.like.theory.emulator.kmax_Mpc
        if self.like.theory.emulator.emulators:
            z_emulator=True
            emu_hyperparams=[]
            for emu in self.like.theory.emulator.emulators:
                emu_hyperparams.append(emu.gp.param_array.tolist())
        else:
            z_emulator=False
            emu_hyperparams=self.like.theory.emulator.gp.param_array.tolist()
        saveDict["z_emulator"]=z_emulator
        saveDict["emu_hyperparameters"]=emu_hyperparams
        saveDict["emu_type"]=self.like.theory.emulator.emu_type
        saveDict["reduce_var"]=self.like.theory.emulator.reduce_var_mf

        ## Likelihood & data settings
        saveDict["prior_Gauss_rms"]=self.like.prior_Gauss_rms
        saveDict["z_list"]=self.like.theory.zs.tolist()
        saveDict["emu_cov_factor"]=self.like.emu_cov_factor
        saveDict["data_basedir"]=self.like.data.basedir
        saveDict["data_sim_number"]=self.like.data.sim_number
        saveDict["data_cov_factor"]=self.like.data.data_cov_factor
        free_params_save=[]
        for par in self.like.free_params:
            free_params_save.append([par.name,par.min_value,par.max_value])
        saveDict["free_params"]=free_params_save
        saveDict["free_param_limits"]=self.like.free_param_limits

        ## Sampler stuff
        saveDict["burn_in"]=self.burnin_nsteps
        saveDict["nwalkers"]=self.nwalkers
        saveDict["lnprob"]=self.lnprob.tolist()
        saveDict["flatchain"]=self.chain.tolist()
        saveDict["autocorr"]=self.autocorr.tolist()

        ## Save dictionary to json file in the
        ## appropriate directory
        with open(self.save_directory+"/config.json", "w") as json_file:
            json.dump(saveDict,json_file)

        self._write_dict_to_text(saveDict)

        ## Save plots
        self.plot_best_fit()
        self.plot_prediction()
        self.plot_corner()
        self.plot_autocorrelation_time()

        return


    def plot_histograms(self,cube=False):
        """Make histograms for all dimensions, using re-normalized values if
            cube=True"""

        # get chain (from sampler or from file)
        chain,lnprob=self.get_chain()
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


    def plot_corner(self,cube=False,mock_values=True):
        """Make corner plot, using re-normalized values if cube=True"""

        # get chain (from sampler or from file)
        values,lnprob=self.get_chain(cube=cube)

        labels=[]
        for p in self.like.free_params:
            if cube:
                labels.append(p.name+' in cube')
            else:
                labels.append(p.name)
        figure = corner.corner(values,labels=labels,
                                hist_kwargs={"density":True,"color":"blue"})

        # Extract the axes
        axes = np.array(figure.axes).reshape((self.ndim, self.ndim))
        if mock_values==True:
            if cube:
                list_mock_values=[self.like.free_params[aa].value_in_cube() for aa in range(
                                                len(self.like.free_params))]
            else:
                list_mock_values=[self.like.free_params[aa].value for aa in range(
                                                len(self.like.free_params))]

            # Loop over the diagonal
            for i in range(self.ndim):
                ax = axes[i, i]
                ax.axvline(list_mock_values[i], color="r")
                prior=self.get_trunc_norm(self.like.free_params[i].value_in_cube(),
                                                    100000)
                if cube:
                    ax.hist(prior,bins=200,alpha=0.4,color="hotpink",density=True)
                else:
                    for aa in range(len(prior)):
                        prior[aa]=self.like.free_params[i].value_from_cube(prior[aa])
                    ax.hist(prior,bins=50,alpha=0.4,color="hotpink",density=True)

            # Loop over the histograms
            for yi in range(self.ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(list_mock_values[xi], color="r")
                    ax.axhline(list_mock_values[yi], color="r")
                    ax.plot(list_mock_values[xi], list_mock_values[yi], "sr")

        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/corner.pdf")
        else:
            plt.show()
        return


    def plot_best_fit(self):

        """ Plot the P1D of the data and the emulator prediction
        for the MCMC best fit
        """

        ## Get best fit values for each parameter
        chain,lnprob=self.get_chain()
        plt.figure()
        mean_value=[]
        for parameter_distribution in np.swapaxes(chain,0,1):
            mean_value.append(np.mean(parameter_distribution))
        print("Mean values:", mean_value)
        self.like.plot_p1d(values=mean_value)
        plt.title("MCMC best fit")
        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/best_fit.pdf")
        else:
            plt.show()

        return


    def plot_prediction(self):

        """ Plot the P1D of the data and the emulator prediction
        for the fiducial model """

        plt.figure()
        self.like.plot_p1d(values=None)
        plt.title("Fiducial model")
        if self.save_directory is not None:
            plt.savefig(self.save_directory+"/fiducial.pdf")
        else:
            plt.show()

        return
