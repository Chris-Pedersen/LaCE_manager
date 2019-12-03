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
import data_PD2013
import mean_flux_model
import thermal_model
import lya_theory
import likelihood
import likelihood_parameter
import scipy.stats


class EmceeSampler(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(self,like=None,emulator=None,free_parameters=None,
                        nwalkers=None,read_chain_file=None,verbose=False):
        """Setup sampler from likelihood, or use default.
            If read_chain_file is provided, read pre-computed chain."""

        self.verbose=verbose
        self.store_distances=False

        if like:
            if self.verbose: print('use input likelihood')
            self.like=like
            if free_parameters:
                self.like.set_free_parameters(free_parameters)
        else:
            if self.verbose: print('use default likelihood')
            data=data_PD2013.P1D_PD2013(blind_data=True)
            zs=data.z
            theory=lya_theory.LyaTheory(zs,emulator=emulator)
            self.like=likelihood.Likelihood(data=data,theory=theory,
                            free_parameters=free_parameters,verbose=False)

        # number of free parameters to sample
        self.ndim=len(self.like.free_params)

        if read_chain_file:
            if self.verbose: print('will read chain from file',read_chain_file)
            self.read_chain_from_file(read_chain_file)
            self.nwalkers=None
            self.sampler=None
            self.p0=None
        else:
            self.chain_from_file=None
            # number of walkers
            if nwalkers:
                self.nwalkers=nwalkers
            else:
                self.nwalkers=10*self.ndim
            if self.verbose: print('setup with',self.nwalkers,'walkers')
            # setup sampler
            self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                            self.log_prob)
            # setup walkers
            self.p0=self.get_initial_walkers()

        self.distances=[]
        for aa in range(len(self.like.data.z)):
            self.distances.append([])

        if self.verbose:
            print('done setting up sampler')
        

    def run_burn_in(self,nsteps,nprint=20):
        """Start sample from initial points, for nsteps"""

        if not self.sampler: raise ValueError('sampler not properly setup')

        if self.verbose: print('start burn-in, will do',nsteps,'steps')

        pos=self.p0
        for i,result in enumerate(self.sampler.sample(pos,iterations=nsteps)):
            pos=result[0]
            if self.verbose and (i % nprint == 0):
                print(i,np.mean(pos,axis=0))

        if self.verbose: print('finished burn-in')

        self.burnin_nsteps=nsteps
        self.burnin_pos=pos
    
        return


    def run_chains(self,nsteps,nprint=20):
        """Run actual chains, starting from end of burn-in"""

        if not self.sampler: raise ValueError('sampler not properly setup')

        # reset and run actual chains
        self.sampler.reset()

        pos=self.burnin_pos
        for i, result in enumerate(self.sampler.sample(pos,iterations=nsteps)):
            if i % nprint == 0:
                print(i,np.mean(result[0],axis=0))

        return


    def get_initial_walkers(self):
        """Setup initial states of walkers in sensible points"""

        if not self.sampler: raise ValueError('sampler not properly setup')

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


    def get_chain(self):
        """Figure out whether chain has been read from file, or computed"""

        if not self.chain_from_file is None:
            chain=self.chain_from_file['chain']
            lnprob=self.chain_from_file['lnprob']
        else:
            if not self.sampler: raise ValueError('sampler not properly setup')
            chain=self.sampler.flatchain
            lnprob=self.sampler.flatlnprobability
        return chain,lnprob


    def read_chain_from_file(self,filename):
        """Read chain from file, and check parameters"""

        # start by reading parameters from file
        input_params=[]
        param_filename=filename+".params"
        param_file=open(param_filename,'r')
        for line in param_file:
            info=line.split()
            name=info[0]
            min_value=float(info[1])
            max_value=float(info[2])
            param=likelihood_parameter.LikelihoodParameter(name=name,
                                    min_value=min_value,max_value=max_value)
            input_params.append(param)

        # check that paramters are the same
        for ip in range(self.ndim):
            par=self.like.free_params[ip]
            assert par.is_same_parameter(input_params[ip]),"wrong parameters"

        # read chain itself
        chain_filename=filename+".chain"
        chain=np.loadtxt(chain_filename,unpack=False)
        lnprob_filename=filename+".lnprob"
        lnprob=np.loadtxt(lnprob_filename,unpack=False)

        # if only 1 parameter, reshape chain to be ndarray as the others
        if len(chain.shape) == 1:
            N=len(chain)
            self.chain_from_file={'chain':chain.reshape(N,1)}
        else:
            self.chain_from_file={'chain':chain}
        self.chain_from_file['lnprob']=lnprob

        # make sure you read file with same number of parameters
        chain_shape=self.chain_from_file['chain'].shape
        assert self.ndim == chain_shape[1],"mismatch"
        assert chain_shape[0] == len(self.chain_from_file['lnprob']),"mismatch"

        return


    def write_chain_to_file(self,filename):
        """Write flat chain to file"""

        if not self.sampler: raise ValueError('sampler not properly setup')

        # start by writing parameters info to file
        param_filename=filename+".params"
        param_file = open(param_filename,'w')
        for par in self.like.free_params:
            info_str="{} {} {} \n".format(par.name,par.min_value,par.max_value)
            param_file.write(info_str)
        param_file.close()

        # now write chain itself
        chain_filename=filename+".chain"
        np.savetxt(chain_filename,self.sampler.flatchain)

        # finally write log likelihood in file
        lnprob_filename=filename+".lnprob"
        np.savetxt(lnprob_filename,self.sampler.flatlnprobability)

        return


    def plot_histograms(self,cube=False):
        """Make histograms for all dimensions, using re-normalized values if
            cube=True"""

        # get chain (from sampler or from file)
        chain,lnprob=self.get_chain()

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


    def plot_corner(self,cube=False,mock_values=False):
        """Make corner plot, using re-normalized values if cube=True"""

        # get chain (from sampler or from file)
        chain,lnprob=self.get_chain()

        labels=[]
        for p in self.like.free_params:
            if cube:
                labels.append(p.name+' in cube')
            else:
                labels.append(p.name)

        if cube:
            values=chain
        else:
            cube_values=chain
            list_values=[self.like.free_params[ip].value_from_cube(
                                cube_values[:,ip]) for ip in range(self.ndim)]
            values=np.array(list_values).transpose()

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

        plt.show()
        return

    def plot_best_fit(self):
        """ Plot the P1D of the data, the likelihood fit, and the MCMC best fit
        the likelihood fit model is fitting the likelihood parameters to the
        emulator parameters in the given simulation
        """
        ## Get best fit values for each parameter
        chain,lnprob=self.get_chain()
        mean_value=[]
        for parameter_distribution in np.swapaxes(chain,0,1):
            mean_value.append(np.mean(parameter_distribution))
        print("Mean values:", mean_value)
        self.like.plot_p1d(values=mean_value)
        return

