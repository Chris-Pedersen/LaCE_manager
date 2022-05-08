# LaCE
Lyman-alpha Cosmology Emulator. This code is a Gaussian process emulator for the 1D flux power spectrum
of the Lyman-alpha forest, and was used to generate the results shown in
https://arxiv.org/abs/2011.15127.

## Installation

1. Set environment variables: `export LACE_MANAGER_REPO=/path/to/repo/LaCE_Manager` and `export LACE_REPO=/path/to/repo/LaCE`. Best to set this in a `.bashrc` or similar.
2. Ensure the python dependencies below are installed
3. Run `git submodule init && git submodule update` in the `LaCE_manager` repo
4. `cd LaCE` and run `python3 setup.py install --user`
5. `cd ..` and run `python3 setup.py install --user`


#### Dependencies:
Python version 3.6 or later is necessary due to `CAMB` version dependencies.

The following modules are required:

`numpy`

`scipy`

`matplotlib`

`configobj`

`emcee` version 3.0.2 (not earlier ones, they are significantly different apparently)

`tqdm` to work with emcee progress bar

`corner`

`chainconsumer`

`CAMB` version 1.1.3 or later https://github.com/cmbant/CAMB (only works with Python 3.6 or later as of 14/01/2021)

`GPy` (only works with Python 3.8 or lower, not compatible with 3.9 as of 14/01/2021)

`cProfile`

To setup/run/postprocess simulations:

`configargparse`

`fake_spectra` branch at https://github.com/Chris-Pedersen/fake_spectra which includes temperature rescalings in postprocessing

`validate`

`classylss`

`asciitable`


### Parameter spaces:

#### Likelihood parameters:
These are the parameters that we will ultimately get posteriors for. For each set of likelihood parameters, N emulator calls are made, where N is the number of redshift bins in the data. The specific emulator calls to be made are determined using the `lya_theory` object, which maps between likelihood and emulator parameters.

The likelihood parameters are:

`g_star`
`f_star`
`Delta2_star`
`n_star`
`alpha_star`
`ln_tau_0`
`ln_tau_1`
`ln_sigT_kms_0`
`ln_sigT_kms_1`
`ln_gamma_0`
`ln_gamma_1`
`ln_kF_0`
`ln_kF_1`

The IGM parameters represent a rescaling of a fiducial simulation run at the centre of the Latin hypercube for that suite. We perform this rescaling using a power law, where the index "0" represents the amplitude, and "1" represents the slope with redshift.

#### Emulator parameters:
These are the parameters that describe each individual P1D(k) power spectrum. We have detached these from redshift and traditional cosmology parameters.

`sigT_Mpc`
`alpha_p`
`n_p`
`gamma`
`Delta2_p`
`mF`
`f_p`
`kF_Mpc`

### Saving and loading emulator hyperparameters
The default operation of the emulator is currently to optimise a new set of hyperparameters on whichever training set it is initialised with. However for sampler runs we suggest setting the `train=False` flag, and use GPEmulator.load_default(). This will load a standardised set of hyperparameters (along with the appropriate parameter rescalings for the X training data) that are optimised on the entire suite.

## Sampler information

### Saved sampler chains
Note that I am currently not storing sampler chains in the repo as the file sizes are too large. This means that many of the notebooks in `lya_sampler/notebooks` will not run with a fresh clone of the repo. Will need to figure out a longer term solution to this.

### Running a sampler
An example script can be found in `lya_sampler/scripts/multiprocess_sampler.py` with a corresponding config file `example.config`. The syntax to run is the following: `python3 multiprocess_sampler.py -c example.config`. This script will create a new folder in `lya_sampler/chains/`, and store everything related to the sampler run there.

The prior volume is defined in `free_param_limits` in `multiprocess_sampler.py`, where the list of the parameter limits must be the same as passed in `free_params`. If a Gaussian prior is chosen, the code is currently set up to centre the Gaussian prior around the truth in the chosen test simulation for the cosmology parameters, and the truth in the fiducial simulation for the IGM parameters.

The procedure of `multi_sampler.py` is as follows:
1. Set up a `P1D_MPGADGET` data object. This reads the P1D from a selected test simulation and converts it into velocity units. Currently we are using the BOSS covariance matrices, z and k bins from https://arxiv.org/abs/1306.5896. Here we have the option to rescale the data covariance matrices by a uniform factor.
2. Set up a training set for the Gaussian process emulator using `ArxivP1D`. Here we have the option to undersample simulations (using `undersample_cube=2` to take 50% of the simulations, for example) or include postprocessing rescalings.
3. Create an emulator object. Here we have several options to use different kernels (asymmetric, RBF only etc), use an indepedent GP at each redshift, or perform some mappings of the training data (`reduce_var` options). This script might be unstable experimenting with these options, so I suggest just using the default arguments with `train=False`, and then running `GPEmulator.load_default()` to load the standard hyperparameters.
4. The `P1D_MPGADGET` and `GPEmulator` objects are then passed to a `Likelihood` object which contains our log_prob function. Here we set which free parameters we want to vary, define the prior volume, and chose priors. Set `prior_Gauss_rms=-1` to use a uniform prior. Any other value will set the 1-sigma width of the prior in unit volume. Currently the same prior width is used for all parameters. There is also an option here to rescale the contribution of the emulator uncertainty to likliehood evalutions, using `emu_cov_factor` (default=1).
5. The `Likelihood` object is then passed to the `EmceeSampler`. NB that when running in parallel, it is best to set `OMP_NUM_THREADS=1`, and `multiprocessing.pool` will automatically find the number of cores available on a given node and parallelise appropriately.
6. Running the sampler with `force_steps=True` will force the sampler to run for whatever step number is set in the config file. Otherwise the default behaviour is to run until autocorrelation time convergence is reached, and the `nsteps` in the config file is just a ceiling.
