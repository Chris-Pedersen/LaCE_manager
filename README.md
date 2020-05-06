# LyaCosmoParams
Notes on cosmological parameters and Lyman alpha simulations

It now contains different folders:
 - notes: LaTeX file with discussions on parameterizations
 - playground: several (disorganized) Jupyter notebooks to play around
 - setup_simulations: python code to setup grid of hydro simulations
 - lya_cosmo: interface with CAMB, and basic cosmology functions
 - lya_nuisance: modelling of nuisance astrophysics (mean flux, temperature...)
 - user_interface: example of CosmoMC module for the marginalized likelihood

Dependencies:
The following modules are required:

`numpy`

`scipy`

`matplotlib`

`emcee` version 3.0.2 (not earlier ones, they are significantly different apparently)

`tqdm` to work with emcee progress bar

`corner`

`CAMB` version 1.0 or later (Jan 2019) https://github.com/cmbant/CAMB

`GPy`

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
`T0_1`
`T0_2`
`T0_3`
`ln_gamma_0`
`ln_gamma_1`
`ln_kF_0`
`ln_kF_1`

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
An example script can be found in `lya_sampler/scripts/multi_sampler.py` with a corresponding config file `example.config`. The syntax to run is the following: `python3 multi_sampler.py -c example.config`. This script will create a new folder in `lya_sampler/chains/`, and store everything related to the sampler run there.

The prior volume is defined in `free_param_limits` in `multi_sampler.py`, where the list of the parameter limits must be the same as passed in `free_params`. If a Gaussian prior is chosen, the code is currently set up to centre the Gaussian prior around the truth in the chosen test simulation for the cosmology parameters, and the truth in the fiducial simulation for the IGM parameters.

The procedure of `multi_sampler.py` is as follows:
1. Set up a `P1D_MPGADGET` data object. This reads the P1D from a selected test simulation and converts it into velocity units. Currently we are using the BOSS covariance matrices, z and k bins from https://arxiv.org/abs/1306.5896. Here we have the option to rescale the data covariance matrices by a uniform factor.
2. Set up a training set for the Gaussian process emulator using `ArxivP1D`. Here we have the option to undersample simulations (using `undersample_cube=2` to take 50% of the simulations, for example) or include postprocessing rescalings.
3. Create an emulator object. Here we have several options to use different kernels (asymmetric, RBF only etc), use an indepedent GP at each redshift, or perform some mappings of the training data (`reduce_var` options). This script might be unstable experimenting with these options, so I suggest just using the default arguments with `train=False`, and then running `GPEmulator.load_default()` to load the standard hyperparameters.
4. The `P1D_MPGADGET` and `GPEmulator` objects are then passed to a `Likelihood` object which contains our log_prob function. Here we set which free parameters we want to vary, define the prior volume, and chose priors. Set `prior_Gauss_rms=-1` to use a uniform prior. Any other value will set the 1-sigma width of the prior in unit volume. Currently the same prior width is used for all parameters. There is also an option here to rescale the contribution of the emulator uncertainty to likliehood evalutions, using `emu_cov_factor` (default=1).
5. The `Likelihood` object is then passed to the `EmceeSampler`. NB that when running in parallel, it is best to set `OMP_NUM_THREADS=1`, and `multiprocessing.pool` will automatically find the number of cores available on a given node and parallelise appropriately.
6. Running the sampler with `force_steps=True` will force the sampler to run for whatever step number is set in the config file. Otherwise the default behaviour is to run until autocorrelation time convergence is reached, and the `nsteps` in the config file is just a ceiling.
