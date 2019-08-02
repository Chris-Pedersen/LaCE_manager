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

`emcee`

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
The optimised Gaussian process hyperparameters are saved alongside each sim suite in /p1d_emulator/sim_suites/.
The X (emulator parameters) and Y (P1D(k) or the polyfit coefficients) training data are rebuilt from the
ArxivP1D on the fly each time the GPEmulator class is initialised, otherwise saving and loading the emulators
becomes too disk-space intensive. The hyperparameters are stored as a .npy object.

In order to ensure that the hyperparamters are loaded for only the training
data and emulator configurations they are optimised on, a .json dictionary is written alongside each .npy
object, containing all the relevant configurations of the emulator. The savenames are simply
`saved_emulator_x.npy`, where x starts at 1. Every time a new emulator is saved, this index increments, and emulator saves
will not overwrite or duplicate.

Every time a new emulator class is initialised with train=True, before running the hyperparameter optimisation, the
code will first look through the relevant basedir for saved hyperparameters with the same configuration. It will prioritise
loading the hyperparameters rather than re-optimising. Emulators trained on anything other than the default P1DArxiv (i.e. with
`max_arxiv_size != None` cannot be saved, as the points that are dropped are randomly selected and we cannot ensure that
the training data is the same.
