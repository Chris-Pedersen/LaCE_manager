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
import pressure_model
import lya_theory
import likelihood
import emcee_sampler
import data_MPGADGET
import z_emulator
import p1d_arxiv


sampler = emcee_sampler.EmceeSampler(read_chain_file=2)

sampler.plot_corner(mock_values=True)
hyperparam_stuff=sampler.like.emulator.emulators[0].gp.to_dict(save_data=False)



