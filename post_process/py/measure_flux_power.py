import numpy as np
import sys
import os

def get_transmitted_flux_fraction(skewers,scale_tau):
    """ Read optical depth from skewers object, and rescale it with scale_tau"""

    tau = skewers.get_tau(elem='H', ion=1, line=1215)
    return np.exp(-scale_tau*tau)


def measure_p1d_Mpc(field,L_Mpc):
    """ Compute 1D power spectrum for input array, using L_Mpc to normalize. """

    # get dimensions of input array
    (nspec, npix) = np.shape(field)

    # get 1D Fourier modes for each skewer
    fourier = np.fft.rfft(field, axis=1)

    # compute amplitude of Fourier modes
    power_skewer = np.abs(fourier)**2    

    # compute mean of power in all spectra
    mean_power = np.sum(power_skewer, axis=0)/nspec
    assert np.shape(mean_power) == (npix//2+1,), 'wrong dimensions in p1d'

    # normalize power spectrum using cosmology convention 
    # white noise power should be P=sigma^2*dx
    p1d_Mpc = mean_power * L_Mpc / npix**2

    # get frequencies from numpy.fft
    k = np.fft.rfftfreq(npix)

    # normalize using box size (first wavenumber should be 2 pi / L_Mpc)
    k_Mpc = k * (2.0*np.pi) * npix / L_Mpc

    return k_Mpc, p1d_Mpc


def measure_F_p1D_Mpc(skewers,scale_tau,L_Mpc):
    """ Measure 1D power spectrum of F, after rescaling optical depth. """

    # obtain transmitted flux fraction, after rescaling
    F = get_transmitted_flux_fraction(skewers,scale_tau)

    # compute power spectrum of F (not delta_F), in units of Mpc
    k_Mpc, p1d_Mpc = measure_p1d_Mpc(F, L_Mpc)

    # return also mean flux (to later on compute power of delta_F)
    mF = np.mean(F)

    return k_Mpc, p1d_Mpc, mF

