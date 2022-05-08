import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spt
import sys
import os
import time

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
    """ Measure 1D power spectrum of delta_F, after rescaling optical depth. """

    # obtain transmitted flux fraction, after rescaling
    F = get_transmitted_flux_fraction(skewers,scale_tau)
    mF = np.mean(F)
    delta_F = F / mF - 1.0

    # compute power spectrum of delta_F, in units of Mpc
    k_Mpc, p1d_Mpc = measure_p1d_Mpc(delta_F, L_Mpc)

    return k_Mpc, p1d_Mpc, mF


def get_box_geometry(grid,L_Mpc):
    """ Figure out description of the 3D box for input grid of skewers. """

    # get box geometry
    n_xyz=grid.shape
    n_xy=int(np.sqrt(n_xyz[0]))
    n_z=n_xyz[1]
    print('n_xy={}, n_z={}'.format(n_xy,n_z))
    d_xy = L_Mpc / n_xy
    d_z = L_Mpc / n_z

    # collect results to return
    results={'n_xy':n_xy,'n_z':n_z,'d_xy':d_xy,'d_z':d_z}

    # specify wavenumbers in box
    k_xy = np.fft.fftfreq(n_xy, d=d_xy) * 2. * np.pi
    k_z = np.fft.fftfreq(n_z, d=d_z) * 2. * np.pi
    # construct 3D grid of wavenumbers
    box_kx = k_xy[:,np.newaxis,np.newaxis]
    box_ky = k_xy[np.newaxis,:,np.newaxis]
    box_kz = k_z[np.newaxis,np.newaxis,:]
    box_k = np.sqrt(box_kx**2 + box_ky**2 + box_kz**2)
    # construct mu in two steps, without NaN warnings
    box_mu = box_kz/np.ones_like(box_k)
    box_mu[box_k>0.] /= box_k[box_k>0.]
    box_mu[box_k == 0.] = np.nan

    results['box_k']=box_k
    results['box_mu']=box_mu

    return results
    

def measure_p3d_Mpc(skewers,L_Mpc,n_k_bins=20,k_Mpc_max=20.0,n_mu_bins=16):
    """ Compute 3D power spectrum of input grid of skewers. """

    # obtain transmitted flux fraction, after rescaling
    F = get_transmitted_flux_fraction(skewers,scale_tau=1.0)
    mF = np.mean(F)
    delta_F = F / mF - 1.0

    # get information about the box geometry
    box = get_box_geometry(delta_F,L_Mpc)
    print(time.asctime(),'got box geometry')

    # collect relevant information 
    results={'z':skewers.red,'mean_flux':mF,'n_xy':box['n_xy'],
                'n_z':box['n_z'],'d_xy':box['d_xy'],'d_z':box['d_z'],
                'n_k_bins':n_k_bins,'k_Mpc_max':k_Mpc_max,'n_mu_bins':n_mu_bins}

    # get Fourier modes
    norm_fac = 1.0 / delta_F.size
    modes = np.fft.fftn(delta_F) * norm_fac
    print(time.asctime(),'got Fourier modes')

    # get raw power
    raw_p3d = (np.abs(modes) ** 2)

    # change format and get rid of first mode (k=0)
    k_box = box['box_k'].flatten()[1:]
    mu_box = np.absolute(box['box_mu']).flatten()[1:]
    p3d_box = raw_p3d.flatten()[1:]

    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_Mpc_max)
    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999*np.min(k_box[k_box > 0.]))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min,lnk_bin_max,n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)
    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    # compute bin averages (not including first mode with k=0, mu=0)
    binned_p3d = spt.binned_statistic_2d(k_box, mu_box, p3d_box,
                statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print(time.asctime(),'got binned power')
    binned_counts = spt.binned_statistic_2d(k_box,mu_box,p3d_box,
            statistic='count', bins=[k_bin_edges,mu_bin_edges])[0]
    print(time.asctime(),'got bin counts')
    # compute binned values for (k,mu)
    binned_k = spt.binned_statistic_2d(k_box, mu_box, k_box,
            statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print(time.asctime(),'got binned k')
    binned_mu = spt.binned_statistic_2d(k_box, mu_box, mu_box,
                    statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    print(time.asctime(),'got binned mu')

    # quantity above is dimensionless, multiply by box size (in Mpc)
    results['p3d_Mpc'] = binned_p3d * L_Mpc**3
    results['k_Mpc'] = binned_k
    results['mu'] = binned_mu
    results['counts'] = binned_counts

    return results


def plot_p3d(results,downsample_mu=3,savefig=None):
    """ Make simple plot for measured P3D, for a few mu bins. """

    p3d_Mpc=results['p3d_Mpc']
    k_Mpc=results['k_Mpc']
    mu=results['mu']
    counts=results['counts']
    # plot p3d in snapshot
    n_mu=results['mu'].shape[1]
    mu_bin_edges = np.linspace(0., 1.,n_mu + 1)
    cm=plt.get_cmap('hsv')
    for i in range(0,n_mu,downsample_mu):
        col=cm(i/n_mu)
        mask=~np.isnan(mu[:,i])
        P_mu=p3d_Mpc[:,i][mask]
        k_mu=k_Mpc[:,i][mask]
        error=P_mu/np.sqrt(counts[:,i][mask])
        plt.errorbar(k_mu,P_mu,yerr=error,
                capsize=3,ecolor=col,color=col,marker="x",
                label=r"%.2f $\leq \mu \leq$ %.2f" % (mu_bin_edges[i],
                                                          mu_bin_edges[i+1]))
    plt.xlabel("k (/Mpc)")
    plt.legend(loc="best",numpoints=1,fancybox=True,fontsize="small")
    plt.ylabel(r"$P_F(k,\mu)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r'P3D at z=%.2f' % results['z'])
    if savefig:
        plt.savefig(savefig+'.pdf')

