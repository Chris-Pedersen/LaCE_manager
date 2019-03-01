import os
import numpy as np
import camb

def get_cosmology(h=0.67, mnu=0.06, omch2=0.12, ombh2=0.022, TCMB=2.7255, 
            As=2.1e-09, ns=0.96):
    """Given set of cosmological parameters, return CAMB cosmology object."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0*h, ombh2=ombh2, omch2=omch2, omk=0, mnu=mnu)
    pars.InitPower.set_params(As=As, ns=ns)
    return pars


def print_info(pars):
    """Given CAMB cosmology object, print relevant parameters"""
    print('h =',pars.H0/100.0,'; Omega_b h^2 =',pars.ombh2,
          '; Omega_c h^2 =',pars.omch2,'; Omega_k =',pars.omk,
          '; ommnuh2 =',int(1e5*pars.omnuh2)/1.e5,
          '; A_s =',pars.InitPower.As,'; n_s =',pars.InitPower.ns)
    return


def get_Plin_hMpc(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of h/Mpc"""
    # kmax here sets the maximum k computed in transfer function (in 1/Mpc)
    pars.set_matter_power(redshifts=zs, kmax=30.0)
    results = camb.get_results(pars)
    # fluid here specifies species we are interested in (8=CDM+baryons)
    fluid=8
    # maxkh and npoints here refer to points where we want to compute the power, in h/Mpc
    kh, zs_out, Ph = results.get_matter_power_spectrum(var1=fluid,var2=fluid,npoints=5000,maxkh=20)
    return kh, zs_out, Ph


def get_Plin_Mpc(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc"""
    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_Plin_hMpc(pars,zs)
    # translate to Mpc
    h = pars.H0 / 100.0
    k_Mpc = k_hMpc * h
    P_Mpc = P_hMpc / h**3
    return k_Mpc, zs_out, P_Mpc


def dkms_dhMpc(cosmo,z):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return rouhgly 100.
    Inputs:
        - cosmo: dictionary with information about cosmological model.
        - z: redshift
    """
    # For now assume only open LCDM universes 
    h=cosmo['hubble']/100.0
    Om_m=(cosmo['omch2']+cosmo['ombh2']+cosmo['omnuh2'])/h**2
    Om_L=1.0-Om_m
    # H(z) = H0 * E(z)
    Ez = np.sqrt(Om_m*(1+z)**3+Om_L)
    # dv / hdX = 100 E(Z)/(1+z)
    dvdX=100*Ez/(1+z)
    return dvdX


def fit_power(k_kms,P_kms,k_p_kms=0.009,alpha_p=None):
    """Fits small scales power spectrum using a power law with running.
    Inputs: 
        - k_kms: wavenumbers in velocity units
        - P_kms: power spectrum in velocity units
        - k_p_kms: pivot wavenumber, in velocity units
        - alpha_p: use fixed value for 2nd derivative. 
            If set to None, it will fit for it.
    Outputs:
        - A_p: amplitude of linear power at k_p
        - n_p: log derivative at k_p
        - alpha_p: second log derivative at k_p 
    """
    # start by defining region that we want to use for the fit
    mask = (k_kms > 0.5*k_p_kms) & (k_kms < 2.0*k_p_kms)

    # start by computing log P, and log k/kp
    y = np.log(P_kms[mask])
    x = np.log(k_kms[mask]/k_p_kms)

    # at this point, y = A + n*x + 1/2*alpha*x*x

    # check if 2nd derivative is fixed
    if alpha_p is not None:
        # subtract 2nd derivative term from log power
        #print('using fixed alpha_p = %f'%alpha_p)
        y -= 0.5*alpha_p*x**2
        # fit only for a straight line
        poly=np.polyfit(x,y,deg=1)
        n_p = poly[0]
        A_p = np.exp(poly[1])
        # we want to report Delta_L^2 = A_p * k_p^3 / (2*pi^2)
        DL2_p = k_p_kms**3 * A_p / (2*np.pi**2)
        return DL2_p, n_p
    else:
        # fit a 2nd order polynomial
        poly=np.polyfit(x,y,deg=2)
        alpha_p = poly[0]
        n_p = poly[1]
        A_p = np.exp(poly[2])
        # we want to report Delta_L^2 = A_p * k_p^3 / (2*pi^2)
        DL2_p = k_p_kms**3 * A_p / (2*np.pi**2)
        return DL2_p, n_p, alpha_p

