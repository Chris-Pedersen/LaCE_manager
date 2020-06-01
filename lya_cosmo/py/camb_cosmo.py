"""Setup CAMB cosmology objects, and compute linear power and similar."""

import os
import numpy as np
import camb

def get_cosmology(params=None,H0=67.0, mnu=0.0, omch2=0.12, ombh2=0.022, 
            omk=0.0, TCMB=2.7255, As=2.1e-09, ns=0.965, nrun=0.0):
    """Given set of cosmological parameters, return CAMB cosmology object.
        One can either pass a dictionary (params), or a set of values for the
        cosmological parameters."""

    pars = camb.CAMBparams()
    if params is None:
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk,
                mnu=mnu,TCMB=TCMB)
        pars.InitPower.set_params(As=As, ns=ns, nrun=nrun)
    else:
        # use default values for those not provided
        cosmo_fid=get_cosmology()
        if 'H0' in params: H0=params['H0']
        else: H0=cosmo_fid.H0
        if 'ombh2' in params: ombh2=params['ombh2']
        else: ombh2=cosmo_fid.ombh2
        if 'omch2' in params: omch2=params['omch2']
        else: omch2=cosmo_fid.omch2
        if 'omk' in params: omk=params['omk']
        else: omk=cosmo_fid.omk
        if 'mnu' in params: mnu=params['mnu']
        else: mnu=cosmo_fid.omnuh2*93.14 # eq 12 in https://arxiv.org/pdf/astro-ph/0603494.pdf
        if 'TCMB' in params: TCMB=params['TCMB']
        else: TCMB=cosmo_fid.TCMB
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2,
                    omk=omk, mnu=mnu, TCMB=TCMB)

        # redo for primorial power
        if 'As' in params: As=params['As']
        else: As=cosmo_fid.InitPower.As
        if 'ns' in params: ns=params['ns']
        else: ns=cosmo_fid.InitPower.ns
        if 'nrun' in params: nrun=params['nrun']
        else: nrun=cosmo_fid.InitPower.nrun
        pars.InitPower.set_params(As=As, ns=ns, nrun=nrun)

    return pars


def print_info(pars,simulation=False):
    """Given CAMB cosmology object, print relevant parameters"""

    if simulation:
        Omh2=(pars.omch2+pars.ombh2)
        Om=Omh2/(pars.H0/100.0)**2
        print('H0 = {:.4E}, Omega_bc = {:.4E}, A_s = {:.4E}, n_s = {:.4E}, alpha_s = {:.4E}'.format(pars.H0,Om,pars.InitPower.As,pars.InitPower.ns,pars.InitPower.nrun))
    else:
        print('H0 = {:.4E}, Omega_b h^2 = {:.4E}, Omega_c h^2 = {:.4E}, Omega_k = {:.4E}, Omega_nu h^2 = {:.4E}, T_CMB = {:.4E}, A_s = {:.4E}, n_s = {:.4E}, alpha_s = {:.4E}'.format(pars.H0,pars.ombh2,pars.omch2,pars.omk,pars.omnuh2,pars.TCMB,pars.InitPower.As,pars.InitPower.ns,pars.InitPower.nrun))
    return


def get_linP_hMpc(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of h/Mpc"""

    # make sure that all models are evaluated at the same points in 1/Mpc
    h=pars.H0/100.0
    kmin_Mpc=1.e-4
    kmax_Mpc=30.0
    kmin_hMpc=kmin_Mpc/h
    kmax_hMpc=kmax_Mpc/h

    # kmax here sets the maximum k computed in transfer function (in 1/Mpc)
    pars.set_matter_power(redshifts=zs, kmax=2.0*kmax_Mpc)
    results = camb.get_results(pars)
    # fluid here specifies species we are interested in (8=CDM+baryons)
    fluid=8
    # maxkh and npoints where we want to compute the power, in h/Mpc
    kh, zs_out, Ph = results.get_matter_power_spectrum(var1=fluid,var2=fluid,
            npoints=5000,minkh=kmin_hMpc,maxkh=kmax_hMpc)
    return kh, zs_out, Ph


def get_linP_Mpc(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc"""

    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs)
    # translate to Mpc
    h = pars.H0 / 100.0
    k_Mpc = k_hMpc * h
    P_Mpc = P_hMpc / h**3
    return k_Mpc, zs_out, P_Mpc


def get_linP_kms(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of s/km"""

    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs)

    # each redshift will now have a different set of wavenumbers
    Nz=len(zs)
    Nk=len(k_hMpc)
    k_kms=np.empty([Nz,Nk])
    P_kms=np.empty([Nz,Nk])
    for iz in range(Nz):
        z = zs[iz]
        dvdX = dkms_dhMpc(pars,z)
        k_kms[iz] = k_hMpc/dvdX
        P_kms[iz] = P_hMpc[iz]*dvdX**3
    return k_kms, zs_out, P_kms


def dkms_dMpc(cosmo,z):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc). At z=3 it should return roughly 70.
    Inputs:
        - cosmo: CAMB model object.
        - z: redshift
    """

    h=cosmo.H0/100.0
    return h*dkms_dhMpc(cosmo,z)


def dkms_dhMpc(cosmo,z):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return roughly 100.
    Inputs:
        - cosmo: CAMB model object.
        - z: redshift
    """

    # Check if cosmology is non-flat
    if abs(cosmo.omk) > 1.e-10:
        results = camb.get_results(cosmo)
        H_z=results.hubble_parameter(z)
        dvdX=H_z/(1+z)/(cosmo.H0/100.0)  
        return dvdX

    # use flat cosmology
    h=cosmo.H0/100.0
    Om_m=(cosmo.omch2+cosmo.ombh2+cosmo.omnuh2)/h**2
    Om_L=1.0-Om_m
    # H(z) = H0 * E(z)
    Ez = np.sqrt(Om_m*(1+z)**3+Om_L)
    # dv / hdX = 100 E(Z)/(1+z)
    dvdX=100*Ez/(1+z)
    return dvdX

