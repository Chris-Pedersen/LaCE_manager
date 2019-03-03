import os
import numpy as np
import camb

def get_cosmology(params=None,H0=67.0, mnu=0.06, omch2=0.12, ombh2=0.022, 
            omk=0.0, TCMB=2.7255, As=2.1e-09, ns=0.96):
    """Given set of cosmological parameters, return CAMB cosmology object.
        One can either pass a dictionary (params), or a set of values for the
        cosmological parameters."""
    pars = camb.CAMBparams()
    if params is None:
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0, 
                mnu=mnu,TCMB=TCMB)
        pars.InitPower.set_params(As=As, ns=ns)
    else:
        pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], 
                omch2=params['omch2'], omk=params['omk'], 
                mnu=params['mnu'],TCMB=params['TCMB'])      
        pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    return pars


def print_info(pars):
    """Given CAMB cosmology object, print relevant parameters"""
    print('H0 =',pars.H0,'; Omega_b h^2 =',pars.ombh2,
          '; Omega_c h^2 =',pars.omch2,'; Omega_k =',pars.omk,
          '; ommnuh2 =',int(1e5*pars.omnuh2)/1.e5,'; T_CMB =',pars.TCMB,
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
    kh, zs_out, Ph = results.get_matter_power_spectrum(var1=fluid,var2=fluid,
            npoints=5000,maxkh=20)
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


def get_Plin_kms(pars,zs=[3]):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc"""
    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_Plin_hMpc(pars,zs)

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


def dkms_dhMpc(pars,z):
    """Compute factor to translate velocity separations (in km/s) to comoving
        separations (in Mpc/h). At z=3 it should return rouhgly 100.
    Inputs:
        - cosmo: dictionary with information about cosmological model.
        - z: redshift
    """
    # For now assume only flat LCDM universes 
    if abs(pars.omk) > 1.e-10:
        raise ValueError("Non-flat cosmologies are not supported (yet)")
    h=pars.H0/100.0
    Om_m=(pars.omch2+pars.ombh2+pars.omnuh2)/h**2
    Om_L=1.0-Om_m
    # H(z) = H0 * E(z)
    Ez = np.sqrt(Om_m*(1+z)**3+Om_L)
    # dv / hdX = 100 E(Z)/(1+z)
    dvdX=100*Ez/(1+z)
    return dvdX


def fit_g_star(pars,z_star):
    """ Compute derivative of Hubble expansion, normalized to EdS"""
    results = camb.get_results(pars)
    dz=z_star/100.0
    z_minus=z_star-dz
    z_plus=z_star+dz
    H_minus=results.hubble_parameter(z=z_minus)
    H_star=results.hubble_parameter(z=z_star)
    H_plus=results.hubble_parameter(z=z_plus)
    gamma_minus=H_minus/H_star*((1+z_star)/(1+z_minus))**1.5
    gamma_plus=H_plus/H_star*((1+z_star)/(1+z_plus))**1.5
    g_star=(gamma_plus-gamma_minus)/(z_plus-z_minus)
    return g_star


def fit_f_star(pars,z_star=3.0,k_p_Mpc=1.0):
    """Given cosmology, compute logarithmic growth rate (f) at z_star, around
        pivot point k_p (in 1/Mpc)"""
    # will compute derivative around z_star
    dz=z_star/100.0
    z_minus=z_star-dz
    z_plus=z_star+dz
    zs=[z_minus,z_star,z_plus]
    k_Mpc, zs_out, P_Mpc = get_Plin_Mpc(pars,zs)
    P_minus=P_Mpc[0]
    P_star=P_Mpc[1]
    P_plus=P_Mpc[2]
    # get linear growth with respect to Einstein-de Sitter
    eta_minus=np.sqrt(P_minus/P_star)*(1+z_minus)/(1+z_star)
    eta_plus=np.sqrt(P_plus/P_star)*(1+z_plus)/(1+z_star)
    # compute derivatives of eta, to compute f_star
    deta_dz = (eta_plus-eta_minus)/(z_plus-z_minus)
    f_star = 1 - (1+z_star) * deta_dz
    # average value of f_star around k_p
    mask=(k_Mpc > 0.8*k_p_Mpc) & (k_Mpc < 1.2*k_p_Mpc)
    f_star_p = np.mean(f_star[mask])
    return f_star_p


def fit_linP_ratio_kms(pars,pars_fid,z_star,kmin_kms,kmax_kms,deg=2):
    """Given two cosmologies, compute ratio of linear power at z_star,
        in units of velocity, and fit polynomial to log ratio"""
    k_kms, _, P_kms = get_Plin_kms(pars,[z_star])
    k_kms_fid, _, P_kms_fid = get_Plin_kms(pars_fid,[z_star])
    # compute ratio
    k_ratio=np.logspace(np.log10(kmin_kms),np.log10(kmax_kms),1000)
    P_ratio=np.interp(k_ratio,k_kms[0],P_kms[0]) \
            / np.interp(k_ratio,k_kms_fid[0],P_kms_fid[0])
    P_ratio_fit=fit_polynomial(kmin_kms,kmax_kms,k_ratio,P_ratio,deg=deg)
    return P_ratio_fit


def fit_polynomial(xmin,xmax,x,y,deg=2):
    """ Fit a polynomial on the log of the function, within range"""
    x_fit= (x > xmin) & (x < xmax)
    poly=np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


