import numpy as np
import os
import camb
import camb_cosmo

def get_linP_zs_Mpc(cosmo,zs,kp_Mpc):
    """For each redshift, fit linear power parameters (in Mpc)"""

    linP_zs=[]
    for z in zs:
        pars=parameterize_linP_Mpc(cosmo,z,kp_Mpc=kp_Mpc,include_f_p=True)
        linP_z={'f_p':pars['f_p'], 'Delta2_p':pars['Delta2_p'],
                    'n_p':pars['n_p'], 'alpha_p':pars['alpha_p']}
        linP_zs.append(linP_z)
    return linP_zs


def compute_gz(cosmo,z):
    """ Compute logarithmic derivative of Hubble expansion, normalized to EdS:
        g(z) = dln H(z) / dln(1+z)^3/2 = 2/3 (1+z)/H(z) dH/dz """

    results = camb.get_results(cosmo)
    # compute derivative of Hubble
    dz=z/100.0
    z_minus=z-dz
    z_plus=z+dz
    H_minus=results.hubble_parameter(z=z_minus)
    H_plus=results.hubble_parameter(z=z_plus)
    dHdz=(H_plus-H_minus)/(z_plus-z_minus)
    # compute hubble at z, and return g(z)
    Hz=results.hubble_parameter(z=z)
    gz=dHdz/Hz*(1+z)*2/3
    return gz


def compute_fz(cosmo,z,kp_Mpc=1.0):
    """Given cosmology, compute logarithmic growth rate (f) at z, around
        pivot point k_p (in 1/Mpc):
        f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz """

    # will compute derivative of linear power at z
    dz=z/100.0
    zs=[z+dz,z,z-dz]
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,zs)
    z_minus=zs_out[0]
    z=zs_out[1]
    z_plus=zs_out[2]
    P_minus=P_Mpc[0]
    Pz=P_Mpc[1]
    P_plus=P_Mpc[2]
    dPdz=(P_plus-P_minus)/(z_plus-z_minus)
    # compute logarithmic growth rate
    fz_k = -0.5*dPdz/Pz*(1+z)
    # compute mean around k_p
    mask=(k_Mpc > 0.5*kp_Mpc) & (k_Mpc < 2.0*kp_Mpc)
    fz = np.mean(fz_k[mask])
    return fz


def fit_polynomial(xmin,xmax,x,y,deg=2):
    """ Fit a polynomial on the log of the function, within range"""
    x_fit= (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly=np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


def fit_linP_Mpc(cosmo,z,kp_Mpc,deg=2):
    """Given input cosmology, compute linear power at z (in Mpc)
        and fit polynomial around kp_Mpc"""
    k_Mpc, _, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,[z])
    # specify wavenumber range to fit
    kmin_Mpc = 0.5*kp_Mpc
    kmax_Mpc = 2.0*kp_Mpc
    # fit polynomial of log power over wavenumber range 
    P_fit=fit_polynomial(kmin_Mpc/kp_Mpc,kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,
            P_Mpc[0],deg=deg)
    return P_fit


def fit_linP_kms(cosmo,z,kp_kms,deg=2):
    """Given input cosmology, compute linear power at (in km/s)
        and fit polynomial around kp_kms"""
    k_kms, _, P_kms = camb_cosmo.get_linP_kms(cosmo,[z])
    # specify wavenumber range to fit
    kmin_kms = 0.5*kp_kms
    kmax_kms = 2.0*kp_kms
    # compute ratio
    P_fit=fit_polynomial(kmin_kms/kp_kms,kmax_kms/kp_kms,k_kms/kp_kms,
            P_kms,deg=deg)
    return P_fit


def parameterize_linP_Mpc(cosmo,z,kp_Mpc,include_f_p=False):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z and wavenumbers kp (in Mpc)."""

    # compute linear power, in Mpc, at z
    # and fit a second order polynomial to the log power, around kp_Mpc
    linP_Mpc = fit_linP_Mpc(cosmo,z,kp_Mpc,deg=2)
    # translate the polynomial to our parameters
    ln_A_p = linP_Mpc[0]
    Delta2_p = np.exp(ln_A_p)*kp_Mpc**3/(2*np.pi**2)
    n_p = linP_Mpc[1]
    # note that the curvature is alpha/2
    alpha_p = 2.0*linP_Mpc[2]

    results={'Delta2_p':Delta2_p,'n_p':n_p,'alpha_p':alpha_p}

    if include_f_p:
        # get logarithmic growth rate at z around kp_Mpc
        f_p = compute_fz(cosmo,z=z,kp_Mpc=kp_Mpc)
        results['f_p']=f_p

    return results


def parameterize_cosmology_kms(cosmo,z_star,kp_kms):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z_star and wavenumbers kp_kms."""

    # compute linear power, in km/s, at z_star
    # and fit a second order polynomial to the log power, around kp_kms
    linP_kms = fit_linP_kms(cosmo,z_star,kp_kms,deg=2)
    # translate the polynomial to our parameters
    ln_A_star = linP_kms[0]
    Delta2_star = np.exp(ln_A_star)*kp_kms**3/(2*np.pi**2)
    n_star = linP_kms[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0*linP_kms[2]

    # get logarithmic growth rate at z_star, around kp_Mpc
    # the exact value here should not matter, f(k) is very flat here
    kp_Mpc=kp_kms*camb_cosmo.dkms_dMpc(cosmo,z_star)
    f_star = compute_fz(cosmo,z=z_star,kp_Mpc=kp_Mpc)

    # compute deviation from EdS expansion
    g_star = compute_gz(cosmo,z=z_star)

    results={'f_star':f_star,'g_star':g_star,'linP_kms':linP_kms,
            'Delta2_star':Delta2_star,'n_star':n_star,'alpha_star':alpha_star}

    return results

