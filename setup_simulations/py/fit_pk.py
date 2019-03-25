import os
import numpy as np
import camb
import camb_cosmo

def get_g_star(pars,z_star):
    """ Compute logarithmic derivative of Hubble expansion, normalized to EdS:
        g(z) = dln H(z) / dln(1+z)^3/2 = 3/2 (1+z)/H(z) dH/dz """
    results = camb.get_results(pars)
    # compute derivative of Hubble
    dz=z_star/100.0
    z_minus=z_star-dz
    z_plus=z_star+dz
    H_minus=results.hubble_parameter(z=z_minus)
    H_plus=results.hubble_parameter(z=z_plus)
    dHdz=(H_plus-H_minus)/(z_plus-z_minus)
    # compute hubble at z_star, and return g(z_star)
    H_star=results.hubble_parameter(z=z_star)
    g_star=dHdz/H_star*(1+z_star)*2/3
    return g_star


def get_f_star(pars,z_star=3.0,k_p_hMpc=1.0):
    """Given cosmology, compute logarithmic growth rate (f) at z_star, around
        pivot point k_p (in h/Mpc):
        f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz """
    # will compute derivative of linear power at z_star
    dz=z_star/100.0
    zs=[z_star+dz,z_star,z_star-dz]
    k_hMpc, zs_out, P_hMpc = camb_cosmo.get_linP_hMpc(pars,zs)
    z_minus=zs_out[0]
    z_star=zs_out[1]
    z_plus=zs_out[2]
    P_minus=P_hMpc[0]
    P_star=P_hMpc[1]
    P_plus=P_hMpc[2]
    dPdz=(P_plus-P_minus)/(z_plus-z_minus)
    # compute logarithmic growth rate
    f_star_k = -0.5*dPdz/P_star*(1+z_star)
    # compute mean around k_p
    mask=(k_hMpc > 0.5*k_p_hMpc) & (k_hMpc < 2.0*k_p_hMpc)
    f_star = np.mean(f_star_k[mask])
    return f_star


def fit_linP_Mpc(pars,z_star,kp_Mpc,deg=2):
    """Given input cosmology, compute linear power at z_star (in Mpc)
        and fit polynomial around kp_Mpc"""
    k_Mpc, _, P_Mpc = camb_cosmo.get_linP_Mpc(pars,[z_star])
    # specify wavenumber range to fit
    kmin_Mpc = 0.5*kp_Mpc
    kmax_Mpc = 2.0*kp_Mpc
    # compute ratio
    P_fit=fit_polynomial(kmin_Mpc/kp_Mpc,kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,
            P_Mpc[0],deg=deg)
    return P_fit


def fit_polynomial(xmin,xmax,x,y,deg=2):
    """ Fit a polynomial on the log of the function, within range"""
    x_fit= (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly=np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


def parameterize_cosmology_Mpc(pars,z_star=3,kp_Mpc=0.7):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z_star and wavenumbers kp (in Mpc)."""
    # get logarithmic growth rate at z_star, around k_p_hMpc
    k_p_hMpc=1.0
    f_star = get_f_star(pars,z_star=z_star,k_p_hMpc=k_p_hMpc)
    # compute deviation from EdS expansion
    g_star = get_g_star(pars,z_star=z_star)
    # compute linear power, in Mpc, at z_star
    # and fit a second order polynomial to the log power, around kp_Mpc
    linP_Mpc = fit_linP_Mpc(pars,z_star,kp_Mpc,deg=2)
    results={'f_star':f_star, 'g_star':g_star, 'linP_Mpc':linP_Mpc}
    return results


def Om_from_Om_star(Om_star,z_star=3):
    z3=(1+z_star)**3
    Om=Om_star/(z3+Om_star-Om_star*z3)
    return Om
