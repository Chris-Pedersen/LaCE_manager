import numpy as np
import os
import camb
import camb_cosmo
import likelihood_parameter

class LinearPowerModel_Mpc(object):
    """Store parameters describing the linear power in comoving coordinates,
        for a given CAMB object."""


    def __init__(self,cosmo,z_star=3.0,kp_Mpc=0.69):
        """Setup model, specifying redshift and pivot point"""

        self.z_star=z_star
        self.kp_Mpc=kp_Mpc

        # parameterize cosmology and store parameters
        self.cosmo=cosmo
        self._setup_from_cosmology()

        return


    def _setup_from_cosmology(self):
        """Compute and store parameters describing the linear power."""

        self.linP_params=parameterize_linP_Mpc(self.cosmo,
                                                    self.z_star,self.kp_Mpc)
        
        return


    def get_params(self):
        """Return dictionary with parameters."""

        params={'f_star':self.get_f_star(), 'g_star':self.get_g_star(), 
                'Delta2_star':self.get_Delta2_star(), 
                'n_star':self.get_n_star(), 'alpha_star':self.get_alpha_star()}

        return params


    def get_f_star(self):
        return self.linP_params['f_star']

    def get_g_star(self):
        return self.linP_params['g_star']

    def get_Delta2_star(self):
        return self.linP_params['Delta2_star']

    def get_n_star(self):
        return self.linP_params['n_star']

    def get_alpha_star(self):
        return self.linP_params['alpha_star']

    def parameterize_z_Mpc(self,zs):
        """For each redshift, fit linear power parameters (in Mpc)"""

        # we could delete this function, and remove self.cosmo from object
        return get_linP_zs_Mpc(self.cosmo,zs,self.kp_Mpc)


def get_linP_zs_Mpc(cosmo,zs,kp_Mpc):
    """For each redshift, fit linear power parameters (in Mpc)"""

    linP_zs=[]
    for z in zs:
        pars=parameterize_linP_Mpc(cosmo,z,kp_Mpc=kp_Mpc)
        # _star is only for parameters at z_star
        linP_z={'f_p':pars['f_star'], 'Delta2_p':pars['Delta2_star'],
                    'n_p':pars['n_star'], 'alpha_p':pars['alpha_star']}
        linP_zs.append(linP_z)
    return linP_zs


def compute_g_star(cosmo,z_star):
    """ Compute logarithmic derivative of Hubble expansion, normalized to EdS:
        g(z) = dln H(z) / dln(1+z)^3/2 = 2/3 (1+z)/H(z) dH/dz """

    results = camb.get_results(cosmo)
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


def compute_f_star(cosmo,z_star=3.0,kp_Mpc=1.0):
    """Given cosmology, compute logarithmic growth rate (f) at z_star, around
        pivot point k_p (in 1/Mpc):
        f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz """

    # will compute derivative of linear power at z_star
    dz=z_star/100.0
    zs=[z_star+dz,z_star,z_star-dz]
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,zs)
    z_minus=zs_out[0]
    z_star=zs_out[1]
    z_plus=zs_out[2]
    P_minus=P_Mpc[0]
    P_star=P_Mpc[1]
    P_plus=P_Mpc[2]
    dPdz=(P_plus-P_minus)/(z_plus-z_minus)
    # compute logarithmic growth rate
    f_star_k = -0.5*dPdz/P_star*(1+z_star)
    # compute mean around k_p
    mask=(k_Mpc > 0.5*kp_Mpc) & (k_Mpc < 2.0*kp_Mpc)
    f_star = np.mean(f_star_k[mask])
    return f_star


def fit_polynomial(xmin,xmax,x,y,deg=2):
    """ Fit a polynomial on the log of the function, within range"""
    x_fit= (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly=np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


def fit_linP_Mpc(cosmo,z_star,kp_Mpc,deg=2):
    """Given input cosmology, compute linear power at z_star (in Mpc)
        and fit polynomial around kp_Mpc"""
    k_Mpc, _, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,[z_star])
    # specify wavenumber range to fit
    kmin_Mpc = 0.5*kp_Mpc
    kmax_Mpc = 2.0*kp_Mpc
    # fit polynomial of log power over wavenumber range 
    P_fit=fit_polynomial(kmin_Mpc/kp_Mpc,kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,
            P_Mpc[0],deg=deg)
    return P_fit


def fit_linP_kms(cosmo,z_star,kp_kms,deg=2):
    """Given input cosmology, compute linear power at z_star (in km/s)
        and fit polynomial around kp_kms"""
    k_kms, _, P_kms = camb_cosmo.get_linP_kms(cosmo,[z_star])
    # specify wavenumber range to fit
    kmin_kms = 0.5*kp_kms
    kmax_kms = 2.0*kp_kms
    # compute ratio
    P_fit=fit_polynomial(kmin_kms/kp_kms,kmax_kms/kp_kms,k_kms/kp_kms,
            P_kms,deg=deg)
    return P_fit


def parameterize_linP_Mpc(cosmo,z,kp_Mpc):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z and wavenumbers kp (in Mpc)."""

    # WE SHOULD NOT HAVE _star PARAMETERS with kp_Mpc 

    # WE SHOULD JUST DELETE THIS WHOLE OBJECT

    # get logarithmic growth rate at z around kp_Mpc
    f_star = compute_f_star(cosmo,z_star=z,kp_Mpc=kp_Mpc)
    # compute deviation from EdS expansion
    g_star = compute_g_star(cosmo,z_star=z)
    # compute linear power, in Mpc, at z
    # and fit a second order polynomial to the log power, around kp_Mpc
    linP_Mpc = fit_linP_Mpc(cosmo,z,kp_Mpc,deg=2)
    # translate the polynomial to our parameters
    ln_A_star = linP_Mpc[0]
    Delta2_star = np.exp(ln_A_star)*kp_Mpc**3/(2*np.pi**2)
    n_star = linP_Mpc[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0*linP_Mpc[2]
    results={'f_star':f_star, 'g_star':g_star,
            'Delta2_star':Delta2_star, 'n_star':n_star, 
            'alpha_star':alpha_star, 'linP_Mpc':linP_Mpc}
    return results


def parameterize_cosmology_kms(cosmo,z_star,kp_kms):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z_star and wavenumbers kp (in km/s)."""

    # get logarithmic growth rate at z_star, around kp_Mpc
    # the exact value here should not matter, f(k) is very flat here
    kp_Mpc=kp_kms*camb_cosmo.dkms_dMpc(cosmo,z_star)
    f_star = compute_f_star(cosmo,z_star=z_star,kp_Mpc=kp_Mpc)
    # compute deviation from EdS expansion
    g_star = compute_g_star(cosmo,z_star=z_star)
    # compute linear power, in km/s, at z_star
    # and fit a second order polynomial to the log power, around kp_kms
    linP_kms = fit_linP_kms(cosmo,z_star,kp_kms,deg=2)
    # translate the polynomial to our parameters
    ln_A_star = linP_kms[0]
    Delta2_star = np.exp(ln_A_star)*kp_kms**3/(2*np.pi**2)
    n_star = linP_kms[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0*linP_kms[2]
    results={'f_star':f_star, 'g_star':g_star,
            'Delta2_star':Delta2_star, 'n_star':n_star, 
            'alpha_star':alpha_star, 'linP_kms':linP_kms}
    return results

