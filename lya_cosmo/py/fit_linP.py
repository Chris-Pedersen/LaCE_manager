import numpy as np
import os
import camb
import camb_cosmo

# no need to go beyond this k_Mpc when fitting linear power only
camb_fit_kmax_Mpc=1.5

def get_linP_Mpc_zs(cosmo,zs,kp_Mpc,include_f_p=True,use_camb_fz=False):
    """For each redshift, fit linear power parameters around kp_Mpc.
        - include_f_p to compute logarithmic groth rate at each z (slow)
        - use_camb_fz will use faster code, but not at kp_Mpc """

    # run slowest part of CAMB computation, to avoid repetition
    camb_results=camb_cosmo.get_camb_results(cosmo,zs,
            kmax_Mpc=camb_fit_kmax_Mpc)

    # compute linear power at all zs
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,zs,
            camb_results=camb_results,kmax_Mpc=camb_fit_kmax_Mpc)

    # if asked for, compute also logarithmic growth rate
    use_camb_fz=False
    if use_camb_fz and include_f_p:
        # fast function (already has results in hand)
        fz=camb_cosmo.get_f_of_z(cosmo,zs,camb_results)

    # specify wavenumber range to fit
    kmin_Mpc = 0.5*kp_Mpc
    kmax_Mpc = 2.0*kp_Mpc

    # loop over all redshifts, and collect linP parameters
    linP_zs=[]
    for iz in range(len(zs)):
        # check that redshifts are properly sorted
        z=zs[iz]
        assert z==zs_out[iz],'redshifts not sorted out correctly'
        # fit polynomial of log power over wavenumber range
        linP_Mpc=fit_polynomial(kmin_Mpc/kp_Mpc,kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,
                P_Mpc[iz])
        # translate the polynomial to our parameters
        ln_A_p = linP_Mpc[0]
        Delta2_p = np.exp(ln_A_p)*kp_Mpc**3/(2*np.pi**2)
        n_p = linP_Mpc[1]
        # note that the curvature is alpha/2
        alpha_p = 2.0*linP_Mpc[2]

        linP_z={'Delta2_p':Delta2_p,'n_p':n_p,'alpha_p':alpha_p}

        if include_f_p:
            # compute logarithmic growth rate at each z
            if use_camb_fz:
                # new version, faster but uses f = f sigma_8 / sigma_8
                f_p = fz[iz]
            else:
                # older version, slower but can ask for particular kp_Mpc
                f_p = compute_fz(cosmo,z=z,kp_Mpc=kp_Mpc)
            linP_z['f_p']=f_p

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


def compute_fz(cosmo,z,kp_Mpc):
    """Given cosmology, compute logarithmic growth rate (f) at z, around
        pivot point k_p (in 1/Mpc):
        f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz """

    # will compute derivative of linear power at z
    dz=z/100.0
    zs=[z+dz,z,z-dz]
    k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,zs,
            kmax_Mpc=camb_fit_kmax_Mpc)
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


def fit_linP_Mpc(cosmo,z,kp_Mpc,deg=2,camb_results=None):
    """Given input cosmology, compute linear power at z (in Mpc)
        and fit polynomial around kp_Mpc.
        - camb_results optional to avoid calling get_results."""

    k_Mpc, _, P_Mpc = camb_cosmo.get_linP_Mpc(cosmo,[z],
            camb_results=camb_results,kmax_Mpc=camb_fit_kmax_Mpc)
    # specify wavenumber range to fit
    kmin_Mpc = 0.5*kp_Mpc
    kmax_Mpc = 2.0*kp_Mpc
    # fit polynomial of log power over wavenumber range 
    P_fit=fit_polynomial(kmin_Mpc/kp_Mpc,kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,
            P_Mpc[0],deg=deg)
    return P_fit


def fit_linP_kms(cosmo,z,kp_kms,deg=2,camb_results=None):
    """Given input cosmology, compute linear power at z
        (in km/s) and fit polynomial around kp_kms.
        - camb_results optional to avoid calling get_results. """

    k_kms, _, P_kms = camb_cosmo.get_linP_kms(cosmo,[z],
            camb_results=camb_results,kmax_Mpc=camb_fit_kmax_Mpc)
    # specify wavenumber range to fit
    kmin_kms = 0.5*kp_kms
    kmax_kms = 2.0*kp_kms
    # compute ratio
    P_fit=fit_polynomial(kmin_kms/kp_kms,kmax_kms/kp_kms,k_kms/kp_kms,
            P_kms,deg=deg)
    return P_fit


def parameterize_cosmology_kms(cosmo,z_star,kp_kms,use_camb_fz=False):
    """Given input cosmology, compute set of parameters that describe 
        the linear power around z_star and wavenumbers kp_kms.
        If use_camb_fz will get f from f sigma_8 / sigma_8."""

    # call get_results first, to avoid calling it twice
    zs=[z_star]
    camb_results = camb_cosmo.get_camb_results(cosmo,zs=zs,
            kmax_Mpc=camb_fit_kmax_Mpc)

    # compute linear power, in km/s, at z_star
    # and fit a second order polynomial to the log power, around kp_kms
    linP_kms = fit_linP_kms(cosmo,z_star,kp_kms,deg=2,camb_results=camb_results)

    # translate the polynomial to our parameters
    ln_A_star = linP_kms[0]
    Delta2_star = np.exp(ln_A_star)*kp_kms**3/(2*np.pi**2)
    n_star = linP_kms[1]
    # note that the curvature is alpha/2
    alpha_star = 2.0*linP_kms[2]

    # get logarithmic growth rate at z_star, around kp_Mpc
    if use_camb_fz:
        # new code, compute from f sigma_8 / sigma_8 at z
        f_of_z=camb_cosmo.get_f_of_z(cosmo,zs,camb_results)
        f_star=f_of_z[0]
    else:
        # old code, compute derivative of power at kp_Mpc
        kp_Mpc=kp_kms*camb_cosmo.dkms_dMpc(cosmo,z_star,camb_results)
        f_star = compute_fz(cosmo,z=z_star,kp_Mpc=kp_Mpc)

    # compute deviation from EdS expansion
    g_star = compute_gz(cosmo,z=z_star)

    results={'f_star':f_star,'g_star':g_star,'linP_kms':linP_kms,
            'Delta2_star':Delta2_star,'n_star':n_star,'alpha_star':alpha_star}

    return results

