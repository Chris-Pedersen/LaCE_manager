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


def get_linP_hMpc(pars,zs=[3],have_power=False):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of h/Mpc"""
    # kmax here sets the maximum k computed in transfer function (in 1/Mpc)
    pars.set_matter_power(redshifts=zs, kmax=30.0)
    results = camb.get_results(pars)
    # fluid here specifies species we are interested in (8=CDM+baryons)
    fluid=8
    # maxkh and npoints here refer to points where we want to compute the power, in h/Mpc
    kh, zs_out, Ph = results.get_matter_power_spectrum(var1=fluid,var2=fluid,
            npoints=5000,maxkh=20,have_power_spectra=have_power)
    return kh, zs_out, Ph


def get_linP_Mpc(pars,zs=[3],have_power=False):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc"""
    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs,have_power=have_power)
    # translate to Mpc
    h = pars.H0 / 100.0
    k_Mpc = k_hMpc * h
    P_Mpc = P_hMpc / h**3
    return k_Mpc, zs_out, P_Mpc


def get_linP_kms(pars,zs=[3],have_power=False):
    """Given a CAMB cosmology, and a set of redshifts, compute the linear
        power spectrum for CDM+baryons, in units of 1/Mpc"""
    # get linear power in units of Mpc/h
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs,have_power=have_power)

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


def get_f_star(pars,z_star=3.0,k_p_hMpc=1.0,have_power=False):
    """Given cosmology, compute logarithmic growth rate (f) at z_star, around
        pivot point k_p (in h/Mpc):
        f(z) = d lnD / d lna = - 1/2 * (1+z)/P(z) dP/dz """
    # will compute derivative of linear power at z_star
    dz=z_star/100.0
    z_minus=z_star-dz
    z_plus=z_star+dz
    zs=[z_minus,z_star,z_plus]
    k_hMpc, zs_out, P_hMpc = get_linP_hMpc(pars,zs,have_power=have_power)
    P_minus=P_hMpc[0]
    P_star=P_hMpc[1]
    P_plus=P_hMpc[2]
    dPdz=(P_plus-P_minus)/(z_plus-z_minus)
    # compute logarithmic growth rate
    f_star_k = -0.5*dPdz/P_star*(1+z_star)
    # compute mean around k_p
    mask=(k_hMpc > 0.8*k_p_hMpc) & (k_hMpc < 1.2*k_p_hMpc)
    f_star = np.mean(f_star_k[mask])
    return f_star


def fit_linP_ratio_kms(pars,pars_fid,z_star,kp_kms,deg=2,have_power=False):
    """Given two cosmologies, compute ratio of linear power at z_star,
        in units of velocity, and fit polynomial to log ratio"""
    k_kms, _, P_kms = get_linP_kms(pars,[z_star],have_power=have_power)
    k_kms_fid, _, P_kms_fid = get_linP_kms(pars_fid,[z_star],
            have_power=have_power)
    # specify wavenumber range to fit
    kmin_kms = kp_kms*0.8
    kmax_kms = kp_kms/0.8
    # compute ratio
    k_ratio=np.logspace(np.log10(kmin_kms),np.log10(kmax_kms),1000)
    P_ratio=np.interp(k_ratio,k_kms[0],P_kms[0]) \
            / np.interp(k_ratio,k_kms_fid[0],P_kms_fid[0])
    P_ratio_fit=fit_polynomial(kmin_kms/kp_kms,kmax_kms/kp_kms,k_ratio/kp_kms,
            P_ratio,deg=deg)
    return P_ratio_fit


def fit_polynomial(xmin,xmax,x,y,deg=2):
    """ Fit a polynomial on the log of the function, within range"""
    x_fit= (x > xmin) & (x < xmax)
    # We could make these less correlated by better choice of parameters
    poly=np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


def parameterize_cosmology(pars,pars_fid,z_star=3,kp_kms=0.009):
    """Given cosmology, and fiducial cosmology, compute set of parameters that 
    describe the linear power around z_star and wavenumbers kp (in km/s)."""
    # get logarithmic growth rate in both cosmologies at z_star, around k_p_hMpc
    k_p_hMpc=1.0
    f_star = get_f_star(pars,z_star=z_star,k_p_hMpc=k_p_hMpc)
    f_star_fid = get_f_star(pars_fid,z_star=z_star,k_p_hMpc=k_p_hMpc)
    # compute deviation from EdS expansion
    g_star = get_g_star(pars,z_star=z_star)
    g_star_fid = get_g_star(pars_fid,z_star=z_star)
    # compute ratio of linear power, in velocity units, at z_star
    # and fit a second order polynomial to the log ratio, around kp_kms
    linP_ratio_kms = fit_linP_ratio_kms(pars,pars_fid,z_star,kp_kms,
            deg=2,have_power=True)
    results={'df_star':f_star-f_star_fid}
    results['dg_star']=g_star-g_star_fid
    results['linP_ratio_kms']=linP_ratio_kms
    return results

def reconstruct_linP_kms(zs,k_kms,pars_fid,linP_params,z_star,kp_kms):
    """Given fiducial cosmology and linear parameters, reconstruct linear
        power spectra"""
    # get linear power and background expansion for fiducial cosmology
    results_fid = camb.get_results(pars_fid)
    H_star_fid = results_fid.hubble_parameter(z=z_star)
    k_kms_fid, zs_out, P_kms_fid = get_linP_kms(pars_fid,zs)
    # get parameters describing linear power
    df_star=linP_params['df_star']
    dg_star=linP_params['dg_star']
    linP_ratio_kms=linP_params['linP_ratio_kms']
    # will store reconstructed linear power here
    Nz=len(zs)
    Nk=len(k_kms)
    linP_kms=np.empty([Nz,Nk])
    for iz in range(Nz):
        z=zs_out[iz]
        # Hubble parameter in fiducial cosmology
        H_fid = results_fid.hubble_parameter(z=z)
        # evaluate fiducial power at slightly different wavenumber 
        x = 1 + 3/2*dg_star*(z-z_star)/(1+z_star)
        P_rec = np.interp(x*k_kms,k_kms_fid[iz],P_kms_fid[iz]) * x**3
        # apply change in shape, taking into account we work in km/s
        y_fid = H_fid/(1+z)/H_star_fid*(1+z_star)
        y=y_fid*x
        P_rec *= np.exp(linP_ratio_kms(np.log(y*k_kms/kp_kms)))
        # correct linear growth
        P_rec *= (1-df_star*(z-z_star)/(1+z_star))**2
        linP_kms[iz]=P_rec
    return linP_kms
