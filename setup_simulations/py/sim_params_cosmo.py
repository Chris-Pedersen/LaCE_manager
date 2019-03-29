import os
import numpy as np
import camb_cosmo
import fit_linP


def cosmo_from_sim_params(param_space,sim_params,linP_model_fid,
			verbose=False):
    """Given list of simulation parameters, and fiducial cosmology, 
		return target cosmology.
        Input:
            - param_space: dictionary describing the parameter space.
            - sim_params: array with values for each parameter in this sim.
            - linP_model_fid: linear power model in fiducial cosmology. 
        Output:
            - cosmo_sim: CAMB object for the target cosmology."""
    
    # get pivot points in linear power parameterization
    z_star = linP_model_fid.z_star
    assert linP_model_fid.k_units is 'Mpc', 'linP_model not in Mpc units'
    kp_Mpc = linP_model_fid.kp

    # translate Omega_star to Omega_0 (in target cosmology)
    ip_Om_star=param_space['Om_star']['ip']
    Om_star=sim_params[ip_Om_star]
    # translate Omega_star to Omega_0 (assumes flat LCDM)
    z3=(1+z_star)**3
    Om=Om_star/(z3+Om_star-Om_star*z3)
    # get parameters from fiducial cosmology
    Obh2=linP_model_fid.cosmo.ombh2
    Och2=linP_model_fid.cosmo.omch2
    Omh2=Obh2+Och2
    h=np.sqrt(Omh2/Om)
    if verbose:
        print('Omega_m_star =', Om_star)
        print('Omega_m =', Om)
        print('h =', h)
    
    # get temporary cosmology to tune primordial power spectrum
    cosmo_temp=camb_cosmo.get_cosmology(H0=100.0*h)
    # get linear power parameters, in comoving units
    linP_model_temp=fit_linP.LinearPowerModel(cosmo_temp,z_star,'Mpc',kp_Mpc)
    Delta2_star_temp=linP_model_temp.get_Delta2_star()
    n_star_temp=linP_model_temp.get_n_star()
    alpha_star_temp=linP_model_temp.get_alpha_star()
    lnA_star_temp=np.log(2*np.pi**2*Delta2_star_temp/kp_Mpc**3)
    
    # difference in linear power at kp between target and fiducial cosmology
	# (once they have the same transfer function)
    ip_Delta2_star=param_space['Delta2_star']['ip']
    Delta2_star=sim_params[ip_Delta2_star]
    lnA_star=np.log(Delta2_star*(2*np.pi**2)/kp_Mpc**3)
    delta_lnA_star=lnA_star-lnA_star_temp
    # slope
    ip_n_star=param_space['n_star']['ip']
    n_star=sim_params[ip_n_star]
    delta_n_star=n_star-n_star_temp
    # running
    if 'alpha_star' in param_space:
        ip_alpha_star=param_space['alpha_star']['ip']
        alpha_star=sim_params[ip_alpha_star]
        delta_alpha_star=alpha_star-alpha_star_temp
    else:
        delta_alpha_star=0.0
    if verbose:
        print('delta_lnA_star =',delta_lnA_star)
        print('delta_n_star =',delta_n_star)
        print('delta_alpha_star =',delta_alpha_star)
    
    # transform differences into differences at kp_CMB
    kCMB=linP_model_fid.cosmo.InitPower.pivot_scalar
    # ratio of pivot points
    ln_kCMB_p=np.log(kCMB/kp_Mpc)
    delta_nrun=delta_alpha_star
    delta_ns=delta_n_star+delta_alpha_star*ln_kCMB_p
    delta_lnAs=delta_lnA_star+delta_n_star*ln_kCMB_p+0.5*delta_alpha_star*ln_kCMB_p**2
    if verbose:
        print('delta_lnAs =',delta_lnAs)
        print('delta_ns =',delta_ns)
        print('delta_nrun =',delta_nrun)
        
    # compute primordial power for target cosmology
    lnAs=np.log(cosmo_temp.InitPower.As)+delta_lnAs
    As=np.exp(lnAs)
    ns=cosmo_temp.InitPower.ns+delta_ns
    nrun=cosmo_temp.InitPower.nrun+delta_nrun
    if verbose:
        print('As =',np.exp(lnAs))
        print('ns =',ns)
        print('nrun =',nrun)
        
    # setup simulation cosmology object
    cosmo_sim=camb_cosmo.get_cosmology(H0=100.0*h,As=As,ns=ns,nrun=nrun)
    if verbose:
        camb_cosmo.print_info(cosmo_sim,simulation=True)

    return cosmo_sim
