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

    # WE SHOULD USE PIVOT POINT IN KM/S TO DESCRIBE SIMULATIONS
    
    # get pivot points in linear power parameterization
    z_star = linP_model_fid.z_star
    kp_Mpc = linP_model_fid.kp_Mpc

    # translate Omega_star to Omega_0 (in target cosmology)
    if 'Om_star' in param_space:
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
    else:
        h=linP_model_fid.cosmo.H0/100.0
        if verbose:
            print('h =', h)
    
    # get temporary cosmology to tune primordial power spectrum
    cosmo_temp=camb_cosmo.get_cosmology(H0=100.0*h)
    # get linear power parameters, in comoving units
    linP_model_temp=fit_linP.LinearPowerModel_Mpc(cosmo=cosmo_temp,
            z_star=z_star,kp_Mpc=kp_Mpc)
    Delta2_star_temp=linP_model_temp.get_Delta2_star()
    n_star_temp=linP_model_temp.get_n_star()
    alpha_star_temp=linP_model_temp.get_alpha_star()
    lnA_star_temp=np.log(2*np.pi**2*Delta2_star_temp/kp_Mpc**3)
    
    # difference in linear power at kp between target and fiducial cosmology
	# (once they have the same transfer function)
    if 'Delta2_star' in param_space:
        ip_Delta2_star=param_space['Delta2_star']['ip']
        Delta2_star=sim_params[ip_Delta2_star]
        lnA_star=np.log(Delta2_star*(2*np.pi**2)/kp_Mpc**3)
        delta_lnA_star=lnA_star-lnA_star_temp
    else:
        delta_lnA_star=0.0
    # slope
    if 'n_star' in param_space:
        ip_n_star=param_space['n_star']['ip']
        n_star=sim_params[ip_n_star]
        delta_n_star=n_star-n_star_temp
    else:
        delta_n_star=0.0
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


def cosmo_from_sim_params_new(param_space,sim_params,verbose=False):
    """Given list of simulation parameters, and fiducial cosmology,
		return target cosmology.
        Input:
            - param_space: SimulationParameterSpace object.
            - sim_params: array with values for each parameter in this sim.
        Output:
            - cosmo_sim: CAMB object for the target cosmology."""

    # WE SHOULD USE PIVOT POINT IN KM/S TO DESCRIBE SIMULATIONS

    # get pivot points in linear power parameterization
    z_star = param_space.z_star
    kp_Mpc = param_space.kp_Mpc

    # get temporary cosmology with correct H_0 (if varying growth)
    if 'Om_star' in param_space.params:
        # translate Omega_star to Omega_0 (in target cosmology)
        ip_Om_star=param_space.params['Om_star']['ip']
        Om_star=sim_params[ip_Om_star]
        # translate Omega_star to Omega_0 (assumes flat LCDM)
        z3=(1+z_star)**3
        Om=Om_star/(z3+Om_star-Om_star*z3)
        # get parameters from fiducial cosmology
        fid_cosmo = camb_cosmo.get_cosmology()
        Obh2=fid_cosmo.ombh2
        Och2=fid_cosmo.omch2
        Omh2=Obh2+Och2
        h=np.sqrt(Omh2/Om)
        temp_cosmo=camb_cosmo.get_cosmology(H0=100.0*h)
        if verbose:
            print('Omega_m_star =', Om_star)
            camb_cosmo.print_info(temp_cosmo)
    else:
        # use fiducial cosmology
        temp_cosmo=camb_cosmo.get_cosmology()
        h=temp_cosmo.H0/100.0
        if verbose:
            camb_cosmo.print_info(temp_cosmo)

    # get linear power parameters in temporary cosmology (comoving units)
    temp_linP=fit_linP.parameterize_linP_Mpc(temp_cosmo,z_star,kp_Mpc)
    Delta2_star_temp=temp_linP['Delta2_star']
    n_star_temp=temp_linP['n_star']
    alpha_star_temp=temp_linP['alpha_star']
    lnA_star_temp=np.log(2*np.pi**2*Delta2_star_temp/kp_Mpc**3)

    # difference in linear power at kp between target and fiducial cosmology
	# (once they have the same transfer function)
    if 'Delta2_star' in param_space.params:
        ip_Delta2_star=param_space.params['Delta2_star']['ip']
        Delta2_star=sim_params[ip_Delta2_star]
        lnA_star=np.log(Delta2_star*(2*np.pi**2)/kp_Mpc**3)
        delta_lnA_star=lnA_star-lnA_star_temp
    else:
        delta_lnA_star=0.0
    # slope
    if 'n_star' in param_space.params:
        ip_n_star=param_space.params['n_star']['ip']
        n_star=sim_params[ip_n_star]
        delta_n_star=n_star-n_star_temp
    else:
        delta_n_star=0.0
    # running
    if 'alpha_star' in param_space.params:
        ip_alpha_star=param_space.params['alpha_star']['ip']
        alpha_star=sim_params[ip_alpha_star]
        delta_alpha_star=alpha_star-alpha_star_temp
    else:
        delta_alpha_star=0.0
    if verbose:
        print('delta_lnA_star =',delta_lnA_star)
        print('delta_n_star =',delta_n_star)
        print('delta_alpha_star =',delta_alpha_star)

    # transform differences into differences at kp_CMB
    kCMB=temp_cosmo.InitPower.pivot_scalar
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
    lnAs=np.log(temp_cosmo.InitPower.As)+delta_lnAs
    As=np.exp(lnAs)
    ns=temp_cosmo.InitPower.ns+delta_ns
    nrun=temp_cosmo.InitPower.nrun+delta_nrun
    if verbose:
        print('As =',np.exp(lnAs))
        print('ns =',ns)
        print('nrun =',nrun)

    # setup simulation cosmology object
    cosmo_sim=camb_cosmo.get_cosmology(H0=100.0*h,As=As,ns=ns,nrun=nrun)
    if verbose:
        camb_cosmo.print_info(cosmo_sim,simulation=True)

    return cosmo_sim
