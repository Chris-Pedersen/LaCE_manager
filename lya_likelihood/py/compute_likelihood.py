import numpy as np
import thermal_model


def emulate_p1d(z,k_Mpc,emu,dkms_dMpc,mf_model,T_model,linP_Mpc_params,
            extra_info=False):
    """Emulate 1D power given model and redshift"""

    # get emulator parameters for linear power, at this redshift (in Mpc)
    model=linP_Mpc_params
    # get emulator parameters for nuisance models, at this redshift
    model['mF']=mf_model.get_mean_flux(z)
    model['gamma']=T_model.get_gamma(z)
    T0=T_model.get_T0(z)
    sigT_kms=thermal_model.thermal_broadening_kms(T0)
    model['sigT_Mpc']=sigT_kms/dkms_dMpc
    emu_p1d=emu.emulate_p1d_Mpc(model,k_Mpc)
    if extra_info:
        nearest=emu.find_nearest_model(model)
        return emu_p1d, nearest
    else:
        return emu_p1d


def get_chi2(data,cosmo_fid,emu,rec_cosmo,mf_model,T_model,
            linP_Mpc_params=None,extra_info=False,verbose=False):
    """Compute chi2 given data, fiducial cosmology, emulator, 
        reconstructed cosmology, nuisance params and linear power params."""

    # get all redshifts measured
    zs=data.z

    # check if linear power parameters have been cached
    if linP_Mpc_params is None:
        if verbose:
            print('compute linear power parameters')
        linP_Mpc_params=rec_cosmo.get_linP_Mpc_params(zs)
    
    Nz=len(zs)
    chi2=0

    if extra_info:
        all_chi2=[]
        all_nearest=[]

    for iz in range(Nz):
        # acess data for this redshift
        z=zs[iz]
        if verbose: print('compute chi2 for z={}'.format(z))
        # get conversion from Mpc to km/s
        dkms_dMpc=rec_cosmo.reconstruct_Hubble(z)/(1+z)
        # get data
        p1d=data.get_Pk_iz(iz)
        cov=data.get_cov_iz(iz)
        # figure out wavenumbers in Mpc
        k_kms=data.k
        k_Mpc=k_kms * dkms_dMpc
        if extra_info:
            emu_p1d_Mpc, nearest = emulate_p1d(z,k_Mpc,emu,dkms_dMpc,
                    mf_model,T_model,linP_Mpc_params[iz],extra_info)
        else:
            emu_p1d_Mpc = emulate_p1d(z,k_Mpc,emu,dkms_dMpc,
                    mf_model,T_model,linP_Mpc_params[iz],extra_info)
        if verbose: print('emulated power')
        # translate to km/s
        emu_p1d_kms = emu_p1d_Mpc * dkms_dMpc
        # compute chi2 for this redshift bin
        icov = np.linalg.inv(cov)
        diff = (p1d-emu_p1d_kms)
        chi2_z = np.dot(np.dot(icov,diff),diff)
        if extra_info:
            all_chi2.append(chi2_z)
            all_nearest.append(nearest)
        chi2 += chi2_z
        if verbose: print('added {} to chi2'.format(chi2_z))
        
    if extra_info:
        return chi2, all_chi2, all_nearest
    else:
        return chi2
