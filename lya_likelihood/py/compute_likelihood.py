import numpy as np
import thermal_model

def get_chi2(data,theory,linP_Mpc_params=None,verbose=False):
    """Compute chi2 given data and lyman alpha theory (uses emulator). """

    # get measured bins from data
    k_kms=data.k
    zs=data.z
    Nz=len(zs)

    # ask emulator prediction for P1D in each bin
    emu_p1d = theory.get_p1d_kms(k_kms,linP_Mpc_params)
    if verbose: print('got P1D from emulator')

    # compute chi2 contribution from each redshift bin
    chi2=0

    for iz in range(Nz):
        # acess data for this redshift
        z=zs[iz]
        if verbose: print('compute chi2 for z={}'.format(z))
        # get data
        p1d=data.get_Pk_iz(iz)
        cov=data.get_cov_iz(iz)
        # compute chi2 for this redshift bin
        icov = np.linalg.inv(cov)
        diff = (p1d-emu_p1d[iz])
        chi2_z = np.dot(np.dot(icov,diff),diff)
        chi2 += chi2_z
        if verbose: print('added {} to chi2'.format(chi2_z))
        
    return chi2
