# adapted from github.com/keirkwame/GenPk/filtering_length.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import fake_spectra.abstractsnapshot as absn
import read_gadget


def power_spectrum_model(k, A, n, kF):
    return A * (k ** n) * np.exp(-1. * ((k /kF) ** 2))


def fit_filtering_length(simdir, kmax_Mpc=None,verbose=False,
                genpk_full_path='/home/dc-font1/Codes/GenPK_Keir/gen-pk',
                write_json=True,show_plots=False,store_plots=False):
    """For each snapshot, measure "real flux" power and fit filtering length"""

    # figure out snapshots in simulation
    paramfile=simdir+'/paramfile.gadget'
    if verbose: print('read GADGET config file',paramfile)
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    # will store measured "real flux" power here
    genpkdir=simdir+'/genpk/'
    os.makedirs(genpkdir,exist_ok=True)

    # snapshot outputs are here
    outdir=simdir+'/output/'

    # store filtering lengths here
    kF_Mpc=[]

    for num in range(Nsnap):
        z=zs[num]
        snap=absn.AbstractSnapshotFactory(num,outdir,Tscale=1.0,gammascale=1.0)
        # normalized Hubble parameter h ~ 0.7)
        hubble = snap.get_header_attr("HubbleParam")
        # box size in kpc/h
        L_hkpc= snap.get_header_attr("BoxSize")
        L_Mpc=L_hkpc/1000.0/hubble

        snap_tag=str(num).rjust(3,'0')
        genpk_filename=genpkdir+'/PK-by-PART_'+snap_tag
        if os.path.exists(genpk_filename):
            if verbose: print('read pre-computed power from',genpk_filename)
        else:
            snap_dir=os.path.join(outdir,"PART_"+snap_tag)
            info_file=genpkdir+'/info_genpk_'+snap_tag
            cmd=genpk_full_path+' -i '+snap_dir+' -o '+genpkdir
            cmd+=' > '+info_file
            if verbose: print('run genpk, cmd =',cmd)
            os.system(cmd)

        # fit filtering length from measured "real flux" power
        genpk_file = np.loadtxt(genpk_filename)
        k_box_units = genpk_file[:, 0]
        # dimensionless power spectrum (P*k**3)
        Pk3 = ((k_box_units * 1.) ** 3) * genpk_file[:, 1]
        # normalize wavenumbers
        k_0_Mpc = 2. * np.pi / L_Mpc
        k_Mpc = k_box_units * k_0_Mpc
        #P_Mpc = Pk3 / k_Mpc**3

        # fit wavenumbers below this one
        if not kmax_Mpc:
            kmax_Mpc = 120. * k_0_Mpc 
            if verbose: print('use kmax_Mpc =',kmax_Mpc)
        # get optimised parameters and covariance
        params_bounds = (np.array([-np.inf, -np.inf, 0.]), np.inf)
        opt_params, params_cov = spo.curve_fit(power_spectrum_model, 
                    k_Mpc[k_Mpc<kmax_Mpc], Pk3[k_Mpc<kmax_Mpc], 
                    bounds=params_bounds)

        if verbose:
            print('A = %e ; n = %e ; k_F = %.4f 1/Mpc'%tuple(opt_params))
            print('Paramater covariance =', params_cov)

        fit_kF_Mpc=opt_params[2]
        kF_Mpc.append(fit_kF_Mpc)

        if store_plots or show_plots:
            plt.figure()
            plt.plot(k_Mpc, Pk3, label=r'Simulation')
            mask=(k_Mpc < kmax_Mpc)
            Pk_model=power_spectrum_model(k_Mpc[mask],*opt_params)
            plt.plot(k_Mpc[mask], Pk_model, ls='--',
                    label=r'Best-fit model, $k_F$ = %.2f 1/Mpc'%fit_kF_Mpc)
            plt.axvline(x = kmax_Mpc, ls=':', color='black')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim([np.min(Pk3) / 5., np.max(Pk3) * 5.])
            plt.legend()
            plt.xlabel(r'k [1 / Mpc]')
            plt.ylabel(r'Dimensionless power')
            plt.title(r'Real-space flux, z = %.3f'%z)
            if store_plots:
                genpk_plot=genpkdir+'kF_'+snap_tag+'.png'
                plt.savefig(genpk_plot)
    if show_plots: 
        plt.show()

    # store information in json file
    json_filename=simdir+'/filtering_length.json'
    kF_data = {'simdir':simdir, 'kF_Mpc':kF_Mpc, 'kmax_Mpc':kmax_Mpc}
    kF_data['zs']=zs.tolist()

    json_file = open(json_filename,"w")
    json.dump(kF_data,json_file)
    json_file.close()

    return

