import numpy as np
import sys
import os
import json
import fake_spectra.tempdens as tdr
import fake_spectra.griddedspectra as grid_spec 
# our modules
import read_gadget
import camb_cosmo
import thermal_model

def get_skewers_filename(num,n_skewers,width_Mpc,scale_T0=None,
            scale_gamma=None):
    """Filename storing skewers for a particular temperature model"""

    filename='skewers_'+str(num)+'_Ns'+str(n_skewers)
    filename+='_wM'+str(int(1000*width_Mpc)/1000)
    if scale_T0:
        filename+='_sT'+str(int(1000*scale_T0)/1000)
    if scale_gamma:
        filename+='_sg'+str(int(1000*scale_gamma)/1000)
    filename+='.hdf5'
    return filename 


def get_snapshot_json_filename(num,n_skewers,width_Mpc):
    """Filename describing the set of skewers for a given snapshot"""

    filename='snap_skewers_'+str(num)+'_Ns'+str(n_skewers)
    filename+='_wM'+str(int(1000*width_Mpc)/1000)
    filename+='.json'
    return filename 


def dkms_dMpc_z(simdir,num):
    """Setup cosmology from Gadget config file, and compute dv/dX"""

    paramfile=simdir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    z=zs[num]
    # read cosmology information from Gadget file
    cosmo_info=read_gadget.camb_from_gadget(paramfile)
    # setup CAMB object
    cosmo=camb_cosmo.get_cosmology(cosmo_info)
    # convert kms to Mpc (should be around 75 km/s/Mpc at z=3)
    dkms_dMpc = camb_cosmo.dkms_dMpc(cosmo,z=z)
    return dkms_dMpc,z

 
def thermal_broadening_Mpc(T_0,dkms_dMpc):
    """Thermal broadening RMS in comoving units, given T_0"""

    sigma_T_kms=thermal_model.thermal_broadening_kms(T_0)
    sigma_T_Mpc=sigma_T_kms/dkms_dMpc
    return sigma_T_Mpc


def rescale_write_skewers_z(simdir,num,skewers_dir=None,n_skewers=50,
            width_Mpc=0.1,scales_T0=None,scales_gamma=None):
    """Extract skewers for a given snapshot, for different temperatures."""

    # don't rescale unless asked to
    if scales_T0 is None:
        scales_T0=[1.0]
    if scales_gamma is None:
        scales_gamma=[1.0]

    # make sure output directory exists (will write skewers there)
    if skewers_dir is None:
        skewers_dir=simdir+'/output/skewers/'
    if os.path.exists(skewers_dir):
        if not os.path.isdir(skewers_dir):
            raise ValueError(skewers_dir+' is not a directory')
    else:
        print('make directory',skewers_dir)
        os.mkdir(skewers_dir)

    # figure out redshift for this snapshot, and dkms/dMpc
    dkms_dMpc, z = dkms_dMpc_z(simdir,num)
    width_kms = width_Mpc * dkms_dMpc

    # figure out temperature-density before scalings
    T0_ini, gamma_ini = tdr.fit_td_rel_plot(num,simdir+'/output/',plot=False)

    sim_info={'simdir':simdir, 'skewers_dir':skewers_dir,
                'z':z, 'snap_num':num, 'n_skewers':n_skewers, 
                'width_Mpc':width_Mpc, 'width_kms':width_kms,
                'T0_ini':T0_ini, 'gamma_ini':gamma_ini,
                'scales_T0':scales_T0, 'scales_gamma':scales_gamma}

    # will also stored measured values
    sim_T0=[]
    sim_gamma=[]
    sim_sigT_Mpc=[]
    sim_mf=[]
    sk_files=[]

    for scale_T0 in scales_T0:
        for scale_gamma in scales_gamma:
            T0=T0_ini*scale_T0
            gamma=gamma_ini*scale_gamma
            sk_filename=get_skewers_filename(num,n_skewers,width_Mpc,
                                    scale_T0,scale_gamma)

            skewers=get_skewers_snapshot(simdir,skewers_dir,num,
                            n_skewers=n_skewers,width_kms=width_kms,
                            set_T0=T0,set_gamma=gamma,
                            skewers_filename=sk_filename)

            # call mean flux, so that the skewers are really computed
            mf=skewers.get_mean_flux()
            skewers.save_file()
            sim_mf.append(mf)
            # store temperature information
            sim_T0.append(T0)
            sim_gamma.append(gamma)
            sim_sigT_Mpc.append(thermal_broadening_Mpc(T0,dkms_dMpc))
            sk_files.append(sk_filename)

    sim_info['sim_T0']=sim_T0
    sim_info['sim_gamma']=sim_gamma
    sim_info['sim_mf']=sim_mf
    sim_info['sim_sigT_Mpc']=sim_sigT_Mpc
    sim_info['sk_files']=sk_files

    snapshot_filename=get_snapshot_json_filename(num,n_skewers,width_Mpc)
    sim_info['snapshot_filename']=snapshot_filename
    json_file = open(skewers_dir+'/'+snapshot_filename,"w")
    json.dump(sim_info,json_file)
    json_file.close()

    print('done')

    return sim_info


def write_default_skewers(simdir,skewers_dir=None,zmax=6.0,n_skewers=50,
            width_kms=10):
    """Extract skewers for all snapshots, default temperature."""

    # make sure output directory exists (will write skewers there)
    if skewers_dir is None:
        skewers_dir=simdir+'/output/skewers/'
    if os.path.exists(skewers_dir):
        if not os.path.isdir(skewers_dir):
            raise ValueError(skewers_dir+' is not a directory')
    else:
        print('make directory',skewers_dir)
        os.mkdir(skewers_dir)

    # figure out number of snapshots and redshifts
    paramfile=simdir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    # will store information to write to JSON file later
    mf_snap=[]
    mf_z=[]
    mf_val=[]
    sk_files=[]

    # loop over snapshots and extract skewers
    for num in range(Nsnap):
        z=zs[num]
        if z < zmax:
            # extract skewers
            skewers=get_skewers_snapshot(simdir,skewers_dir,num,
                        n_skewers=n_skewers,width_kms=width_kms)
            # call mean flux, so that the skewers are really computed
            mf=skewers.get_mean_flux()
            # figure out filename for skewers
            set_skewers_filename(simdir,skewers)
            skewers.save_file()
            mf_snap.append(num)
            mf_z.append(z)
            mf_val.append(mf)
            sk_files.append(skewers.savefile)

    # dictionary for mean flux in default settings
    mf_info={'number':mf_snap,'z':mf_z,'mean_flux':mf}
    mf_info['skewer_files']=sk_files
    mf_info['simdir']=simdir
    mf_info['skewers_dir']=skewers_dir
    mf_info['zmax']=zmax
    mf_info['n_skewers']=n_skewers
    mf_info['width_kms']=width_kms

    return mf_info


def get_skewers_snapshot(simdir,skewers_dir,snap_num,n_skewers=50,width_kms=10,
                set_T0=None,set_gamma=None,skewers_filename=None):
    """Extract skewers for a particular snapshot"""

    if not skewers_filename:
        skewers_filename="skewers_"+str(snap_num)+"_"+str(n_skewers)
        skewers_filename+="_"+str(width_kms)
        if set_T0:
            skewers_filename+='_T0_'+str(set_T0)
        if set_gamma:
            skewers_filename+='_gamma_'+str(set_gamma)
        skewers_filename+='.hdf5'

    # check that spectra file does not exist yet (should crash)
    if os.path.exists(skewers_dir+'/'+skewers_filename):
        print(skewers_filename,'already exists in',skewers_dir)

    skewers = grid_spec.GriddedSpectra(snap_num,simdir+'/output/',
                nspec=n_skewers,res=width_kms,savefile=skewers_filename,
                savedir=skewers_dir,
                reload_file=True,set_T0=set_T0,set_gamma=set_gamma)

    return skewers

