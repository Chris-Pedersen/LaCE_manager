import numpy as np
import sys
import os
import fake_spectra.griddedspectra as grid_spec 
import read_gadget

def write_default_skewers(basedir,skewers_dir=None,paramfile_gadget=None,
        zmax=6.0,n_skewers=50,width_kms=10):
    """Extract skewers for all snapshots, default temperature."""

    # make sure output directory exists (will write skewers there)
    if skewers_dir is None:
        skewers_dir=basedir+'/output/skewers/'
    if os.path.exists(skewers_dir):
        if not os.path.isdir(skewers_dir):
            raise ValueError(skewers_dir+' is not a directory')
    else:
        print('make directory',skewers_dir)
        os.mkdir(skewers_dir)

    # figure out number of snapshots and redshifts
    paramfile=basedir+'/paramfile.gadget'
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
            skewers=get_skewers_snapshot(basedir,skewers_dir,num,
                        n_skewers=n_skewers,width_kms=width_kms)
            # call mean flux, so that the skewers are really computed
            mf=skewers.get_mean_flux()
            skewers.save_file()
            mf_snap.append(num)
            mf_z.append(z)
            mf_val.append(mf)
            sk_files.append(skewers.savefile)

    # dictionary for mean flux in default settings
    mf_info={'number':mf_snap,'z':mf_z,'mean_flux':mf}
    mf_info['skewer_files']=sk_files
    mf_info['basedir']=basedir
    mf_info['skewers_dir']=skewers_dir
    mf_info['zmax']=zmax
    mf_info['n_skewers']=n_skewers
    mf_info['width_kms']=width_kms

    return mf_info


def get_skewers_snapshot(basedir,skewers_dir,snap_num,n_skewers=50,width_kms=10,
                set_T0=None,set_gamma=None):
    """Extract skewers for a particular snapshot"""

    spectra_name="skewers_"+str(snap_num)+"_"+str(n_skewers)
    spectra_name+="_"+str(width_kms)
    if set_T0:
        spectra_name+='_T0_'+str(set_T0)
    if set_gamma:
        spectra_name+='_gamma_'+str(set_gamma)
    spectra_name+='.hdf5'

    # check that spectra file does not exist yet (should crash)
    if os.path.exists(skewers_dir+'/'+spectra_name):
        print(spectra_name,'already exists in',skewers_dir)

    skewers = grid_spec.GriddedSpectra(snap_num,basedir+'/output/',
                nspec=n_skewers,res=width_kms,savefile=spectra_name,
                savedir=skewers_dir,
                reload_file=True,set_T0=set_T0,set_gamma=set_gamma)

    return skewers

