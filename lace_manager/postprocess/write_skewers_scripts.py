import numpy as np
import os
from lace_manager.setup_simulations import read_gadget
from lace_manager.postprocess import get_job_script

def string_from_list(in_list):
    """ Given a list of floats, return a string with comma-separated values"""
    out_string=str(in_list[0])
    if len(in_list)>1:
        for x in in_list[1:]:
            out_string += ', '+str(x)
    return out_string


def get_options_string(raw_dir,post_dir,snap_num,n_skewers,width_Mpc,
                scales_T0,scales_gamma,verbose):
    """ Option string to pass to python script in SLURM"""

    # make sure scales are comma-separated string (and not lists)
    if isinstance(scales_T0,str):
        str_scales_T0=scales_T0
    else:
        str_scales_T0=string_from_list(scales_T0)
    if isinstance(scales_gamma,str):
        str_scales_gamma=scales_gamma
    else:
        str_scales_gamma=string_from_list(scales_gamma)

    options='--raw_dir {} --post_dir {} --snap_num {} '.format(raw_dir,
                post_dir,snap_num)
    options+='--n_skewers {} --width_Mpc {} '.format(n_skewers,width_Mpc)
    options+='--scales_T0 {} '.format(str_scales_T0)
    options+='--scales_gamma {} '.format(str_scales_gamma)
    if verbose:
        options+='--verbose'

    return options


def write_skewers_script(script_name,raw_dir,post_dir,
                snap_num,n_skewers,width_Mpc,
                scales_T0,scales_gamma,time,verbose):
    """ Generate a SLURM file to extract skewers for a given snapshot."""

    # construct string with options to be passed to python script
    options=get_options_string(raw_dir,post_dir,snap_num,n_skewers,width_Mpc,
                scales_T0,scales_gamma,verbose)

    if verbose:
        print('print options: '+options)

    # set output files (.out and .err)
    output_files=post_dir+'/slurm_skewers_'+str(snap_num)

    submit_string=get_job_script.get_job_script("extract_skewers",
                    "extract_tdr_skewers.py",options,time,output_files)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


def write_skewer_scripts_in_sim(raw_dir,post_dir,n_skewers,width_Mpc,
                scales_T0,scales_gamma,time,zmax,verbose,run=False):
    """ Generate a SLURM file for each snapshot in the simulation, to extract
        skewers for different thermal histories."""
    
    if verbose:
        print('in write_skewer_scripts_in_sim',raw_dir,post_dir)

    # get redshifts / snapshots Gadget parameter file 
    paramfile=raw_dir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for snap in range(Nsnap):
        z=zs[snap]
        if z < zmax:
            if verbose:
                print('will extract skewers for snapshot',snap)
            slurm_script=post_dir+'/skewers_%s.sub'%snap
            write_skewers_script(script_name=slurm_script,raw_dir=raw_dir,
                        post_dir=post_dir,
                        snap_num=snap,n_skewers=n_skewers,width_Mpc=width_Mpc,
                        scales_T0=scales_T0,scales_gamma=scales_gamma,
                        time=time,verbose=verbose)
            if run:
                info_file=post_dir+'/info_sub_skewers_'+str(snap)
                if verbose:
                    print('print submit info to',info_file)
                cmd='sbatch '+slurm_script+' > '+info_file
                os.system(cmd)
        else:
            if verbose:
                print('will NOT extract skewers for snapshot',snap)

