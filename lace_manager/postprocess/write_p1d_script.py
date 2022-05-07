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


def get_options_string(post_dir,snap_num,n_skewers,width_Mpc,
                scales_tau,p1d_label,verbose):
    """ Option string to pass to python script in SLURM"""

    # make sure scales are comma-separated string (and not lists)
    if isinstance(scales_tau,str):
        str_scales_tau=scales_tau
    else:
        str_scales_tau=string_from_list(scales_tau)

    options='--post_dir {} --snap_num {} '.format(post_dir,snap_num)
    options+='--n_skewers {} --width_Mpc {} '.format(n_skewers,width_Mpc)
    options+='--scales_tau {} '.format(str_scales_tau)
    if p1d_label is not None:
        options+='--p1d_label {} '.format(p1d_label)
    if verbose:
        options+='--verbose'

    return options


def write_p1d_script(script_name,post_dir,snap_num,n_skewers,width_Mpc,
                scales_tau,time,p1d_label,verbose):
    """ Generate a SLURM file to measure p1d for a given snapshot."""

    # construct string with options to be passed to python script
    options=get_options_string(post_dir,snap_num,n_skewers,width_Mpc,
                scales_tau,p1d_label,verbose)

    if verbose:
        print('print options: '+options)

    # set output files (.out and .err)
    output_files=post_dir+'/slurm_p1d_'+str(snap_num)

    submit_string=get_job_script.get_job_script("calc_flux_p1d",
                    "archive_flux_power.py",options,time,output_files)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


def write_p1d_scripts_in_sim(raw_dir,post_dir,n_skewers,width_Mpc,
                scales_tau,time,zmax,verbose,p1d_label=None,run=False):
    """ Generate a SLURM file for each snapshot in the simulation, to read
        skewers for different thermal histories and measure p1d."""
    
    if verbose:
        print('in write_p1d_scripts_in_sim',raw_dir,post_dir)

    # get redshifts / snapshots Gadget parameter file 
    paramfile=raw_dir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for snap in range(Nsnap):
        z=zs[snap]
        if z < zmax:
            if verbose:
                print('will measure p1d for snapshot',snap)
            slurm_script=post_dir+'/p1d_%s.sub'%snap
            write_p1d_script(script_name=slurm_script,post_dir=post_dir,
                        snap_num=snap,n_skewers=n_skewers,width_Mpc=width_Mpc,
                        scales_tau=scales_tau,time=time,
                        p1d_label=p1d_label,verbose=verbose)
            if run:
                info_file=post_dir+'/info_sub_p1d_'+str(snap)
                if verbose:
                    print('print submit info to',info_file)
                cmd='sbatch '+slurm_script+' > '+info_file
                os.system(cmd)
        else:
            if verbose:
                print('will NOT measure p1d for snapshot',snap)

