import numpy as np
import os
from lace.setup_simulations import read_gadget
from lace.postprocess import flux_real_genpk
from lace.postprocess import get_job_script

def get_options_string(raw_dir,post_dir,snap_num,verbose):
    """ Option string to pass to python script in SLURM"""

    options='--raw_dir {} --post_dir {} --snap_num {} '.format(raw_dir,
                post_dir,snap_num)
    if verbose:
        options+='--verbose'

    return options


def write_genpk_script(script_name,raw_dir,post_dir,snap_num,time,verbose):
    """ Generate a SLURM file to run GenPk for a given snapshot."""

    # construct string with options to be passed to python script
    options=get_options_string(raw_dir,post_dir,snap_num,verbose)

    if verbose:
        print('print options: '+options)

    # set output files (.out and .err)
    output_files=post_dir+'/slurm_genpk_'+str(snap_num)

    # get string with submission script
    submit_string=get_job_script.get_job_script("genpk_fluxreal",
                    "run_genpk_flux_real.py",options,time,output_files)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


def write_genpk_scripts_in_sim(raw_dir,post_dir,time,verbose):
    """ Generate a SLURM file for each snapshot to run GenPk"""
    
    if verbose:
        print('in write_genpk_scripts_in_sim',raw_dir,post_dir)

    # get redshifts / snapshots Gadget parameter file 
    paramfile=raw_dir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for snap in range(Nsnap):
        # figure out if GenPk was already computed
        genpk_filename=flux_real_genpk.flux_real_genpk_filename(post_dir,snap)
        print('genpk filename =',genpk_filename)
        if os.path.exists(genpk_filename):
            if verbose: print('GenPk file existing',genpk_filename)
            continue
        else:
            if verbose: print('Will generate genpk file',genpk_filename)

        slurm_script=post_dir+'/genpk_%s.sub'%snap
        write_genpk_script(script_name=slurm_script,
                            raw_dir=raw_dir,post_dir=post_dir,
                            snap_num=snap,time=time,verbose=verbose)
        info_file=post_dir+'/info_sub_genpk_'+str(snap)
        if verbose:
            print('print submit info to',info_file)
        cmd='sbatch '+slurm_script+' > '+info_file
        os.system(cmd)
