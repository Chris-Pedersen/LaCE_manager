import numpy as np
import os
from LyaCosmoParams.setup_simulations import read_gadget
from LyaCosmoParams.postprocess import flux_real_genpk
from LyaCosmoParams.postprocess import get_job_script

def get_options_string(simdir,snap_num,verbose):
    """ Option string to pass to python script in SLURM"""

    options='--simdir {} --snap_num {} '.format(simdir,snap_num)
    if verbose:
        options+='--verbose'

    return options


def write_genpk_script(script_name,simdir,snap_num,time,verbose):
    """ Generate a SLURM file to run GenPk for a given snapshot."""

    # construct string with options to be passed to python script
    options=get_options_string(simdir,snap_num,verbose)

    if verbose:
        print('print options: '+options)

    # set output files (.out and .err)
    output_files=simdir+'/slurm_genpk_'+str(snap_num)

    # get string with submission script
    submit_string=get_job_script.get_job_script("genpk_fluxreal",
                    "run_genpk_flux_real.py",options,time,output_files)

    submit_script = open(script_name,'w')
    for line in submit_string:
        submit_script.write(line)
    submit_script.close()


def write_genpk_scripts_in_sim(simdir,time,verbose):
    """ Generate a SLURM file for each snapshot to run GenPk"""
    
    if verbose:
        print('in write_genpk_scripts_in_sim',simdir)

    # get redshifts / snapshots Gadget parameter file 
    paramfile=simdir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for snap in range(Nsnap):
        # figure out if GenPk was already computed
        genpk_filename=flux_real_genpk.flux_real_genpk_filename(simdir,snap)
        print('genpk filename =',genpk_filename)
        if os.path.exists(genpk_filename):
            if verbose: print('GenPk file existing',genpk_filename)
            continue
        else:
            if verbose: print('Will generate genpk file',genpk_filename)

        slurm_script=simdir+'/genpk_%s.sub'%snap
        write_genpk_script(script_name=slurm_script,simdir=simdir,
                            snap_num=snap,time=time,verbose=verbose)
        info_file=simdir+'/info_sub_genpk_'+str(snap)
        if verbose:
            print('print submit info to',info_file)
        cmd='sbatch '+slurm_script+' > '+info_file
        os.system(cmd)
