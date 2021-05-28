import os
import numpy as np

def flux_real_genpk_filename(post_dir, snap_num):
    genpkdir=post_dir+'/genpk/'
    snap_tag=str(snap_num).rjust(3,'0')
    return genpkdir+'/PK-by-PART_'+snap_tag


def compute_flux_real_power(raw_dir, post_dir, snap_num,verbose=False,
                genpk_full_path='/home/chrisp/Codes/GenPK_Keir/gen-pk'):
    """Measure power spectrum of exp(-tau_noRSD), using GenPk"""

    # will store measured "flux real" power here
    genpkdir=post_dir+'/genpk/'
    os.makedirs(genpkdir,exist_ok=True)

    # snapshot outputs are here
    outdir=raw_dir+'/output/'

    snap_tag=str(snap_num).rjust(3,'0')
    genpk_filename=flux_real_genpk_filename(post_dir,snap_num)
    if os.path.exists(genpk_filename):
        if verbose: print('GenPk was alreay run')
    else:
        snap_dir=os.path.join(outdir,"PART_"+snap_tag)
        info_file=genpkdir+'/info_genpk_'+snap_tag
        cmd=genpk_full_path+' -i '+snap_dir+' -o '+genpkdir
        cmd+=' > '+info_file
        if verbose: print('run genpk, cmd =',cmd)
        os.system(cmd)

    return

