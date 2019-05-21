import numpy as np
import os
import fake_spectra.abstractsnapshot as absn
import read_gadget


def fit_filtering_length(simdir, kmax_Mpc=15.0,
        genpk_full_path='/home/dc-font1/Codes/GenPK_Keir/gen-pk'):
    """For each snapshot, measure "real flux" power and fit filtering length"""

    # figure out snapshots in simulation
    paramfile=simdir+'/paramfile.gadget'
    print('read GADGET config file',paramfile)
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    # will store measured "real flux" power here
    genpkdir=simdir+'/genpk/'
    os.makedirs(genpkdir,exist_ok=True)

    # snapshot outputs are here
    outdir=simdir+'/output/'

    for num in range(Nsnap):
        z=zs[num]
        print(num,'z =',z)
        snap=absn.AbstractSnapshotFactory(num,outdir,Tscale=1.0,gammascale=1.0)
        hubble = snap.get_header_attr("HubbleParam")
        print('HubbleParam =',hubble)

        snap_tag=str(num).rjust(3,'0')
        genpk_filename=genpkdir+'PK-by-PART_'+snap_tag
        if os.path.exists(genpk_filename):
            print('will read pre-compute power from',genpk_filename)
        else:
            snap_dir=os.path.join(outdir,"PART_"+snap_tag)
            print(snap_tag,'snap dir =',snap_dir)
            cmd=genpk_full_path+' -i '+snap_dir+' -o '+genpkdir
            cmd+=' > info_'+snap_tag
            print('cmd =',cmd)
            os.system(cmd)
            print('done')

        # fit filtering length from measured "real flux" power
