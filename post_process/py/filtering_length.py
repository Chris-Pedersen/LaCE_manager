import numpy as np
import os
import fake_spectra.abstractsnapshot as absn
import read_gadget


def fit_filtering_length(simdir, kmax_Mpc=15.0,
        genpk_full_path='/home/dc-font1/Codes/GenPK_Keir/gen-pk'):
    """Fit filtering length from measured "real flux" power"""

    paramfile=simdir+'/paramfile.gadget'
    print('read GADGET config file',paramfile)
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    for num in range(Nsnap):
        z=zs[num]
        snap=absn.AbstractSnapshotFactory(num,simdir,Tscale=1.0,gammascale=1.0)
        # sanity check
        z_snap= 1./snap.get_header_attr("Time") - 1.0
        print(num,'z =',z,'z_snap =',z_snap)
        hubble = snap.get_header_attr("HubbleParam")
        print('HubbleParam =',hubble)

        snap_tag=str(num).rjust(3,'0')
        snap_dir=os.path.join(simdir,"PART_"+snap_tag)
        print(snap_tag,'snap dir =',snap_dir)

        cmd=genpk_full_path+' -i '+snap_dir+' -o '+simdir
        cmd+=' > info_'+snap_tag
        print('cmd =',cmd)
        #os.system(cmd)

        print('done')
