import numpy as np
import fake_spectra.tempdens as tdr
import read_gadget

def compute_TDR(simdir,zmax=20):
    """Measure temperature-density relation for all snapshots above zmax"""

    paramfile=simdir+'/paramfile.gadget'
    zs=read_gadget.redshifts_from_paramfile(paramfile)
    Nsnap=len(zs)

    # store information to write to file later
    thermal_snap=[]
    thermal_z=[]
    thermal_T0=[]
    thermal_gamma=[]
    for num in range(Nsnap):
        z=zs[num]
        if z < zmax:
            T0,gamma=tdr.fit_td_rel_plot(num,simdir+'/output',plot=False)
            thermal_snap.append(num)
            thermal_z.append(z)
            thermal_T0.append(T0)
            thermal_gamma.append(gamma)

    thermal_info={'number':thermal_snap,'z':thermal_z,'T0':thermal_T0,
            'gamma':thermal_gamma}
    thermal_info['zmax']=zmax

    return thermal_info

