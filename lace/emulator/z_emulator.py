import numpy as np
import matplotlib.pyplot as plt
from lace.emulator import gp_emulator
from lace.emulator import p1d_archive
import copy

class ZEmulator:
    """
    Composite emulator, with different emulators for different ranges
    of mean flux values.
    """
    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                verbose=False,kmax_Mpc=10.0,paramList=None,train=False,
                max_archive_size=None,
                z_max=5,passarchive=None,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                keep_every_other_rescaling=False,checkHulls=False,
                emu_type="k_bin",set_noise_var=1e-10,N_mf=10,z_list=None,
                paramLimits=None):

        # read all files with P1D measured in simulation suite
        if passarchive==None:
            self.custom_archive=False
            self.archive=p1d_archive.archiveP1D(basedir,p1d_label,skewers_label,
                        max_archive_size=max_archive_size,verbose=verbose,
                        drop_tau_rescalings=drop_tau_rescalings,
                        drop_temp_rescalings=drop_temp_rescalings,z_max=z_max,
                        keep_every_other_rescaling=keep_every_other_rescaling)
        else:
            self.custom_archive=True
            print("Loading emulator using a specific archive, not the one set in basedir")
            self.archive=passarchive

        self._split_archive_up(z_list)
        self.emulators=[]
        self.paramList=paramList
        self.kmax_Mpc=kmax_Mpc
        self.emu_type=emu_type
        self.paramLimits=paramLimits

        for archive in self.archive_list:
            emu=gp_emulator.GPEmulator(verbose=verbose,
                    kmax_Mpc=kmax_Mpc,paramList=paramList,train=train,
                    emu_type=emu_type,set_noise_var=set_noise_var,
                    passarchive=archive,checkHulls=checkHulls,paramLimits=self.paramLimits)
            self.emulators.append(emu)

        self.training_k_bins=self.emulators[0].training_k_bins



    def _split_archive_up(self,z_list):
        """ Split up the archive into a list of archive objects, one for
        each redshift """

        ## First identify which redshifts are in the archive:
        self.zs=[]
        for item in self.archive.data:
            self.zs.append(item["z"]) if item["z"] not in self.zs else self.zs
        
        ## Remove unwanted redshifts if a list of redshifts is provided
        if z_list is not None:
            removes=[] ## List of indices of self.zs to remove
            for aa,z in enumerate(self.zs):
                if z not in z_list:
                    removes.append(aa)
            for aa in sorted(removes, reverse=True):
                del self.zs[aa]

        self.archive_list=[]

        ## For each redshift, create a new archive object
        for z in self.zs:
            copy_archive=copy.deepcopy(self.archive)
            aa=0
            while aa<len(copy_archive.data):
                if copy_archive.data[aa]["z"]==z:
                    aa+=1
                else:
                    del copy_archive.data[aa]
            self.archive_list.append(copy_archive)
            
        return
        

    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False,z=None):
        """ Emulate p1d for a given model & redshift """

        assert z is not None, "z is not provided, cannot emulate p1d"
        assert z in self.zs, "cannot emulate for z=%.1f" % z

        ## Find the appropriate emulator to call from
        return self.emulators[self.zs.index(z)].emulate_p1d_Mpc(model=model,
                                                k_Mpc=k_Mpc,
                                                return_covar=return_covar)

    def get_nearest_distance(self, model, z=None):
        """ Call the get_nearest_distance method for the
        appropriate emulator """

        assert z is not None, "z is not provided, cannot get distance"
        assert z in self.zs, "cannot work for z=%.1f" % z

        return self.emulators[self.zs.index(z)].get_nearest_distance(model,z=z)
        