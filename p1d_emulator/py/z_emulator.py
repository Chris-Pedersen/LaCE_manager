import numpy as np
import matplotlib.pyplot as plt
import gp_emulator
import p1d_arxiv
import copy

class ZEmulator:
    """
    Composite emulator, with different emulators for different ranges
    of mean flux values.
    """
    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                verbose=False,kmax_Mpc=10.0,paramList=None,train=False,
                max_arxiv_size=None,
                z_max=5,passArxiv=None,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                keep_every_other_rescaling=False,checkHulls=False,
                emu_type="k_bin",set_noise_var=1e-10,N_mf=10,z_list=None):

        # read all files with P1D measured in simulation suite
        if passArxiv==None:
            self.custom_arxiv=False
            self.arxiv=p1d_arxiv.ArxivP1D(basedir,p1d_label,skewers_label,
                        max_arxiv_size=max_arxiv_size,verbose=verbose,
                        drop_tau_rescalings=drop_tau_rescalings,
                        drop_temp_rescalings=drop_temp_rescalings,z_max=z_max,
                        keep_every_other_rescaling=keep_every_other_rescaling)
        else:
            self.custom_arxiv=True
            print("Loading emulator using a specific arxiv, not the one set in basedir")
            self.arxiv=passArxiv

        self._split_arxiv_up(z_list)
        self.emulators=[]
        self.paramList=paramList
        self.kmax_Mpc=kmax_Mpc
        self.emu_type=emu_type

        for arxiv in self.arxiv_list:
            emu=gp_emulator.GPEmulator(verbose=verbose,
                    kmax_Mpc=kmax_Mpc,paramList=paramList,train=train,
                    emu_type=emu_type,set_noise_var=set_noise_var,
                    passArxiv=arxiv,checkHulls=checkHulls)
            self.emulators.append(emu)

        self.training_k_bins=self.emulators[0].training_k_bins



    def _split_arxiv_up(self,z_list):
        """ Split up the arxiv into a list of arxiv objects, one for
        each redshift """

        ## First identify which redshifts are in the arxiv:
        self.zs=[]
        for item in self.arxiv.data:
            self.zs.append(item["z"]) if item["z"] not in self.zs else self.zs
        
        ## Remove unwanted redshifts if a list of redshifts is provided
        if z_list is not None:
            removes=[] ## List of indices of self.zs to remove
            for aa,z in enumerate(self.zs):
                if z not in z_list:
                    removes.append(aa)
            for aa in sorted(removes, reverse=True):
                del self.zs[aa]

        self.arxiv_list=[]

        ## For each redshift, create a new arxiv object
        for z in self.zs:
            copy_arxiv=copy.deepcopy(self.arxiv)
            aa=0
            while aa<len(copy_arxiv.data):
                if copy_arxiv.data[aa]["z"]==z:
                    aa+=1
                else:
                    del copy_arxiv.data[aa]
            self.arxiv_list.append(copy_arxiv)
            
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
        