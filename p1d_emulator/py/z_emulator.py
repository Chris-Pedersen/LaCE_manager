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
                z_max=5,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                keep_every_other_rescaling=False,
                emu_type="k_bin",set_noise_var=1e-10,N_mf=10,z_list=None):

        # load full arxiv
        self.arxiv=p1d_arxiv.ArxivP1D(basedir=basedir,p1d_label=p1d_label,
                skewers_label=skewers_label,verbose=verbose,
                max_arxiv_size=max_arxiv_size,
                z_max=z_max,
                drop_tau_rescalings=drop_tau_rescalings,
                drop_temp_rescalings=drop_temp_rescalings,
                keep_every_other_rescaling=keep_every_other_rescaling)

        self._split_arxiv_up(z_list)
        self.emulators=[]

        for arxiv in self.arxiv_list:
            emu=gp_emulator.GPEmulator(verbose=verbose,
                    kmax_Mpc=kmax_Mpc,paramList=paramList,train=train,
                    emu_type=emu_type,set_noise_var=set_noise_var,
                    passArxiv=arxiv)
            self.emulators.append(emu)


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
        

    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False,z=None):
        """ Emulate p1d for a given model & redshift """

        assert z is not None, "z is not provided, cannot emulate p1d"
        assert z in self.zs, "cannot emulate for z=%.1f" % z

        ## Find the appropriate emulator to call from
        print("Desired z = ",z)
        print("z we are using for emulator = ", self.zs[self.zs.index(z)])
        return self.emulators[self.zs.index(z)].emulate_p1d_Mpc(model=model,
                                                k_Mpc=k_Mpc,
                                                return_covar=return_covar)
