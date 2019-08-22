import numpy as np
import gp_emulator
import p1d_arxiv

class MeanFluxEmulator:
    """
    Composite emulator, with different emulators for different ranges
    of mean flux values.
    """
    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                verbose=False,kmax_Mpc=10.0,paramList=None,train=False,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                emu_type="k_bin",set_noise_var=1e-3,N_mf=10):

        # setup central mean flux and edges
        self._setup_mean_flux_bins(N_mf)
        if verbose:
            for i in range(self.N_mf):
                print(i,self.cen_mf[i],self.min_mf[i],'<mf<',self.max_mf[i])

        # load full arxiv
        self.arxiv=p1d_arxiv.ArxivP1D(basedir=basedir,p1d_label=p1d_label,
                skewers_label=skewers_label,verbose=verbose,
                drop_tau_rescalings=drop_tau_rescalings,
                drop_temp_rescalings=drop_temp_rescalings)

        self.emulators=[]
        for i in range(self.N_mf):
            # select entries within mean flux range
            mf_arxiv=self.arxiv.sub_arxiv_mf(min_mf=self.min_mf[i],
                                        max_mf=self.max_mf[i])

            # create GP emulator using only entries in mean flux range
            mf_emu=gp_emulator.GPEmulator(verbose=verbose,
                    kmax_Mpc=kmax_Mpc,paramList=paramList,train=train,
                    drop_tau_rescalings=drop_tau_rescalings,
                    drop_temp_rescalings=drop_temp_rescalings,
                    emu_type=emu_type,set_noise_var=set_noise_var,
                    passArxiv=mf_arxiv)

            self.emulators.append(mf_emu)

        self.verbose=verbose


    def _setup_mean_flux_bins(self,N_mf):
        """ Setup central mean flux and edges. """
 
        self.N_mf=N_mf
        dmf=1.0/N_mf
        # central values
        self.cen_mf=np.linspace(start=0.5*dmf,stop=1.0-0.5*dmf, num=N_mf)
        # minimum values
        self.min_mf=np.fmax(0.0,self.cen_mf - dmf)
        # maximum values
        self.max_mf=np.fmin(1.0,self.cen_mf + dmf)

        return


    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False):
        """
        Method to return the trained P(k) for an arbitrary set of k bins
        by interpolating the trained data
        """

        if self.verbose: print('asked to emulate model',model)

        # look for best emulator to use (closer central value)
        model_mF=model['mF']
        delta_mF=[abs(cen_mf-model_mF) for cen_mf in self.cen_mf]
        i_emu=np.argmin(delta_mF)

        return self.emulators[i_emu].emulate_p1d_Mpc(model=model,k_Mpc=k_Mpc,
                    return_covar=return_covar)


