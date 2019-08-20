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
                emu_type="k_bin",set_noise_var=1e-3):

        # as a start, use 5 mean flux bins
        self.N_mf=5
        self.central_mf=[0.1,0.3,0.5,0.7,0.9]
        self.min_mf=[0.0,0.15,0.35,0.55,0.75]
        self.max_mf=[0.25,0.45,0.65,0.85,1.0]

        # load full arxiv
        arxiv=p1d_arxiv.ArxivP1D(basedir=basedir,p1d_label=p1d_label,
                skewers_label=skewers_label,verbose=verbose,
                drop_tau_rescalings=drop_tau_rescalings,
                drop_temp_rescalings=drop_temp_rescalings,
                max_arxiv_size=500)

        self.emulators=[]
        for i in range(N_mf):
            print(i,self.central_mf[i],';',self.min_mf[i],'<F<',self.max_mf[i])

            mf_arxiv=arxiv
            mf_emu=gp_emulator.GPEmulator(verbose=verbose,
                    kmax_Mpc=kmax_Mpc,paramList=paramList,train=train,
                    drop_tau_rescalings=drop_tau_rescalings,
                    drop_temp_rescalings=drop_temp_rescalings,
                    emu_type=emu_type,set_noise_var=set_noise_var,
                    passArxiv=mf_arxiv)

            self.emulators.append(mf_emu)

        self.verbose=verbose


    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False):
        '''
        Method to return the trained P(k) for an arbitrary set of k bins
        by interpolating the trained data
        '''

        if self.verbose: print('asked to emulate model',model)

        return self.emulators[0].emulate_p1d_Mpc(model=model,k_Mpc=k_Mpc,
                    return_covar=return_covar)

