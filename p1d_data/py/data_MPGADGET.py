import os
import numpy as np
import base_p1d_data
import data_PD2013
import poly_p1d
import json
import matplotlib.pyplot as plt
import p1d_arxiv
import recons_cosmo
from scipy.interpolate import interp1d

class P1D_MPGADGET(base_p1d_data.BaseDataP1D):
    def __init__(self,basedir=None,zmin=None,zmax=None,blind_data=False,
                        sim_number=0,skewers_label=None,
                        z_list=None):
        """ Read mock P1D from MP-Gadget sims, and return
        using the k bins and covariance from PD2013 """

        # folder storing P1D measurement
        if not basedir:
            assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
            repo=os.environ['LYA_EMU_REPO']
            basedir=repo+"/p1d_emulator/sim_suites/emulator_256_28082019/"
            skewers_label="Ns256_wM0.05"

        self.basedir=basedir
        self.sim_number=sim_number

        z,k,Pk,cov=self._load_p1d(basedir,sim_number,skewers_label)

        # drop low-z or high-z bins
        if zmin or zmax:
            z,k,Pk,cov=_drop_zbins(z,k,Pk,cov,zmin,zmax)
        if z_list is not None:
            z,k,Pk,cov=_select_zs(z,k,Pk,cov,z_list)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)
        self._set_true_values()

    def _load_p1d(self,basedir,sim_number,skewers_label):
        ## Load PD2013 data to get covmats
        PD2013=data_PD2013.P1D_PD2013(blind_data=False)
        k=PD2013.k
        z_PD=PD2013.z

        ## Load mock data as arxiv object
        self.mock_data=p1d_arxiv.ArxivP1D(basedir=basedir,
                            drop_tau_rescalings=True,z_max=4,
                            pick_sim_number=sim_number,
                            drop_temp_rescalings=True,skewers_label=skewers_label)

        z_sim=np.empty(len(self.mock_data.data))
        
        ## Populate in reverse order as the archive data is stored from high z to low
        ## whereas the rest of our code works starting with lowest z first
        for aa in range(len(self.mock_data.data)):
            z_sim[aa]=self.mock_data.data[len(self.mock_data.data)-aa-1]["z"]
        
        ## Import cosmology object to get Mpc -> kms conversion factor
        cosmo=recons_cosmo.ReconstructedCosmology(z_sim)

        ## Get k_min for the sim data, & cut k values below that
        k_min_kms=self.mock_data.data[0]["k_Mpc"][1]/(cosmo.reconstruct_Hubble_iz(0,cosmo.linP_model_fid)/(1+min(z_sim)))
        Ncull=np.sum(k<k_min_kms)
        k=k[Ncull:]

        Pk=[]
        cov=[]
        ## Latest commit
        for aa,item in enumerate(self.mock_data.data):
            ## Archive in reverse..
            p1d_Mpc=np.asarray(self.mock_data.data[len(self.mock_data.data)-aa-1]["p1d_Mpc"][1:])
            k_Mpc=np.asarray(self.mock_data.data[len(self.mock_data.data)-aa-1]["k_Mpc"][1:])
            conversion_factor=cosmo.reconstruct_Hubble_iz(aa,cosmo.linP_model_fid)/(1+z_sim[aa])

            interpolator=interp1d(k_Mpc,p1d_Mpc, "cubic")
            k_interp=k*conversion_factor
            interpolated_P=interpolator(k_interp)
            p1d_sim=interpolated_P*conversion_factor

            Pk.append(p1d_sim)
            ## Now get covariance from the nearest
            ## z bin in PD2013
            cov_mat=PD2013.get_cov_iz(np.argmin(abs(z_PD-z_sim[aa])))
            ## Cull low k cov data
            cov_mat=cov_mat[Ncull:,Ncull:]
            cov.append(cov_mat)

        return z_sim,k,Pk,cov
    
    def _set_true_values(self):
        """ For each emulator parameter, generate an array of
        true values from the arxiv """

        self.truth={} ## Dictionary to hold true values
        paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p","n_p"]
        for param in paramList:
            self.truth[param]=[]
        
        for item in self.mock_data.data:
            for param in paramList:
                self.truth[param].insert(0,item[param])

        return

def _drop_zbins(z_in,k_in,Pk_in,cov_in,zmin,zmax):
    """Drop redshift bins below zmin or above zmax"""

    # size of input arrays
    Nz_in=len(z_in)
    Nk=len(k_in)

    # figure out how many z to keep
    keep=np.ones(Nz_in, dtype=bool)
    if zmin:
        keep = np.logical_and(keep,z_in>zmin)
    if zmax:
        keep = np.logical_and(keep,z_in<zmax)
    Nz_out=np.sum(keep)

    # setup new arrays
    z_out=np.empty(Nz_out)
    Pk_out=np.empty((Nz_out,Nk))
    cov_out=[]
    i=0
    for j in range(Nz_in):
        if keep[j]:
            z_out[i]=z_in[j]
            Pk_out[i]=Pk_in[j]
            Pk_out[i]=Pk_in[j]
            cov_out.append(cov_in[j])
            i+=1
    return z_out,k_in,Pk_out,cov_out

def _select_zs(z_in,k_in,Pk_in,cov_in,zs):
    args=np.array([],dtype=int)
    for z in zs:
        args=np.append(args,np.argmin(abs(z_in-z)))

    ## Remove duplicates
    args=np.unique(args)

    z_out=np.empty(len(args))
    cov_out=[]
    Pk_out=np.empty((len(args),len(k_in)))
    for aa,arg in enumerate(args):
        z_out[aa]=z_in[arg]
        Pk_out[aa]=Pk_in[arg]
        cov_out.append(cov_in[arg])

    return z_out,k_in,Pk_out,cov_out
