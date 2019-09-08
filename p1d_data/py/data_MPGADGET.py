import os
import numpy as np
import base_p1d_data
import data_PD2013
import poly_p1d
import json
import matplotlib.pyplot as plt

class P1D_MPGADGET(base_p1d_data.BaseDataP1D):

    def __init__(self,basedir=None,zmin=None,zmax=None,blind_data=False,
                        filename="1024_mock_0.json",z_list=None):
        """ Read mock P1D from MP-Gadget sims, and return
        using the k bins and covariance from PD2013 """

        # folder storing P1D measurement
        if not basedir:
            assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
            repo=os.environ['LYA_EMU_REPO']
            basedir=repo+'/p1d_data/data_files/MP-Gadget_data/'

        z,k,Pk,cov=self._load_p1d(basedir,filename)

        # drop low-z or high-z bins
        if zmin or zmax:
            z,k,Pk,cov=_drop_zbins(z,k,Pk,cov,zmin,zmax)
        if z_list is not None:
            z,k,Pk,cov=_select_zs(z,k,Pk,cov,z_list)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

    def _load_p1d(self,basedir,filename):
        ## Load in dictionaries
        with open(basedir+filename) as json_file:
            data_file = json.load(json_file)
        sim_data = data_file["data"]
        ## Load PD2013 data
        PD2013=data_PD2013.P1D_PD2013(blind_data=False)
        k=PD2013.k
        z_PD=PD2013.z

        ## For each redshift, fit the data and return
        ## P1D(k) with PD2013 bins
        z=np.array([])
        Pk=[]
        cov=[]
        for item in sim_data:
            z=np.append(z,item["z"])
            p1d_sim=np.asarray(item["p1d_kms"]) ## I saved the data as lists :/
            k_sim=np.asarray(item["k_kms"])

            ## Only fit where we have PD2013 data
            kfit=(k_sim < 0.02) & (k_sim > 0.001)
            #print(k_sim)
            lnP_fit = np.polyfit(np.log(k_sim[kfit]),np.log(p1d_sim[kfit]), 4)
            poly=np.poly1d(lnP_fit)
            p1d_rebin=np.exp((poly(np.log(k))))
            Pk.append(p1d_rebin)
            ## Now get covariance from the nearest
            ## z bin in PD2013
            cov.append(PD2013.get_cov_iz(np.argmin(abs(z_PD-z[-1]))))
        return z,k,Pk,cov


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
    args=np.flip(np.unique(args))

    z_out=np.empty(len(args))
    cov_out=[]
    Pk_out=np.empty((len(args),len(k_in)))
    for aa,arg in enumerate(args):
        z_out[aa]=z_in[arg]
        Pk_out[aa]=Pk_in[arg]
        cov_out.append(cov_in[arg])

    return z_out,k_in,Pk_out,cov_out
