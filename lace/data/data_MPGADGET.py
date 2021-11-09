import os
import numpy as np
import json
from numpy.lib.polynomial import poly
from scipy.interpolate import interp1d
import camb
from lace.data import base_p1d_data
from lace.data import data_PD2013
from lace.data import data_Chabanier2019
from lace.data import data_Karacayli_DESI
from lace.data import data_Karacayli_HIRES
from lace.emulator import p1d_archive
from lace.emulator import test_simulation
from lace.setup_simulations import read_genic
from lace.cosmo import camb_cosmo

class P1D_MPGADGET(base_p1d_data.BaseDataP1D):
    """ Class to load an MP-Gadget simulation as a mock
    data object. Can use PD2013 or Chabanier2019 covmats """

    def __init__(self,basedir=None,sim_label=None,skewers_label=None,
            zmin=None,zmax=None,z_list=None,kp_Mpc=0.7,
            data_cov_label="Chabanier2019",data_cov_factor=1.,
            add_syst=True,pivot_scalar=0.05,polyfit=False):
        """ Read mock P1D from MP-Gadget sims, and returns mock measurement:
            - basedir: directory with simulations outputs for a given suite
            - sim_label: can be either:
                -- an integer, index from the Latin hypercube for that suite
                -- "nu", which corresponds to the 0.3eV neutrino sim
                -- "h", which corresponds to the simulation with h=0.74
            - skewers_label: string identifying skewer extraction from sims
            - zmin, zmax, z_list: different ways to specify redshifts to use
            - kp_Mpc: specify pivot point to compute linP parameters at each z
            - data_cov_label: P1D covariance to use (Chabanier2019 or PD2013)
            - data_cov_factor: multiply covariance by this factor
            - add_syst: Include systematic estimates in covariance matrices
            - polyfit: Smooth the mock data by using a polynomial fit to the P1D
        """

        if basedir:
            self.basedir=basedir
        else:
            self.basedir="/p1d_emulator/sim_suites/Australia20/"

        if skewers_label:
            self.skewers_label=skewers_label
        else:
            self.skewers_label='Ns500_wM0.05'

        if sim_label:
            self.sim_label=sim_label
        else:
            self.sim_label=0

        self.data_cov_factor=data_cov_factor
        self.data_cov_label=data_cov_label
        self.kp_Mpc=kp_Mpc
        self.polyfit=polyfit

        # read P1D from simulation
        z,k,Pk,cov=self._load_p1d(add_syst,pivot_scalar=pivot_scalar)

        # drop low-z or high-z bins
        if zmin or zmax:
            z,k,Pk,cov=base_p1d_data._drop_zbins(z,k,Pk,cov,zmin,zmax)
        if z_list is not None:
            z,k,Pk,cov=_select_zs(z,k,Pk,cov,z_list)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

        # store true emulator calls
        self._set_true_values()


    def _load_p1d(self,add_syst,pivot_scalar):

        if self.data_cov_label=="Chabanier2019":
            data_file=data_Chabanier2019.P1D_Chabanier2019(add_syst=add_syst)
        elif self.data_cov_label=="PD2013":
            data_file=data_PD2013.P1D_PD2013(add_syst=add_syst)
        elif self.data_cov_label=="Karacayli_DESI":
            data_file=data_Karacayli_DESI.P1D_Karacayli_DESI()
        elif self.data_cov_label=="Karacayli_HIRES":
            data_file=data_Karacayli_HIRES.P1D_Karacayli_HIRES()
        else:
            print("Unknown data_cov_label",self.data_cov_label)
            quit()

        k=data_file.k
        z_data=data_file.z

        # setup TestSimulation object to read json files from sim directory
        self.mock_sim=test_simulation.TestSimulation(basedir=self.basedir,
                sim_label=self.sim_label,skewers_label=self.skewers_label,
                z_max=10,kmax_Mpc=8,kp_Mpc=self.kp_Mpc,
                pivot_scalar=pivot_scalar)

        # get redshifts in simulation
        z_sim=self.mock_sim.zs
        zmin_sim=min(z_sim)

        # get cosmology in simulation to convert units
        sim_cosmo=self.mock_sim.sim_cosmo
        camb_cosmo.print_info(sim_cosmo)
        sim_camb_results=camb_cosmo.get_camb_results(sim_cosmo)

        # unit conversion, at zmin to get lowest possible k_min_kms
        dkms_dMpc_zmin=sim_camb_results.hubble_parameter(zmin_sim)/(1+zmin_sim)

        # Get k_min for the sim data, & cut k values below that
        k_min_Mpc=self.mock_sim.k_Mpc[1]
        k_min_kms=k_min_Mpc/dkms_dMpc_zmin
        Ncull=np.sum(k<k_min_kms)
        k=k[Ncull:]

        Pk=[]
        cov=[]
        ## Set P1D and covariance for each redshift
        for iz,z in enumerate(z_sim):
            # store P1D in Mpc, except k=0
            if self.polyfit==True:
                ## Get "smoothed" polyfit p1d
                k_Mpc,p1d_Mpc=self.mock_sim.get_polyfit_p1d_Mpc(z)
                p1d_Mpc=p1d_Mpc[1:]
                k_Mpc=k_Mpc[1:]
            else:
                p1d_Mpc=np.asarray(self.mock_sim.p1d_Mpc[iz][1:])
                k_Mpc=np.asarray(self.mock_sim.k_Mpc[1:])
            conversion_factor=sim_camb_results.hubble_parameter(z)/(1+z)
            
            # evaluate P1D in data wavenumbers (in velocity units)
            interpolator=interp1d(k_Mpc,p1d_Mpc,"cubic")
            k_interp=k*conversion_factor
            interpolated_P=interpolator(k_interp)
            p1d_sim=interpolated_P*conversion_factor
            Pk.append(p1d_sim)

            # Now get covariance from the nearest z bin in data
            cov_mat=data_file.get_cov_iz(np.argmin(abs(z_data-z)))
            # Cull low k cov data and multiply by input factor
            cov_mat=self.data_cov_factor*cov_mat[Ncull:,Ncull:]
            cov.append(cov_mat)

        return z_sim,k,Pk,cov
    

    def _set_true_values(self):
        """ For each emulator parameter, generate an array of
        true values from the archive """

        # Dictionary to hold true values
        self.truth={} 
        paramList=["mF","sigT_Mpc","gamma","kF_Mpc","Delta2_p","n_p"]
        for param in paramList:
            self.truth[param]=[]
        
        for z in self.z:
            emu_call=self.mock_sim.get_emulator_calls(z)
            for param in paramList:
                self.truth[param].append(emu_call[param])

        return


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
