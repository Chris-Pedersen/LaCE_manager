import json
import read_gadget
import read_genic
import camb_cosmo
import fit_linP
import numpy as np
import os

class TestSimulation(object):
    """ Object to store parameters and data for one
    specific test simulation. Used for performing tests
    either on the emulator directly in Mpc or on the sampler
    in velocity units """

    def __init__(self,basedir,sim_label,skewers_label,
            z_max,kmax_Mpc,kp_Mpc):
        """ Extract data from a chosen simulation
            - basedir sets which sim suite to work with
            - sim_label can be either:
                -- an integer, index from the Latin hypercube for that suite
                -- "nu", which corresponds to the 0.3eV neutrino sim
                -- "h", which corresponds to the simulation with h=0.74
                -- "central", the central simulation of the initial LH,
                   which is used as the fiducial IGM model
            - skewers_label: string identifying skewer extraction from sims
            - z_max sets the highest z cut
            - kmax_Mpc sets the highest k bin to store the P_1D for
            - kp_Mpc sets the comoving pivot scale used to calculate the
              emulator linear power parameters
        """

        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']

        if type(sim_label)==int:
            self.fulldir=repo+basedir+"sim_pair_"+str(sim_label)
        elif sim_label[0].isdigit():
            self.fulldir=repo+basedir+"sim_pair_"+sim_label
        elif sim_label=="nu":
            self.fulldir=repo+basedir+"nu_sim"
        elif sim_label=="h":
            self.fulldir=repo+basedir+"h_sim"
        elif sim_label=="central":
            self.fulldir=repo+basedir+"central"
            
        self.kp_Mpc=kp_Mpc ## Pivot point for Delta2_p, n_p, alpha_p

        self.skewers_label=skewers_label
        self.p1d_label="p1d"
        
        self._read_json_files(z_max,kmax_Mpc)

        return


    def _read_json_files(self,z_max,kmax_Mpc):
        """ Read the json files for the given sim suite. Store the P1D
        and emulator parameters for the non-rescaled entries
            - z_max: discard redshifts above this cut
            - kmax_Mpc: take only k bins below this cut """
        
        # There is a lot of overlap between this and functions in p1d_arxiv.py
        
        ## First get zs from paramfile
        sim_config=read_gadget.read_gadget_paramfile(self.fulldir+
                            "/sim_plus/paramfile.gadget")
        zs=read_gadget.snapshot_redshifts(sim_config)

        ## Cut over z_max
        self.zs=zs[zs<=z_max]
        drop=zs[:(len(zs)-len(self.zs))] ## Redshifts we don't want to keep
        ## Reverse order of redshifts to match the rest of the code (earliest first)
        self.zs=np.flip(self.zs)
        
        ## Set up remaining lists
        self.p1d_Mpc=[]
        self.k_Mpc=[] ## These should all be the same, will double check
        self.emu_calls=[]
        
        ## Get cosmology from IC file to get linear power parameters
        genic_fname=self.fulldir+"/sim_plus/paramfile.genic"
        sim_cosmo_dict=read_genic.camb_from_genic(genic_fname)
        # setup CAMB object
        self.sim_cosmo=camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)
        # compute linear power parameters at each z (in Mpc units)
        linP_zs=fit_linP.get_linP_Mpc_zs(self.sim_cosmo,self.zs,self.kp_Mpc,
                include_f_p=True)
        #print('linP_zs',linP_zs)
        linP_values=list(linP_zs)

        ## Now loop over each p1d.json file
        ## in reverse order to have smallest redshift first
        for aa in reversed(range(len(drop),len(zs))):
            ## Load json files
            json_path_plus=self.fulldir+"/sim_plus/{}_{}_{}.json".format(
                                self.p1d_label,aa,self.skewers_label)
            json_path_minus=self.fulldir+"/sim_minus/{}_{}_{}.json".format(
                                self.p1d_label,aa,self.skewers_label)

            with open(json_path_plus) as json_file:
                    plus_file = json.load(json_file)
            with open(json_path_minus) as json_file:
                    minus_file = json.load(json_file)

            ## Add p1d to list
            ## Find index of non-rescaled entry
            for bb in range(len(plus_file["p1d_data"])):
                if plus_file["p1d_data"][bb]["scale_tau"]==1.0:
                    plus_data=plus_file["p1d_data"][bb]
                    minus_data=minus_file["p1d_data"][bb]
                    
                    
            assert plus_data["scale_tau"]==minus_data["scale_tau"]
            
            ## P1D
            p1d_plus=np.asarray(plus_data["p1d_Mpc"])
            p1d_minus=np.asarray(minus_data["p1d_Mpc"])
            
            ## k bins
            k_Mpc_plus=np.asarray(plus_data['k_Mpc'])
            k_Mpc_minus=np.asarray(minus_data['k_Mpc'])
            assert len(k_Mpc_plus)==len(k_Mpc_minus), "k bins are different in plus and minus sims"
            
            ## IGM parameters
            mf_plus=plus_data["mF"]
            mf_minus=minus_data["mF"]
            pair_mf=0.5*(plus_data["mF"]
                             +minus_data["mF"])
            
            kF_Mpc=0.5*(plus_data["kF_Mpc"]+minus_data["kF_Mpc"])
            gamma=0.5*(plus_data["sim_gamma"]+minus_data["sim_gamma"])
            sigT_Mpc=0.5*(plus_data["sim_sigT_Mpc"]+minus_data["sim_sigT_Mpc"])
            
            ## Find paired P_1D
            p1d_combined=0.5*(p1d_plus * mf_plus**2
                                + p1d_minus * mf_minus**2) / pair_mf**2
            
            ## Cut higher than k_max
            self.k_Mpc.append(k_Mpc_plus[k_Mpc_plus<kmax_Mpc])
            self.p1d_Mpc.append(p1d_combined[k_Mpc_plus<kmax_Mpc])
            
            ## Save emulator parameters
            emu_dict={}
            ## Add IGM parameters
            emu_dict["mF"]=pair_mf
            emu_dict["gamma"]=gamma
            emu_dict["kF_Mpc"]=kF_Mpc
            emu_dict["sigT_Mpc"]=sigT_Mpc
            ## Add linear power parameters
            ## These are stored starting earliest redshift first
            ## so we flip the ordering of the index
            emu_dict["Delta2_p"]=linP_values[len(zs)-aa-1]["Delta2_p"]
            emu_dict["n_p"]=linP_values[len(zs)-aa-1]["n_p"]
            emu_dict["alpha_p"]=linP_values[len(zs)-aa-1]["alpha_p"]
            emu_dict["f_p"]=linP_values[len(zs)-aa-1]["f_p"]
            self.emu_calls.append(emu_dict)
            
        # Not all redshifts will have the same number of wavenumbers, because
        # of annoying numerical errors
        # self.k_Mpc=self.k_Mpc[0] ## Discard other k bins, they are the same

        return
        
        
    def get_emulator_calls(self,z):
        """ For a given z, return the emulator parameters for this sim 
        in the form of the dictionary that is passed to our emulator """
        
        assert z in self.zs, "Do not have data for that redshift"
        
        return self.emu_calls[np.argwhere(self.zs==z)[0][0]]
    
    
    def get_p1d_Mpc(self,z):
        """ Return the P_1D and corresponding k bins for a given z """
        
        assert z in self.zs, "Do not have data for that redshift"
        
        return self.k_Mpc, self.p1d_Mpc[np.argwhere(self.zs==z)[0][0]]

