import numpy as np
import os
import json
import h5py
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.emulator import poly_p1d
from lace_manager.nuisance import thermal_model

class TestNyxSimulation(object):
    """ Object to store parameters and data for one
    specific test simulation from the Nyx suite. Used for performing tests
    either on the emulator directly in Mpc or on the sampler
    in velocity units """

    def __init__(self,fname,sim_label,z_max=4.0,kp_Mpc=0.7,verbosity=0):
        """ Extract data from a chosen Nyx simulation
            - fname file containing the whole Nyx fuite
            - sim_label can be either:
                -- an integer, index from the training suite
                -- "central", fiducial simulation in Nyx
            - z_max sets the highest z cut
            - kp_Mpc sets the comoving pivot scale used to calculate the
              emulator linear power parameters
        """

        if not fname:
            assert 'LACE_MANAGER_REPO' in os.environ,'export LACE_MANAGER_REPO'
            repo=os.environ['LACE_MANAGER_REPO']
            fname=repo+'/lace_manager/emulator/sim_suites/test_nyx/models.hdf5'
            if verbosity>0:
                print('read Nyx archive from file',fname)
        self.fname=fname
        self.verbosity=verbosity

        if type(sim_label)==int:
            self.model_key='cosmo_grid_{}'.format(sim_label)
            self.thermal_key='thermal_grid_0'
        elif sim_label=='central':
            self.model_key='fiducial'
            self.thermal_key='rescale_Fbar_fiducial'
        else:
            print(sim_label," simulation not found")
            
        self._read_file(z_max,kp_Mpc)

        return


    def _read_file(self,z_max,kp_Mpc):
        """ Read the Nyx file and store P1D and emulator parameters.
            - z_max: discard redshifts above this cut
            - kp_Mpc: pivot point to compute linear power params """

        # open HDF5 file
        if self.verbosity>0: print('will read Nyx file',self.fname)
        f = h5py.File(self.fname, 'r')

        # figure out redshifts in the simulation
        redshiftstrlist=list(f[self.model_key].keys())
        self.zs=[float(s.split('_')[1]) for s in redshiftstrlist]
        if self.verbosity>0: print('zs =',self.zs)

        # setup simulation cosmology
        attrs_global=dict(f[self.model_key].attrs.items())
        if 'grid' not in self.model_key:
            attrs_global['A_s']=2.1e-9
            attrs_global['n_s']=0.966
        self.sim_cosmo=camb_cosmo.get_Nyx_cosmology(attrs_global)
        if self.verbosity>0: camb_cosmo.print_info(self.sim_cosmo)

        # compute linear power parameters at each z (in Mpc units)
        linP_zs=fit_linP.get_linP_Mpc_zs(self.sim_cosmo,self.zs,kp_Mpc,
                include_f_p=True)
        if self.verbosity>1: print('linP_zs',linP_zs)
        linP_values=list(linP_zs)

        # store metadata and measurements at each redshift
        self.p1d_Mpc=[]
        self.k_Mpc=[] 
        self.emu_calls=[]

        for iz,z in enumerate(self.zs):
            print(iz,'z',z)
            zstr=redshiftstrlist[iz]
            print(zstr)

            # convertion factor from Mpc to km/s
            dkms_dMpc=camb_cosmo.dkms_dMpc(self.sim_cosmo,z=z)
            if self.verbosity>1:
                print('at z = {}, 1 Mpc = {:.2f} km/s'.format(z,dkms_dMpc))

            # get linear power parameters describing snapshot
            linP_params = linP_zs[iz]
            emu_call = {}
            emu_call['Delta2_p'] = linP_params['Delta2_p']
            emu_call['n_p'] = linP_params['n_p']
            emu_call['alpha_p'] = linP_params['alpha_p']
            emu_call['f_p'] = linP_params['f_p']

            # read IGM parameters for a particular thermal rescaling
            thermal_str='{}/{}/{}'.format(self.model_key,zstr,self.thermal_key)
            print('thermal string',thermal_str)
            thermal_params=dict(f[thermal_str].attrs.items())
            print(thermal_params)
            emu_call['mF']=thermal_params['Fbar']
            T0=thermal_params['T_0']
            emu_call['T0']=T0
            emu_call['gamma']=thermal_params['gamma']
            # compute thermal broadening in Mpc
            sigma_T_kms=thermal_model.thermal_broadening_kms(T0)
            sigma_T_Mpc=sigma_T_kms/dkms_dMpc
            emu_call['sigT_Mpc']=sigma_T_Mpc

            # store emulator call for this redshift
            print('emulator call',emu_call)
            self.emu_calls.append(emu_call)

            ## store 1D power spectrum (cut higher than k_max=30 1/Mpc)
            kmax_Mpc=30.0
            p1d_data=f[thermal_str]['1d power']
            k_Mpc=p1d_data['k']
            print('k_0',k_Mpc[0])
            p1d_Mpc=p1d_data['Pk1d']
            self.k_Mpc.append(k_Mpc[k_Mpc<kmax_Mpc])
            self.p1d_Mpc.append(p1d_Mpc[k_Mpc<kmax_Mpc])
            
        self.k_Mpc=self.k_Mpc[0] ## Discard other k bins, they are the same

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


    def get_polyfit_p1d_Mpc(self,z,fit_kmax_Mpc,deg=4):
        """ Return "smoothed" P_1D and correspondong k bins """
        
        assert z in self.zs, "Do not have data for that redshift"

        ## Fit polynomial
        fit_p1d = poly_p1d.PolyP1D(self.k_Mpc,
                    self.p1d_Mpc[np.argwhere(self.zs==z)[0][0]],
                    kmin_Mpc=1.e-3,kmax_Mpc=fit_kmax_Mpc,deg=deg)

        p1d_poly=fit_p1d.P_Mpc(self.k_Mpc)
        
        return self.k_Mpc, p1d_poly
