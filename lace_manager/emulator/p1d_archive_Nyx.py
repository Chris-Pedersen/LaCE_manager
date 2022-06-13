import numpy as np
import os
import json
import h5py
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace_manager.nuisance import thermal_model

class archiveP1D_Nyx(object):
    """Book-keeping of flux P1D measured in a suite of Nyx simulations."""

    def __init__(self,fname=None,kp_Mpc=0.7,verbose=False):
        """Load archive from models file.
            linP params will be computed around kp_Mpc."""

        if not fname:
            assert ('LACE_MANAGER_REPO' in os.environ),'export LACE_MANAGER_REPO'
            repo=os.environ['LACE_REPO']
            fname=repo+'/lace_manager/emulator/sim_suites/test_nyx/models.hdf5'
            if verbose:
                print('read Nyx archive from file',fname)

        self.fname=fname
        self.basedir='basedir_Nyx'
        self.fulldir='fulldir_Nyx'
        self.p1d_label='p1d_Nyx'
        self.skewers_label='skewers_Nyx'
        self.verbose=verbose
        self.drop_tau_rescalings=False
        self.drop_temp_rescalings=False
        self.nearest_tau=False
        self.z_max=None
        self.undersample_cube=False
        self.drop_sim_number=None
        self.kp_Mpc=kp_Mpc

        self._load_data()
        
        return


    def _load_data(self):
        """Setup archive by looking at all measured power spectra in sims"""

        # each measured power will have a dictionary, stored here
        self.data=[]

        # open Nyx file with all models and measured power
        if self.verbose:
            print('will read Nyx file',self.fname)
        f = h5py.File(self.fname, 'r')

        # figure out models in simulation grid
        modellist=list(f.keys())
        redshiftstrlist=list((f[m].keys() for m in modellist))
        used_modellist= [m for m,r in zip(modellist,redshiftstrlist) \
                        if len(r)>0 and 'grid' in m]
        used_attrs_global=[dict(f[m].attrs.items())  for m in used_modellist]
        self.nsamples=len(used_attrs_global)
        self.cube_data={'param_names':list(used_attrs_global[0].keys()),
                        'nsamples':self.nsamples,
                        'samples':used_attrs_global}
        if self.verbose:
            print('number of samples',self.cube_data['nsamples'])
            print('parameter names',self.cube_data['param_names'])

        # figure out redshifts that are available for every model
        used_redshiftstrlist=[r for m,r in zip(modellist,redshiftstrlist) \
                        if len(r)>0 and 'grid' in m]
        redshiftslist=[[float(s.split('_')[1]) for s in r] \
                        for r in used_redshiftstrlist]
        all_redshifts=np.unique([z for r in redshiftslist for z in r])        
        used_redshifts=[]
        for z in all_redshifts:
            if np.all([z in r for r in redshiftslist]):
                used_redshifts.append(z)
        if self.verbose:
            print('will use redshift grid',used_redshifts)

        # get thermal parameters in grid
        thermal_grid_str=[[[t for t in f[f'{m}/redshift_{s:.1f}'].keys() \
                        if 'thermal' in t] for m in used_modellist] \
                        for s in used_redshifts]
        #might add a check here for assuring all are same length etc
        thermal_grid_str=thermal_grid_str[0][0] 
        if self.verbose:
            print('thermal grid',thermal_grid_str)

        #this is for sorting by z first, then cosmology, then thermal state
        modelgroupstrings=[[[f'{m}/redshift_{z}/{t}' \
                        for t in thermal_grid_str] for m in used_modellist] \
                        for z in used_redshifts] 
        #the stuff inside dict() is for adding thermal and cosmo pars together
        model_grid_attrs=[[[dict(f[m3].attrs.items(),**cosmo_attrs) \
                        for m3 in m2] for m2,cosmo_attrs \
                        in zip(m1,used_attrs_global)] \
                        for m1 in modelgroupstrings] 
        model_grid_data=[[[f[m3]['1d power'][:] for m3 in m2] for m2 in m1] \
                        for m1 in modelgroupstrings]
        model_grid_redshifts=np.array(used_redshifts)

        # loop over cosmo, z, thermal
        for im,sim in enumerate(used_modellist):
            sim_params=used_attrs_global[im]
            if self.verbose:
                print(sim,'params',sim_params)
            # setup CAMB object from sim_params
            sim_cosmo=camb_cosmo.get_Nyx_cosmology(sim_params)
            # compute linear power parameters at each z (in Mpc units)
            linP_zs=fit_linP.get_linP_Mpc_zs(sim_cosmo,
                        used_redshifts,self.kp_Mpc,include_f_p=True)

            # loop over z, sim, thermal
            for iz,z in enumerate(used_redshifts):
                # convert kms to Mpc (should be around 75 km/s/Mpc at z=3)
                dkms_dMpc=camb_cosmo.dkms_dMpc(sim_cosmo,z=z)
                if self.verbose:
                    print('at z = {}, 1 Mpc = {} km/s'.format(z,dkms_dMpc))

                # get linear power parameters describing snapshot
                linP_params = linP_zs[iz]
                snap_p1d_data = {}
                snap_p1d_data['Delta2_p'] = linP_params['Delta2_p']
                snap_p1d_data['n_p'] = linP_params['n_p']
                snap_p1d_data['alpha_p'] = linP_params['alpha_p']
                snap_p1d_data['f_p'] = linP_params['f_p']
                snap_p1d_data['z']=z

                for it,thermal in enumerate(thermal_grid_str):
                    # deep copy of dictionary (thread safe, why not)
                    p1d_data = json.loads(json.dumps(snap_p1d_data))
                    # add measured P1D
                    p1d_data['k_Mpc']=np.array(model_grid_data[iz][im][it]['k'])
                    p1d_data['p1d_Mpc']=np.array(
                                model_grid_data[iz][im][it]['Pk1d'])
                    # add thermal parameters
                    thermal_str=modelgroupstrings[iz][im][it]
                    thermal_params=dict(f[thermal_str].attrs.items())
                    if self.verbose:
                        print(thermal,'params',thermal_params)
                    p1d_data['mF']=thermal_params['Fbar']
                    T0=thermal_params['T_0']
                    p1d_data['T0']=T0
                    p1d_data['gamma']=thermal_params['gamma']
                    # compute thermal broadening in Mpc
                    sigma_T_kms=thermal_model.thermal_broadening_kms(T0)
                    sigma_T_Mpc=sigma_T_kms/dkms_dMpc
                    p1d_data['sigT_Mpc']=sigma_T_Mpc
                    self.data.append(p1d_data)                

        N=len(self.data)
        if self.verbose:
            print('archive setup, containing %d entries'%len(self.data))

        # create 1D arrays with all entries for a given parameter
        self._store_param_arrays()

        return


    def _store_param_arrays(self):
        """ create 1D arrays with all entries for a given parameter. """

        N=len(self.data)

        # store linear power parameters
        self.Delta2_p=np.array([self.data[i]['Delta2_p'] for i in range(N)])
        self.n_p=np.array([self.data[i]['n_p'] for i in range(N)])
        self.alpha_p=np.array([self.data[i]['alpha_p'] for i in range(N)])
        self.f_p=np.array([self.data[i]['f_p'] for i in range(N)])
        self.z=np.array([self.data[i]['z'] for i in range(N)])

        # store IGM parameters (if present)
        if 'mF' in self.data[0]:
            self.mF=np.array([self.data[i]['mF'] for i in range(N)])
        if 'sigT_Mpc' in self.data[0]:
            self.sigT_Mpc=np.array([self.data[i]['sigT_Mpc'] for i in range(N)])
        if 'gamma' in self.data[0]:
            self.gamma=np.array([self.data[i]['gamma'] for i in range(N)])
        if 'kF_Mpc' in self.data[0]:
            self.kF_Mpc=np.array([self.data[i]['kF_Mpc'] for i in range(N)])

        return


    def print_entry(self,entry,keys=['z','Delta2_p','n_p','alpha_p','f_p',
                                    'mF','sigT_Mpc','gamma','kF_Mpc']):
        """Print basic information about a particular entry in the archive"""

        if entry >= len(self.data):
            raise ValueError('{} entry does not exist in archive'.format(entry))

        data = self.data[entry]
        info='entry = {}'.format(entry)
        for key in keys:
            info += ', {} = {:.4f}'.format(key,data[key])
        print(info)
        return


    def plot_samples(self,param_1,param_2):
        """For parameter pair (param1,param2), plot each point in the archive"""

        emu_data=self.data
        Nemu=len(emu_data)

        # figure out values of param_1,param_2 in archive
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu)])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu)])

        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu)])
        zmin=min(emu_z)
        zmax=max(emu_z)
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return


    def plot_3D_samples(self,param_1,param_2, param_3):
        """For parameter trio (param1,param2,param3), plot each point in the archive"""
        from mpl_toolkits import mplot3d

        emu_data=self.data
        Nemu=len(emu_data)

        # figure out values of param_1,param_2 in archive
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu)])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu)])
        emu_3=np.array([emu_data[i][param_3] for i in range(Nemu)])

        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu)])
        zmin=min(emu_z)
        zmax=max(emu_z)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(emu_1, emu_2, emu_3, c=emu_z, cmap='brg',s=8)
        ax.set_xlabel(param_1)
        ax.set_ylabel(param_2)
        ax.set_zlabel(param_3)
        plt.show()

        return

