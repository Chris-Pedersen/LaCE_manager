import numpy as np
import camb_cosmo
import fit_linP
import recons_cosmo
import mean_flux_model
import thermal_model
import pressure_model
import linear_emulator

class LyaTheory(object):
    """Translator between the likelihood object and the emulator."""

    def __init__(self,zs,emulator=None,cosmo_fid=None,
            mf_model=None,T_model=None,kF_model=None,verbose=False):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - cosmo_fid: CAMB object with the fiducial cosmology (optional)
            - verbose: print information, useful to debug."""

        self.verbose=verbose
        self.zs=zs
        if emulator:
            self.emulator=emulator
        else:
            self.emulator=linear_emulator.LinearEmulator(verbose=verbose)

        # setup object to compute linear power for any cosmology
        self.cosmo=recons_cosmo.ReconstructedCosmology(zs,cosmo_fid=cosmo_fid)

        # setup mean flux model
        if mf_model:
            self.mf_model = mf_model
        else:
            if self.verbose: print('use default mean flux model')
            self.mf_model = mean_flux_model.MeanFluxModel()

        # setup thermal model
        if T_model:
            self.T_model = T_model
        else:
            if self.verbose: print('use default thermal model')
            self.T_model = thermal_model.ThermalModel()

        # setup pressure smoothing model
        if kF_model:
            self.kF_model = kF_model
        else:
            if self.verbose: print('use default pressure model')
            self.kF_model = pressure_model.PressureModel()


    def set_mf_model(self,mf_model):
        self.mf_model=mf_model

    def set_T_model(self,T_model):
        self.T_model=T_model

    def set_kF_model(self,kF_model):
        self.kF_model=kF_model


    def set_cosmo_model(self,linP_model):
        self.cosmo.linP_model=linP_model

    
    def get_emulator_calls(self):
        """Compute models that will be emulated, one per redshift bin"""

        # compute linear power parameters at all redshifts
        linP_Mpc_params=self.cosmo.get_linP_Mpc_params()

        # loop over redshifts and store emulator calls
        emu_calls=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # emulator parameters for linear power, at this redshift (in Mpc)
            model=linP_Mpc_params[iz]
            # emulator parameters for nuisance models, at this redshift
            model['mF']=self.mf_model.get_mean_flux(z)
            model['gamma']=self.T_model.get_gamma(z)
            T0=self.T_model.get_T0(z)
            sigT_kms=thermal_model.thermal_broadening_kms(T0)
            dkms_dMpc=self.cosmo.reconstruct_Hubble_iz(iz)/(1+z)
            model['sigT_Mpc']=sigT_kms/dkms_dMpc
            kF_kms=self.kF_model.get_kF_kms(z)
            model['kF_Mpc']=kF_kms*dkms_dMpc
            if self.verbose: print(iz,z,'model',model)
            emu_calls.append(model)

        return emu_calls


    def get_p1d_kms(self,k_kms):
        """Emulate P1D in velocity units, for all redshift bins"""

        # figure out emulator calls, one per redshift
        emu_calls=self.get_emulator_calls()

        # loop over redshifts and compute P1D
        p1d_kms=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # will call emulator for this model
            model=emu_calls[iz]
            # emulate p1d
            dkms_dMpc=self.cosmo.reconstruct_Hubble_iz(iz)/(1+z)
            k_Mpc = k_kms * dkms_dMpc
            p1d_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc)
            if p1d_Mpc is None:
                if self.verbose: print('emulator did not provide P1D')
                p1d_kms.append(None)
            else:
                p1d_kms.append(p1d_Mpc * dkms_dMpc)

        return p1d_kms


    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        params=self.cosmo.linP_model.get_likelihood_parameters()
        for par in self.mf_model.get_parameters():
            params.append(par)
        for par in self.T_model.get_T0_parameters():
            params.append(par)
        for par in self.T_model.get_gamma_parameters():
            params.append(par)
        for par in self.kF_model.get_parameters():
            params.append(par)

        if self.verbose:
            print('got parameters')
            for par in params:
                print(par.info_str())

        return params


    def update_parameters(self,parameters):
        """Update internal theories with input list of parameters"""

        # count how many have been updated
        counts=self.cosmo.linP_model.update_parameters(parameters)
        if self.verbose: print('updated',counts,'linP parameters')
        counts+=self.mf_model.update_parameters(parameters)
        if self.verbose: print('updated',counts,'after mean flux parameters')
        counts+=self.T_model.update_parameters(parameters)
        if self.verbose: print('updated',counts,'after thermal parameters')
        counts+=self.kF_model.update_parameters(parameters)
        if self.verbose: print('updated',counts,'after pressure parameters')

        return

