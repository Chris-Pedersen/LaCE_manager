import numpy as np
import camb_cosmo
import fit_linP
import recons_cosmo
import thermal_model

class LyaTheory(object):
    """Translator between the likelihood object and the emulator."""

    def __init__(self,zs,emulator,cosmo_fid=None,verbose=True): 
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - cosmo_fid: CAMB object with the fiducial cosmology (optional)
            - verbose: print information, useful to debug."""

        self.verbose=verbose
        self.zs=zs
        self.emulator=emulator
        # setup object to compute linear power for any cosmology
        self.cosmo=recons_cosmo.ReconstructedCosmology(zs,cosmo_fid=cosmo_fid)

        # at this point we do not know the models to evaluate
        self.mf_model = None
        self.T_model = None


    def set_mf_model(self,mf_model):
        self.mf_model=mf_model


    def set_T_model(self,T_model):
        self.T_model=T_model


    def set_cosmo_model(self,linP_model):
        self.cosmo.linP_model=linP_model

    
    def get_emulator_calls(self,linP_Mpc_params=None):
        """Compute models that will be emulated, one per redshift bin"""

        # compute linear power parameters at all redshifts
        if not linP_Mpc_params:
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
            if self.verbose: print(iz,z,'model',model)
            emu_calls.append(model)

        return emu_calls


    def get_p1d_kms(self,k_kms,linP_Mpc_params=None):
        """Emulate P1D in velocity units, for all redshift bins"""

        # figure out emulator calls, one per redshift
        emu_calls=self.get_emulator_calls(linP_Mpc_params)

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
            p1d_kms.append(p1d_Mpc * dkms_dMpc)

        return p1d_kms

