import numpy as np
import matplotlib.pyplot as plt
import camb_cosmo
import fit_linP
import recons_cosmo
import mean_flux_model
import thermal_model
import pressure_model
import linear_emulator
import CAMB_model

class FullTheory(object):
    """Translator between the likelihood object and the emulator. This object
    will map from a set of CAMB parameters directly to emulator calls, without
    going through our Delta^2_\star parametrisation """

    def __init__(self,zs,emulator=None,camb_model_fid=None,verbose=False,
                    mf_model_fid=None,T_model_fid=None,kF_model_fid=None):
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
        if camb_model_fid:
            self.camb_model_fid=camb_model_fid
        else:
            self.camb_model_fid=CAMB_model.CAMBModel(zs=self.zs)

        # setup fiducial IGM models
        if mf_model_fid:
            self.mf_model_fid = mf_model_fid
        else:
            self.mf_model_fid = mean_flux_model.MeanFluxModel()
        if T_model_fid:
            self.T_model_fid = T_model_fid
        else:
            self.T_model_fid = thermal_model.ThermalModel()
        if kF_model_fid:
            self.kF_model_fid = kF_model_fid
        else:
            self.kF_model_fid = pressure_model.PressureModel()


    def get_emulator_calls(self,like_params=[],return_M_of_z=True):
        """Compute models that will be emulated, one per redshift bin"""

        # setup IMG models using list of likelihood parameters
        igm_models=self.get_igm_models(like_params)
        mf_model=igm_models['mf_model']
        T_model=igm_models['T_model']
        kF_model=igm_models['kF_model']

        camb_model=self.camb_model_fid.get_new_model(like_params)

        # compute linear power parameters at all redshifts
        # (ideally, we would specify here the pivot point from the emulator)
        linP_Mpc_params=camb_model.get_linP_Mpc_params()
        M_of_zs=camb_model.get_M_of_zs()

        # loop over redshifts and store emulator calls
        emu_calls=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # emulator parameters for linear power, at this redshift (in Mpc)
            model=linP_Mpc_params[iz]
            # emulator parameters for nuisance models, at this redshift
            model['mF']=mf_model.get_mean_flux(z)
            model['gamma']=T_model.get_gamma(z)
            sigT_kms=T_model.get_sigT_kms(z)
            model['sigT_Mpc']=sigT_kms/M_of_zs[iz]
            kF_kms=kF_model.get_kF_kms(z)
            model['kF_Mpc']=kF_kms*M_of_zs[iz]
            if self.verbose: print(iz,z,'model',model)
            emu_calls.append(model)

        if return_M_of_z==True:
            return emu_calls,M_of_zs
        else:
            return emu_calls


    def get_p1d_kms(self,k_kms,like_params=[],return_covar=False):
        """Emulate P1D in velocity units, for all redshift bins,
            as a function of input likelihood parameters.
            It might also return a covariance from the emulator."""

        if self.emulator is None:
            raise ValueError('no emulator provided')

        # figure out emulator calls, one per redshift
        emu_calls,M_of_z=self.get_emulator_calls(like_params=like_params)

        # loop over redshifts and compute P1D
        p1d_kms=[]
        if return_covar:
            covars=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # will call emulator for this model
            model=emu_calls[iz]
            # emulate p1d
            k_Mpc = k_kms * M_of_z[iz]
            if self.verbose:
                print('M(z={}) = {}'.format(z,M_of_z[iz]))
            if return_covar:
                p1d_Mpc, cov_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=True,
                                                        z=z)
            else:
                p1d_Mpc = self.emulator.emulate_p1d_Mpc(model,k_Mpc,
                                                        return_covar=False,
                                                        z=z)
            if p1d_Mpc is None:
                if self.verbose: print('emulator did not provide P1D')
                p1d_kms.append(None)
                if return_covar:
                    covars.append(None)
            else:
                p1d_kms.append(p1d_Mpc * M_of_z[iz])
                if return_covar:
                    if cov_Mpc is None:
                        covars.append(None)
                    else:
                        covars.append(cov_Mpc * M_of_z[iz]**2)

        if return_covar:
            return p1d_kms,covars
        else:
            return p1d_kms


    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        # get parameters from CAMB model
        params=self.camb_model_fid.get_likelihood_parameters()

        # get parameters from nuisance models
        for par in self.mf_model_fid.get_parameters():
            params.append(par)
        for par in self.T_model_fid.get_sigT_kms_parameters():
            params.append(par)
        for par in self.T_model_fid.get_gamma_parameters():
            params.append(par)
        for par in self.kF_model_fid.get_parameters():
            params.append(par)

        if self.verbose:
            print('got parameters')
            for par in params:
                print(par.info_str())

        return params


    def get_igm_models(self,like_params=[]):
        """Setup IGM models from input list of likelihood parameters"""

        mf_model = self.mf_model_fid.get_new_model(like_params)
        T_model = self.T_model_fid.get_new_model(like_params)
        kF_model = self.kF_model_fid.get_new_model(like_params)

        models={'mf_model':mf_model,'T_model':T_model,'kF_model':kF_model}

        return models


    def plot_p1d(self,k_kms,like_params=[],plot_every_iz=1):
        """Emulate and plot P1D in velocity units, for all redshift bins,
            as a function of input likelihood parameters"""

        # ask emulator prediction for P1D in each bin
        emu_p1d=self.get_p1d_kms(k_kms,like_params)

        # plot only few redshifts for clarity
        Nz=len(self.zs)
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=self.zs[iz]
            p1d=emu_p1d[iz]
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.plot(k_kms,p1d*k_kms/np.pi,color=col,label='z=%.1f'%z)
        plt.yscale('log')
        plt.legend()
        plt.xlabel('k [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.show()

        return
