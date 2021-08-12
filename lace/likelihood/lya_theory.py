import numpy as np
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.likelihood import recons_cosmo
from lace.nuisance import mean_flux_model
from lace.nuisance import thermal_model
from lace.nuisance import pressure_model

class LyaTheory(object):
    """Translator between the likelihood object and the emulator."""

    def __init__(self,zs,emulator,cosmo_fid=None,verbose=False,
                    mf_model_fid=None,T_model_fid=None,kF_model_fid=None,
                    use_camb_fz=True,fit_kmin_kp=0.5,fit_kmax_kp=2.0):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - emulator: object to interpolate simulated p1d
            - cosmo_fid: CAMB object with the fiducial cosmology (optional)
            - verbose: print information, useful to debug
            - fit_kmin_kp: minimum k to use in linP fit (over kp_kms)
            - fit_kmax_kp: maximum k to use in linP fit (over kp_kms). """

        self.verbose=verbose
        self.zs=zs
        self.emulator=emulator

        # specify pivot point to be used in emulator calls
        if self.emulator is None:
            print('using default values for emulator pivot point')
            emu_kp_Mpc=0.7
        else:
            emu_kp_Mpc=self.emulator.archive.kp_Mpc

        # for now, used default pivot point for likelihood parameters
        like_z_star=3.0
        like_kp_kms=0.009

        # setup object to compute linear power for any cosmology
        self.cosmo=recons_cosmo.ReconstructedCosmology(zs,
                emu_kp_Mpc=emu_kp_Mpc,
                like_z_star=like_z_star,like_kp_kms=like_kp_kms,
                cosmo_fid=cosmo_fid,use_camb_fz=use_camb_fz,
                fit_kmin_kp=fit_kmin_kp,fit_kmax_kp=fit_kmax_kp,
                verbose=verbose)

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


    def get_emulator_calls(self,like_params=[],return_blob=False):
        """Compute models that will be emulated, one per redshift bin.
            - return_blob will return extra information about the call."""

        # setup linear power using list of likelihood parameters
        linP_model=self.cosmo.get_linP_model(like_params)
        # setup IMG models using list of likelihood parameters
        igm_models=self.get_igm_models(like_params)
        mf_model=igm_models['mf_model']
        T_model=igm_models['T_model']
        kF_model=igm_models['kF_model']

        # compute linear power parameters at all redshifts
        # (recons_cosmo already knows the pivot point emu_kp_Mpc)
        linP_Mpc_params=self.cosmo.get_linP_Mpc_params(linP_model)

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
            dkms_dMpc=self.cosmo.reconstruct_Hubble_iz(iz,linP_model)/(1+z)
            model['sigT_Mpc']=sigT_kms/dkms_dMpc
            kF_kms=kF_model.get_kF_kms(z)
            model['kF_Mpc']=kF_kms*dkms_dMpc
            if self.verbose: print(iz,z,'model',model)
            emu_calls.append(model)

        if return_blob:
            blob=self.get_blob(linP_model=linP_model)
            return emu_calls,blob
        else:
            return emu_calls


    def get_blobs_dtype(self):
        """Return the format of the extra information (blobs) returned
            by get_p1d_kms and used in emcee_sampler. """

        blobs_dtype = [('Delta2_star', float),('n_star', float),
                        ('alpha_star', float),('f_star', float),
                        ('g_star', float)]
        return blobs_dtype


    def get_blob(self,linP_model=None):
        """Return extra information (blob) for the emcee_sampler. """

        if linP_model is None:
            Nblob=len(self.get_blobs_dtype())
            if Nblob==1:
                return np.nan
            else:
                out=np.nan,*([np.nan]*(Nblob-1))
                return out
        else:
            params=linP_model.get_params()
            return params['Delta2_star'],params['n_star'], \
                    params['alpha_star'],params['f_star'],params['g_star']


    def get_p1d_kms(self,k_kms,like_params=[],return_covar=False,
                    camb_evaluation=None,return_blob=False):
        """Emulate P1D in velocity units, for all redshift bins,
            as a function of input likelihood parameters.
            It might also return a covariance from the emulator,
            or a blob with extra information for the emcee_sampler."""

        if self.emulator is None:
            raise ValueError('no emulator in LyaTheory')

        # figure out emulator calls, one per redshift
        if return_blob:
            emu_calls,blob=self.get_emulator_calls(like_params=like_params,
                                                    return_blob=True)
        else:
            emu_calls=self.get_emulator_calls(like_params=like_params,
                                                    return_blob=False)

        # setup linear power using list of likelihood parameters
        # we will need this to reconstruct H(z)
        linP_model=self.cosmo.get_linP_model(like_params=like_params)

        # loop over redshifts and compute P1D
        p1d_kms=[]
        if return_covar:
            covars=[]
        Nz=len(self.zs)
        for iz,z in enumerate(self.zs):
            # will call emulator for this model
            model=emu_calls[iz]
            # emulate p1d
            dkms_dMpc=self.cosmo.reconstruct_Hubble_iz(iz,linP_model)/(1+z)

            k_Mpc = k_kms * dkms_dMpc
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
                p1d_kms.append(p1d_Mpc * dkms_dMpc)
                if return_covar:
                    if cov_Mpc is None:
                        covars.append(None)
                    else:
                        covars.append(cov_Mpc * dkms_dMpc**2)

        # decide what to return, and return it
        if return_covar:
            if return_blob:
                return p1d_kms,covars,blob
            else:
                return p1d_kms,covars
        else:
            if return_blob:
                return p1d_kms,blob
            else:
                return p1d_kms


    def get_parameters(self):
        """Return parameters in models, even if not free parameters"""

        params=self.cosmo.linP_model_fid.get_likelihood_parameters()
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


def get_mock_theory(zs,emulator=None,cosmo_fid=None,verbose=False,
                alternative_pressure=False):
    """Setup LyaTheory with nuisance models close to that from a mock
        dataset from a MP-Gadget simulation."""

    raise ValueError('update function if ever needed')

    # setup mean flux matching simulation outputs
    ln_tau_0 = -1.03436530241446
    ln_tau_1 = 3.6744666006830182
    ln_tau_coeff=[ln_tau_1,ln_tau_0]
    mf_model_fid = mean_flux_model.MeanFluxModel(ln_tau_coeff=ln_tau_coeff)

    # setup pressure model matching simulation
    if alternative_pressure:
        # pressure in current 1024 simulation is crazy low
        ln_kF_coeff=[0.5,np.log(0.25)]
    else:
        ln_kF_0 =  -0.8077668277205104
        ln_kF_1 =  1.9001923998886694
        ln_kF_coeff=[ln_kF_1,ln_kF_0]
    kF_model_fid = pressure_model.PressureModel(ln_kF_coeff=ln_kF_coeff)

    # setup thermal model matching simulation
    ln_gamma_0 =  0.3295042060454974
    ln_gamma_1 =  -0.2521703939255174
    ln_gamma_coeff=[ln_gamma_1,ln_gamma_0]
    T0_1 =  0.13626544653787526
    T0_2 =  9.546039892898634
    T0_3 =  -1.2041429220366868
    ln_T0_coeff=[T0_1,T0_2,T0_3]
    T_model_fid = thermal_model.ThermalModel(ln_T0_coeff=ln_T0_coeff,
                                                ln_gamma_coeff=ln_gamma_coeff)

    theory=LyaTheory(zs,emulator=emulator,cosmo_fid=cosmo_fid,
                    verbose=verbose,mf_model_fid=mf_model_fid,
                    T_model_fid=T_model_fid,kF_model_fid=kF_model_fid)

    return theory
