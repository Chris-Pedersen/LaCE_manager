import numpy as np
import os
import camb
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from lace.likelihood import linear_power_model

class ReconstructedCosmology(object):
    """Given fiducial cosmology, and set of linear power parameters, 
        reconstruct a cosmology object."""

    def __init__(self,zs,emu_kp_Mpc,like_z_star,like_kp_kms,
                cosmo_fid=None,use_camb_fz=True,
                fit_kmin_kp=0.5,fit_kmax_kp=2.0,verbose=False):
        """Setup object to reconstruct cosmology from linear power parameters.
            - zs: redshifts where we want predictions (call emulator)
            - emu_kp_Mpc: pivot point in Mpc used in the emulator
            - like_z_star: central redshift in likelihood parameterization
            - like_kp_kms: pivot point in likelihood parameterization (s/km)
            - cosmo_fid: CAMB object describing fiducial cosmology
            - use_camb_fz: use CAMB to compute f_star (faster)
            - fit_kmin_kp: minimum k to use in linP fit (over kp_kms)
            - fit_kmax_kp: maximum k to use in linP fit (over kp_kms)
            - verbose: print information for debugging."""

        self.verbose=verbose
        # store redshifts that will be evaluated
        self.zs=zs
        # store pivot point used in emulator to define linear parameters
        self.emu_kp_Mpc=emu_kp_Mpc
        # use CAMB to compute f_star
        self.use_camb_fz=use_camb_fz

        # fiducial cosmology
        if cosmo_fid:
            if self.verbose: print('use input fiducial cosmology')
            self.cosmo_fid=cosmo_fid
        else:
            if self.verbose: print('use default fiducial cosmology')
            self.cosmo_fid=camb_cosmo.get_cosmology()

        # compute CAMB results for fiducial cosmology
        self.camb_results_fid=camb_cosmo.get_camb_results(self.cosmo_fid,
                zs=self.zs,fast_camb=True)
        if self.verbose: print('got camb_results for fiducial cosmology')

        # compute linear power model for fiducial cosmology
        self.linP_model_fid=linear_power_model.LinearPowerModel(
                cosmo=self.cosmo_fid,
                camb_results=self.camb_results_fid,
                z_star=like_z_star,kp_kms=like_kp_kms,
                use_camb_fz=self.use_camb_fz,
                fit_kmin_kp=fit_kmin_kp,fit_kmax_kp=fit_kmax_kp)
        if self.verbose: print('setup linP model for fiducial cosmology')

        # store pivot point for convenience
        self.z_star=self.linP_model_fid.z_star
        self.kp_kms=self.linP_model_fid.kp_kms

        # get Hubble at z_star for fiducial cosmology
        self.H_star_fid=self.camb_results_fid.hubble_parameter(self.z_star)

        # store Hubble parameter at all redshifts for fiducial cosmology
        self._cache_Hz_fid()
        if self.verbose: print('cached H(z) for fiducial cosmology')

        # store linear power at all redshifts for fiducial cosmology
        self._cache_f_p_fid()
        if self.verbose: print('cached f(z) for fiducial cosmology')

        # store fiducial linear power at all redshifts, in km/s
        self._cache_linP_kms_fid()
        if self.verbose: print('cached linP_kms for fiducial cosmology')
    
        # when running with fixed cosmology it is useful to keep this
        self.linP_Mpc_params_fid=self._compute_linP_Mpc_params(
                                            linP_model=self.linP_model_fid)
        if self.verbose: print('got linP_Mpc_params for fiducial cosmology')

        return


    def _cache_Hz_fid(self):
        """Compute Hubble parameter in fiducial cosmology, at all redshifts."""

        self.Hz_fid=[]
        for z in self.zs:
            Hz = self.camb_results_fid.hubble_parameter(z)
            self.Hz_fid.append(Hz)

        return


    def _cache_f_p_fid(self):
        """ Compute growth rate in fiducial cosmology, at all redshifts. """

        self.f_p_fid=[]
        for z in self.zs:
            if self.use_camb_fz:
                f_p=camb_cosmo.get_f_of_z(self.cosmo_fid,
                            self.camb_results_fid,z)
            else:
                f_p=fit_linP.compute_fz(self.cosmo_fid,z=z,
                            kp_Mpc=self.emu_kp_Mpc)
            self.f_p_fid.append(f_p)

        return


    def _cache_linP_kms_fid(self):
        """Fiducial linear power in velocity units, at all redshifts."""

        # call CAMB to get linear power for fiducial cosmology, in km/s
        k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(self.cosmo_fid,
                self.zs,camb_results=self.camb_results_fid)

        # make sure we didn't change the order of the redshift outputs
        assert zs_out[0] == self.zs[0], 'CAMB redshifts not sorted'

        self.k_kms_fid = k_kms
        self.linP_kms_fid = P_kms


    def _compute_linP_Mpc_params(self,linP_model):
        """Reconstruct linear power (in Mpc) for input linP_model and fit
            linear power parameters at each redshift."""

        # to better compare with CAMB_model, compute power in same k-range
        kmax_Mpc=camb_cosmo.camb_fit_kmax_Mpc
        kmin_Mpc=camb_cosmo.camb_kmin_Mpc
        npoints=camb_cosmo.camb_npoints
        k_Mpc=np.logspace(np.log10(kmin_Mpc),np.log10(kmax_Mpc),num=npoints)

        # we are interested in the linear power around kp_Mpc
        kp_Mpc=self.emu_kp_Mpc
        fit_kmin_Mpc=0.5*kp_Mpc
        fit_kmax_Mpc=2.0*kp_Mpc

        linP_Mpc_params=[]
        for iz,z in enumerate(self.zs):
            # reconstruct linear power at the redshift (in Mpc)
            linP_Mpc=self.reconstruct_linP_Mpc(iz,k_Mpc=k_Mpc,
                    linP_model=linP_model)
            # fit polynomial describing log linear power
            linP_fit=fit_linP.fit_polynomial(fit_kmin_Mpc/kp_Mpc,
                    fit_kmax_Mpc/kp_Mpc,k_Mpc/kp_Mpc,linP_Mpc,deg=2)
            # compute parameters used in emulator
            lnA_p=linP_fit[0]
            Delta2_p=np.exp(lnA_p)*kp_Mpc**3/(2*np.pi**2)
            n_p=linP_fit[1]
            # note that the curvature is alpha/2
            alpha_p=2.0*linP_fit[2]
            params={'Delta2_p':Delta2_p,'n_p':n_p,'alpha_p':alpha_p}

            # reconstruct logarithmic growth rate at the redshift
            if False:
                # currently not using it in the emulator
                f_p=self.reconstruct_f_p_iz(iz,linP_model)
                params['f_p']=f_p

            linP_Mpc_params.append(params)

        return linP_Mpc_params


    def linP_model_is_fiducial(self,linP_model):
        """Check if input linP model is same than fiducial"""

        in_params=linP_model.get_params()
        fid_params=self.linP_model_fid.get_params()

        for key,value in in_params.items():
            if value is not fid_params[key]:
                if self.verbose: print('not using fiducial linP_model')
                return False
        if self.verbose: print('using fiducial linP_model')
        return True


    def get_linP_Mpc_params(self,linP_model=None):
        """Reconstruct linear power (in Mpc) for input linP_model and fit
            linear power parameters at each redshift. """

        # check if we are asking for the fiducial model
        if not linP_model:
            if self.verbose: print('no input linP_model, use fiducial')
            return self.linP_Mpc_params_fid
        if self.linP_model_is_fiducial(linP_model):
            if self.verbose: print('input linP_model is fiducial')
            return self.linP_Mpc_params_fid

        if self.verbose: print('will compute parameters for input linP_model')
        return self._compute_linP_Mpc_params(linP_model)


    def reconstruct_linP_Mpc(self,iz,k_Mpc,linP_model):
        """ Use fiducial cosmology and linP_model to reconstruct linear power"""

        z = self.zs[iz]
        # will reconstruct power in units of km/s, then transform to Mpc
        dkms_dMpc = self.reconstruct_Hubble_iz(iz,linP_model)/(1+z)
        k_kms = k_Mpc / dkms_dMpc
        # reconstruct linear power in velocity units
        P_kms = self.reconstruct_linP_kms(iz,k_kms,linP_model)
        P_Mpc = P_kms / dkms_dMpc**3

        return P_Mpc


    def reconstruct_linP_kms(self,iz,k_kms,linP_model,true_cosmo=None,
            ignore_g_star=False,ignore_f_star=False):
        """ Use fiducial cosmology and linP_model to reconstruct power (km/s)
            Options for debugging / testing:
                - if true_cosmo is passed, use it to compute m(z) and g(z)
                - if ignore_g_star, use m(z)=1
                - if ignore_f_star, use g(z)=1
        """

        # evaluate linP at this redshift
        z = self.zs[iz]

        # pivot points
        z_star=self.z_star
        kp_kms=self.kp_kms

        # get parameters describing linear power for input cosmology
        linP_kms_params=linP_model.linP_params['linP_kms']
        # get parameters describing linear power for fiducial cosmology
        linP_kms_params_fid=self.linP_model_fid.linP_params['linP_kms']

        # conversion from Mpc to km/s in fiducial cosmology
        M_star_fid = self.H_star_fid / (1+z_star)
        Mz_fid = self.camb_results_fid.hubble_parameter(z) / (1+z)

        # use the true cosmology to test the approximations
        if true_cosmo is not None:
            # compute true m(z)
            true_camb_results=camb_cosmo.get_camb_results(true_cosmo)
            M_star = true_camb_results.hubble_parameter(z_star) / (1+z_star)
            Mz = true_camb_results.hubble_parameter(z) / (1+z)
            mz = (Mz/M_star) / (Mz_fid/M_star_fid)
            if self.verbose:
                print(z,(Mz/M_star),(Mz_fid/M_star_fid),'m(z)',mz)
            # compute true d(z)
            Dz_Dstar=compute_D_Dstar(true_cosmo,z,z_star)
            Dz_Dstar_fid=compute_D_Dstar(self.cosmo_fid,z,z_star)
            dz = Dz_Dstar/Dz_Dstar_fid
            if self.verbose:
                print(z,Dz_Dstar,Dz_Dstar_fid,'d(z)',dz)
        else:
            # use f_star to approximate d(z)
            if ignore_f_star:
                dz=1
            else:
                dz=self.reconstruct_dz(z,linP_model)
            # use g_star to approximate m(z)
            if ignore_g_star:
                mz=1
            else:
                mz=self.reconstruct_mz(z,linP_model)

        # B(q) describes the ratio of linear power at z_star
        # we want to evaluate it at q' = m(z) M_0(z) / M^0_star q

        qB = mz * Mz_fid / M_star_fid * k_kms
        if self.verbose:
            print(qB.shape,'qB',qB)
        # linP_kms_params actually store log(B) vs log(k_kms/kp_kms)
        lnqB_kp = np.log(qB / kp_kms)
        if self.verbose:
            print(lnqB_kp.shape,'lnqB_kp',lnqB_kp)
        lnB = linP_kms_params(lnqB_kp)-linP_kms_params_fid(lnqB_kp)
        if self.verbose:
            print(lnB.shape,'lnB',lnB)

        # Evaluate the fiducial linear power at q' = m(z) q
        qP = mz * k_kms
        # note that interp here might extrapolate with a constant value
        linP_kms_fid = np.interp(qP,self.k_kms_fid[iz],self.linP_kms_fid[iz])

        return linP_kms_fid * mz**3 * dz**2 * np.exp(lnB)


    def reconstruct_mz(self,z,linP_model):
        """ Use g_star differences to reconstruct m(z) function"""

        # compute difference in acceleration
        g_star=linP_model.get_g_star()
        g_star_fid=self.linP_model_fid.get_g_star()

        # approximate m(z) function
        z_star=self.z_star
        mz = ((1+z)/(1+z_star))**(1.5*(g_star-g_star_fid))
        return mz


    def reconstruct_dz(self,z,linP_model):
        """ Use f_star differences to reconstruct d(z) function"""

        # compute difference in acceleration
        f_star=linP_model.get_f_star()
        f_star_fid=self.linP_model_fid.get_f_star()

        # approximate d(z) function
        z_star=self.z_star
        dz = ((1+z)/(1+z_star))**(f_star_fid-f_star)
        return dz


    def reconstruct_Hubble_iz(self,iz,linP_model):
        """ Use fiducial cosmology and g_star to reconstruct Hubble parameter"""

        Hz_fid=self.Hz_fid[iz]
        z=self.zs[iz]

        # check if we are asking for the fiducial model
        if not linP_model:
            if self.verbose: print('use fiducial linP_model')
            return Hz_fid

        return self.reconstruct_Hubble(z,linP_model,Hz_fid=Hz_fid)


    def reconstruct_Hubble(self,z,linP_model,Hz_fid=None):
        """ Use fiducial cosmology and g_star to reconstruct Hubble parameter"""

        if not Hz_fid:
            Hz_fid=self.camb_results_fid.hubble_parameter(z)

        # modifications will be computed around z_star
        z_star=self.z_star

        # m(z) = M(z) / M_fid(z) = H(z) / H_fid(z)
        mz=self.reconstruct_mz(z,linP_model)
        Hz = Hz_fid * mz

        return Hz


    def reconstruct_M_of_zs(self,linP_model):
        """ Reconstruct a list of M(z)=H(z)/(1+z) for all zs """

        M_of_zs=[]

        for iz,z in enumerate(self.zs):
            M_of_zs.append(self.reconstruct_Hubble_iz(iz,linP_model)/(1+z))

        return M_of_zs


    def reconstruct_f_p_iz(self,iz,linP_model):
        """ Use fiducial cosmology and f_star to reconstruct logarithmic
            growth rate f (around kp_Mpc)"""

        # THIS FUNCTION IS NOT BEING USED CURRENTLY

        f_p_fid=self.f_p_fid[iz]
        z=self.zs[iz]

        # check if we are asking for the fiducial model
        if not linP_model:
            if self.verbose: print('use fiducial linP_model')
            return f_p_fid

        return self.reconstruct_f_p(z,linP_model,f_p_fid=f_p_fid)


    def reconstruct_f_p(self,z,linP_model,f_p_fid=None):
        """ Use fiducial cosmology and f_star to reconstruct logarithmic
            growth rate f (around kp_Mpc)"""

        # THIS FUNCTION IS NOT BEING USED CURRENTLY

        # compute f in fiducial cosmology
        if not f_p_fid:
            if self.use_camb_fz:
                f_p_fid=camb_cosmo.get_f_of_z(self.cosmo_fid,
                            self.camb_results_fid,z)
            else:
                f_p_fid=fit_linP.compute_fz(self.cosmo_fid,z=z,
                            kp_Mpc=self.emu_kp_Mpc)
        # correct using difference in f_star
        f_star=linP_model.get_f_star()
        f_star_fid=self.linP_model_fid.get_f_star()
        df_star=f_star-f_star_fid
        # it is not clear to me that this is the best approximation...
        f_p=f_p_fid+(f_star-f_star_fid)
        return f_p


    def get_linP_model(self,like_params):
        """Use likelihood parameters to construct and return a linP_model"""

        # create dummy linP_model, to be updated next
        fid_params=self.linP_model_fid.get_params()
        # I'm pretty sure we can set this up from fiducial cosmology directly
        linP_model = linear_power_model.LinearPowerModel(params=fid_params,
                                    z_star=self.z_star,kp_kms=self.kp_kms,
                                    use_camb_fz=self.use_camb_fz)
        # update model with likelihood parameters
        linP_model.update_parameters(like_params)

        return linP_model


def compute_D_Dstar(cosmo,z,z_star,kp_Mpc=0.7):
    """Approximate linear growth factor between z_star and z, for input model.
        Computed from ratio of power at kp_Mpc"""

    # FUNCTION NOT USED IN MAIN CODE, ONLY FOR TESTING

    if z==z_star:
        return 1.0

    k_Mpc,zs,P_Mpc=camb_cosmo.get_linP_Mpc(cosmo,zs=[z_star,z])

    # check if CAMB sorted the outputs
    if zs[0]==z:
        Pz=np.interp(kp_Mpc,k_Mpc,P_Mpc[0])
        P_star=np.interp(kp_Mpc,k_Mpc,P_Mpc[1])
    else:
        Pz=np.interp(kp_Mpc,k_Mpc,P_Mpc[1])
        P_star=np.interp(kp_Mpc,k_Mpc,P_Mpc[0])
    D_Dstar=np.sqrt(Pz/P_star)

    return D_Dstar

