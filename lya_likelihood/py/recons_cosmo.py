import numpy as np
import os
import camb
import camb_cosmo
import fit_linP
from scipy import interpolate

class ReconstructedCosmology(object):
    """Given fiducial cosmology, and set of linear power parameters, 
        reconstruct a cosmology object."""

    def __init__(self,zs,cosmo_fid=None,verbose=False):
        """Setup from linear power model and redshifts to evaluate (zs)."""

#    def __init__(self,emu_zs,emu_kp_Mpc=0.69,cosmo_fid=None,
#            like_z_star=3.0,like_kp_kms=0.009,verbose=False):
        """Setup object to reconstruct cosmology from linear power parameters.
            - emu_zs: redshifts where we want predictions (call emulator)
            - emu_kp_Mpc: pivot point in Mpc used in the emulator
            - cosmo_fid: CAMB object describing fiducial cosmology
            - like_z_star: central redshift in likelihood parameterization
            - like_kp_kms: pivot point in likelihood parameterization (s/km)
            - verbose: print information for debugging."""

        self.verbose=verbose
        # store redshifts that will be evaluated
        self.zs=zs

        # fiducial cosmology
        if cosmo_fid:
            if self.verbose: print('use input fiducial cosmology')
            self.cosmo_fid=cosmo_fid
        else:
            if self.verbose: print('use default fiducial cosmology')
            self.cosmo_fid=camb_cosmo.get_cosmology()

        # compute CAMB results for fiducial cosmology
        self.results_fid=camb.get_results(self.cosmo_fid)

        # compute linear power model for fiducial cosmology
        self.linP_model_fid=fit_linP.LinearPowerModel(cosmo=self.cosmo_fid)
        self.z_star=self.linP_model_fid.z_star
        # note this is the pivot point for likelihood parameter
        # not necessarily pivot point for emulator (in Mpc)
        self.kp_kms=self.linP_model_fid.kp

        # get Hubble at z_star for fiducial cosmology, used to compute kp_Mpc
        self.H_star_fid=self.results_fid.hubble_parameter(self.z_star)
        self.dkms_dMpc_star_fid=self.H_star_fid/(1+self.z_star)
        # in general, emulator pivot point could be different. SHOULD FIX.
        self.kp_Mpc=self.kp_kms*self.dkms_dMpc_star_fid

        # store Hubble parameter at all redshifts for fiducial cosmology
        self._cache_Hz_fid()

        # store linear power at all redshifts for fiducial cosmology
        self._cache_f_p_fid()

        # store linear power at all redshifts for fiducial cosmology
        self._cache_linP_Mpc_fid()

        # store fiducial linear power at all redshifts, in km/s
        self._cache_linP_kms_fid()
    
        # when running with fixed cosmology it is useful to keep this
        self.linP_Mpc_params_fid=self._compute_linP_Mpc_params(
                                                linP_model=self.linP_model_fid)

        return


    def _cache_Hz_fid(self):
        """Compute Hubble parameter in fiducial cosmology, at all redshifts."""

        self.Hz_fid=[]
        for z in self.zs:
            Hz = self.results_fid.hubble_parameter(z)
            self.Hz_fid.append(Hz)

        return


    def _cache_f_p_fid(self):
        """ Compute growth rate in fiducial cosmology, at all redshifts. """

        self.f_p_fid=[]
        for z in self.zs:
            f_p = fit_linP.compute_f_star(self.cosmo_fid,z_star=z,
                                                kp_Mpc=self.kp_Mpc)
            self.f_p_fid.append(f_p)

        return


    def _cache_linP_Mpc_fid(self):
        """ Compute linear power in fiducial cosmology, at all redshifts. """

        # call CAMB to get linear power for fiducial cosmology, in Mpc
        k_Mpc, zs_out, P_Mpc = camb_cosmo.get_linP_Mpc(self.cosmo_fid,self.zs)
        # make sure we didn't change the order of the redshift outputs
        assert zs_out[0] == self.zs[0], 'CAMB redshifts not sorted'

        self.k_Mpc = k_Mpc
        self.linP_Mpc_fid = P_Mpc

        return


    def _cache_linP_kms_fid(self):
        """Fiducial linear power in velocity units, at all redshifts."""

        # call CAMB to get linear power for fiducial cosmology, in km/s
        k_kms, zs_out, P_kms = camb_cosmo.get_linP_kms(self.cosmo_fid,self.zs)
        # make sure we didn't change the order of the redshift outputs
        assert zs_out[0] == self.zs[0], 'CAMB redshifts not sorted'

        self.k_kms_fid = k_kms
        self.linP_kms_fid = P_kms


    def _compute_linP_Mpc_params(self,linP_model):
        """Reconstruct linear power (in Mpc) for input linP_model and fit
            linear power parameters at each redshift."""

        # wavenumbers that will be used in fit
        kp_Mpc=self.kp_Mpc
        kmin_Mpc=0.5*kp_Mpc
        kmax_Mpc=2.0*kp_Mpc
        xmin=kmin_Mpc/kp_Mpc
        xmax=kmax_Mpc/kp_Mpc
        x=self.k_Mpc/kp_Mpc

        linP_Mpc_params=[]
        for iz,z in enumerate(self.zs):
            # get information from fiducial cosmology at this redshift
            linP_Mpc_fid=self.linP_Mpc_fid[iz]
            # reconstruct logarithmic growth rate at the redshift
            f_p=self.reconstruct_f_p_iz(iz,linP_model)
            # reconstruct linear power at the redshift (in Mpc)
            linP_Mpc=self.reconstruct_linP_Mpc(iz,linP_model=linP_model)
            # fit polynomial describing log linear power
            linP_fit=fit_linP.fit_polynomial(xmin,xmax,x,linP_Mpc,deg=2)
            # compute parameters used in emulator
            lnA_p=linP_fit[0]
            Delta2_p=np.exp(lnA_p)*kp_Mpc**3/(2*np.pi**2)
            n_p=linP_fit[1]
            # note that the curvature is alpha/2
            alpha_p=2.0*linP_fit[2]
            params={'Delta2_p':Delta2_p,'n_p':n_p,'alpha_p':alpha_p,'f_p':f_p}
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
            linear power parameters at each redshift.
            kp_Mpc should specify pivot point, given by emulator."""

        # check if we are asking for the fiducial model
        if not linP_model:
            if self.verbose: print('use fiducial linP_model')
            return self.linP_Mpc_params_fid
        if self.linP_model_is_fiducial(linP_model):
            if self.verbose: print('use fiducial linP_model')
            return self.linP_Mpc_params_fid

        if self.verbose: print('will compute parameters for input linP_model')
        return self._compute_linP_Mpc_params(linP_model)


    def reconstruct_linP_Mpc(self,iz,linP_model):
        """ Use fiducial cosmology and linP_model to reconstruct linear power"""

        z = self.zs[iz]
        # we want to return the reconstructed power at these wavenumbers
        k_Mpc = self.k_Mpc
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
        Mz_fid = self.results_fid.hubble_parameter(z) / (1+z)

        # use the true cosmology to test the approximations
        if true_cosmo is not None:
            # compute true m(z)
            results=camb.get_results(true_cosmo)
            M_star = results.hubble_parameter(z_star) / (1+z_star)
            Mz = results.hubble_parameter(z) / (1+z)
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
        lnqB_kp = np.log(qB / self.kp_kms)
        if self.verbose:
            print(lnqB_kp.shape,'lnqB_kp',lnqB_kp)
        lnB = linP_kms_params(lnqB_kp)-linP_kms_params_fid(lnqB_kp)
        if self.verbose:
            print(lnB.shape,'lnB',lnB)

        # Evaluate the fiducial linear power at q' = m(z) q
        qP = mz * k_kms
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
            Hz_fid=self.results_fid.hubble_parameter(z)

        # modifications will be computed around z_star
        z_star=self.z_star

        # m(z) = M(z) / M_fid(z) = H(z) / H_fid(z)
        mz=self.reconstruct_mz(z,linP_model)
        Hz = Hz_fid * mz

        return Hz


    def reconstruct_f_p_iz(self,iz,linP_model):
        """ Use fiducial cosmology and f_star to reconstruct logarithmic
            growth rate f (around kp_Mpc)"""

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

        # compute f in fiducial cosmology
        if not f_p_fid:
            f_p_fid=fit_linP.compute_f_star(self.cosmo_fid,z_star=z,
                            kp_Mpc=self.kp_Mpc)
        # correct using difference in f_star
        f_star=linP_model.get_f_star()
        f_star_fid=self.linP_model_fid.get_f_star()
        df_star=f_star-f_star_fid
        # it is not clear to me that this is the best approximation...
        f_p=f_p_fid+(f_star-f_star_fid)
        return f_p


    def get_linP_model(self,like_params):
        """Uset likelihood parameters to construct and return a linP_model"""

        # create dummy linP_model, to be updated next
        fid_params=self.linP_model_fid.get_params()
        linP_model = fit_linP.LinearPowerModel(params=fid_params,
                                z_star=self.z_star,k_units='kms',kp=self.kp_kms)
        # update model with likelihood parameters
        linP_model.update_parameters(like_params)

        return linP_model



def compute_D_Dstar(cosmo,z,z_star,k_hMpc=1.0):
    """Approximate linear growth factor between z_star and z, for input model.
        Computed from ratio of power at k_hMpc"""

    if z==z_star:
        return 1.0

    kh,zs,Ph=camb_cosmo.get_linP_hMpc(cosmo,zs=[z_star,z])

    # check if CAMB sorted the outputs
    if zs[0]==z:
        Pz=np.interp(k_hMpc,kh,Ph[0])
        P_star=np.interp(k_hMpc,kh,Ph[1]) 
    else:
        Pz=np.interp(k_hMpc,kh,Ph[1])
        P_star=np.interp(k_hMpc,kh,Ph[0])    
    D_Dstar=np.sqrt(Pz/P_star)

    return D_Dstar

