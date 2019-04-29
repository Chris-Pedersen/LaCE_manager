import os
import numpy as np
import camb
import camb_cosmo
import fit_linP

class ReconstructedCosmology(object):
    """Given fiducial cosmology, and set of linear power parameters, 
        reconstruct a cosmology object."""

    def __init__(self,linP_model,cosmo_fid=None,use_constant_f=False):
        """Setup from linear power model and fiducial cosmology (optional)."""

        # whether to model z-evolution of logarithmic growth rate with fiducial
        self.use_constant_f=use_constant_f

        # input model describing linear power around z_star and kp_kms
        self.linP_model=linP_model
        assert linP_model.k_units is 'kms', 'ReconstructCosmo only works in kms'

        # fiducial cosmology
        if cosmo_fid:
            self.cosmo_fid=cosmo_fid
        else:
            self.cosmo_fid=camb_cosmo.get_cosmology()

        # compute CAMB results for fiducial cosmology
        self.results_fid=camb.get_results(cosmo_fid)

        # compute linear power model for fiducial cosmology
        z_star=linP_model.z_star
        kp_kms=linP_model.kp
        self.linP_model_fid=fit_linP.LinearPowerModel(cosmo_fid,z_star=z_star,
                    k_units='kms',kp=kp_kms)

        # get Hubble at z_star for fiducial cosmology, used to compute kp_Mpc
        self.H_star_fid=self.results_fid.hubble_parameter(z_star)
        self.A_star_fid=self.H_star_fid/(1+z_star)
        self.kp_Mpc=kp_kms*self.A_star_fid


    def reconstruct_Hubble(self,z):
        """ Use fiducial cosmology and g_star to reconstruct Hubble parameter"""

        # compute difference in acceleration
        g_star=self.linP_model.get_g_star()
        g_star_fid=self.linP_model_fid.get_g_star()
        # compute Hubble parameter in fiducial cosmology
        Hz_fid = self.results_fid.hubble_parameter(z)
        # compute Hubble parameter in input cosmology
        z_star=self.linP_model.z_star
        Hz = Hz_fid * (1+3/2*(g_star-g_star_fid)*(z-z_star)/(1+z_star))
        return Hz


    def reconstruct_f_p(self,z):
        """ Use fiducial cosmology and g_star to reconstruct logarithmic
            growth rate f (around kp_Mpc)"""

        if self.use_constant_f:
            return self.linP_model.get_f_star()
        # compute f in fiducial cosmology
        f_p_fid=fit_linP.compute_f_star(self.cosmo_fid,z_star=z,
                            kp_Mpc=self.kp_Mpc)
        # correct using difference in f_star
        f_star=self.linP_model.get_f_star()
        f_star_fid=self.linP_model_fid.get_f_star()
        df_star=f_star-f_star_fid
        f_p=f_p_fid+(f_star-f_star_fid)
        return f_p


    def reconstruct_linP_kms(self,zs,k_kms):
        """Given fiducial cosmology and linear parameters,
           reconstruct linear power spectra of input cosmology (in km/s) """

        # get linear power for fiducial cosmology
        k_kms_fid,zs_out,P_kms_fid = camb_cosmo.get_linP_kms(self.cosmo_fid,zs)
        assert (zs==zs_out).all(), 'zs != zs_out in reconstruct_linP_kms'

        # get parameters describing linear power for input cosmology
        f_star=self.linP_model.get_f_star()
        g_star=self.linP_model.get_g_star()
        linP_kms=self.linP_model.linP_params['linP_kms']

        # get parameters describing linear power for fiducial cosmology
        f_star_fid=self.linP_model_fid.get_f_star()
        g_star_fid=self.linP_model_fid.get_g_star()
        linP_kms_fid=self.linP_model_fid.linP_params['linP_kms']

        # get relative parameters
        df_star=f_star-f_star_fid
        dg_star=g_star-g_star_fid

        # get z_star and kp_kms from linear power model
        z_star=self.linP_model.z_star
        kp_kms=self.linP_model.kp

        # will store reconstructed linear power here
        Nz=len(zs)
        rec_linP_kms=np.empty_like(k_kms)

        for iz in range(Nz):
            z=zs[iz]
            # Hubble parameter in fiducial cosmology
            H_fid = self.results_fid.hubble_parameter(z=z)
            # evaluate fiducial power at slightly different wavenumber
            x = 1 + 3/2*dg_star*(z-z_star)/(1+z_star)
            P_rec = np.interp(x*k_kms[iz],k_kms_fid[iz],P_kms_fid[iz]) * x**3
            # apply change in shape, taking into account we work in km/s
            y_fid = H_fid/(1+z)/self.H_star_fid*(1+z_star)
            y=y_fid*x
            lnky=np.log(y*k_kms[iz]/kp_kms)
            P_rec *= np.exp(linP_kms(lnky)-linP_kms_fid(lnky))
            # correct linear growth
            P_rec *= (1-df_star*(z-z_star)/(1+z_star))**2
            rec_linP_kms[iz]=P_rec

        return rec_linP_kms


    def reconstruct_linP_Mpc(self,zs,k_Mpc):
        """Given fiducial cosmology and linear parameters,
           reconstruct linear power spectra of input cosmology (in Mpc) """

        Nz=len(zs)
        # reconstruct Hubble parameter at the input redshifts
        Hz=[self.reconstruct_Hubble(z) for z in zs]
        Az=[Hz[iz]/(1+zs[iz]) for iz in range(Nz)]
        k_kms=[k_Mpc/Az[iz] for iz in range(Nz)]
        rec_linP_Mpc=np.empty([Nz,len(k_Mpc)])
        # reconstruct linear power in velocity units
        rec_linP_kms=self.reconstruct_linP_kms(zs,k_kms)
        # note that we expect z to be a float, not a list
        for iz in range(Nz):
            rec_linP_Mpc[iz]=rec_linP_kms[0]/Az[iz]**3
        return rec_linP_Mpc


    def get_linP_Mpc_params(self,zs):
        """Reconstruct linear power (in Mpc) at input redshifts, and fit
            linear power parameters"""

        # wavenumbers that will be used in fit
        kp_Mpc=self.kp_Mpc
        kmin_Mpc=0.5*kp_Mpc
        kmax_Mpc=2.0*kp_Mpc
        k_Mpc = np.logspace(np.log10(kmin_Mpc),np.log10(kmax_Mpc),100)

        # reconstruct linear power in Mpc
        linP_Mpc_rec=self.reconstruct_linP_Mpc(zs,k_Mpc)

        linP_Mpc_params=[]
        for iz in range(len(zs)):
            xmin=kmin_Mpc/kp_Mpc
            xmax=kmax_Mpc/kp_Mpc
            x=k_Mpc/kp_Mpc
            # fit polynomial describing log linear power
            linP_fit=fit_linP.fit_polynomial(xmin,xmax,x,linP_Mpc_rec[iz],deg=2)
            # compute parameters used in emulator
            lnA_p=linP_fit[0]
            Delta2_p=np.exp(lnA_p)*kp_Mpc**3/(2*np.pi**2)
            n_p=linP_fit[1]
            # note that the curvature is alpha/2
            alpha_p=2.0*linP_fit[2]
            # reconstruct logarithmic growth rate at the redshift
            f_p=self.reconstruct_f_p(zs[iz])
            params={'Delta2_p':Delta2_p,'n_p':n_p,'alpha_p':alpha_p,'f_p':f_p}
            linP_Mpc_params.append(params)

        return linP_Mpc_params
