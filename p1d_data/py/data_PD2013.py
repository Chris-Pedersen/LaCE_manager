import numpy as np
import base_p1d_data

class P1D_PD2013(base_p1d_data.BaseDataP1D):

    def __init__(self,basedir='../../p1d_data/data_files/PD2013/',use_FFT=True,
                add_syst=True,blind_data=False):
        """Read measured P1D from files, either FFT or likelihood version.
            If blind_data=True, use analytical formula instead."""

        if use_FFT:
            z,k,Pk,cov=self._setup_FFT(basedir,add_syst)
        else:
            z,k,Pk,cov=self._setup_like(basedir,add_syst)

        if blind_data:
            Nz=len(z)
            for iz in range(Nz):
                Pk[iz] = analytic_p1d_PD2013_z_kms(z[iz],k)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)


    def _setup_FFT(self,basedir,add_syst=True):
        """Setup measurement using FFT approach"""
    
        # start by reading Pk file
        p1d_file=basedir+'/table4a.dat'
        iz,ik,inz,ink,inPk,inPkstat,inPknoise,inPkmetal,inPksyst=np.loadtxt(
                    p1d_file,unpack=True)

        # store unique values of redshift and wavenumber
        z=np.unique(inz)
        Nz=len(z)
        Nz=Nz
        k=np.unique(ink)
        Nk=len(k)
        Nk=Nk

        # store P1D, statistical error, noise power, metal power and systematic 
        Pk=np.reshape(inPk,[Nz,Nk])
        Pkstat=np.reshape(inPkstat,[Nz,Nk])    
        Pknoise=np.reshape(inPknoise,[Nz,Nk])
        Pkmetal=np.reshape(inPkmetal,[Nz,Nk])
        Pksyst=np.reshape(inPksyst,[Nz,Nk])

        # now read correlation matrices and compute covariance matrices
        cov=[]
        for i in range(Nz):
            corr_file=basedir+'/cct4b'+str(i+1)+'.dat'
            corr=np.loadtxt(corr_file,unpack=True)
            # compute covariance matrix (stats only)
            sigma=Pkstat[i]
            zcov=np.dot(corr,np.outer(sigma,sigma))
            if add_syst:
                syst=Pksyst[i]
                zcov+=np.diag(syst)
            cov.append(zcov)

        return z,k,Pk,cov
        

    def _setup_like(self,basedir,add_syst=True):
        """Setup measurement using likelihood approach"""

        p1d_file=basedir+'/table5a.dat'
        raise ValueError('implement _setup_like to read likelihood P1D') 
        

def analytic_p1d_PD2013_z_kms(z,k_kms):
    """Fitting formula for 1D P(z,k) from Palanque-Delabrouille et al. (2013).
        Wavenumbers and power in units of km/s. Corrected to be flat at low-k"""

    # numbers from Palanque-Delabrouille (2013)
    A_F = 0.064
    n_F = -2.55
    alpha_F = -0.1
    B_F = 3.55
    beta_F = -0.28
    k0 = 0.009
    z0 = 3.0
    n_F_z = n_F + beta_F * np.log((1+z)/(1+z0))
    # this function would go to 0 at low k, instead of flat power
    k_min=k0*np.exp((-0.5*n_F_z-1)/alpha_F)
    flatten=(k_kms < k_min)
    k_kms[flatten] = k_min
    exp1 = 3 + n_F_z + alpha_F * np.log(k_kms/k0)
    toret = np.pi * A_F / k0 * pow(k_kms/k0, exp1-1) * pow((1+z)/(1+z0), B_F)

    return toret
