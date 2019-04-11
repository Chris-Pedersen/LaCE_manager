import numpy as np
import base_p1d_data

class P1D_PD2013(base_p1d_data.BaseDataP1D):

    def __init__(self,basedir='../data_files/PD2013/',use_FFT=True,
                add_syst=True):
        """Read measured P1D from files, either FFT or likelihood version"""

        if use_FFT:
            z,k,Pk,cov=self._setup_FFT(basedir,add_syst)
        else:
            z,k,Pk,cov=self._setup_like(basedir,add_syst)

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
        


