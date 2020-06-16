import numpy as np

class BaseDataP1D(object):

    def __init__(self,z,k,Pk,cov):
        self.z=z
        self.k=k
        self.Pk=Pk
        self.cov=cov


    def get_Pk_iz(self,iz):
        if self.Pk is None:
            raise ValueError('power spectrum is blinded')
        return self.Pk[iz]


    def get_cov_iz(self,iz):
        return self.cov[iz]


    def _cull_data(self,kmin_kms):
        """Remove bins with wavenumber k < kmin_kms. """

        if kmin_kms is None: return
        # figure out number of bins to cull
        Ncull=np.sum(self.k<kmin_kms)
        # cull wavenumbers, power spectra, and covariances
        self.k=self.k[Ncull:]
        self.Pk=self.Pk[:,Ncull:]
        for i in range(len(self.cov)):
            self.cov[i]=self.cov[i][Ncull:,Ncull:]

        return

