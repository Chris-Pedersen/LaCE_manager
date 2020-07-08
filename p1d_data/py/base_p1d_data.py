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


def _drop_zbins(z_in,k_in,Pk_in,cov_in,zmin,zmax):
    """Drop redshift bins below zmin or above zmax"""

    # size of input arrays
    Nz_in=len(z_in)
    Nk=len(k_in)

    # figure out how many z to keep
    keep=np.ones(Nz_in, dtype=bool)
    if zmin:
        keep = np.logical_and(keep,z_in>zmin)
    if zmax:
        keep = np.logical_and(keep,z_in<zmax)
    Nz_out=np.sum(keep)

    # setup new arrays
    z_out=np.empty(Nz_out)
    Pk_out=np.empty((Nz_out,Nk))
    cov_out=[]
    i=0
    for j in range(Nz_in):
        if keep[j]:
            z_out[i]=z_in[j]
            Pk_out[i]=Pk_in[j]
            Pk_out[i]=Pk_in[j]
            cov_out.append(cov_in[j])
            i+=1
    return z_out,k_in,Pk_out,cov_out