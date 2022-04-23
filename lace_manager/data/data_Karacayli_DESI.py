import os
import numpy as np
from lace_manager.data import base_p1d_data

class P1D_Karacayli_DESI(base_p1d_data.BaseDataP1D):

    def __init__(self,diag_cov=True,kmax_kms=0.04):
        """Read measured P1D from file"""

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(diag_cov,kmax_kms)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

        return


    def _read_file(self,diag_cov,kmax_kms):
        """Read file containing mock P1D"""

        # folder storing P1D measurement
        assert ('LACE_MANAGER_REPO' in os.environ),'export LACE_MANAGER_REPO'
        repo=os.environ['LACE_MANAGER_REPO']
        basedir=repo+'/lace_manager/data/data_files/Karacayli_DESI/'
    
        # start by reading the file with measured band power
        p1d_file=basedir+'desilite-oqe-mock-power-spectrum.txt'
        with open(p1d_file, 'r') as reader:
            lines=reader.readlines()
        # read number of bins from line 42
        bins = lines[41].split()
        Nz = int(bins[1])
        Nk = int(bins[2])
        print('read Nz = {} , Nk = {}'.format(Nz,Nk))
        # z k1 k2 kc Pfid ThetaP Pest ErrorP d b t
        data = lines[44:]

        # store unique redshifts 
        inz=[float(line.split()[0]) for line in data]
        z=np.unique(inz)
        # store unique wavenumbers 
        ink=[float(line.split()[3]) for line in data]
        k=np.unique(ink)

        # store measured P1D
        inPk=[float(line.split()[6]) for line in data]
        Pk=np.array(inPk).reshape([Nz,Nk])

        # will keep only wavenumbers with k < kmax_kms
        kmask=k<kmax_kms
        k=k[kmask]
        Nkmask=len(k)
        print('will only use {} k bins below {}'.format(Nkmask,kmax_kms))
        Pk=Pk[:,:Nkmask]

        # now read covariance matrix
        assert diag_cov, 'implement code to read full covariance'

        # for now only use diagonal elements
        inErr=[float(line.split()[7]) for line in data]
        cov=[]
        for i in range(Nz):
            err=inErr[i*Nk:(i+1)*Nk]
            var=np.array(err)[:Nkmask]**2
            cov.append(np.diag(var))

        return z,k,Pk,cov
