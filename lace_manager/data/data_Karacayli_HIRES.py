import os
import numpy as np
from lace_manager.data import base_p1d_data

class P1D_Karacayli_HIRES(base_p1d_data.BaseDataP1D):

    def __init__(self,diag_cov=True,kmax_kms=0.09):
        """Read measured P1D from file.
            - diag_cov: for now, use diagonal covariance
            - kmax_kms: limit to low-k where we trust emulator """

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(diag_cov,kmax_kms)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

        return


    def _read_file(self,diag_cov,kmax_kms):
        """Read file containing mock P1D"""

        # folder storing P1D measurement
        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']
        basedir=repo+'/lace/data/data_files/Karacayli_HIRES/'

        # start by reading the file with measured band power
        p1d_file=basedir+'highres-mock-power-spectrum.txt'
        with open(p1d_file, 'r') as reader:
            lines=reader.readlines()
        # read number of bins from line 42
        bins = lines[41].split()
        Nz = int(bins[1])
        Nk = int(bins[2])
        print('Nz = {} , Nk = {}'.format(Nz,Nk))
        # z k1 k2 kc Pfid ThetaP Pest ErrorP d b t
        data = lines[44:]

        # store unique redshifts 
        inz=[float(line.split()[0]) for line in data]
        z=np.unique(inz)

        # store unique wavenumbers
        ink=[float(line.split()[3]) for line in data]
        k=np.unique(ink)

        # store P1D, statistical error, noise power, metal power and systematic
        inPk=[float(line.split()[6]) for line in data]
        inPk=np.array(inPk).reshape([Nz,Nk])

        # for now only use diagonal elements
        assert diag_cov, 'implement code to read full covariance'
        inErr=[float(line.split()[7]) for line in data]

        # limit only to modes < 0.1 s/km
        Ncull=np.sum(k>kmax_kms)
        k=k[:Nk-Ncull]
        Pk=[]
        cov=[]
        for i in range(Nz):
            Pk.append(inPk[i,:Nk-Ncull])
            err=inErr[i*Nk:(i+1)*Nk]
            cov.append(np.diag(np.array(err[:Nk-Ncull])**2))

        return z,k,Pk,cov
