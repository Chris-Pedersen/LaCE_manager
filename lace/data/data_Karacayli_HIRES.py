import os
import numpy as np
from lace.data import base_p1d_data

class P1D_Karacayli_HIRES(base_p1d_data.BaseDataP1D):

    def __init__(self,diag_cov=True):
        """Read measured P1D from file"""

        # folder storing P1D measurement
        assert ('LYA_EMU_REPO' in os.environ),'export LYA_EMU_REPO'
        repo=os.environ['LYA_EMU_REPO']
        basedir=repo+'/lace/data/data_files/Karacayli_HIRES/'

        # read redshifts, wavenumbers, power spectra and covariance matrices
        z,k,Pk,cov=self._read_file(basedir,diag_cov)

        base_p1d_data.BaseDataP1D.__init__(self,z,k,Pk,cov)

        return


    def _read_file(self,basedir,diag_cov):
        """Read file containing mock P1D"""
    
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
        # store unique redshifts 
        ink=[float(line.split()[3]) for line in data]
        k=np.unique(ink)

        # store P1D, statistical error, noise power, metal power and systematic
        inPk=[float(line.split()[6]) for line in data]
        Pk=np.array(inPk).reshape([Nz,Nk])

        # now read covariance matrix
        assert diag_cov, 'implement code to read full covariance'

        # for now only use diagonal elements
        inErr=[float(line.split()[7]) for line in data]
        cov=[]
        for i in range(Nz):
            err=inErr[i*Nk:(i+1)*Nk]
            cov.append(np.diag(np.array(err)**2))

        return z,k,Pk,cov
