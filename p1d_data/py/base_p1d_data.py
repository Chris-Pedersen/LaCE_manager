import numpy as np

class BaseDataP1D(object):

    def __init__(self,z,k,Pk,cov):
        self.z=z
        self.k=k
        self.Pk=Pk
        self.cov=cov


    def get_Pk_iz(self,iz):
        return self.Pk[iz]


    def get_cov_iz(self,iz):
        return self.cov[iz]


