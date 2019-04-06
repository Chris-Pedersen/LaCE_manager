import numpy as np
import sys
import os
import json
import p1d_arxiv

class SimplestEmulator(object):
    """Nearest-grid point emulator for flux P1D."""

    def __init__(self,basedir='../mini_sim_suite/',skewers_label='Ns50_wM0.1',
                verbose=True):
        """Setup emulator from base sim directory and label identifying skewer
            configuration (number, width)"""

        self.verbose=verbose

        # read all files with P1D measured in simulation suite
        self.arxiv=p1d_arxiv.ArxivP1D(basedir,skewers_label,verbose) 

        # define metric to compute distances between models
        self.metric=self.set_distance_metric()
    

    def set_distance_metric(self):
        """Set parameter uncertainties used to compute distances"""

        # completely made up for now
        metric={'mF':0.02,'kF_Mpc':0.1,'sigT_Mpc':0.01,'gamma':0.1,
                'Delta2_p':0.02,'n_p':0.001,'alpha_p':0.001,'f_p':0.002}

        if self.verbose:
            print('will use metric',metric)

        return metric


    def get_distance(self,model1,model2):
        """Compute distance between two models"""

        distance=0.0
        for key,value in model1.items():
            dx=model1[key]-model2[key]
            sigma=self.metric[key]
            distance += (dx/sigma)**2

        return distance


    def get_distances(self,model):
        """Compute distances from input model to all arxived models"""

        # loop over all models in arxiv
        Nm=len(self.arxiv.data)
        distances=np.empty(Nm)
        for i in range(Nm):
            distances[i]=self.get_distance(model,self.arxiv.data[i])

        return distances


    def get_nearest_model(self,model):
        """Given input model, find nearest model in arxiv"""

        # compute distance to all models in arxiv
        distances = self.get_distances(model)
        # identify nearest model
        nearest = np.argmin(distances)

        return self.arxiv.data[nearest]
        

    def emulate_p1d(self,model):
        """Return (k,p1d) for nearest model in arxiv"""

        nearest_model = self.get_nearest_model(model)

        return nearest_model['k_Mpc'], nearest_model['p1d_Mpc']
