import numpy as np
import sys
import os
import json
from lace.emulator import p1d_archive

class SimplestEmulator(object):
    """Nearest-grid point emulator for flux P1D."""

    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                keep_every_other_rescaling=False,
                max_archive_size=None,undersample_z=1,verbose=False):
        """Setup emulator from base sim directory and label identifying skewer
            configuration (number, width)"""

        self.verbose=verbose

        # read all files with P1D measured in simulation suite
        self.archive=p1d_archive.archiveP1D(basedir,p1d_label,skewers_label,
                        drop_tau_rescalings=drop_tau_rescalings,
                        drop_temp_rescalings=drop_temp_rescalings,
                        keep_every_other_rescaling=keep_every_other_rescaling,
                        max_archive_size=max_archive_size,
                        undersample_z=undersample_z,verbose=verbose)

        # define metric to compute distances between models
        self.metric=self.set_distance_metric()
    

    def set_distance_metric(self):
        """Set parameter uncertainties used to compute distances"""

        # completely made up for now
        metric={'mF':0.01,'kF_Mpc':0.1,'sigT_Mpc':0.02,'gamma':0.02,
                'Delta2_p':0.01,'n_p':0.01,'alpha_p':0.01,'f_p':0.005}

        if self.verbose: print('will use metric',metric)

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
        """Compute distances from input model to all archiveed models"""

        # loop over all models in archive
        Nm=len(self.archive.data)
        distances=np.empty(Nm)
        for i in range(Nm):
            distances[i]=self.get_distance(model,self.archive.data[i])

        return distances


    def find_nearest_model(self,model):
        """Given input model, find nearest model in archive"""

        # compute distance to all models in archive
        distances = self.get_distances(model)
        # identify nearest model
        nearest = np.argmin(distances)

        return nearest
        

    def get_nearest_model(self,model):
        """Given input model, return earest model in archive"""

        nearest = self.find_nearest_model(model)
        if self.verbose: self.archive.print_entry(nearest)

        return self.archive.data[nearest]


    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False):
        """Return (k,p1d) for nearest model in archive"""

        if self.verbose: print('asked to emulate model',model)

        nearest_model = self.get_nearest_model(model)

        p1d = np.interp(k_Mpc,nearest_model['k_Mpc'],nearest_model['p1d_Mpc'])

        if return_covar:
            N=len(p1d)
            return p1d, np.zeros([N,N])
        else:
            return p1d
