import numpy as np
import sys
import os
import json
import fake_spectra.griddedspectra as grid_spec 
import read_gadget
import camb_cosmo

class SnapshotAdmin(object):
    """Book-keeping of all elements related to a snapshot. 
        For now, it reads pre-computed skewers, for different temperatures."""

    def __init__(self,snap_json, scales_tau=None):
        """Setup from JSON file with information about skewers extracted.
            One can also specify tau rescalings. """

        # read snapshot information from file (including temperature scalings)
        with open(snap_json) as json_data:
            self.data = json.load(json_data)

        # number of temperature models present in file
        self.NT = len(self.data['sim_T0'])

        # store number of optical depth rescalings we want to do
        if scales_tau:
            self.scales_tau=scales_tau
        else:
            self.scales_tau=[1.0]


    def get_all_flux_power(self):
        """Loop over all skewers, and return flux power for each"""

        info_p1d=[]
        
        return 1.0

