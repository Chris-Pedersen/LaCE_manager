import numpy as np
import sys
import os
import json
import fake_spectra.griddedspectra as grid_spec
import read_genic
import measure_flux_power as powF

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

        # get box size from GenIC file, to normalize power
        genic_file=self.data['basedir']+'/paramfile.genic'
        L_Mpc=read_genic.L_Mpc_from_paramfile(genic_file,verbose=True)

        basedir=self.data['basedir']
        skewers_dir=self.data['skewers_dir']
        snap_num=self.data['snap_num']

        # loop over all temperature models in snapshot
        Nsk=len(self.data['sk_files'])
        # collect all measured powers, with information about skewers
        arxiv_p1d=[]

        for isk in range(Nsk):
            sk_file=self.data['sk_files'][isk]
            # read skewers from HDF5 file
            skewers=grid_spec.GriddedSpectra(snap_num, basedir+'/output/',
                    savedir=skewers_dir, savefile=sk_file, reload_file=False)

            # loop over tau scalings
            for scale_tau in self.scales_tau:
                k,p1d,mF=powF.measure_F_p1D_Mpc(skewers,scale_tau,L_Mpc=L_Mpc)
                info_p1d={'k_Mpc':k,'p1d_Mpc':p1d,'mF':mF}
                info_p1d['scale_tau']=scale_tau
                info_p1d['sk_file']=sk_file
                arxiv_p1d.append(info_p1d)

        return arxiv_p1d

