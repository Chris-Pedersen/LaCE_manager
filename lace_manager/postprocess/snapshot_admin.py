import numpy as np
import sys
import os
import json
import fake_spectra.spectra as spec
from lace.setup_simulations import read_genic
from lace_manager.postprocess import measure_flux_power as powF

class SnapshotAdmin(object):
    """Book-keeping of all elements related to a snapshot.
        For now, it reads pre-computed skewers, for different temperatures."""

    def __init__(self,snap_json,scales_tau=None,kF_Mpc=None,post_dir=None):
        """Setup from JSON file with information about skewers extracted.
            One can also specify tau rescalings, and (optionally) provide
            the measured filtering length.
            If post_dir is provided, use it as postprocessing directory."""

        # read snapshot information from file (including temperature scalings)
        with open(snap_json) as json_data:
            self.data = json.load(json_data)

        if post_dir:
            print('use',post_dir,'as post-processing directory')
            self.data['post_dir']=post_dir
            self.data['raw_dir']='NA'
            self.data['skewers_dir']=post_dir+'/skewers/'
        else:
            if 'post_dir' in self.data:
                print('no post_dir provided, will use',self.data['post_dir'])
                self.data['skewers_dir']=self.data['post_dir']+'/skewers/'
            else:
                raise ValueError('must provide post_dir when using old files')

        # see if you have access filtering length information
        if kF_Mpc:
            print('received kF_Mpc =',kF_Mpc)
            self.data['kF_Mpc']=kF_Mpc

        # number of temperature models present in file
        self.NT = len(self.data['sim_T0'])

        # store number of optical depth rescalings we want to do
        if scales_tau:
            self.scales_tau=scales_tau
        else:
            self.scales_tau=[1.0]


    def get_all_flux_power(self,add_p3d=False):
        """Loop over all skewers, and return flux power for each"""

        post_dir=self.data['post_dir']
        genic_file=post_dir+'/paramfile.genic'
        L_Mpc=read_genic.L_Mpc_from_paramfile(genic_file,verbose=True)
        skewers_dir=self.data['skewers_dir']
        snap_num=self.data['snap_num']

        # will loop over all temperature models in snapshot
        Nsk=len(self.data['sk_files'])
        # collect all measured powers, with information about skewers
        p1d_data=[]

        for isk in range(Nsk):
            sk_file=self.data['sk_files'][isk]
            sim_T0=self.data['sim_T0'][isk]
            sim_gamma=self.data['sim_gamma'][isk]
            sim_sigT_Mpc=self.data['sim_sigT_Mpc'][isk]
            sim_scale_T0=self.data['sim_scale_T0'][isk]
            sim_scale_gamma=self.data['sim_scale_gamma'][isk]

            # read skewers from HDF5 file
            skewers=spec.Spectra(snap_num,base="NA",cofm=None,axis=None,
                    savedir=skewers_dir,savefile=sk_file,res=None,
                    reload_file=False,load_snapshot=False,quiet=False)

            # loop over tau scalings
            for scale_tau in self.scales_tau:
                k,p1d,mF=powF.measure_F_p1d_Mpc(skewers,scale_tau,L_Mpc=L_Mpc)
                info_p1d={'k_Mpc':k.tolist(),'p1d_Mpc':p1d.tolist(),'mF':mF}
                info_p1d['scale_tau']=scale_tau
                # add information about skewers and temperature rescaling
                info_p1d['sk_file']=sk_file
                info_p1d['sim_T0']=sim_T0
                info_p1d['sim_gamma']=sim_gamma
                info_p1d['sim_sigT_Mpc']=sim_sigT_Mpc
                info_p1d['sim_scale_T0']=sim_scale_T0
                info_p1d['sim_scale_gamma']=sim_scale_gamma
                if 'kF_Mpc' in self.data:
                    info_p1d['kF_Mpc'] = self.data['kF_Mpc']
                # try to add also P3D
                if add_p3d:
                    # being lazy for now
                    info_p3d=powF.measure_p3d_Mpc(skewers,scale_tau,L_Mpc=L_Mpc)
                    info_p1d['p3d_data']=info_p3d

                p1d_data.append(info_p1d)

        self.p1d_data=p1d_data
        return p1d_data


    def get_p1d_json_filename(self,p1d_label):
        """Use metadata information to figure filename for JSON with P1D"""

        num=self.data['snap_num']
        n_skewers=self.data['n_skewers']
        width_Mpc=self.data['width_Mpc']
        
        filename=p1d_label+'_'+str(num)+'_Ns'+str(n_skewers)
        filename+='_wM'+str(int(1000*width_Mpc)/1000)
        filename+='.json'

        return filename


    def write_p1d_json(self,p1d_label=None):
        """ Write JSON file with P1D measured in all post-processing"""

        if p1d_label is None:
            p1d_label='p1d'

        filename=self.data['post_dir']+'/'+self.get_p1d_json_filename(p1d_label)
        print('will print P1D to file',filename)

        # make sure we have already computed P1D
        if not self.p1d_data:
            print('computing P1D before writing JSON file')
            self.get_all_flux_power()

        p1d_info={'snapshot_data':self.data, 'scales_tau':self.scales_tau,
                    'p1d_data': self.p1d_data}

        json_file = open(filename,"w")
        json.dump(p1d_info,json_file)
        json_file.close()

