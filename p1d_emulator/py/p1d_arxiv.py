import numpy as np
import sys
import os
import json

class ArxivP1D(object):
    """Book-keeping of flux P1D measured in a suite of simulations."""

    def __init__(self,basedir='../mini_sim_suite/',skewers_label='Ns50_wM0.1',
                verbose=True):
        """Load arxiv from base sim directory and label identifying skewer
            configuration (number, width)"""

        self.basedir=basedir
        self.skewers_label=skewers_label
        self.verbose=verbose

        self._load_data()


    def _load_data(self):
        """Setup arxiv by looking at all measured power spectra in sims"""

        # each measured power will have a dictionary, stored here
        self.data=[]

        # read file containing information about latin hyper-cube
        cube_json=self.basedir+'/latin_hypercube.json'
        with open(cube_json) as json_file:  
            self.cube_data = json.load(json_file)
        if self.verbose:
            print('latin hyper-cube data',self.cube_data)
        self.nsamples=self.cube_data['nsamples']
        if self.verbose:
            print('simulation suite has %d samples'%self.nsamples)

        # read info from all sims, all snapshots, all rescalings
        for sample in range(self.nsamples):
            # store parameters for simulation pair / model
            sim_params = self.cube_data['samples']['%d'%sample]
            if self.verbose:
                print(sample,'sample has sim params =',sim_params)
            model_dict ={'sample':sample,'sim_param':sim_params}

            # read number of snapshots (should be the same in all sims)
            pair_dir=self.basedir+'/sim_pair_%d'%sample
            pair_json=pair_dir+'/parameter.json'
            with open(pair_json) as json_file:  
                pair_data = json.load(json_file)
            #print(sample,'pair data',pair_data)
            zs=pair_data['zs']
            Nz=len(zs)
            print('simulation has %d redshifts'%Nz) 

            for snap in range(Nz):        
                # make sure that we have skewers for this snapshot (z < zmax)
                plus_p1d_json=pair_dir+'/sim_plus/p1d_{}_{}.json'.format(
                                snap,self.skewers_label)
                if not os.path.isfile(plus_p1d_json):
                    if self.verbose:
                        print(pair_dir,'does not have this snapshot',snap)
                    continue

                # get linear power parameters describing snapshot
                linP_params = pair_data['linP_zs'][snap]
                snap_p1d_data = {}
                snap_p1d_data['Delta2_p'] = linP_params['Delta2_p']
                snap_p1d_data['n_p'] = linP_params['n_p']
                snap_p1d_data['alpha_p'] = linP_params['alpha_p']
                snap_p1d_data['f_p'] = linP_params['f']
                snap_p1d_data['z']=zs[snap]
        
                # open file with 1D power measured in snapshot for sim_plus
                with open(plus_p1d_json) as json_file:
                    plus_data = json.load(json_file)
                # open file with 1D power measured in snapshot for sim_minus
                minus_p1d_json=pair_dir+'/sim_minus/p1d_{}_{}.json'.format(
                                snap,self.skewers_label)
                with open(minus_p1d_json) as json_file: 
                    minus_data = json.load(json_file)

                # number of post-process rescalings for each snapshot
                Npp=len(plus_data['p1d_data'])
                # read info for each post-process
                for pp in range(Npp):
                    # deep copy of dictionary (thread safe, why not)
                    p1d_data = json.loads(json.dumps(snap_p1d_data))
                    k_Mpc = np.array(plus_data['p1d_data'][pp]['k_Mpc'])
                    if len(k_Mpc) != len(minus_data['p1d_data'][pp]['k_Mpc']):
                        print(sample,snap,pp)
                        print(len(k_Mpc),'!=',
                                    len(minus_data['p1d_data'][pp]['k_Mpc']))
                        raise ValueError('different k_Mpc in minus/plus')
                    # average plus + minus stats
                    plus_mF = plus_data['p1d_data'][pp]['mF']
                    minus_mF = minus_data['p1d_data'][pp]['mF']
                    pair_mF = 0.5*(plus_mF+minus_mF)
                    p1d_data['mF'] = pair_mF 
                    p1d_data['T0'] = 0.5*(plus_data['p1d_data'][pp]['sim_T0']+minus_data['p1d_data'][pp]['sim_T0'])
                    p1d_data['gamma'] = 0.5*(plus_data['p1d_data'][pp]['sim_gamma']+minus_data['p1d_data'][pp]['sim_gamma'])
                    p1d_data['sigT_Mpc'] = 0.5*(plus_data['p1d_data'][pp]['sim_sigT_Mpc']+minus_data['p1d_data'][pp]['sim_sigT_Mpc'])
                    # compute average of < F F >, not <delta delta> 
                    plus_p1d = np.array(plus_data['p1d_data'][pp]['p1d_Mpc'])
                    minus_p1d = np.array(minus_data['p1d_data'][pp]['p1d_Mpc'])
                    pair_p1d = (plus_p1d * plus_mF**2 + minus_p1d * minus_mF**2) / pair_mF
                    p1d_data['k_Mpc'] = k_Mpc
                    p1d_data['p1d_Mpc'] = pair_p1d
                    self.data.append(p1d_data)                
        
        if self.verbose:
            print('Arxiv setup, containing %d measured P1D'%len(self.data))

        # store IGM parameters
        Ntot=len(self.data)
        self.mF=np.array([self.data[i]['mF'] for i in range(Ntot)])
        self.sigT_Mpc=np.array([self.data[i]['sigT_Mpc'] for i in range(Ntot)])
        self.gamma=np.array([self.data[i]['gamma'] for i in range(Ntot)])

        # store linear power parameters 
        self.Delta2_p=np.array([self.data[i]['Delta2_p'] for i in range(Ntot)])
        self.n_p=np.array([self.data[i]['n_p'] for i in range(Ntot)])
        self.alpha_p=np.array([self.data[i]['alpha_p'] for i in range(Ntot)])
        self.f_p=np.array([self.data[i]['f_p'] for i in range(Ntot)])
        

