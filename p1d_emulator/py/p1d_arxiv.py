import numpy as np
import sys
import os
import json

class ArxivP1D(object):
    """Book-keeping of flux P1D measured in a suite of simulations."""

    def __init__(self,basedir='../mini_sim_suite/',
                p1d_label='p1d',skewers_label=None,
                drop_tau_rescalings=False,drop_temp_rescalings=False,
                max_arxiv_size=None,verbose=True):
        """Load arxiv from base sim directory and (optional) label
            identifying skewer configuration (number, width)"""

        self.basedir=basedir
        self.p1d_label=p1d_label
        self.skewers_label=skewers_label
        self.verbose=verbose

        self._load_data(drop_tau_rescalings,drop_temp_rescalings,max_arxiv_size)


    def _load_data(self,drop_tau_rescalings,drop_temp_rescalings,max_arxiv_size):
        """Setup arxiv by looking at all measured power spectra in sims"""

        # each measured power will have a dictionary, stored here
        self.data=[]

        # if we didn't provide skewers label, read only general sim info
        no_skewers=(self.skewers_label is None)

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
            if self.verbose:
                print('simulation has %d redshifts'%Nz) 

            for snap in range(Nz):        
                # get linear power parameters describing snapshot
                linP_params = pair_data['linP_zs'][snap]
                snap_p1d_data = {}
                snap_p1d_data['Delta2_p'] = linP_params['Delta2_p']
                snap_p1d_data['n_p'] = linP_params['n_p']
                snap_p1d_data['alpha_p'] = linP_params['alpha_p']
                if 'f' in linP_params:
                    snap_p1d_data['f_p'] = linP_params['f']
                else:
                    snap_p1d_data['f_p'] = linP_params['f_p']
                snap_p1d_data['z']=zs[snap]

                # check if we have extracted skewers yet
                if no_skewers:
                    self.data.append(snap_p1d_data)
                    continue

                # make sure that we have skewers for this snapshot (z < zmax)
                plus_p1d_json=pair_dir+'/sim_plus/{}_{}_{}.json'.format(
                                self.p1d_label,snap,self.skewers_label)
                if not os.path.isfile(plus_p1d_json):
                    if self.verbose:
                        print(plus_p1d_json,'snapshot does not have p1d')
                    continue
                # open file with 1D power measured in snapshot for sim_plus
                with open(plus_p1d_json) as json_file:
                    plus_data = json.load(json_file)
                # open file with 1D power measured in snapshot for sim_minus
                minus_p1d_json=pair_dir+'/sim_minus/{}_{}_{}.json'.format(
                                self.p1d_label,snap,self.skewers_label)
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
                    plus_pp=plus_data['p1d_data'][pp]
                    minus_pp=minus_data['p1d_data'][pp]
                    plus_mF = plus_pp['mF']
                    minus_mF = minus_pp['mF']
                    pair_mF = 0.5*(plus_mF+minus_mF)
                    p1d_data['mF'] = pair_mF 
                    p1d_data['T0'] = 0.5*(plus_pp['sim_T0']+minus_pp['sim_T0'])
                    p1d_data['gamma'] = 0.5*(plus_pp['sim_gamma']
                                            +minus_pp['sim_gamma'])
                    p1d_data['sigT_Mpc'] = 0.5*(plus_pp['sim_sigT_Mpc']
                                            +minus_pp['sim_sigT_Mpc'])
                    # store also scalings used (not present in old versions)
                    if 'sim_scale_T0' in plus_pp:
                        p1d_data['scale_T0'] = plus_pp['sim_scale_T0']
                    if 'sim_scale_gamma' in plus_pp:
                        p1d_data['scale_gamma'] = plus_pp['sim_scale_gamma']
                    # store also filtering length (not present in old versions)
                    if 'kF_Mpc' in plus_pp:
                        p1d_data['kF_Mpc'] = plus_pp['kF_Mpc']
                    p1d_data['scale_tau'] = plus_pp['scale_tau']
                    # compute average of < F F >, not <delta delta> 
                    plus_p1d = np.array(plus_pp['p1d_Mpc'])
                    minus_p1d = np.array(minus_pp['p1d_Mpc'])
                    pair_p1d = (plus_p1d * plus_mF**2
                                + minus_p1d * minus_mF**2) / pair_mF
                    p1d_data['k_Mpc'] = k_Mpc
                    p1d_data['p1d_Mpc'] = pair_p1d
                    self.data.append(p1d_data)                

        if drop_tau_rescalings:
            if self.verbose: print('will drop tau scalings from arxiv')
            self._drop_tau_rescalings()
        if drop_temp_rescalings:
            if self.verbose: print('will drop temperature scalings from arxiv')
            self._drop_temperature_rescalings()

        if max_arxiv_size is not None:
            Ndata=len(self.data)
            if Ndata > max_arxiv_size:
                if self.verbose: print('will keep only',max_arxiv_size,'entries')
                keep=np.random.randint(0,Ndata,max_arxiv_size)
                keep_data=[self.data[i] for i in keep]
                self.data=keep_data

        N=len(self.data)
        if self.verbose:
            print('Arxiv setup, containing %d entries'%len(self.data))

        # store linear power parameters
        self.Delta2_p=np.array([self.data[i]['Delta2_p'] for i in range(N)])
        self.n_p=np.array([self.data[i]['n_p'] for i in range(N)])
        self.alpha_p=np.array([self.data[i]['alpha_p'] for i in range(N)])
        self.f_p=np.array([self.data[i]['f_p'] for i in range(N)])
        self.z=np.array([self.data[i]['z'] for i in range(N)])

        # store IGM parameters
        if not no_skewers:
            self.mF=np.array([self.data[i]['mF'] for i in range(N)])
            self.sigT_Mpc=np.array([self.data[i]['sigT_Mpc'] for i in range(N)])
            self.gamma=np.array([self.data[i]['gamma'] for i in range(N)])
            if self.data[0]['kF_Mpc']:
                self.kF_Mpc=np.array([self.data[i]['kF_Mpc'] for i in range(N)])

        return


    def _drop_tau_rescalings(self):
        """Keep only entries with scale_tau=1"""

        data = [x for x in self.data if x['scale_tau']==1.0]
        self.data = data
        return


    def _drop_temperature_rescalings(self):
        """Keep only entries with scale_T0=1 and scale_gamma=1"""

        data = [x for x in self.data if ((x['scale_T0']==1.0)
                                                & (x['scale_gamma']==1.0))]
        self.data = data
        return


    def print_entry(self,entry,keys=['z','Delta2_p','n_p','alpha_p','f_p',
                                    'mF','sigT_Mpc','gamma']):
        """Print basic information about a particular entry in the arxiv"""

        if entry >= len(self.data):
            raise ValueError('{} entry does not exist in arxiv'.format(entry))

        data = self.data[entry]
        info='entry = {}'.format(entry)
        for key in keys:
            info += ', {} = {:.4f}'.format(key,data[key])
        print(info)


