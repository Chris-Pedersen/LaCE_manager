"""Dictionary describing the simulation parameter space."""

import numpy as np

class SimulationParameterSpace(object):
    """Describe simulation parameter space, to be used by latin hypercube."""

    def __init__(self,file_name=None,add_running=False,add_mu_H=False):
        """Construct space from file, or using default setting"""

        if file_name is None:
            self._default_setup(add_running,add_mu_H)
        else:
            self._setup_from_file(filename,add_running,add_mu_H)


    def _default_setup(self,add_running,add_mu_H):
        """Default setup of parameter space"""

        z_star=3.0
        kp_Mpc=0.7
        params={}
        params['Om_star']={'ip':len(params), 'min_val':0.950, 'max_val':0.975, 
                'z_star':z_star, 'latex':r'$\Omega_\star$'}
        params['Delta2_star']={'ip':len(params), 'min_val':0.25, 'max_val':0.45,
                'z_star':z_star, 'kp_Mpc':kp_Mpc, 'latex':r'$\Delta^2_\star$'}
        params['n_star']={'ip':len(params), 'min_val':-2.35, 'max_val':-2.25, 
                'z_star':z_star, 'kp_Mpc':kp_Mpc, 'latex':r'$n_\star$'}
        if add_running:
            params['alpha_star']={'ip':len(params), 'min_val':-0.265, 
                    'max_val':-0.165, 'z_star':z_star, 'kp_Mpc':kp_Mpc, 
                    'latex':r'$\alpha_\star$'}
        params['mu_He']={'ip':len(params), 'min_val':0.5, 'max_val':2.0, 
                    'latex':r'$\mu_{\rm He}$'}
        if add_mu_H:
            params['mu_H']={'ip':len(params), 'min_val':0.5, 'max_val':2.0, 
                '   latex':r'$\mu_{\rm H}$'}
        self.params=params


    def _setup_from_file(self,filename,add_running,add_mu_H):
        print('setup from file')
        raise ValueError('implement setup_from_file')

