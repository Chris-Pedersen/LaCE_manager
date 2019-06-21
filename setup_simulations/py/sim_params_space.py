"""Dictionary describing the simulation parameter space."""

import numpy as np

class SimulationParameterSpace(object):
    """Describe simulation parameter space, to be used by latin hypercube."""

    def __init__(self,filename=None,add_growth=True,add_amplitude=True,
                add_slope=True,add_running=False,
                add_heat_amp=True,add_heat_slo=True,add_z_rei=True):
        """Construct space from file, or using default setting"""

        if filename is None:
            self._default_setup(add_growth,add_amplitude,add_slope,add_running,
                                add_heat_amp,add_heat_slo,add_z_rei)
        else:
            self._setup_from_file(filename,add_growth,add_amplitude,
                            add_slope,add_running,
                            add_heat_amp,add_heat_slo,add_z_rei)

    def _default_setup(self,add_growth,add_amplitude,add_slope,add_running,
                                add_heat_amp,add_heat_slo,add_z_rei):
        """Default setup of parameter space"""

        self.z_star=3.0
        self.kp_Mpc=0.7
        params={}
        if add_growth:
            params['Om_star']={'ip':len(params),
                    'min_val':0.955, 'max_val':0.975,
                    'z_star':self.z_star, 'latex':r'$\Omega_\star$'}
        if add_amplitude:
            params['Delta2_star']={'ip':len(params),
                    'min_val':0.25, 'max_val':0.45,
                    'z_star':self.z_star, 'kp_Mpc':self.kp_Mpc,
                    'latex':r'$\Delta^2_\star$'}
        if add_slope:
            params['n_star']={'ip':len(params),
                    'min_val':-2.35, 'max_val':-2.25,
                    'z_star':self.z_star, 'kp_Mpc':self.kp_Mpc,
                    'latex':r'$n_\star$'}
        if add_running:
            params['alpha_star']={'ip':len(params),
                    'min_val':-0.265, 'max_val':-0.165,
                    'z_star':self.z_star, 'kp_Mpc':self.kp_Mpc,
                    'latex':r'$\alpha_\star$'}
        if add_heat_amp:
            params['heat_amp']={'ip':len(params), 'min_val':0.5, 'max_val':2.0,
                    'latex':r'$H_A$'}
        if add_heat_slo:
            params['heat_slo']={'ip':len(params), 'min_val':-0.5, 'max_val':0.5,
                    'latex':r'$H_S$'}
        if add_z_rei:
            params['z_rei']={'ip':len(params), 'min_val':5.5, 'max_val':15.0,
                    'latex':r'$z_r$'}

        self.params=params


    def _setup_from_file(self,filename,add_growth,add_amplitude,
                            add_slope,add_running,
                            add_heat_amp,add_heat_slo,add_z_rei):
        print('should implement setup from file')
        self._default_setup(add_growth,add_amplitude,add_slope,add_running,
                            add_heat_amp,add_heat_slo,add_z_rei)
        #raise ValueError('implement setup_from_file')
