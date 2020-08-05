import numpy as np
import matplotlib.pyplot as plt
# our own modules
import likelihood
import iminuit_minimizer


class Scan1D(object):
    """Set a 1D grid of values, and minimize chi2 in each point"""

    def __init__(self,like,param_grid,verbose=True):
        """Setup with input likelihood, and parameter grid.
            - like: input likelihood (parameter to scan should be free param)
            - param_grid: {name,minval,maxval,nval} for parameter to scan """

        self.verbose=verbose
        self.like=like

        # setup parameter values to scan
        self.param_grid=param_grid
        self.param_values=np.linspace(self.param_grid['minval'],
                self.param_grid['maxval'],self.param_grid['nval'])

        # best-fit values (when all parameters are free) will be stored later
        self.global_best_fit=None
        # scan will be computed when needed
        self.chi2_scan=None


    def _cache_global_best_fit(self):
        """Maximize likelihood for all parameters and store results"""

        # start at the center of the unit cube
        start=len(self.like.free_params)*[0.5]
        # setup iminuit minimizer
        minimizer = iminuit_minimizer.IminuitMinimizer(self.like,start=start)

        # run minimizer and get best-fit values
        minimizer.minimize(compute_hesse=False)
        self.global_best_fit = minimizer.minimizer.np_values()

        if self.verbose:
            chi2 = self.like.get_chi2(values=self.global_best_fit)
            print('global chi2 =',chi2,'; best-fit =',self.global_best_fit)


    def _min_chi2(self,param_value):

        # figure out index of parameter that is being scanned
        pname=self.param_grid['name']
        ip=[i for i,p in enumerate(self.like.free_params) if p.name == pname][0]
        par=self.like.free_params[ip]
        fixed_cube_value=par.value_in_cube(param_value)
        if self.verbose:
            print(ip,par.name,param_value,'in cube',fixed_cube_value)

        # setup array with starting values, fixed for our scanned parameter
        start=np.copy(self.global_best_fit)
        start[ip]=fixed_cube_value

        # setup array of booleans to specify fixed parameter
        fix=len(self.like.free_params)*[False]
        fix[ip]=True

        # setup iminuit minimizer
        minimizer = iminuit_minimizer.IminuitMinimizer(self.like,
                start=start,fix=fix)

        # run minimizer and get best-fit values
        minimizer.minimize(compute_hesse=False)
        local_best_fit = minimizer.minimizer.np_values()

        # compute chi2 for best-fit values
        chi2 = self.like.get_chi2(values=local_best_fit)

        return chi2


    def _cache_scan(self):
        """Compute and cache 1D chi2 scan"""

        # the global fit will help chose the starting point
        if self.global_best_fit is None:
            self._cache_global_best_fit()

        # define array of parameter values to use
        self.chi2_scan=np.empty_like(self.param_values)
        for i in range(len(self.param_values)):
            self.chi2_scan[i] = self._min_chi2(self.param_values[i])


    def get_chi2_scan(self):
        """Compute scan if not computed yet, and return it"""

        if self.chi2_scan is None:
            if self.verbose: print('will compute chi2 scan')
            self._cache_scan()

        return self.param_values, self.chi2_scan


    def plot_chi2_scan(self,true_value=None):
        """Plot chi2 scan (compute it if needed)"""

        values, chi2 = self.get_chi2_scan()

        # find out name and index of scanned parameter
        pname=self.param_grid['name']
        ip=[i for i,p in enumerate(self.like.free_params) if p.name == pname][0]
        # add vertical line with value from global fit
        par=self.like.free_params[ip]
        global_best_fit=par.value_from_cube(self.global_best_fit[ip])
        plt.axvline(x=global_best_fit,ls=':',label='global fit')
        # add vertical line with value from global fit
        if true_value:
            plt.axvline(x=true_value,ls='-.',label='truth')

        # plot chi2 vs parameter scanned
        plt.plot(values,chi2,'-',label='scan')

        plt.xlabel(self.param_grid['name'])
        plt.ylabel('chi2')
        plt.grid(True)
        plt.legend()



class Scan2D(object):
    """Set a 2D grid of values, and minimize likelihood in each point"""

    def __init__(self,like,param_grid_1,param_grid_2,verbose=True):
        """Setup with input likelihood, and parameter grids.
            - like: input likelihood
            - param_grid_1: {name,minval,maxval,nval} for parameter 1
            - param_grid_2: same for second parameter (y axis in plot) """

        self.verbose=verbose
        self.like=like
        self.param_grid_1=param_grid_1
        self.param_grid_2=param_grid_2
        # setup parameter values to scan
        self.param_values_1=np.linspace(self.param_grid_1['minval'],
                self.param_grid_1['maxval'],self.param_grid_1['nval'])
        self.param_values_2=np.linspace(self.param_grid_2['minval'],
                self.param_grid_2['maxval'],self.param_grid_2['nval'])
        # scan will be computed when needed
        self.chi2_scan=None


    def _min_chi2(self,par_val_1,par_val_2):
        """Find minimum chi2 for a particular point in the parameter grid"""

        # for now use hack
        hack_mean_1=np.mean(self.param_values_1)
        hack_error_1=np.std(self.param_values_1)
        hack_mean_2=np.mean(self.param_values_2)
        hack_error_2=np.std(self.param_values_2)
        chi2 = ((par_val_1-hack_mean_1)/hack_error_1)**2 \
                + ((par_val_2-hack_mean_2)/hack_error_2)**2

        return chi2


    def _cache_scan(self):
        """Compute and cache 2D chi2 scan"""

        # will loop over all points in the 2D grid
        vals_1=self.param_values_1
        vals_2=self.param_values_2
        chi2_list=[self._min_chi2(v1,v2) for v2 in vals_2 for v1 in vals_1]

        # convert into numpy array, and reshape to matrix to plot
        self.chi2_scan=np.array(chi2_list).reshape([len(vals_2),len(vals_1)])


    def get_chi2_scan(self):
        """Compute scan if not computed yet, and return it"""

        if self.chi2_scan is None:
            if self.verbose: print('will compute chi2 scan')
            self._cache_scan()

        return self.param_values_1, self.param_values_2, self.chi2_scan


    def plot_chi2_scan(self):
        """Plot chi2 scan (compute it if needed)"""

        vals_1, vals_2, chi2_scan = self.get_chi2_scan()

        # set range of values in grid (used by pyplot.contour)
        extent=[np.min(vals_1),np.max(vals_1),np.min(vals_2),np.max(vals_2)]
        plt.contour(chi2_scan,extent=extent,levels=[1,4,11],origin='lower')
        plt.xlabel(self.param_grid_1['name'])
        plt.ylabel(self.param_grid_2['name'])
        plt.grid(True)
