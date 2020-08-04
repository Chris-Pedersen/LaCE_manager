import numpy as np
import matplotlib.pyplot as plt
# our own modules
import likelihood
import iminuit_minimizer


class Scan1D(object):
    """Set a 1D grid of values, and minimize chi2 in each point"""

    def __init__(self,like,param_grid,verbose=True):
        """Setup with input likelihood, and parameter grid.
            - like: input likelihood
            - param_grid: {name,minval,maxval,nval} for parameter to scan """

        self.verbose=verbose
        self.in_like=like
        self.param_grid=param_grid
        # setup parameter values to scan
        self.param_values=np.linspace(self.param_grid['minval'],
                self.param_grid['maxval'],self.param_grid['nval'])
        # scan will be computed when needed
        self.chi2_scan=None


    def _min_chi2(self,param_value):
        """Find minimum chi2 for a particular value of the grid parameter"""

        # for now use hack
        hack_mean=np.mean(self.param_values)
        hack_error=np.std(self.param_values)
        chi2 = ((param_value-hack_mean)/hack_error)**2

        return chi2


    def _cache_scan(self):
        """Compute and cache 1D chi2 scan"""

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


    def plot_chi2_scan(self):
        """Plot chi2 scan (compute it if needed)"""

        values, chi2 = self.get_chi2_scan()

        plt.plot(values,chi2)
        plt.xlabel(self.param_grid['name'])
        plt.ylabel('chi2')
        plt.grid(True)

class Scan2D(object):
    """Set a 2D grid of values, and minimize likelihood in each point"""

    def __init__(self,like,param_grid_1,param_grid_2,verbose=True):
        """Setup with input likelihood, and parameter grids.
            - like: input likelihood
            - param_grid_1: {name,minval,maxval,nval} for parameter 1
            - param_grid_2: same for second parameter (y axis in plot) """

        self.verbose=verbose
        self.in_like=like
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
