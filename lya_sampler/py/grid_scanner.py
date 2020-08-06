import numpy as np
import matplotlib.pyplot as plt
# our own modules
import likelihood
import iminuit_minimizer


class Scan1D(object):
    """Set a 1D grid of values, and minimize minus_log_prob in each point"""

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
        self.grid_scan=None


    def _cache_global_best_fit(self):
        """Maximize posterior for all parameters and store results"""

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


    def _max_log_like(self,param_value):
        """Return max log_like for best-fit model in a particular point
            (note that the minimization is done in minus_log_prob)."""

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

        # compute minus_log_prob for best-fit values
        max_log_like = self.like.get_log_like(values=local_best_fit)

        return max_log_like


    def _cache_scan(self):
        """Compute and cache 1D grid scan"""

        # the global fit will help chose the starting point
        if self.global_best_fit is None:
            self._cache_global_best_fit()

        # define array of parameter values to use
        self.grid_scan=np.empty_like(self.param_values)
        for i in range(len(self.param_values)):
            self.grid_scan[i] = self._max_log_like(self.param_values[i])


    def get_grid_scan(self):
        """Compute scan if not computed yet, and return it"""

        if self.grid_scan is None:
            if self.verbose: print('will compute grid scan')
            self._cache_scan()

        return self.param_values, self.grid_scan


    def plot_grid_scan(self,true_value=None,cube_values=False,
            yaxis='DeltaChi2'):
        """Plot grid scan (compute it if needed)
            - true_value: if provided, will add vertical line
            - cube_values: use cube values [0,1] in x-axis
            - yaxis: quantity to plot (DeltaChi2,DeltaMinusLogLike')"""

        # get parameter values in grid, and maximum likelihood in each point
        values, max_log_like = self.get_grid_scan()

        # find out name and index of scanned parameter
        pname=self.param_grid['name']
        ip=[i for i,p in enumerate(self.like.free_params) if p.name == pname][0]
        par=self.like.free_params[ip]

        # add vertical lines with truth and value from global fit
        global_best_fit_in_cube=self.global_best_fit[ip]
        if cube_values:
            xval=[par.value_in_cube(val) for val in values]
            plt.xlabel(pname+' (in cube)')
            plt.axvline(x=global_best_fit_in_cube,ls=':',label='global fit')
            if true_value:
                true_value_in_cube=par.value_in_cube(true_value)
                plt.axvline(x=true_value_in_cube,ls='--',label='truth')
        else:
            xval=values
            plt.xlabel(pname)
            global_best_fit=par.value_from_cube(global_best_fit_in_cube)
            plt.axvline(x=global_best_fit,ls=':',label='global fit')
            if true_value:
                plt.axvline(x=true_value,ls='--',label='truth')

        # figure out quantity to use in y axis
        global_log_like=self.like.get_log_like(values=self.global_best_fit)
        minus_delta_log_like=-1.0*(max_log_like-global_log_like)
        if yaxis is 'DeltaChi2':
            yval = 2.0*minus_delta_log_like
            plt.ylabel('Delta Chi2')
        elif yaxis is 'DeltaMinusLogLike':
            yval = minus_delta_log_like
            plt.ylabel('minus Delta log like')
        else:
            raise ValueError('implement plotting for ',yaxis)

        plt.plot(xval,yval,'-',label='scan')
        plt.legend()
        plt.grid(True)



class Scan2D(object):
    """Set a 2D grid of values, and minimize minus_log_prob in each point"""

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

        # best-fit values (when all parameters are free) will be stored later
        self.global_best_fit=None
        # scan will be computed when needed
        self.grid_scan=None


    def _cache_global_best_fit(self):
        """Maximize posterior for all parameters and store results"""

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


    def _max_log_like(self,par_val_1,par_val_2):
        """Return max log_like for best-fit model in a particular point
            (note that the minimization is done in minus_log_prob)."""

        # figure out indices of parameters that are being scanned
        pname_1=self.param_grid_1['name']
        pname_2=self.param_grid_2['name']
        ip1=[i for i,p in enumerate(self.like.free_params) if p.name == pname_1][0]
        ip2=[i for i,p in enumerate(self.like.free_params) if p.name == pname_2][0]
        par1=self.like.free_params[ip1]
        par2=self.like.free_params[ip2]
        fixed_cube_val_1=par1.value_in_cube(par_val_1)
        fixed_cube_val_2=par2.value_in_cube(par_val_2)
        if self.verbose:
            print(ip1,par1.name,par_val_1,'in cube',fixed_cube_val_1)
            print(ip2,par2.name,par_val_2,'in cube',fixed_cube_val_2)

        # setup array with starting values, fixed for our scanned parameter
        start=np.copy(self.global_best_fit)
        start[ip1]=fixed_cube_val_1
        start[ip2]=fixed_cube_val_2

        # setup array of booleans to specify fixed parameter
        fix=len(self.like.free_params)*[False]
        fix[ip1]=True
        fix[ip2]=True

        # setup iminuit minimizer
        minimizer = iminuit_minimizer.IminuitMinimizer(self.like,
                start=start,fix=fix)

        # run minimizer and get best-fit values
        minimizer.minimize(compute_hesse=False)
        local_best_fit = minimizer.minimizer.np_values()

        # compute max log-like for best-fit values
        max_log_like = self.like.get_log_like(values=local_best_fit)

        return max_log_like


    def _cache_scan(self):
        """Compute and cache 2D grid scan"""

        # the global fit will help chose the starting point
        if self.global_best_fit is None:
            self._cache_global_best_fit()

        # will loop over all points in the 2D grid
        vals_1=self.param_values_1
        vals_2=self.param_values_2
        grid_list=[self._max_log_like(v1,v2) for v2 in vals_2 for v1 in vals_1]

        # convert into numpy array, and reshape to matrix to plot
        self.grid_scan=np.array(grid_list).reshape([len(vals_2),len(vals_1)])


    def get_grid_scan(self):
        """Compute scan if not computed yet, and return it"""
        if self.grid_scan is None:
            if self.verbose: print('will compute grid scan')
            self._cache_scan()

        return self.param_values_1, self.param_values_2, self.grid_scan


    def plot_grid_scan(self,true_values=None,cube_values=False,
            zaxis='DeltaChi2',levels=[2.30,6.18,11.83]):
        """Plot grid scan (compute it if needed)
            - true_values: if provided, will add vertical line
            - cube_values: use cube values [0,1] in x-axis
            - zaxis: quantity to plot (DeltaChi2,DeltaMinusLogLike')
            - levels: contours to plot. """

        # get parameter values in grid, and maximum likelihood in each point
        vals_1, vals_2, max_log_like = self.get_grid_scan()

        # find out names and indices of scanned parameters
        pname_1=self.param_grid_1['name']
        pname_2=self.param_grid_2['name']
        ip1=[i for i,p in enumerate(self.like.free_params) if p.name == pname_1][0]
        ip2=[i for i,p in enumerate(self.like.free_params) if p.name == pname_2][0]
        par1=self.like.free_params[ip1]
        par2=self.like.free_params[ip2]

        # add vertical lines with truth and value from global fit
        global_1_in_cube=self.global_best_fit[ip1]
        global_2_in_cube=self.global_best_fit[ip2]
        if cube_values:
            xval=[par1.value_in_cube(val) for val in vals_1]
            yval=[par2.value_in_cube(val) for val in vals_2]
            plt.xlabel(pname_1+' (in cube)')
            plt.ylabel(pname_2+' (in cube)')
            plt.axvline(x=global_1_in_cube,ls=':',color='gray',label='global')
            plt.axhline(y=global_2_in_cube,ls=':',color='gray')
            if true_values:
                true_1_in_cube=par1.value_in_cube(true_values[0])
                true_2_in_cube=par2.value_in_cube(true_values[1])
                plt.axvline(x=true_1_in_cube,ls='--',color='gray',label='truth')
                plt.axhline(y=true_2_in_cube,ls='--',color='gray')
        else:
            xval=vals_1
            yval=vals_2
            plt.xlabel(pname_1)
            plt.ylabel(pname_2)
            global_1=par1.value_from_cube(global_1_in_cube)
            global_2=par2.value_from_cube(global_2_in_cube)
            plt.axvline(x=global_1,ls=':',color='gray',label='global')
            plt.axhline(y=global_2,ls=':',color='gray')
            if true_values:
                plt.axvline(x=true_values[0],ls='--',color='gray',label='truth')
                plt.axhline(x=true_values[1],ls='--',color='gray')

        # set range of values in grid (used by pyplot.contour)
        extent=[np.min(xval),np.max(xval),np.min(yval),np.max(yval)]

        # figure out quantity to use in contour
        global_log_like=self.like.get_log_like(values=self.global_best_fit)
        minus_delta_log_like=-1.0*(max_log_like-global_log_like)
        if zaxis is 'DeltaChi2':
            zval = 2.0*minus_delta_log_like
            plt.title('Delta Chi2 = '+str(levels))
        elif zaxis is 'DeltaMinusLogLike':
            zval = minus_delta_log_like
            plt.title('minus Delta log like = '+str(levels))
        else:
            raise ValueError('implement plotting for ',zaxis)

        plt.contour(zval,extent=extent,levels=levels,origin='lower')
        plt.grid(True)
        plt.legend()
