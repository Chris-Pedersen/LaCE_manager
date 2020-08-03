import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
# our own modules
import likelihood


class IminuitMinimizer(object):
    """Wrapper around an iminuit minimizer for Lyman alpha likelihood"""

    def __init__(self,like,error=0.02,verbose=True):
        """Setup minimizer from likelihood."""

        self.verbose=verbose
        self.like=like

        # set initial values (for now, center of the unit cube)
        ini_values=0.5*np.ones(len(self.like.free_params))

        # setup iminuit object (errordef=0.5 if using log-likelihood)
        self.minimizer = Minuit.from_array_func(like.minus_log_prob,ini_values,
                error=error,errordef=0.5)


    def minimize(self,compute_hesse=True):
        """Run migrad optimizer, and optionally compute Hessian matrix"""

        if self.verbose: print('will run migrad')
        self.minimizer.migrad()
        
        if compute_hesse:
            if self.verbose: print('will compute Hessian matrix')
            self.minimizer.hesse()


    def plot_best_fit(self,plot_every_iz=1):
        """ Plot best-fit P1D vs data.
            - plot_every_iz (int): skip some redshift bins. """

        # get best-fit values from minimizer (should check that it was run)
        best_fit_values=self.minimizer.np_values()
        if self.verbose: print('best-fit values =',best_fit_values)

        plt.title("iminuit best fit")
        self.like.plot_p1d(plot_every_iz=2,values=best_fit_values)
        plt.show()

        return

