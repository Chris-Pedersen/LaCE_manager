import numpy as np
import matplotlib.pyplot as plt
# our own modules
import likelihood
import iminuit_minimizer

class GridScanner(object):
    """Set a 1D / 2D grid of values, and minimize likelihood in each point"""

    def __init__(self,like,grid_1,grid_2=None,verbose=True):
        """Setup with input likelihood, and parameter grid(s).
            - like: input likelihood
            - grid_1: [pname,minval,maxval,nval] for parameter 1
            - grid_2: same than grid_1, but optional for 2D scans """

        self.verbose=verbose
        self.in_like=like
        self.grid_1=grid_1
        self.grid_2=grid_2
        self.like_scan=None


    def _cache_scan(self):
        """Compute and cache likelihood scan"""

        self.like_scan=123


    def get_scan(self):
        """Compute scan if not computed yet, and return it"""

        if self.like_scan is None:
            if self.verbose: print('will compute likelihood scan')
            self._cache_scan()

        return self.like_scan


