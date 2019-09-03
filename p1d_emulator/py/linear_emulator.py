import numpy as np
import sys
import os
import json
import scipy.interpolate
import p1d_arxiv
import poly_p1d


class LinearEmulator(object):
    """Linear interpolation emulator for flux P1D."""

    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
            emulate_slope=True,emulate_running=False,
            emulate_growth=False,emulate_pressure=True,
            drop_tau_rescalings=False,drop_temp_rescalings=False,
            keep_every_other_rescaling=False,
            deg=4,kmax_Mpc=10.0,max_arxiv_size=None,
            undersample_z=1,verbose=False):
        """Setup emulator from base sim directory and label identifying skewer
            configuration (number, width)"""

        self.verbose=verbose

        # read all files with P1D measured in simulation suite
        self.arxiv=p1d_arxiv.ArxivP1D(basedir,p1d_label,skewers_label,
                    drop_tau_rescalings=drop_tau_rescalings,
                    drop_temp_rescalings=drop_temp_rescalings,
                    keep_every_other_rescaling=keep_every_other_rescaling,
                    max_arxiv_size=max_arxiv_size,undersample_z=undersample_z,
                    verbose=verbose)

        # for each model in arxiv, fit smooth function to P1D
        self._fit_p1d_in_arxiv(deg,kmax_Mpc)

        # setup parameter space to be used in emulator
        self._setup_param_space(emulate_slope=emulate_slope,
                    emulate_running=emulate_running,
                    emulate_growth=emulate_growth,
                    emulate_pressure=emulate_pressure)

        # for each order in polynomial, setup interpolation object
        self._setup_interp(deg)
        

    def _fit_p1d_in_arxiv(self,deg,kmax_Mpc):
        """For each entry in arxiv, fit polynomial to log(p1d)"""
        
        for entry in self.arxiv.data:
            k_Mpc = entry['k_Mpc']
            p1d_Mpc = entry['p1d_Mpc']
            fit_p1d = poly_p1d.PolyP1D(k_Mpc,p1d_Mpc,kmin_Mpc=1.e-3,
                    kmax_Mpc=kmax_Mpc,deg=deg)
            entry['fit_p1d'] = fit_p1d


    def _setup_param_space(self,emulate_slope,emulate_running,
                            emulate_growth,emulate_pressure):
        """Set order of parameters in emulator"""

        self.params=['Delta2_p']
        if emulate_slope:
            self.params.append('n_p')
        if emulate_running:
            self.params.append('alpha_p')
        if emulate_growth:
            self.params.append('f_p')
        self.params += ['mF','sigT_Mpc','gamma']
        if emulate_pressure:
            self.params.append('kF_Mpc')
        if self.verbose:
            print('parameter names in emulator',self.params)


    def _setup_interp(self,deg):
        """For each order in polynomial, setup interpolation object"""

        # for each parameter in params, get values from arxiv
        point_params=[]
        for par in self.params:
            values = np.array([entry[par] for entry in self.arxiv.data])
            point_params.append(values)
        self.points=np.vstack(point_params).transpose()

        N=len(self.arxiv.data)
        self.linterps=[]
        for p in range(deg+1):
            print('setup interpolator for coefficient',p)
            values = [entry['fit_p1d'].lnP[p] for entry in self.arxiv.data] 
            linterp = scipy.interpolate.LinearNDInterpolator(self.points,values)
            self.linterps.append(linterp)
            # it is good to try the interpolator to finish the setup
            # (it might help to avoid thread issues later on)
            test_point=np.median(self.points,axis=0)
            print(test_point,'test',linterp(test_point))


    def _point_from_model(self,model):
        """Extract model parameters from dictionary in the right order"""

        point=[]
        for par in self.params:
            point.append(model[par])

        return np.array(point)


    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False):
        """Return emulate 1D power spectrum at input k values"""

        if self.verbose: print('asked to emulate model',model)

        # get interpolation point from input model
        point = self._point_from_model(model)
        if self.verbose: print('evaluate point',point)

        # emulate coefficients for PolyP1D object (note strange order of coeffs)
        Npar=len(self.linterps)
        coeffs=np.empty(Npar)
        for i in range(Npar):
            coeffs[Npar-i-1] = self.linterps[i](point)
            # linear interpolation can not extrapolate
            if np.isnan(coeffs[Npar-i-1]):
                if self.verbose:
                    print('linear emulator failed',point)
                    if return_covar:
                        return None,None
                    else:
                        return None
        if self.verbose: print('got coefficients',coeffs)

        # set P1D object
        kmin_Mpc=self.arxiv.data[0]['fit_p1d'].kmin_Mpc
        smooth_p1d = poly_p1d.PolyP1D(lnP_fit=coeffs,kmin_Mpc=kmin_Mpc)
        p1d_Mpc = smooth_p1d.P_Mpc(k_Mpc)

        if return_covar:
            N=len(p1d_Mpc)
            return p1d_Mpc, np.zeros([N,N])
        else:
            return p1d_Mpc

