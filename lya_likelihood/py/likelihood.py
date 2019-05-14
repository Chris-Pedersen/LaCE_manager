import numpy as np
import matplotlib.pyplot as plt
import data_PD2013
import lya_theory
import likelihood_parameter


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(self,data=None,theory=None,emulator=None,
                    free_parameters=None,verbose=True):
        """Setup likelihood from theory and data"""

        self.verbose=verbose

        if data:
            self.data=data
        else:
            if self.verbose: print('use default data')
            self.data=data_PD2013.P1D_PD2013(blind_data=True)

        if theory:
            self.theory=theory
        else:
            zs=self.data.z
            if self.verbose: print('use default theory')
            self.theory=lya_theory.LyaTheory(zs,emulator=emulator)

        # setup parameters
        if not free_parameters:
            free_parameters=['ln_tau_0']
        self._setup_free_parameters(free_parameters)

        if self.verbose: print(len(self.free_params),'free parameters')

        return


    def _setup_free_parameters(self,free_parameter_names):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params=[]

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        # select parameters using input list of names
        for par in params:
            if par.name in free_parameter_names:
                self.free_params.append(par)

        Nfree=len(self.free_params)
        Nin=len(free_parameter_names)
        assert (Nfree==Nin), 'could not setup free paremeters'

        if self.verbose:
            print('likelihood setup with {} free parameters'.format(Nfree))

        return


    def update_parameters(self,values):
        """Use input array of values (in cube) to update parameters,
            and update theories."""

        assert len(values)==len(self.free_params),'size mismatch'
        Npar=len(values)
        for ip in range(Npar):
            self.free_params[ip].set_from_cube(values[ip])

        if self.verbose: print('updated parameters, update theories')

        # pass parameters to internal theories to update their models
        self.theory.update_parameters(self.free_params)

        return


    def get_chi2(self,linP_Mpc_params=None):
        """Compute chi2 using data and theory"""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d = self.theory.get_p1d_kms(k_kms,linP_Mpc_params)
        if self.verbose: print('got P1D from emulator')

        # compute chi2 contribution from each redshift bin
        chi2=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            p1d=self.data.get_Pk_iz(iz)
            cov=self.data.get_cov_iz(iz)
            # compute chi2 for this redshift bin
            icov = np.linalg.inv(cov)
            diff = (p1d-emu_p1d[iz])
            chi2_z = np.dot(np.dot(icov,diff),diff)
            chi2 += chi2_z
            if self.verbose: print('added {} to chi2'.format(chi2_z))
        
        return chi2


    def log_prob(self,values,linP_Mpc_params=None):

        # for now priors are top hats in 0 < x < 1
        if max(values) > 1.0:
            return -np.inf
        if min(values) < 0.0:
            return -np.inf

        # update parameter and theories
        self.update_parameters(values)

        # compute chi2
        chi2=self.get_chi2(linP_Mpc_params=linP_Mpc_params)
        loglike=-0.5*chi2

        return loglike


    def go_silent(self):
        """Turn off verbosity on all objects in likelihood object"""

        self.verbose=False
        self.theory.verbose=False
        self.theory.cosmo.verbose=False
        self.theory.emulator.verbose=False
        self.theory.emulator.arxiv.verbose=False


    def go_loud(self):
        """Turn on verbosity on all objects in likelihood object"""

        self.verbose=True
        self.theory.verbose=True
        self.theory.cosmo.verbose=True
        self.theory.emulator.verbose=True
        self.theory.emulator.arxiv.verbose=True


    def plot_p1d(self,linP_Mpc_params=None,plot_only_few_bins=False):
        """Plot P1D in theory vs data. If plot_only_iz is set,
            plot only few redshift bins"""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d = self.theory.get_p1d_kms(k_kms,linP_Mpc_params)
        if self.verbose: print('got P1D from emulator')

        # plot only few redshifts for clarity
        for iz in range(0,Nz,3):
            # acess data for this redshift
            z=zs[iz]
            p1d_data=self.data.get_Pk_iz(iz)
            p1d_cov=self.data.get_cov_iz(iz)
            p1d_theory=emu_p1d[iz]
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.errorbar(k_kms,p1d_data*k_kms/np.pi,color=col,
                    yerr=np.sqrt(np.diag(p1d_cov))*k_kms/np.pi,label='z=%.1f'%z)
            plt.plot(k_kms,p1d_theory*k_kms/np.pi,color=col)
        plt.yscale('log')
        plt.legend()
        plt.xlabel('k [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.show()

        return


    def overplot_emulator_calls(self,param_1,param_2,linP_Mpc_params=None):
        """For parameter pair (param1,param2), overplot emulator calls
            with values stored in arxiv, color coded by redshift"""

        # get emulator calls
        emu_calls=self.theory.get_emulator_calls(linP_Mpc_params)

        # figure out values of param_1,param_2 called
        call_1=[emu_call[param_1] for emu_call in emu_calls]
        call_2=[emu_call[param_2] for emu_call in emu_calls]

        # figure out values of param_1,param_2 in arxiv
        emu_data=self.theory.emulator.arxiv.data
        Nemu=len(emu_data)
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu)])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu)])

        # overplot
        zs=self.data.z
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu)])
        zmin=min(min(emu_z),min(zs))
        zmax=max(max(emu_z),max(zs))
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        plt.scatter(call_1,call_2,c=zs,s=50,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return
