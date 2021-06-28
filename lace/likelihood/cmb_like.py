import numpy as np
from lace.cosmo import camb_cosmo

class CMBLikelihood(object):
    """ Object to return estimates of a CMB likelihood for a
    given cosmology. We do this by storing the parameter covariances
    approximated as Gaussians from the Planck chains """

    def __init__(self,cosmo,nu_mass=False):
        """ Use the target cosmology as the centre of the posterior
        to represent the truth, so we are consistent between the CMB
        and P1D. """

        ## Ordering of parameters here is important:
             ## omegabh2, omegach2, 100theta_MC, As, ns
             ## These are taken from the Planck chain

        self.cosmo=cosmo
        self.results=camb_cosmo.get_camb_results(self.cosmo)

        ## In the same units as the CMB covariance
        self.mock_values=np.array([self.cosmo.ombh2,
                            self.cosmo.omch2,
                            self.results.cosmomc_theta()*100,
                            self.cosmo.InitPower.As*1e9,
                            self.cosmo.InitPower.ns])

        ## With As and cosmomc_theta in the same units as the rest of the code
        self.true_values=np.array([self.cosmo.ombh2,
                            self.cosmo.omch2,
                            self.results.cosmomc_theta(),
                            self.cosmo.InitPower.As,
                            self.cosmo.InitPower.ns])

        ## List of parameters in LaTeX form
        self.param_list=["$\omega_b$","$\omega_c$","$\\theta_{MC}$","$A_s$","$n_s$"]

        ## Add neutrino mass info if we are using it
        self.nu_mass=nu_mass
        if self.nu_mass==True:
            self.cov_cmb=cmb_mnu_cov
            self.mock_values=np.append(self.mock_values,
                        self.cosmo.omnuh2/0.0107333333)
            self.true_values=np.append(self.true_values,
                        self.cosmo.omnuh2/0.0107333333)
            self.param_list.append("$\Sigma m_\\nu$")
        else:
            self.cov_cmb=cmb_nonu_cov
        self.icov_cmb=np.linalg.inv(self.cov_cmb)


    def get_cmb_like(self,cosmo_dic,cosmo_fid):
        """ For a given target cosmology, return an approximation of the
        Planck likelihood for this cosmology using the stored parameter
        covariance. We need a dictionary of the cosmology parameters that are
        being varied, and a fiducial cosmology for any parameters that aren't varied """

        ## Need to create some "diff", the difference between the parameter
        ## values for a given model and the values in the mock cosmology

        if "ombh2" in cosmo_dic:
            ombh2=cosmo_dic["ombh2"]
        else:
            ombh2=cosmo_fid.ombh2
        if "omch2" in cosmo_dic:
            omch2=cosmo_dic["omch2"]
        else:
            omch2=cosmo_fid.omch2
        if "cosmomc_theta" in cosmo_dic:
            cosmomc_theta=cosmo_dic["cosmomc_theta"]*100
        else:
            cosmo_fid_results=camb_cosmo.get_camb_results(cosmo_fid)
            cosmomc_theta=cosmo_fid_results.cosmomc_theta()*100
        if "As" in cosmo_dic:
            As=cosmo_dic["As"]*1e9
        else:
            As=cosmo_fid.InitPower.As*1e9
        if "ns" in cosmo_dic:
            ns=cosmo_dic["ns"]
        else:
            ns=cosmo_fid.InitPower.ns
        
        test_values=np.array([ombh2,omch2,cosmomc_theta,As,ns])

        if self.nu_mass==True:
            if "mnu" in cosmo_dic:
                mnu=cosmo_dic["mnu"]
            else:
                mnu=cosmo_fid.omnuh2/0.0107333333
            test_values=np.append(test_values,mnu)

        diff=self.mock_values-test_values

        return -0.5*np.dot(np.dot(self.icov_cmb,diff),diff)


    def return_CMB_only(self,nsamp=100000):
        """ Return the CMB likelihood distribution """

        data = np.random.multivariate_normal(self.mock_values,
                            self.cov_cmb, size=nsamp)
        ## have to convert As back to units that the rest of the code uses
        data[:,3]*=1e-9
        ## have to convert As back to units that the rest of the code uses
        data[:,2]*=0.01

        return data


cmb_nonu_cov=np.array([[ 2.49435335e-08, -1.15686924e-07,  1.86073091e-08,
         9.03007066e-07,  3.28031640e-07],
       [-1.15686924e-07,  1.92020194e-06, -1.64274248e-07,
        -2.76307775e-06, -4.29827806e-06],
       [ 1.86073091e-08, -1.64274248e-07,  9.97926903e-08,
         7.73656782e-07,  5.14558108e-07],
       [ 9.03007066e-07, -2.76307775e-06,  7.73656782e-07,
         1.17136384e-03,  1.76531323e-05],
       [ 3.28031640e-07, -4.29827806e-06,  5.14558108e-07,
         1.76531323e-05,  1.99589130e-05]])

cmb_mnu_cov=np.array([[ 2.88023813e-08, -1.27089673e-07,  2.43676928e-08,
            7.60667530e-07,  4.01978397e-07, -7.56495151e-06],
        [-1.27089673e-07,  1.92181010e-06, -1.94174370e-07,
            -6.29402507e-07, -4.58358014e-06,  4.13128633e-05],
        [ 2.43676928e-08, -1.94174370e-07,  1.10834884e-07,
            9.89076692e-07,  6.01273667e-07, -1.31572811e-05],
        [ 7.60667530e-07, -6.29402507e-07,  9.89076692e-07,
            1.15088641e-03,  1.11767973e-05,  8.01147431e-05],
        [ 4.01978397e-07, -4.58358014e-06,  6.01273667e-07,
            1.11767973e-05,  2.15653471e-05, -1.73038156e-04],
        [-7.56495151e-06,  4.13128633e-05, -1.31572811e-05,
            8.01147431e-05, -1.73038156e-04,  3.15689398e-02]])
