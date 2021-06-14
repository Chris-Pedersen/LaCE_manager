import numpy as np
from lace.cosmo import camb_cosmo

class CMBLikelihood(object):
    """ Object to return estimates of a CMB likelihood for a
    given cosmology. We do this by storing the parameter covariances
    approximated as Gaussians from the Planck chains """

    def __init__(self,cosmo):
        """ Use the target cosmology as the centre of the posterior
        to represent the truth, so we are consistent between the CMB
        and P1D. """

        ## Ordering of parameters here is important:
             ## omegabh2, omegach2, 100theta_MC, As, ns
             ## These are taken from the Planck chain
        self.cov_cmb=np.array([[ 2.21402189e-08, -1.17862732e-07,  1.67858338e-08,
                        7.25645861e-07,  2.95418057e-07],
                    [-1.17862732e-07,  1.86632441e-06, -1.47759444e-07,
                        -3.52821105e-06, -4.35537473e-06],
                    [ 1.67858338e-08, -1.47759444e-07,  9.58159747e-08,
                        6.82338866e-07,  4.71612574e-07],
                    [ 7.25645861e-07, -3.52821105e-06,  6.82338866e-07,
                        1.11972760e-03,  1.68665934e-05],
                    [ 2.95418057e-07, -4.35537473e-06,  4.71612574e-07,
                        1.68665934e-05,  1.90804173e-05]])
        self.cosmo=cosmo
        self.results=camb_cosmo.get_camb_results(self.cosmo)
        self.icov_cmb=np.linalg.inv(self.cov_cmb)
        self.mock_values=np.array([self.cosmo.ombh2,
                            self.cosmo.omch2,
                            self.results.cosmomc_theta(),
                            self.cosmo.InitPower.As*1e9,
                            self.cosmo.InitPower.ns])


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
            cosmomc_theta=cosmo_dic["cosmomc_theta"]
        else:
            cosmo_fid_results=camb_cosmo.get_camb_results(cosmo_fid)
            cosmomc_theta=cosmo_fid_results.cosmomc_theta()
        if "As" in cosmo_dic:
            As=cosmo_dic["As"]*1e9
        else:
            As=cosmo_fid.InitPower.As*1e9
        if "ns" in cosmo_dic:
            ns=cosmo_dic["ns"]
        else:
            ns=cosmo_fid.InitPower.ns

        test_values=np.array([ombh2,omch2,cosmomc_theta,As,ns])
        diff=self.mock_values-test_values

        return -0.5*np.dot(np.dot(self.icov_cmb,diff),diff)


    def return_CMB_only(nsamp=100000):
        """ Return the CMB likelihood distribution """

        data = np.random.multivariate_normal(self.mock_values,
                            self.cov_cmb, size=nsamp)
        ## have to convert As back to units that the rest of the code uses
        data[:,3]*=1e-9

        return data
