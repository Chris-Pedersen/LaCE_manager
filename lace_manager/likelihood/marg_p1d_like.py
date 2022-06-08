import numpy as np
import scipy.interpolate

class MargP1DLike(object):
    """ Object to return a Gaussian approximation to the marginalised
        likelihood on Delta2_star and n_star from a mock simulation """
    
    def __init__(self,kde_fname=None,
                sim_label=None,reduced_IGM=False,polyfit=False):
        """  - kde_fname: read marginalised P1D from KDE filed (ignore other)
             - sim_label: simulation we are using as a mock dataset
             - reduced_IGM: use a covariance matrix after
               marginalising over only ln_tau_0, in the interests
               of running faster chains.
             - polyfit: Use Gaussian approximations fit from polyfit
               constraints instead of k_bin emulator
            
            We set the truth values from the sim_label, and the cov
            from the reduced_IGM flag as the covariance is roughly
            the same for each test sim """
            ### ANDREU: I don't think we should ever use the "truth" here!

        if kde_fname:
            print('will setup marg_p1d from KDE file',kde_fname)
            self.kde_fname=kde_fname
            self.kde_lnprob=read_kde(self.kde_fname)
            self.Gauss_mean=None
            self.Gauss_icov=None
        else:
            print('will setup Gaussian marg_p1d')
            self.kde_fname=None
            self.kde_lnprob=None
            # need to store this in chain info file
            self.sim_label=sim_label
            self.polyfit=polyfit
            self.reduced_IGM=reduced_IGM

            if polyfit==False:
                if sim_label=="central":
                    self.Gauss_mean=np.array([0.3462089,-2.29839649])
                elif sim_label=="h":
                    self.Gauss_mean=np.array([0.3393979,-2.29942019])
                elif sim_label=="nu":
                    self.Gauss_mean=np.array([0.3425163,-2.29975242])
                else:
                    raise Exception("Do not have Gaussian marg_p1d")
                # Covariance should be independent of simulation
                if reduced_IGM==False:
                    # When marginalising over 8 IGM parameters
                    cov=np.array([[1.77781998e-04, 4.56077117e-05],
                                    [4.56077117e-05, 4.47283109e-05]])
                else:
                    # When marginalising only over ln_tau_0
                    cov=np.array([[8.20389465e-05, 3.19638330e-05],
                                    [3.19638330e-05, 1.94088790e-05]])
            else:
                if sim_label=="central":
                    self.Gauss_mean=np.array([0.3462,-2.2986])
                elif sim_label=="nu":
                    self.Gauss_mean=np.array([0.356,-2.3041])
                elif sim_label=="running":
                    self.Gauss_mean=np.array([0.348,-2.3041])
                else:
                    raise Exception("Do not have Gaussian marg_p1d")
                # Covariance should be independent of simulation
                if reduced_IGM==False:
                    # When marginalising over 8 IGM parameters
                    cov=np.array([[1.56e-04, 4.67e-05],
                                    [4.67e-05, 5.4e-05]])
                else:
                    # When marginalising only over ln_tau_0
                    cov=np.array([[7.75e-05,2.59e-05 ],
                                    [2.59e-05, 1.35e-05]])

            self.Gauss_icov=np.linalg.inv(cov)


    def return_lya_like(self,vals):
        """ vals is a numpy array of the Delta2_star and n_star
            values we want to evaluate the likelihood at """

        if self.kde_lnprob:
            return self.get_kde_lnprob(vals)
        else:
            diff=self.Gauss_mean-vals
            return -0.5*np.dot(np.dot(self.Gauss_icov,diff),diff)


    def get_kde_lnprob(self,vals):
        """ Compute lnprob from KDE"""

        # get compressed parameters
        D2_star=vals[0]
        n_star=vals[1]

        # check parameters are within range
        x,y=self.kde_lnprob.get_knots()
        if (np.min(x)>D2_star) or (np.max(x)<D2_star):
            #print('D2_star={:.5f} out of bounds'.format(D2_star))
            return -np.inf
        if (np.min(y)>n_star) or (np.max(y)<n_star):
            #print('n_star={:.5f} out of bounds'.format(n_star))
            return -np.inf

        return self.kde_lnprob.ev(D2_star,n_star)


def read_kde(kde_fname):
    """ Read KDE from file, and setup 2D interpolator"""

    # open binary file
    data = np.load(kde_fname)

    # read 2D grid of parameters and KDE density
    D2_star=data['D2_star']
    n_star=data['n_star']
    lnprob=np.log(data['density'])
    max_lnprob=np.max(lnprob)

    # setup interpolator for normalised log probability
    return scipy.interpolate.RectBivariateSpline(D2_star,n_star,
                lnprob-max_lnprob)

