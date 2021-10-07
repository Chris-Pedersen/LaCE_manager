import numpy as np

class MargP1DLike(object):
    """ Object to return a Gaussian approximation to the marginalised
        likelihood on Delta2_star and n_star from a mock simulation """
    
    def __init__(self,sim_label,reduced_IGM=False):
        """ sim_label will be the simulation we are using
            as a mock dataset (will add later)
            reduced_IGM is a flag that will use a
            covariance matrix after marginalising over only
            ln_tau_0, purely in the interests of running faster chains
            
            We set the truth values from the sim_label, and the cov
            from the reduced_IGM flag as the covariance is roughly
            the same for each test sim """

        ## Hardcoding the central sim values for now
        self.sim_label=sim_label
        if sim_label=="central":
            self.true=np.array([0.3462089,-2.29839649])
        elif sim_label=="h":
            self.true=np.array([0.3393979,-2.29942019])
        elif sim_label=="nu":
            self.true=np.array([0.3425163,-2.29975242])
        else:
            print("Do not have truth values stored for the mock simulation")

        self.reduced_IGM=reduced_IGM
        if self.reduced_IGM==False:
            ## Use covariance after a full marginalisation
            self.cov=np.array([[1.77781998e-04, 4.56077117e-05],
                                [4.56077117e-05, 4.47283109e-05]])
        else:
            ## Use covariance after a marginalisation over only ln_tau_0
            self.cov=np.array([[8.20389465e-05, 3.19638330e-05],
                                [3.19638330e-05, 1.94088790e-05]])
        self.icov=np.linalg.inv(self.cov)

        
    def return_lya_like(self,vals):
        """ vals is a numpy array of the Delta2_star and n_star
            values we want to evaluate the likelihood at """

        diff=self.true-vals

        return -0.5*np.dot(np.dot(self.icov,diff),diff)
