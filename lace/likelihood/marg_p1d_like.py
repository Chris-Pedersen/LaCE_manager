import numpy as np

class MargP1DLike(object):
    """ Object to return a Gaussian approximation to the marginalised
        likelihood on Delta2_star and n_star from a mock simulation """
    
    def __init__(self):
        """ sim_label will be the simulation we are using
            as a mock dataset (will add later) """

        ## Hardcoding the central sim values for now
        self.true=np.array([0.34589603058122564,-2.299873036928861])
        self.cov=np.array([[2.64662445e-04, 4.53424261e-05],
                            [4.53424261e-05, 4.35919691e-05]])
        self.icov=np.linalg.inv(self.cov)

        
    def return_lya_like(self,vals):
        """ vals is a numpy array of the Delta2_star and n_star
            values we want to evaluate the likelihood at """

        diff=self.true-vals

        return -0.5*np.dot(np.dot(self.icov,diff),diff)