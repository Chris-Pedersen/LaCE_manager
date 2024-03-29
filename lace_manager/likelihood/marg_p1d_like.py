import numpy as np
import scipy.interpolate

class MargP1DLike(object):
    """ Object to return a Gaussian approximation to the marginalised
        likelihood on Delta2_star and n_star from a mock simulation """
    
    def __init__(self,grid_fname=None,Gnedin=False,
                sim_label=None,reduced_IGM=False,polyfit=False):
        """  - grid_fname: read 2D grid of marg P1D from file (ignore others)
             - Gnedin: Gaussianize n_star with monotonic function from Nick
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

        if grid_fname:
            print('will setup marg_p1d from grid file',grid_fname)
            self.grid_fname=grid_fname
            # setup grid for likelihood or log-likelihood
            self.setup_grid()
            self.Gauss_mean=None
            self.Gauss_icov=None
            self.Gnedin=None
        elif Gnedin:
            print('will setup Gaussianized marg_p1d')
            self.grid_fname=None
            self.grid_like=None
            self.grid_log_like=None
            self.Gnedin=True
            assert polyfit and sim_label=="central" and not reduced_IGM, "Gaussianize"
            # need to store this in chain info file
            self.sim_label=sim_label
            self.polyfit=polyfit
            self.reduced_IGM=reduced_IGM
            # When marginalising over 8 IGM parameters
            self.Gauss_mean=np.array([0.34939,0.19337])
            cov=np.array([[1.7815e-04, 5.0512e-03], [5.0512e-03, 1.1383e+00]])
            self.Gauss_icov=np.linalg.inv(cov)
        else:
            print('will setup Gaussian marg_p1d')
            self.grid_fname=None
            self.grid_like=None
            self.grid_log_like=None
            self.Gnedin=False
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
                    self.Gauss_mean=np.array([0.3494,-2.3026])
                elif sim_label=="nu":
                    self.Gauss_mean=np.array([0.356,-2.3041])
                elif sim_label=="running":
                    self.Gauss_mean=np.array([0.348,-2.3041])
                else:
                    raise Exception("Do not have Gaussian marg_p1d")
                # Covariance should be independent of simulation
                if reduced_IGM==False:
                    # When marginalising over 8 IGM parameters
                    cov=np.array([[1.782e-04, 3.9623e-05],
                                    [3.9623e-05, 6.5565e-05]])
                else:
                    # When marginalising only over ln_tau_0
                    cov=np.array([[7.75e-05,2.59e-05 ],
                                    [2.59e-05, 1.35e-05]])

            self.Gauss_icov=np.linalg.inv(cov)


    def get_log_like(self,Delta2_star,n_star,z_star=3.0,kp_kms=0.009):
        """ Return (marginalised) log likelihood on compressed parameters"""

        # for now, assume we use the traditional pivot point
        assert (z_star==3.0) and (kp_kms==0.009), 'update pivot point'

        if self.grid_like or self.grid_log_like:
            return self.get_grid_log_like(Delta2_star,n_star)
        else:
            if self.Gnedin:
                # hard-coded for now
                mean_n_star=-2.302567
                rms_n_star=0.0080972
                u=(n_star-mean_n_star)/rms_n_star
                us=u+0.2
                uf = us*(3.5-1.5*np.tanh(0.5*us))/3 - 0.03
                diff=self.Gauss_mean-np.array([Delta2_star,uf])
            else:
                diff=self.Gauss_mean-np.array([Delta2_star,n_star])
            return -0.5*np.dot(np.dot(self.Gauss_icov,diff),diff)


    def get_grid_log_like(self,Delta2_star,n_star,min_prob=0.0001):
        """ Compute log-likelihood from 2D grid"""

        # minimum log-likelihood to avoid numerical noise
        min_log_like=np.log(min_prob)

        # check parameters are within range
        if self.grid_log_like:
            x,y=self.grid_log_like.get_knots()
        else:
             x,y=self.grid_like.get_knots()
        if (np.min(x)>np.min(Delta2_star)) or (np.max(x)<np.max(Delta2_star)):
            return min_log_like
        if (np.min(y)>np.min(n_star)) or (np.max(y)<np.max(n_star)):
            return min_log_like

        if self.grid_log_like:
            log_like=self.grid_log_like.ev(Delta2_star,n_star)
        elif self.grid_like:
            log_like=np.log(self.grid_like.ev(Delta2_star,n_star))
        else:
            raise ValueError('we need either grid_like or grid_log_like')

        return np.fmax(min_log_like,log_like)


    def plot_log_like(self,min_Delta2_star=0.31,max_Delta2_star=0.38,
                min_n_star=-2.33,max_n_star=-2.26,plot_min_prob=0.01):
        """Plot 2D contour with marginalised posterior"""
        import matplotlib.pyplot as plt

        # number of points per dimension
        Nj=200j
        X, Y = np.mgrid[min_Delta2_star:max_Delta2_star:Nj,
                        min_n_star:max_n_star:Nj]
        # evalute log-likelihood in 2D grid
        Z = np.empty_like(X)
        N=int(Nj.imag)
        assert X.shape==(N,N)
        for ix in range(N):
            for iy in range(N):
                Z[ix,iy] = self.get_log_like(Delta2_star=X[ix,iy],
                                                        n_star=Y[ix,iy])

        # specify color map and range
        cmap=plt.cm.gist_earth_r
        extent=[min_Delta2_star,max_Delta2_star,min_n_star,max_n_star]
        plt.imshow(np.rot90(Z),cmap=cmap,extent=extent,
                    vmin=np.log(plot_min_prob),vmax=0.0,label='density')
        plt.xlabel(r'$\Delta_\star$',fontsize=16)
        plt.ylabel(r'$n_\star$',fontsize=16)
        plt.colorbar()
        plt.show()


    def setup_grid(self):
        """ Read 2D grid of (log) likelihood from file, and set interpolator"""

        # open binary file
        data = np.load(self.grid_fname)

        # read 2D grid of parameters and KDE density
        if 'D2_star' in data:
            Delta2_star=data['D2_star']
        else:
            Delta2_star=data['Delta2_star']
        n_star=data['n_star']
        if 'density' in data:
            like=data['density']
            max_like=np.max(like)
            self.grid_like=scipy.interpolate.RectBivariateSpline(
                        Delta2_star,n_star,like/max_like)
            self.grid_log_like=None
        elif 'log_like' in data:
            log_like=data['log_like']
            max_log_like=np.max(log_like)
            self.grid_log_like=scipy.interpolate.RectBivariateSpline(
                        Delta2_star,n_star,log_like-max_log_like)
            self.grid_like=None

        return
