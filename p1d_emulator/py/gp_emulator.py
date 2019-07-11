import GPy
import numpy as np
import p1d_arxiv
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import poly_p1d

class GPEmulator:
    """
    Gaussian process emulator to emulate P1D from a simulation suite.
    This will train on the data in an 'arxiv' object, and will return
    a given P_1D(k) for the same k-bins used in training.
    GPEmulator.predict takes models in a dictionary format currently.
    """
    def __init__(self,basedir=None,p1d_label=None,skewers_label=None,
                max_arxiv_size=None,verbose=False,kmax_Mpc=10.0,
                paramList=None,train=False,drop_tau_rescalings=False,
                drop_temp_rescalings=False,undersample_z=1,emu_type="k_bin",
                passArxiv=None,set_noise_var=1e-10):

        self.kmax_Mpc=kmax_Mpc
        self.basedir=basedir
        self.emu_type=emu_type
        self.emu_noise=set_noise_var
        # read all files with P1D measured in simulation suite
        if passArxiv==None:
            self.arxiv=p1d_arxiv.ArxivP1D(basedir,p1d_label,skewers_label,
                                max_arxiv_size=max_arxiv_size,verbose=verbose,
                                drop_tau_rescalings=drop_tau_rescalings,
                                drop_temp_rescalings=drop_temp_rescalings,
                                undersample_z=undersample_z)
        else:
            print("Loading emulator using a specific arxiv, not the one set in basedir")
            self.arxiv=passArxiv

        ## Find max k bin
        self.k_bin=np.max(np.argwhere(self.arxiv.data[0]["k_Mpc"]<self.kmax_Mpc))
        self.training_k_bins=self.arxiv.data[0]["k_Mpc"][:self.k_bin]
        ## If none, take all parameters
        if paramList==None:
        	self.paramList=["mF","Delta2_p","alpha_p","sigT_Mpc","f_p","n_p","gamma","kF_Mpc"]
        else:
        	self.paramList=paramList

        self.trained=False

        if train==True:
            if verbose: print('will train GP emulator')
            self.train()

    def _training_points_k_bin(self,arxiv):
        ''' Method to get the Y training points in the form of the P1D
        at different k values '''
        P1D_k=np.empty([len(self.arxiv.data),self.k_bin])
        for aa in range(len(self.arxiv.data)):
            P1D_k[aa]=self.arxiv.data[aa]['p1d_Mpc'][:self.k_bin] ## Collect P1D data for all k bins
        return P1D_k

    def _training_points_polyfit(self,arxiv):
        ''' Method to get the Y training points in the form of polyfit 
        coefficients '''
        self._fit_p1d_in_arxiv(4,self.kmax_Mpc)
        coeffs=np.empty([len(self.arxiv.data),5]) ## Hardcoded to use 4th degree polynomial
        for aa in range(len(self.arxiv.data)):
            coeffs[aa]=self.arxiv.data[aa]['fit_p1d'] ## Collect P1D data for all k bins
        return coeffs

    def _rescale_params(self,params,paramLimits):
        ''' Rescale a set of parameters to have a unit volume '''
        for aa in range(len(params)):
            params[aa]=((params[aa]-paramLimits[aa,0])/(paramLimits[aa,1]-paramLimits[aa,0]))
        return params

    def _buildTrainingSets(self,arxiv,paramList):
        ## Grid that will contain all training params
        params=np.empty([len(self.arxiv.data),len(paramList)])

        if self.emu_type=="k_bin":
            trainingPoints=self._training_points_k_bin(arxiv)
        elif self.emu_type=="polyfit":
            trainingPoints=self._training_points_polyfit(arxiv)
        else:
            print("Unknown emulator type, terminating")
            quit()

        for aa in range(len(self.arxiv.data)):
            for bb in range(len(paramList)):
                params[aa][bb]=arxiv.data[aa][paramList[bb]] ## Populate parameter grid
        return params,trainingPoints

    def _fit_p1d_in_arxiv(self,deg,kmax_Mpc):
        """For each entry in arxiv, fit polynomial to log(p1d)"""
        
        for entry in self.arxiv.data:
            k_Mpc = entry['k_Mpc']
            p1d_Mpc = entry['p1d_Mpc']
            fit_p1d = poly_p1d.PolyP1D(k_Mpc,p1d_Mpc,kmin_Mpc=1.e-3,
                    kmax_Mpc=kmax_Mpc,deg=deg)
            entry['fit_p1d'] = fit_p1d.lnP_fit ## Add coeffs for each model to arxiv

    def _build_interp(self,arxiv,paramList):
        ''' Method to build an GP object from a spectra archive and list of parameters
        Currently the parameter rescaling is done by taking the min and max
        of the provided params, not by defining our own prior volume. Need to decide
        whether or not this is what we want. '''

        params,Ypoints=self._buildTrainingSets(arxiv,paramList)

        ## Get parameter limits for rescaling
        self.paramLimits=self._get_param_limits(params)

        ## Rescaling to unit volume
        for cc in range(len(self.arxiv.data)):
            params[cc]=self._rescale_params(params[cc],self.paramLimits)
        print("Rescaled params to unity volume")

        ## Factors by which to rescale the flux to set a mean of 0
        self.scalefactors = np.median(Ypoints, axis=0)

        #Normalise by the median value
        normspectra = (Ypoints/self.scalefactors) -1.

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(len(paramList))
        kernel += GPy.kern.RBF(len(paramList))

        print("Training GP on %d points" % len(self.arxiv.data))
        self.gp = GPy.models.GPRegression(params,normspectra,kernel=kernel, noise_var=self.emu_noise)
        status = self.gp.optimize(messages=False) #True
        print("Optimised")

    def _get_param_limits(self,paramGrid):
        paramLimits=np.empty((np.shape(paramGrid)[1],2))
        for aa in range(len(paramLimits)):
            paramLimits[aa,0]=min(paramGrid[:,aa])
            paramLimits[aa,1]=max(paramGrid[:,aa])
        return paramLimits

    def saveEmulator(self,saveName):
        pickle.dump(self.gp,open(saveName+".p", "wb" ))
        print("GP emulator object saved as:" + saveName + ".p")

    def train(self):
        self._build_interp(self.arxiv,self.paramList)
        self.trained=True

    def printPriorVolume(self):
        for aa in range(len(self.paramList)):
            print(self.paramList[aa],self.paramLimits[aa])

    def predict(self,model):
        ## Method to return P1D or polyfit coeffs for a given parameter set
        ## For the k bin emulator this will be in the training k bins
        #assert len(model)==len(self.paramList), "Emulator has %d parameters, you have asked for a model with %d" % (len(self.paramList),len(model))
        if self.trained==False:
            print("Emulator not trained, cannot make a prediction")
            return
        param=[]
        for par in self.paramList:
            ## Rescale input parameters
            param.append(model[par])
        for aa in range(len(self.paramList)):
            param[aa]=(param[aa]-self.paramLimits[aa,0])/(self.paramLimits[aa,1]-self.paramLimits[aa,0])
        pred,err=self.gp.predict(np.array(param).reshape(1,-1))
        return np.ndarray.flatten((pred+1)*self.scalefactors),np.ndarray.flatten(np.sqrt(err)*self.scalefactors)

    def emulate_p1d_Mpc(self,model,k_Mpc,return_covar=False):
        '''
        Method to return the trained P(k) for an arbitrary set of k bins
        by interpolating the trained data
        '''
        if max(k_Mpc)>max(self.training_k_bins) and verbose:
            print(max(k_Mpc))
            print(max(self.training_k_bins))
            print("Warning! Your requested k bins are higher than the training values.")
        pred,err=self.predict(model)
        if self.emu_type=="k_bin":
            interpolator=interp1d(self.training_k_bins,pred, "cubic")
            interpolated_P=interpolator(k_Mpc)
        elif self.emu_type=="polyfit":
            poly=np.poly1d(pred)
            err=np.abs(err)
            interpolated_P=np.exp(poly(np.log(k_Mpc)))
            err=(err[0]*interpolated_P**4+err[1]*interpolated_P**3+err[2]*interpolated_P**2+err[3]*interpolated_P)
            covar = np.outer(err, err)
        if return_covar==True:
            if self.emu_type=="k_bin":
                error_interp=interp1d(self.training_k_bins,err, "cubic")
                error=error_interp(k_Mpc)
                # for now, assume that we have fully correlated errors
                covar = np.outer(error, error)
                #covar = np.diag(error**2)
                return interpolated_P, covar
            else:
                return interpolated_P, covar
        else:
            return interpolated_P

    def crossValidation(self,testSample=0.25):
        '''
        Method to run a cross validation test on the given
        data arxiv.
        '''
        print("Running cross-validation")
        if self.trained:
            print("Cannot run cross validation on an already-trained emulator.")
            quit()

        ## Split the arxiv into test and training samples
        test=[]
        numTest=int(len(self.arxiv.data)*testSample)
        numTrain=len(self.arxiv.data)-testSample
        for aa in range(numTest):
            test.append(self.arxiv.data.pop(np.random.randint(len(self.arxiv.data))))

        self.train()

        accuracy=np.array([])
        for aa in range(len(test)):
            ## Set up model for emu calls
            model={}
            for parameter in self.paramList:
                model[parameter]=test[aa][parameter]
            ## Make emu calls for each test point
            pred,err=self.predict(model)
            ## Find inaccuracy/error
            accuracy=np.append(accuracy,(pred-test[aa]["p1d_Mpc"][:self.k_bin])/err)

        ## Generate mock Gaussian to overlay
        x=np.linspace(-6,6,200)
        y=(np.exp(-0.5*(x*x)))/np.sqrt(2*np.pi)

        ## Plot results
        plt.figure()
        plt.title("%d training, %d test samples, noise=%.1e" % (numTrain,numTest,self.emu_noise))
        plt.hist(accuracy,bins=500,density=True)
        plt.plot(x,y)
        plt.xlim(-6,6)
        plt.xlabel("(predicted-truth)/std")
        plt.show()