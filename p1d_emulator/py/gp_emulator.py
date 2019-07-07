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
                passArxiv=None):

        self.kmax_Mpc=kmax_Mpc
        self.basedir=basedir
        self.emu_type=emu_type
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

        if train:
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
        self.gp = GPy.models.GPRegression(params,normspectra,kernel=kernel, noise_var=1e-10)
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

    def printPriorVolume(self):
        for aa in range(len(self.paramList)):
            print(self.paramList[aa],self.paramLimits[aa])

    def crossValidation(self,testSample=0.25,plotIndividual=False):
        paramList=self.paramList
        arxiv=self.arxiv
        ''' Code to cross validate. testSample determines the proportion
        of the data to save for testing, default=0.25.
        In progress of extending this to all k bins but its not working properly at the moment
        '''

        trainingLen=int(len(self.arxiv.data)*(1-testSample))
        testLen=len(self.arxiv.data)- trainingLen

        params,P1D_k=self._buildTrainingSets(arxiv,paramList)

        ## Get parameter limits for rescaling
        self.paramLimits=self._get_param_limits(params)

        ## Rescaling to unit volume
        for cc in range(len(self.arxiv.data)):
            params[cc]=self._rescale_params(params[cc],self.paramLimits)
        print("Rescaled params to unity volume")

        ## Factors by which to rescale the flux to set a mean of 0
        self.scalefactors = np.median(P1D_k, axis=0)

        #Normalise by the median value
        P1D_k = (P1D_k/self.scalefactors) -1.

        ## Split into test and training
        params_train=params[:trainingLen]
        P1D_k_train=P1D_k[:trainingLen]
        params_test=params[-testLen:]
        P1D_k_test=P1D_k[-testLen:]

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(len(paramList))
        kernel += GPy.kern.RBF(len(paramList))

        print("Training GP")
        self.gp = GPy.models.GPRegression(params_train,P1D_k_train,kernel=kernel, noise_var=1e-10)
        status = self.gp.optimize(messages=False) #True
        print("Optimised")

        ## Now predict from the remaining test values
        ## Mean comes out in shape (# of models, # of k bins)
        ## Std comes out with # of models
        print("test params shape",np.shape(params_test))
        mean,std=self.gp.predict(params_test)

        std=np.sqrt(std)
        error=(mean-P1D_k_test)/std
        error=np.hstack(error)

        if plotIndividual==True:
            ## Take a random model and plot the true P(k)
            ## along with the predicted P(k)
            sample=np.random.randint(trainingLen,trainingLen+testLen)
            predictedP,err=self.gp.predict(params[sample].reshape(1,-1))
            #err=np.full(len(predictedP),np.sqrt(err)*self.scalefactors)
            predictedP=np.hstack(predictedP)
            predictedP=(predictedP+1)*self.scalefactors
            truek=self.arxiv.data[sample]["k_Mpc"][:self.k_bin]
            trueP=self.arxiv.data[sample]["p1d_Mpc"][:self.k_bin]
            plt.figure()
            plt.title("Truth and predicted for a random test model #%d" % sample)
            plt.plot(np.log10(truek),truek*trueP,label="True")
            plt.plot(np.log10(truek[:len(predictedP)]),predictedP*truek[:len(predictedP)],label="Predicted",linestyle="dashed")
            #plt.errorbar(np.log10(truek[:len(predictedP)]),predictedP*truek[:len(predictedP)],yerr=np.sqrt(err),label="Predicted",linestyle="dashed")
            plt.xlabel("log k")
            plt.legend()

        ## Generate mock Gaussian to overlay
        x=np.linspace(-6,6,200)
        y=(np.exp(-0.5*(x*x)))/np.sqrt(2*np.pi)
        plt.figure()
        plt.title("%d training, %d test samples" % (trainingLen,testLen))
        plt.hist(error,bins=500,density=True)
        plt.plot(x,y)
        plt.xlim(-6,6)
        plt.xlabel("(predicted-truth)/std")
        plt.show()

    def predict(self,model):
        ## Method to return P1D for a given parameter set
#assert len(model)==len(self.paramList), "Emulator has %d parameters, you have asked for a model with %d" % (len(self.paramList),len(model))
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
        if max(k_Mpc)>max(self.training_k_bins):
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
            error_interp=interp1d(self.training_k_bins,err, "cubic")
            error=error_interp(k_Mpc)
            # for now, assume that we have fully correlated errors
            covar = np.outer(error, error)
            #covar = np.diag(error**2)
            return interpolated_P, covar
        else:
            return interpolated_P
