import GPy
import numpy as np
import p1d_arxiv
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import poly_p1d
import os
import json

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
                drop_temp_rescalings=False,keep_every_other_rescaling=False,
                undersample_z=1,emu_type="k_bin",z_max=5,
                passArxiv=None,set_noise_var=1e-3,asymmetric_kernel=False):

        self.kmax_Mpc=kmax_Mpc
        self.basedir=basedir
        self.emu_type=emu_type
        self.emu_noise=set_noise_var
        self.max_arxiv_size=max_arxiv_size
        self.drop_tau_rescalings=drop_tau_rescalings
        self.drop_temp_rescalings=drop_temp_rescalings
        self.keep_every_other_rescaling=keep_every_other_rescaling
        self.undersample_z=undersample_z
        self.verbose=verbose
        self.asymmetric_kernel=asymmetric_kernel
        self.z_max=z_max

        # read all files with P1D measured in simulation suite
        if passArxiv==None:
            self.custom_arxiv=False
            self.arxiv=p1d_arxiv.ArxivP1D(basedir,p1d_label,skewers_label,
                        max_arxiv_size=self.max_arxiv_size,verbose=verbose,
                        drop_tau_rescalings=drop_tau_rescalings,
                        drop_temp_rescalings=drop_temp_rescalings,z_max=self.z_max,
                        keep_every_other_rescaling=keep_every_other_rescaling,
                        undersample_z=undersample_z)
        else:
            self.custom_arxiv=True
            print("Loading emulator using a specific arxiv, not the one set in basedir")
            self.arxiv=passArxiv

        ## Find max k bin
        self.k_bin=int(np.max(np.argwhere(self.arxiv.data[0]["k_Mpc"]<self.kmax_Mpc)))
        self.training_k_bins=self.arxiv.data[0]["k_Mpc"][:self.k_bin]
        ## If none, take all parameters
        if paramList==None:
        	self.paramList=["mF","Delta2_p","alpha_p","sigT_Mpc","f_p","n_p","gamma","kF_Mpc"]
        else:
        	self.paramList=paramList

        self._build_interp(self.arxiv,self.paramList)
        self.trained=False

        if train==True:
            ## First try load the emulator
            self.loadEmulator()
            if self.trained==False:
                if self.verbose: print('will train GP emulator')
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
        kernel = GPy.kern.Linear(len(paramList),ARD=self.asymmetric_kernel)
        kernel += GPy.kern.RBF(len(paramList),ARD=self.asymmetric_kernel)

        self.gp = GPy.models.GPRegression(params,normspectra,kernel=kernel,
                        noise_var=self.emu_noise,initialize=False)


    def _get_param_limits(self,paramGrid):
        paramLimits=np.empty((np.shape(paramGrid)[1],2))
        for aa in range(len(paramLimits)):
            paramLimits[aa,0]=min(paramGrid[:,aa])
            paramLimits[aa,1]=max(paramGrid[:,aa])
        return paramLimits


    def train(self):
        self.gp.initialize_parameter()
        print("Training GP on %d points" % len(self.arxiv.data))
        status = self.gp.optimize(messages=False)
        print("Optimised")
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
        try:
            if max(k_Mpc)>max(self.training_k_bins):
                print(max(k_Mpc))
                print(max(self.training_k_bins))
                print("Warning! Your requested k bins are higher than the training values.")
        except:
            if k_Mpc>max(self.training_k_bins):
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

        ## We are now training the emulator on a customised arxiv
        self.custom_arxiv=True

        ## Split the arxiv into test and training samples
        test=[] ## List for test data
        numTest=int(len(self.arxiv.data)*testSample)
        numTrain=len(self.arxiv.data)-numTest
        for aa in range(numTest):
            test.append(self.arxiv.data.pop(np.random.randint(len(self.arxiv.data))))

        ## Rebuild parameter grids using the new reduced arxiv
        self._build_interp(self.arxiv,self.paramList)
        self.train()

        accuracy=np.array([])
        if self.emu_type=="k_bin":
            for aa in range(len(test)):
                ## Set up model for emu calls
                model={}
                for parameter in self.paramList:
                    model[parameter]=test[aa][parameter]
                ## Make emu calls for each test point
                pred,err=self.predict(model)
                ## Find inaccuracy/error
                accuracy=np.append(accuracy,(pred-test[aa]["p1d_Mpc"][:self.k_bin])/err)
        elif self.emu_type=="polyfit":
            for aa in range(len(test)):
                ## Set up model for emu calls
                model={}
                for parameter in self.paramList:
                    model[parameter]=test[aa][parameter]
                ## Make emu calls for each test point
                pred,err=self.predict(model)
                poly=np.poly1d(pred)
                err=np.abs(err)
                interpolated_P=np.exp(poly(np.log(self.training_k_bins)))
                err=(err[0]*interpolated_P**4+err[1]*interpolated_P**3+err[2]*interpolated_P**2+err[3]*interpolated_P)
                ## Find inaccuracy/error
                accuracy=np.append(accuracy,(interpolated_P-test[aa]["p1d_Mpc"][:self.k_bin])/err)

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


    def saveEmulator(self):
        ''' Method to save a trained emulator. The emulator
        hyperparameters are saved alongside a .json dictionary
        detailing the initial configuration of the emulator.
        When an emulator is saved it will check the basedir
        for existing emulator saves.
        
        We currently do not save emulators on custom arxivs
        as it is impossible to know if the training points
        are all the same. Training points are currently
        reassembled using the snapshot arxiv, and not
        saved alongside the emulator hyperparameters. '''

        ## Perform checks
        if self.custom_arxiv or self.max_arxiv_size:
            print("Cannot save emulators trained on custom arxivs")
            return
        if not self.trained:
            print("Cannot save an emulator that is not trained")
            return

        ## Create and save dictionary of emulator parameters
        ## Can't think of a way to do this iteratively so we write it out
        initParams={}
        initParams["k_bin"]=self.k_bin
        initParams["emu_type"]=self.emu_type
        initParams["emu_noise"]=self.emu_noise
        initParams["drop_tau_rescalings"]=self.drop_tau_rescalings
        initParams["drop_temp_rescalings"]=self.drop_temp_rescalings
        initParams["keep_every_other_rescaling"]=self.keep_every_other_rescaling
        initParams["undersample_z"]=self.undersample_z
        initParams["paramList"]=self.paramList
        initParams["asymmetric_kernel"]=self.asymmetric_kernel
        initParams["z_max"]=self.z_max

        saveString=self.basedir+"/saved_emulator_"

        ## Here we check to see if an emulator matching
        ## our initial parameters is already saved
        aa=1
        while os.path.isfile(saveString+str(aa)+".json"):
            ## Load dictionary and check if the values are different
            ## to the emulator we want to save
            with open(saveString+str(aa)+".json") as json_file:  
                fileInitDict=json.load(json_file)
            if fileInitDict==initParams: ## If so we can break the loop
                if self.verbose:
                    print("This emulator is already saved.")
                return
            else: ## If not keep looking through saved emulators
                aa+=1

        saveString=saveString+str(aa)
        with open("%s.json" % saveString, 'w') as fp:
            json.dump(initParams, fp)

        ## Create and save dictionary of hyperparameters
        saveDict={}
        saveDict["paramList"]=self.paramList
        saveDict["kmax_Mpc"]=self.kmax_Mpc
        np.save('%s.npy' % saveString, self.gp.param_array)
        if self.verbose:
            print("Model saved as %s.npy" % saveString)


    def loadEmulator(self):
        ''' Method to load a saved set of emulator
        hyperparameters. We need to make sure the
        dataset and emulator configurations that the
        hyperparameters were optimised on is the same
        as the dataset we will use as training data.
        So we will check the .json files for a perfect match '''

        ## Perform same checks as when saving an emulator
        ## as save/load does not work with non-standard
        ## data arxivs

        if self.custom_arxiv or self.max_arxiv_size:
            print("Cannot load emulators with non-standard training data")
            return
        if self.trained:
            print("Cannot load an emulator after training")
            return

        initParams={}
        initParams["k_bin"]=self.k_bin
        initParams["emu_type"]=self.emu_type
        initParams["emu_noise"]=self.emu_noise
        initParams["drop_tau_rescalings"]=self.drop_tau_rescalings
        initParams["drop_temp_rescalings"]=self.drop_temp_rescalings
        initParams["keep_every_other_rescaling"]=self.keep_every_other_rescaling
        initParams["undersample_z"]=self.undersample_z
        initParams["paramList"]=self.paramList
        initParams["asymmetric_kernel"]=self.asymmetric_kernel
        initParams["z_max"]=self.z_max

        saveString=self.basedir+"/saved_emulator_"

        ## Here we check to see if an emulator matching
        ## our initial parameters is already saved
        aa=1
        while os.path.isfile(saveString+str(aa)+".json"):
            ## Load dictionary and check if the values are different
            ## to the emulator we want to load
            with open(saveString+str(aa)+".json") as json_file:  
                fileInitDict=json.load(json_file)
            if fileInitDict==initParams: ## If so, load it
                saveString=saveString+str(aa)
                self.gp.update_model(False)
                self.gp.initialize_parameter()
                self.gp[:]=np.load(saveString+".npy")
                self.gp.update_model(True)
                self.trained=True
                if self.verbose:
                    print("Loading emulator from %s.npy" % saveString)
                return
                
            else: ## If not keep looking through saved emulators
                aa+=1
        if self.verbose:
            print("Could not find a matching emulator to load")

