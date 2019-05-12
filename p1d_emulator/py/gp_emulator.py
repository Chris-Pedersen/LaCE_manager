import GPy
import numpy as np
import p1d_arxiv
import pickle
import matplotlib.pyplot as plt

class GPEmulator:
    """ Gaussian process emulator to emulate P1D from a simulation suite.
    Will start off just emulating the power in the lowest k-bin in units of 1/Mpc
    and we can worry about changing this later on.
    """
    def __init__(self,basedir='../mini_sim_suite/',
		p1d_label='p1d',skewers_label='Ns50_wM0.1',
                max_arxiv_size=None,verbose=True,kmax_Mpc=10.0,
                paramList=None):
        self.kmax_Mpc=kmax_Mpc
        # read all files with P1D measured in simulation suite
        self.arxiv=p1d_arxiv.ArxivP1D(basedir,p1d_label,skewers_label,
        	max_arxiv_size,verbose)

        ## Find max k bin
        self.k_bin=np.max(np.argwhere(self.arxiv.data[0]["k_Mpc"]<self.kmax_Mpc))
        ## If none, take all parameters
        if paramList==None:
        	self.paramList=["mF","Delta2_p","alpha_p","sigT_Mpc","f_p","n_p","gamma"]
        else:
        	self.paramList=paramList

    def _fit_p1d_in_arxiv(self,kmax_Mpc):
        """For each entry in arxiv, fit polynomial to log(p1d)"""    
        for entry in self.arxiv.data:
            k_Mpc = entry['k_Mpc']
            p1d_Mpc = entry['p1d_Mpc']

    def _build_interp(self,arxiv,paramList):
        ''' Method to build an GP object from a spectra archive and list of parameters
        Currently the parameter rescaling is done by taking the min and max
        of the provided params, not by defining our own prior volume. Need to decide
        whether or not this is what we want. '''

        ## Grid that will contain all training params
        params=np.empty([len(self.arxiv.data),len(paramList)])

        ## Array to contain our training data
        P1D_k=np.empty([len(self.arxiv.data),self.k_bin-1]) ## -1 as we ignore the 0th k bin
        #print(arxiv.data[1]["k_Mpc"])
        for aa in range(len(self.arxiv.data)):
            P1D_k[aa]=self.arxiv.data[aa]['p1d_Mpc'][1:self.k_bin] ## Collect P1D data for all k bins
            for bb in range(len(paramList)):
                params[aa][bb]=arxiv.data[aa][paramList[bb]] ## Populate parameter grid

        ## Rescaling to unit volume
        self.maxParams=np.empty(len(paramList))
        self.minParams=np.empty(len(paramList))
        for aa in range(len(paramList)):
            self.maxParams[aa]=max(params[:,aa])
            self.minParams[aa]=min(params[:,aa])
        for cc in range(len(self.arxiv.data)):
            for dd in range(len(paramList)):
                params[cc][dd]=((params[cc][dd]-self.minParams[dd])/(self.maxParams[dd]-self.minParams[dd]))
        print("Rescaled params to unity volume")

        ## Factors by which to rescale the flux to set a mean of 0
        self.scalefactors = np.median(P1D_k, axis=0)

        #Normalise by the median value
        normspectra = (P1D_k/self.scalefactors) -1.

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(len(paramList))
        kernel += GPy.kern.RBF(len(paramList))

        print("Training GP")
        self.gp = GPy.models.GPRegression(params,normspectra,kernel=kernel, noise_var=1e-10)
        status = self.gp.optimize(messages=False) #True
        print("Optimised")

    def predict(self,model):
        assert len(model)==len(self.paramList), "Emulator has %d parameters, you have asked for a model with %d" % (len(model),len(self.paramList))
        param=[]
        ## Method to return P1D for a given parameter set
        for par in self.paramList:
            ## Rescale input parameters
            param.append(model[par])
        for aa in range(len(self.paramList)):
            param[aa]=(param[aa]-self.minParams[aa])/(self.maxParams[aa]-self.minParams[aa])
        pred,err=self.gp.predict(np.array(param).reshape(1,-1))
        return (pred+1)*self.scalefactors,np.sqrt(err)*self.scalefactors

    def saveEmulator(self,saveName):
        pickle.dump(self.gp,open(saveName+".p", "wb" ))
        print("GP emulator object saved as:" + saveName + ".p")

    def train(self,spects):
        self._build_interp(spects,self.paramList)

    def crossValidation(self,kbin,testSample=0.25):
        paramList=self.paramList
        arxiv=self.arxiv
        ''' Code to cross validate. testSample determines the proportion
        of the data to save for testing, default=0.25.
        Currently only uses a single k bin.
        '''

        ## Grid that will contain all training params
        trainingLen=int(len(self.arxiv.data)*(1-testSample))
        testLen=len(self.arxiv.data)- trainingLen

        params=np.empty([len(self.arxiv.data),len(paramList)])

        ## Array to contain our training data
        P1D_k=np.empty(len(self.arxiv.data))
        print(arxiv.data[1]["k_Mpc"][kbin])

        for aa in range(len(self.arxiv.data)):
            P1D_k[aa]=self.arxiv.data[aa]['p1d_Mpc'][kbin] ## Collect P1D data for a given k bin
            for bb in range(len(paramList)):
                params[aa][bb]=arxiv.data[aa][paramList[bb]]#print(entry['k_Mpc'][1])

        ## Rescaling to unit volume
        maxParams=np.empty(len(paramList))
        minParams=np.empty(len(paramList))
        maxTrain=np.empty(len(paramList))
        minTrain=np.empty(len(paramList))
        for aa in range(len(paramList)):
            ## Max params for total
            maxParams[aa]=max(params[:,aa])
            minParams[aa]=min(params[:,aa])
            ## Max & min params for training
            maxTrain[aa]=max(params[:trainingLen,aa])
            minTrain[aa]=min(params[:trainingLen,aa])
        for cc in range(len(self.arxiv.data)):
            for dd in range(len(paramList)):
                params[cc][dd]=((params[cc][dd]-minParams[dd])/(maxParams[dd]-minParams[dd]))
        print("Rescaled params to unity volume")

        ## Rescale flux to get mean 0
        P1D_k=P1D_k/np.mean(P1D_k)-1 ##
        print("Mean Pk=",np.mean(P1D_k))

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(len(paramList))
        kernel = GPy.kern.RBF(len(paramList))

        ## Split into test and training
        params_train=params[:trainingLen]
        P1D_k_train=P1D_k[:trainingLen]

        params_test=params[-testLen:]
        P1D_k_test=P1D_k[-testLen:]

        ## Find new min/max params to determine whether a test value is outside the convex hull
        for aa in range(len(paramList)):
            ## Max params for total
            maxParams[aa]=max(params[:,aa])
            minParams[aa]=min(params[:,aa])
            ## Max & min params for training
            maxTrain[aa]=max(params[:trainingLen,aa])
            minTrain[aa]=min(params[:trainingLen,aa])

        ## Identify the points where the test data lies outside the training
        outside=0
        for aa in range(len(params_test)):
            #print("Max:",maxTrain)
            #print("Test:",params_test[aa])
            ## Check if this point lies outside the convex hull of the training set
            moreThan=params_test[aa]>maxTrain
            lessThan=params_test[aa]<minTrain
            if np.sum(np.logical_or(moreThan,lessThan))>0:
                print("#### outside value ###\n",params_test[aa])
                outside+=1

        print("There are %d test values outside the training convex hull" % outside)
        
        print("Training GP")
        self.gp = GPy.models.GPRegression(params_train,P1D_k_train.reshape(-1,1),kernel=kernel, noise_var=1e-10)
        status = self.gp.optimize(messages=False) #True
        print("Optimised")

        ## Now predict from the remaining test values
        mean,std=self.gp.predict(params_test)
        mean=np.hstack(mean)

        std=np.sqrt(np.hstack(std)) ## Take std from var
        error=(mean-P1D_k_test)/std

        ## Generate mock Gaussian to overlay
        x=np.linspace(-6,6,500)
        y=(np.exp(-0.5*(x*x)))/np.sqrt(2*np.pi)
        plt.figure()
        plt.title("%d training, %d test samples, k=%.3f" % (trainingLen,testLen,arxiv.data[1]["k_Mpc"][kbin]))
        plt.hist(error,bins=250,density=True)
        plt.plot(x,y)
        plt.xlim(-6,6)
        plt.xlabel("(predicted-truth)/std")
        
        plt.figure()
        plt.title("Fractional error distribution")
        plt.hist(std/mean,bins=100,density=True)
        plt.xlim(-0.1,0.1)
        plt.show()

'''
class PolyfitGPEmulator:
    """ Gaussian process emulator to learn the 4th degree polynomial coefficients
    to the P1D as a function of parameter """

basedir='../../p1d_emulator/sim_suites/emulator_04052019/'
p1d_label='mf_p1d'
skewers_label='Ns100_wM0.1'



## Full list is ["mF","Delta2_p","alpha_p","sigT_Mpc","f_p","n_p","gamma"]
#Params=["mF","Delta2_p","sigT_Mpc","f_p","gamma"]
GP_EMU=GPEmulator(basedir,p1d_label,skewers_label,max_arxiv_size=1000,verbose=True,paramList=None,kmax_Mpc=5)

GP_EMU.train(GP_EMU.arxiv)


# identify mean model
median_mF=np.median(GP_EMU.arxiv.mF)
median_sigT_Mpc=np.median(GP_EMU.arxiv.sigT_Mpc)
median_gamma=np.median(GP_EMU.arxiv.gamma)
median_Delta2_p=np.median(GP_EMU.arxiv.Delta2_p)
median_n_p=np.median(GP_EMU.arxiv.n_p)
median_alpha_p=np.median(GP_EMU.arxiv.alpha_p)
median_f_p=np.median(GP_EMU.arxiv.f_p)

median_model={'mF':median_mF,'sigT_Mpc':median_sigT_Mpc,'gamma':median_gamma,
            'Delta2_p':median_Delta2_p,'n_p':median_n_p,'alpha_p':median_alpha_p,'f_p':median_f_p}

print('mean model =',median_model)

pred,err=GP_EMU.predict(median_model)

print(pred)


#GP_EMU.crossValidation(18)
#GP_EMU.predict()
#GP_EMU.saveEmulator("test5000")
#archive=GP_EMU.arxiv
#print(archive[1])
'''