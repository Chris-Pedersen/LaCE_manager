# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Playing with kernels in GPs

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import GPy


# ### Setting up the problem, and training data
#
# Simple 1D example, where the truth is a sinusoidal function, and we'll have a few noisy training points. 

# define truth underlying function we are trying to model
def true_function(x,x0=0):
    #return np.exp(x-x0)
    return x-x0+0.1*np.sin(10*(x-x0))


# +
# define test data (_s stands for \star)
x_max=3
x_min=-x_max

n_s = 81
x_s = np.linspace(x_min,x_max,n_s).reshape(-1,1)
# -

# training data (_t stands for training)
x_t =np.append(np.linspace(-1,0,int(n_s/7.0)),np.linspace(1,x_max,int(n_s/3.5)))
n_t = len(x_t)
x_t = np.array(x_t).reshape(n_t,1)
# noise variance
noise_var=1.e-4
noise_rms=np.sqrt(noise_var)
y_t = true_function(x_t) + noise_rms*np.random.normal(size=[n_t,1])

plt.errorbar(x_t,y_t,yerr=noise_rms*np.ones(n_t),label='(noisy) training points')
plt.plot(x_s,true_function(x_s),':',color='red',label='truth')
plt.legend()
plt.show()

# ### Squared-Exponential kernel (or RBF)

# decide whether to optimise noise variance or use true value
fix_noise_var=False

# setup RBF kernel
k_RBF = GPy.kern.RBF(input_dim=1)
k_RBF2 = GPy.kern.RBF(input_dim=1)

k_RBF.lengthscale=0.2
k_RBF.variance=2.0827777483791223
#m_RBF.Gaussian_noise.variance=2.053290096700064e-11
#m_RBF.Gaussian_noise.variance.unfix()
#k_RBF.lengthscale.unfix()

# +
#k_RBF2.lengthscale=2.70365255520041
#k_RBF2.variance=6.102532893477933
#m_RBF2.Gaussian_noise.variance=1.232232723201087e-05
# -

# setup main GP model
m_RBF = GPy.models.GPRegression(x_t,y_t,k_RBF,noise_var=noise_var)
print('log marginal likelihood',m_RBF.log_likelihood())
if fix_noise_var:
    m_RBF.Gaussian_noise.variance.fix()
# setup main GP model
m_RBF2 = GPy.models.GPRegression(x_t,y_t,k_RBF2,noise_var=noise_var)
print('log marginal likelihood',m_RBF2.log_likelihood())
if fix_noise_var:
    m_RBF2.Gaussian_noise.variance.fix()    

m_RBF.optimize(messages=True)

m_RBF2.optimize(messages=True)

fig = m_RBF.plot()
plt.plot(x_s,true_function(x_s),':',color='red')
#plt.xlim([-.8,0.8])
#xplot=[0.99,1.0]
#plt.xlim(xplot)
#plt.xlim([x_min,x_max])
#plt.ylim([0.98,1.02])
#plt.ylim((xplot))
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.show()

fig2 = m_RBF2.plot()
plt.plot(x_s,true_function(x_s),':',color='red')
#plt.xlim([-.8,0.8])
#xplot=[0.99,1.0]
#plt.xlim(xplot)
#plt.xlim([x_min,x_max])
#plt.ylim([0.98,1.02])
#plt.ylim((xplot))
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.show()

# This worked quite well. Depending on the random seed, some of the hyper-parameters are not properly fitted (the noise variance, for instance, should not move from its true value), but this is expected. If we have fewer points, it is even harder to fit them. Let's try now with a linear kernel!

# ### Dot kernel (or linear)
#
# $K(x_1,x_2) = \sigma_L^2 x_1 x_2$
#
# This kernel is useful in linear model, i.e., it could be useful when doing first-order Taylor expansions around a fiducial model.

k_lin = GPy.kern.Linear(input_dim=1,variances=1.0,ARD=False)

# setup main GP object
m_lin = GPy.models.GPRegression(x_t,y_t,k_lin,noise_var=noise_var)
print('log marginal likelihood',m_lin.log_likelihood())
if fix_noise_var:
    m_lin.Gaussian_noise.variance.fix()

m_lin.optimize(messages=True)

fig = m_lin.plot()
plt.plot(x_s,true_function(x_s),':',color='red',)
#plt.xlim([0,1])
#plt.ylim([0,1])
plt.show()

# This is not too bad, probably because the truth is not too far from linear (sin x ~ x) on the scales tested. However, if you re-run the notebook with an offset (x0=2), the GP with linear kernel fails completely. This is a situation similar to the one in our P1D problem, since our parameters are defined in the range [0,1] and the fiducial model that we should be Taylor-expanding around is somewhere near x=0.5.

# add both kernels
k_com = k_RBF + k_lin
# setup main GP object
m_com = GPy.models.GPRegression(x_t,y_t,k_com,noise_var=noise_var)
if fix_noise_var:
    m_com.Gaussian_noise.variance.fix()

print('log marginal likelihood',m_com.log_likelihood())

m_com.optimize(messages=True)

fig = m_com.plot()
plt.plot(x_s,true_function(x_s),':',color='red')
#plt.xlim([x_min,x_max])
#plt.xlim([0,1])
#plt.ylim([0,1])
plt.show()


