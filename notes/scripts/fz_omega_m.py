import numpy as np
import matplotlib.pyplot as plt

def get_Om(Om0,z):
    Ez2 = Om0*(1+z)**3+(1-Om0)
    return Om0 * (1+z)**3 / Ez2

def get_f(Om0,z):
    Omz = get_Om(Om0,z)
    return Omz**0.6

z = np.arange(2.0,5.0,0.1)

fz_02 = get_f(0.2,z)
fz_03 = get_f(0.3,z)
fz_04 = get_f(0.4,z)

zx = 3.0
fx_02 = get_f(0.2,zx)
fx_03 = get_f(0.3,zx)
fx_04 = get_f(0.4,zx)

plt.plot(z,fz_04/fz_03,color='red',ls='-',label=r'$\Omega_m=0.4$')
plt.plot(z,fz_03/fz_03,color='black',ls=':',label=r'$\Omega_m=0.3$')
plt.plot(z,fz_02/fz_03,color='blue',ls='-',label=r'$\Omega_m=0.2$')
plt.plot(z,np.ones_like(z)*fx_02/fx_03,color='blue',ls=':')
plt.plot(z,np.ones_like(z)*fx_04/fx_03,color='red',ls=':')
plt.title(r'Growth rate for different $\Omega_m$')
plt.xlabel('z')
plt.ylabel(r'$f(z)/f_0(z)$')
plt.legend(loc=4)
plt.savefig('fz_omega_m.pdf')
plt.show()

