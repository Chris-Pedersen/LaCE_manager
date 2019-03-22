import numpy as np
import matplotlib.pyplot as plt

def get_Hz(h,Om0,z):
    Ez2 = Om0*(1+z)**3+(1-Om0)
    return 100.0*h*np.sqrt(Ez2)

z = np.arange(2.0,5.0,0.1)

Hz_07_02 = get_Hz(0.7,0.2,z)
Hz_07_03 = get_Hz(0.7,0.3,z)
Hz_07_04 = get_Hz(0.7,0.4,z)

Hx_07_02 = get_Hz(0.7,0.2,3.0)
Hx_07_03 = get_Hz(0.7,0.3,3.0)
Hx_07_04 = get_Hz(0.7,0.4,3.0)

plt.plot(z,Hz_07_04/Hz_07_03,color='red',ls='-',label=r'$\Omega_m=0.4$')
plt.plot(z,Hz_07_03/Hz_07_03,color='black',ls=':',label=r'$\Omega_m=0.3$')
plt.plot(z,Hz_07_02/Hz_07_03,color='blue',ls='-',label=r'$\Omega_m=0.2$')
plt.plot(z,np.ones_like(z)*Hx_07_02/Hx_07_03,color='blue',ls=':')
plt.plot(z,np.ones_like(z)*Hx_07_04/Hx_07_03,color='red',ls=':')
plt.title(r'Expansion history for different $\Omega_m$ (fixed h=0.7)')
plt.xlabel('z')
plt.ylabel(r'$H(z)/H_0(z)$')
plt.legend()
plt.savefig('Hz_omega_m_fixed_h.pdf')
plt.show()

z32=((1+z)/(1+3))**1.5

plt.plot(z,Hz_07_04/Hx_07_04/z32,color='red',ls='-',label=r'$\Omega_m=0.4$')
plt.plot(z,Hz_07_03/Hx_07_03/z32,color='black',ls='-',label=r'$\Omega_m=0.3$')
plt.plot(z,Hz_07_02/Hx_07_02/z32,color='blue',ls='-',label=r'$\Omega_m=0.2$')
plt.plot(z,np.ones_like(z),'gray',ls=':',label='EdS')
plt.title(r'Expansion history for different $\Omega_m$')
plt.xlabel('z')
plt.ylabel(r'$\frac{H(z)}{H(z=3)} \left(\frac{1+z}{1+3}\right)^{-3/2}$')
plt.legend()
plt.savefig('Hz3_omega_m.pdf')
plt.show()

