import numpy as np


def gaussian_chi2(neff,DL2,neff_val,DL2_val,neff_err,DL2_err,r):
    """Given central values and errors for Delta_L^2 and n_eff, and its 
        cross-correlation coefficient r, compute Gaussian delta chi^2 at
        points (neff,DL2).
    """
    chi2 = ((DL2-DL2_val)**2/DL2_err**2+(neff-neff_val)**2/neff_err**2
                -2*r*(neff-neff_val)*(DL2-DL2_val)/DL2_err/neff_err)/(1-r*r)
    return chi2


def gaussian_chi2_McDonald2005(neff,DL2):
    """Compute Gaussian Delta chi^2 for a particular point(s) (neff,DL2),
    using the measurement from McDonald et al. (2005).
    """
    # DL2 = k^3 P(k) / (2 pi^2)
    DL2_val=0.452
    DL2_err=0.072
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val=-2.321
    neff_err=0.069
    # correlation coefficient
    r=0.63
    return gaussian_chi2(neff,DL2,neff_val,DL2_val,neff_err,DL2_err,r)


def gaussian_chi2_Chabanier2019(neff,DL2):
    """Compute Gaussian Delta chi^2 for a particular point(s) (neff,DL2),
    using the measurement from Chabanier et al. (2019, Figure 20).
    """
    # DL2 = k^3 P(k) / (2 pi^2)
    DL2_val=0.312
    DL2_err=0.02
    # neff = effective slope at kp = 0.009 s/km, i.e., d ln P / dln k
    neff_val=-2.338
    neff_err=0.0065
    # correlation coefficient
    r=0.55
    return gaussian_chi2(neff,DL2,neff_val,DL2_val,neff_err,DL2_err,r)

