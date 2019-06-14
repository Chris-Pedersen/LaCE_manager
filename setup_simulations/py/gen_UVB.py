# adapted from http://thermal.joseonorbe.com/doc/gen_UVB.py

import numpy
import os
import asciitable
import scipy.interpolate as spi
import scipy.special as sps

#################
### Constants ###
#################
TCMBz0=2.725 # TCMB at z=0
mproton=1.6726231E-24 #g
kB=1.3806488e-16 #boltzmann constant erg/K -> g cm2 s-2 K-1
gamma_m_1=5./3.-1.
cm2Mpc=3.240779291E-25 
cm2Mpc_cube=cm2Mpc**3.
H0=3.24077929e-18 # 1/s
lightspeed=2.9979e10 #cgs cm/s
##Critical density with h = 1, in g/cm^3.
##rho_crit = 3 H_0^2 / (8 pi G)
rho_crit_100_pMpc = 3.29940063E68  # protons/Mpc^3

#### Set Clumping factors
#HI
CiHI=1.0 #ionization clumping HI
CeHI=1.0 # collisional ion clumping HI
#HII
CrHII=1.5 #recombination clumping HI 
# HeI
CiHeI=1.0 #ionization clumping HeI
CeHeI=1.0 # collisional ion clumping HeI
# HeII
CdHeII=1.0 # dielectric clumping HeII
CrHeII=1.5 # recombination clumping HeII
# HeIII
CrHeIII=1.5 # recombination clumping HeIII
CeHeIII=1.0 # collisional clumping HeIII 
CiHeIII=1.0 # ionization clumping HeIII
#### Minimum value of photoio so that code does not crach
GammaHmin=1E-18
#### Minimum qdot values
qHmin=1E-33
qHemin=1E-34

lowval=1E-5
###########################
### Auxiliary Functions ###
###########################

# write TREECOOL file
def write_TREECOOL(output_file,lz,photo_new,QDeltaT=False):
    pf=open(output_file,'w')
    if QDeltaT==False: # standard out
        for i in range(len(lz)):
            pf.write("%f %e %e %e %e %e %e\n" %
                (lz[i],photo_new[0,i],photo_new[1,i],photo_new[2,i],
                    photo_new[3,i],photo_new[4,i],photo_new[5,i]) )
    else: # Delta T format
        dTHdz,dTHedz=QDeltaT
        for i in range(len(lz)):
            pf.write("%f %e %e %e %e %e %e %e %e\n" %
                (lz[i],photo_new[0,i],photo_new[1,i],photo_new[2,i],
                    photo_new[3,i],photo_new[4,i],photo_new[5,i],
                    dTHdz[i],dTHedz[i]) )
    return

def fzcut(z,zcut):
    #fcut=z>zcut
    #ncut=numpy.where(d==numpy.min(d))[0][0]
    #return ncut
    return numpy.where(z == numpy.max(z[z <= zcut]))[0][0]

def fint(z,y):
    yint=numpy.trapz(y,z)
    return yint

def fintarray(z,y):
    yinta=numpy.zeros(len(z))
    for i in range(len(z)-1):
        yinta[i]=numpy.trapz(y[i:],z[i:])
    return yinta

def fdQdz(z,Q):
    dQdz=numpy.zeros(len(z))
    fDQ=spi.interp1d(z,Q,kind='linear')
    # Create new bins to better define the derivate at our desired
    # values which are z values
    bz=(z[:-1]+z[1:])/2. #new z values; binning done so center are the z values
    Q_new=fDQ(bz) # values of DeltaT at the new z values
    dQ=numpy.abs(Q_new[1:]-Q_new[:-1]) # just interested in the absolute value
    dz=bz[1:]-bz[:-1] # bin sizes 
    # First and last value is always =0 
    dQdz[1:-1]=dQ/dz
    return dQdz

def interpUVB(input_file):
    # UVB model that will be used once reionization is finished
    data=asciitable.read(input_file)
    lz = data['col1']
    fpiHI=spi.interp1d(lz,data['col2'],kind='linear')
    fpiHeI=spi.interp1d(lz,data['col3'],kind='linear')
    fpiHeII=spi.interp1d(lz,data['col4'],kind='linear')
    fphHI=spi.interp1d(lz,data['col5'],kind='linear')
    fphHeI=spi.interp1d(lz,data['col6'],kind='linear')
    fphHeII=spi.interp1d(lz,data['col7'],kind='linear')
    return [lz,fpiHI,fpiHeI,fpiHeII,fphHI,fphHeI,fphHeII]

####### COSMOLOGY
#Get Hubble parameter
def Hubble(z,cosmo=[0.702,0.046,0.275,0.725,0.76]):
    h,omega_b,omega_m,omega_lambda,Xp=cosmo
    return h*H0*numpy.sqrt(omega_m*(1.0+z)**3.+omega_lambda)

###### Hydrogen density in the universe
# calc nH: Mean hydrogen density in the universe <nH>:
def calc_nH(cosmo=[0.702,0.046,0.275,0.725,0.76]):
    h,omega_b,omega_m,omega_lambda,Xp=cosmo
    nH=Xp*omega_b*rho_crit_100_pMpc*(h**2) #Number density of protons per Mpc^3.
    return nH

def falpha(T,a,b,Tf0,Tf1):
    x0=numpy.sqrt(T/Tf0);x1=numpy.sqrt(T/Tf1)
    return a/ ( x0 * ((1.+x0)**(1.-b)) * ((1.+x1)**(1+b)) )

#######  Recombination times
# calc case B HI recombination coeff
def calc_alphaB(T0=2E4,alpha='Nyx'):
    #Td=(T0/1E4)**-0.7
    #alphaB=4.2e-13*cm2Mpc_cube*Td # From Abel et al. 1997
    ###### Nyx
    if alpha=='Nyx':
        a=7.982E-11;b=0.7480;Tf0=3.148;Tf1=7.036E5
        alphaB_F=falpha(T0,a,b,Tf0,Tf1) 
        alphaB=alphaB_F*cm2Mpc_cube
    elif alpha=='cte_low':
        alphaB=1.6e-13*cm2Mpc_cube #the case B recombination coefficient: our reion paper dwarf
    elif alpha=='cte_Kuhlen':
        #alphaB=2.5114e-13*cm2Mpc_cube #Kuhlen value
        alphaB=2.5114e-13*cm2Mpc_cube  #alpha from NORAD: http://www.astronomy.ohio-state.edu/~nahar/nahar_radiativeatomicdata/h1/h1.rrc.txt.  Unit conversion to Mpc^3/s
    else:
        print("alpha not defined!")
    return alphaB # in Mpc^3/s

# calc case B HeII recombination coeff
def calc_alphaBHeII(T0=2E4):
    ###### Nyx
    if isinstance(T0, (float,int)): T0=numpy.array([T0]) #,long
    alphaB_F=numpy.zeros(len(T0))
    for i in range(len(T0)):
        T=T0[i]
        if T<= 1E6:
            a=3.294E-11;b=0.6910;Tf0=15.54;Tf1=3.676E7
        else:
            a=9.356E-10;b=0.7892;Tf0=4.266E-2;Tf1=4.677E6
        alphaB_F[i]=falpha(T,a,b,Tf0,Tf1) 
        alphaB_F[i]=alphaB_F[i]*(cm2Mpc**3.)
    if len(T0)==1:
        alphaB_F=alphaB_F[0]
    return alphaB_F # in Mpc^3/s

# calc case B HeIII recombination coeff
def calc_alphaBHeIII(T0=2E4):
    ###### Nyx
    a=1.891E-10;b=0.7524;Tf0=9.370;Tf1=2.774E6
    alphaB_F=falpha(T0,a,b,Tf0,Tf1) 
    alphaB_F=alphaB_F*(cm2Mpc**3.)
    return alphaB_F # in Mpc^3/s

# calc case B HeII dielectric coeff
def calc_alphadHeII(T0=2E4):
    ###### Nyx
    d0=1+0.3*numpy.exp(-9.4E4/T0)
    d1=numpy.exp(-4.7E5/T0)
    d2=T0**(-3./2.)
    alphad=1.9E-3*d0*d1*d2
    alphad=alphad*(cm2Mpc**3.)
    return alphad # in Mpc^3/s

def get_clumpingHI(z,model):
    if model=='Pawlik':
        Clump=1.+43.*(z**(-1.71)) #Pawlik et al 2009 CHII ovdens 100 r19.5L6N256
        #remember this fit only valid for z>6
    elif model=='Iliev07':
        Clump=26.2917*numpy.exp(-0.1822*z+0.003505*(z**2.)) #Iliev et al. 2007
    elif model=='sim':
        Clump=z*0.0+1.5 #=1.5 z>=10
        Clump[z<10]=2.0 #=2.0 z<10
    else:
        Clump=z*0.0+model # model is a constant
        #Clump=z*0.0+3. #Robertson Page 2
    return Clump


## Collisional ionization from Lukic et al.
def fGe(T0,A,E,P,X,m):
    U=11604.5*E/T0
    d0=(1.+P*numpy.sqrt(U))/(X+U)
    d1=U**m
    d2=numpy.exp(-U)
    Ge=A*d0*d1*d2
    return Ge
def calc_GammaeHI(T0):
    A=2.91E-8;E=13.6;P=0.;X=0.232;m=0.39
    Ge=fGe(T0,A,E,P,X,m)
    return Ge*(cm2Mpc**3.)
def calc_GammaeHeI(T0):
    A=1.75E-8;E=24.6;P=0.;X=0.180;m=0.35
    Ge=fGe(T0,A,E,P,X,m)
    return Ge*(cm2Mpc**3.)
def calc_GammaeHeII(T0):
    A=2.05E-9;E=54.4;P=1.;X=0.265;m=0.25
    Ge=fGe(T0,A,E,P,X,m)
    return Ge*(cm2Mpc**3.)


#########################
###      Functions    ###
### to generate rates ###
#########################

########### Calc Gamma analytical from Q
#   Assuming Ionization Eq.
# Inverse relation Gamma from QHII 
def QHII2Gamma(z,QHII,T0=1E4,cosmo=[0.702,0.046,0.275,0.725,0.76],
        clump='sim',Gcut=True):
    alpha=calc_alphaB(T0=T0)
    Ge=calc_GammaeHI(T0)
    nH=calc_nH(cosmo=cosmo)*((1+z)**3.)
    QHI=1.-QHII
    CrHII=get_clumpingHI(z,clump) #
    Xp=cosmo[4];Yp=1.-Xp;YHELIUM=Yp/(4.*Xp)
    HeHcorr=1.+YHELIUM # this is ~1.08 
    # Lukic et al.
    # averaging & <ne=HeH*nHII> & <nHII^2/nHI>=CHII*<nHII>^2/<nHI>
    ne=HeHcorr*nH*QHII
    Gamma=alpha*HeHcorr*CrHII*nH*(QHII**2.)/(CiHI*QHI) #- CeHI*Ge*ne/CiHI #Ge is only relevant in shocks
    if Gcut:
        Gamma[Gamma<GammaHmin]=0.0 # fix for very low values
    return Gamma

# Inverse relation Gamma from QHeII
def QHeII2Gamma(z,QHeII,T0=1E4,cosmo=[0.702,0.046,0.275,0.725,0.76],
        clump='sim',Gcut=True):
    alphar=calc_alphaBHeII(T0=T0)
    alphad=calc_alphadHeII(T0=T0)
    Ge=calc_GammaeHeI(T0)
    nH=calc_nH(cosmo=cosmo)*((1+z)**3.)
    QHeI=1.-QHeII
    CrHeII=get_clumpingHI(z,clump) #recombination rate clumping
    Xp=cosmo[4];Yp=1.-Xp;YHELIUM=Yp/(4.*Xp)
    HeHcorr=1.+YHELIUM # this is ~1.08 
    ne=HeHcorr*nH*QHeII
    Gamma=(CrHeII*alphar+CdHeII*alphad)*HeHcorr*nH*(QHeII**2.)/(CiHeI*QHeI)  #-CeHeI*Ge*ne/CiHeI #Ge is only relevant in shocks
    if Gcut:
        Gamma[Gamma<GammaHmin]=0.0 # fix for very low values
    return Gamma

# Inverse relation Gamma from QHeIII 
# Assuming that nHeI=0 so H reionization finished already 
def QHeIII2Gamma(z,QHII,QHeIII,T0=1E4,cosmo=[0.702,0.046,0.275,0.725,0.76]):
    alphar=calc_alphaBHeIII(T0=T0)
    Ge=calc_GammaeHeII(T0)
    alpharHeII=calc_alphaBHeII(T0=T0)
    alphadHeII=calc_alphadHeII(T0=T0)
    nH=calc_nH(cosmo=cosmo)*((1+z)**3.)
    QHeII=1.-QHeIII
    Xp=cosmo[4];Yp=1.-Xp;YHELIUM=Yp/(4.*Xp)
    HeHcorr=1.+YHELIUM # this is ~1.08 
    ne=nH*(HeHcorr*QHII+YHELIUM*QHeIII)
    f1=HeHcorr+YHELIUM*QHeIII
    Gamma=CrHeIII*alphar*f1*nH*QHeIII/QHeII
    #    -CeHeII*Ge*ne/CiHeII 
    #    -CrHeII*alpharHeII*ne/CiHeII # why ignore this? 
    #    -CdHeII*alphadHeII*ne/CiHeII 
    Gamma[Gamma<0]=0.0 #alpharHeII and alphadHeII
    return Gamma

#   Assuming Ionization Eq.
# Inverse relation q_HI from QHII 
def QHII2qHI(z,QHII,dTdz,cosmo=[0.702,0.046,0.275,0.725,0.76]):
    nH=calc_nH(cosmo=cosmo)*((1+z)**3.) #protons/Mpc^3 
    QHI=1.-QHII
    dtdzcorr=(1.+z)*Hubble(z,cosmo=cosmo) # to transform dT/dz to dT/dt in seconds 
    Xp=cosmo[4];Yp=1.-Xp;YHELIUM=Yp/(4.*Xp)
    HeHcorr=1.+YHELIUM # this is ~1.08 
    CqHI=1.0
    xe=HeHcorr*QHII #=ne/nH 
    mu=(1.+4*YHELIUM)/(1.+YHELIUM+xe)
    f1=(kB*dTdz*dtdzcorr)/(gamma_m_1*mproton*mu) #dT/dz in erg/s
    f2= mproton/(Xp*QHI)
    q=CqHI*f1*f2
    q[q<qHmin]=0.0
    return q # this must be in erg/s per ion

#   Assuming Ionization Eq.
# Inverse relation q_HeII from QHeIII 
def QHeIII2qHeII(z,QHeIII,dTdz,cosmo=[0.702,0.046,0.275,0.725,0.76]):
    nH=calc_nH(cosmo=cosmo)*((1+z)**3.) #protons/Mpc^3
    QHeII=1.-QHeIII
    dtdzcorr=(1.+z)*Hubble(z,cosmo=cosmo) # to transform to seconds
    Xp=cosmo[4];Yp=1.-Xp;YHELIUM=Yp/(4.*Xp)
    HeHcorr=1.+YHELIUM # this is ~1.08 
    CqHeII=1.0
    xe=HeHcorr*QHeII+YHELIUM*QHeIII #ne/nH
    mu=(1.+4.*YHELIUM)/(1.+YHELIUM+xe)
    f1=(kB*dTdz*dtdzcorr)/(gamma_m_1*mproton*mu) #dT/dz in erg/s
    f2= 4*mproton/(Yp*QHeII)
    q=CqHeII*f1*f2 
    q[q<qHemin]=0.0
    return q # this must be in erg/s per ion

#######################
###  Main Function  ###
#######################

def genQ2G_DeltaT(z,QHII,zcutH,QHeIII,zcutHe,
        DeltaTHI,DeltaTHeII,
        input_file='data/TREECOOL_P18.txt',
        cosmo=[0.702,0.046,0.275,0.725,0.76],
        Gthreshold=True,
        output_file=None):

    ### First we need to do assumption about temperature evolution
    TCMB=TCMBz0*(1.+z)
    TzH=QHII*DeltaTHI
    T0HI=1.0E4 # We assume that once HI is finished T0 of IGM is T0HI
    TzHe=T0HI+QHeIII*DeltaTHeII
    TzH[TzH<=TCMB]=TCMB[TzH<=TCMB] # When T<Tcmb assume T=Tcmb
    TzHe[TzHe<=TCMB]=TCMB[TzHe<=TCMB] # When T<Tcmb assume T=Tcmb
    #### Now we create empty arrays
    lz=numpy.log10(z+1)
    newlen=len(z) 
    photo_new=numpy.zeros((6,newlen))
    dTHdz=numpy.zeros(newlen);dTHedz=numpy.zeros(newlen)
    ncutH=fzcut(z,zcutH) # highest redhist below or equal to the cut
    ncutHe=fzcut(z,zcutHe) # highest redhist below or equal to the cut

    #### Now we obtain the functions to get photoionization 
    #### and photoheating rates at z lower than reionization and fill values
    lzmodel,fpiHI,fpiHeI,fpiHeII,fphHI,fphHeI,fphHeII=interpUVB(input_file)
    #### IONIZATIOn HI
    # For ncut and below (z<zcut) we use model values
    photo_new[0,:ncutH+1]=fpiHI(lz[:ncutH+1])
    photo_new[1,:ncutH+1]=fpiHeI(lz[:ncutH+1])
    photo_new[3,:ncutH+1]=fphHI(lz[:ncutH+1])
    photo_new[4,:ncutH+1]=fphHeI(lz[:ncutH+1])
    # # H & HeI above ncut z>zcut
    GammaHI=QHII2Gamma(z[ncutH+1:],QHII[ncutH+1:],T0=TzH[ncutH+1:],cosmo=cosmo)
    GammaHeI=QHeII2Gamma(z[ncutH+1:],QHII[ncutH+1:],T0=TzH[ncutH+1:],cosmo=cosmo)  # Assuming QHeII=QHII
    if Gthreshold:
        ### The new values of Gamma can not be higher that the value where we join with the old
        fmaxH=photo_new[0,ncutH];GammaHI[GammaHI>fmaxH]=fmaxH
        fmaxHeI=photo_new[1,ncutH];GammaHeI[GammaHeI>fmaxHeI]=fmaxHeI
    photo_new[0,ncutH+1:]=GammaHI # ionization 
    photo_new[1,ncutH+1:]=GammaHeI # ionization
    #### IONIZATIOn HEII
    # For ncut and below (z<zcut) we use model values
    photo_new[2,:ncutHe+1]=fpiHeII(lz[:ncutHe+1])
    photo_new[5,:ncutHe+1]=fphHeII(lz[:ncutHe+1])
    # HeII above ncut z>zcut
    GammaHeII=QHeIII2Gamma(z[ncutHe+1:],QHII[ncutHe+1:],QHeIII[ncutHe+1:],T0=TzHe[ncutHe+1:],cosmo=cosmo)
    if Gthreshold:
        fmaxHeII=photo_new[2,ncutHe];GammaHeII[GammaHeII>fmaxHeII]=fmaxHeII
    photo_new[2,ncutHe+1:]=GammaHeII # ionization
    #### HEATING HI 
    dQdz=fdQdz(z,QHII)
    dTHdz=DeltaTHI*dQdz/abs(fint(z,dQdz))
    dTHdz[:ncutH+1]=0.0 # this is to force dTdz=0 when z<=zreion
    dTHdz[ncutH+1:][dTHdz[ncutH+1:]==0.0]=lowval # above z_reion, never have zero value
    gDTH=fintarray(z,dTHdz)
    fH=dTHdz!=0.0 # select values of photoheating we want to change
    photo_new[3,fH]=QHII2qHI(z[fH],QHII[fH],dTHdz[fH],cosmo=cosmo)
    ### HEATING HeliumII
    dQdz=fdQdz(z,QHeIII)
    dTHedz=DeltaTHeII*dQdz/abs(fint(z,dQdz))
    dTHedz[ncutHe+1:][dTHedz[ncutHe+1:]==0.0]=lowval # above z_reion, never have zero value
    dTHedz[:ncutHe+1]=0.0 # this is to force  dTdz=0 when z<=zreion
    gDTHe=fintarray(z,dTHedz)
    fHe=dTHedz!=0.0 # select values of photoheating we want to change
    photo_new[5,fHe]=QHeIII2qHeII(z[fHe],QHeIII[fHe],dTHedz[fHe],cosmo=cosmo)
    # The new values of dot{q} can not be higher that the value  where we join with the old
    if Gthreshold:
        fmaxH=photo_new[3,numpy.min(numpy.where(fH))-1]
        ffixH=numpy.logical_and(photo_new[3,:]>fmaxH,fH)
        photo_new[3,ffixH]=fmaxH
        fmaxHe=photo_new[5,numpy.min(numpy.where(fHe))-1]
        ffixHe=numpy.logical_and(photo_new[5,:]>fmaxHe,fHe)
        photo_new[5,ffixHe]=fmaxHe
    # Final check for dot{q}. They have to be zero if Gamma is zero
    fcH=photo_new[0,:]==0.0;photo_new[3,:][fcH]=0.0
    fcHe=photo_new[2,:]==0.0;photo_new[5,:][fcHe]=0.0
    #### write to file
    if output_file!=None:
        write_TREECOOL(output_file,lz,photo_new,QDeltaT=False)
    return z,photo_new


######################################################
# A note on Eq. 11 in https://arxiv.org/abs/1607.04218:
# Due to a poor description on my part this equation
# is confusing. In fact, I am using the "regularized" or
# "normalized" lower incomplete gamma function (and not
# simply the lower incomplete gamma function as stated
# in the text). 
# I copy below the whole
# python function that should reproduce the results
# presented in the paper. 
import scipy.special

def myfQHII_2(listz,zzero,n1=50.,n2=1.,norm=0.5):
    QHII=numpy.zeros(len(listz))
    for i in range(len(listz)):
        z=listz[i]
        x=numpy.abs(z-zzero)
        sgn=numpy.sign(z-zzero)
        if (z-zzero)<=0:
            f=0.5+norm*scipy.special.gammainc(1./n1,x**n1)
        else:
            f=0.5-norm*scipy.special.gammainc(1./n2,x**n2)
        if f>1.0:
            f=1.0
        elif f<0.0:
            f=0.0
        QHII[i]=f
        QHII[0]=1.
        QHII[-1]=0.
    return QHII


#####################################################################
#### Equations by Keir Rogers to get alternative Q_HII evolution ####
#####################################################################

def volume_filling_factor_HII(z, z_mid, n_1=50., n_2=1.):
    """Simple parametric model for input volume filling factor of ionised hydrogen (Q_HII) as a function of redshift z"""
    sign_array = numpy.zeros_like(z) + (z <= z_mid) - (z > z_mid)
    gamma_a = numpy.zeros_like(z) + ((z <= z_mid) / n_1)  + ((z > z_mid) / n_2)
    gamma_x = numpy.zeros_like(z) + ((z <= z_mid) * (numpy.abs(z - z_mid) ** n_1))  + ((z > z_mid) * (numpy.abs(z - z_mid) ** n_2))
    return 0.5 + (0.5 * sign_array * sps.gammainc(gamma_a, gamma_x))

def volume_filling_factor_HeIII(z, z_end):
    """Simple parametric model for input volume filling factor of doubly-ionised helium (Q_HeIII) as a function of z"""
    arctan_model = 1. - numpy.arctan(z - z_end)
    arctan_model[arctan_model > 1.] = 1.
    arctan_model[arctan_model < 0.] = 0.
    return arctan_model

def find_z_end_HI_reion(z, z_mid, n_1=50., n_2=1.):
    """Approximate the redshift when HI reionisation has 'finished' (Q_HII = 1)"""
    return numpy.max(z[volume_filling_factor_HII(z, z_mid, n_1=n_1, n_2=n_2) == 1.])


##############################################
#### Main equation used in LyaCosmoParams ####
##############################################

def generate_treecool_file(output_file='test_TREECOOL.txt',
                    z_mid_HI_reion=7.65, z_end_HeII_reion = 3.5,
                    DeltaTHI = 2.e4, DeltaTHeII = 1.5e+4):
    # input model will be, for now, that of HM2012
    treecool_dir=os.environ['LYA_EMU_REPO']+'/setup_simulations/test_sim/'
    input_file=treecool_dir+'/TREECOOL_HM12.txt'
    # cosmology in HM2012
    Om=0.3
    OL=1.0-Om
    Ob=0.045
    h=0.7
    Y_He=0.245
    cosmoHM = [h,Ob,Om,OL,1.-Y_He] #X_P = 1 - Y_P (???)    
    # list of redshifts to go into the TREECOOL file
    listz=numpy.arange(0.,18.,0.01)
    
    # Figure out filling factor of hydrogen
    QHII = myfQHII_2(listz,z_mid_HI_reion,n1=50.,n2=1.,norm=0.5)
    #QHII = volume_filling_factor_HII(listz, z_mid_HI_reion)
    z_end_HI_reion = find_z_end_HI_reion(listz, z_mid_HI_reion)
    print('z_end for HI reionisation model (Q_HII = 1) =', z_end_HI_reion)

    # Filling factor of helium
    QHeIII = volume_filling_factor_HeIII(listz, z_end_HeII_reion)
    
    # generate new treecool file, and write to disk
    genQ2G_DeltaT(listz,QHII=QHII,zcutH=z_end_HI_reion,
                QHeIII=QHeIII,zcutHe=z_end_HeII_reion,
                DeltaTHI=DeltaTHI,DeltaTHeII=DeltaTHeII,
                input_file=input_file,cosmo=cosmoHM,Gthreshold=True,
                output_file=output_file)
