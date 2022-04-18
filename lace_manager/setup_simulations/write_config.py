"""Write GenIC / Gadget configuration file from a given cosmology."""

import numpy as np
import os
import json
from lace_manager.setup_simulations import gen_UVB as UVB
from lace_manager.cosmo import fit_linP

def write_genic_file(simdir,cosmo,Ngrid=256,box_Mpc=90,z_ini=99,
        seed=123,paired=False):
    """Write a GenIC file for a given cosmology"""

    # make sure they are not asking what we can not deliver
    if cosmo.omnuh2 > 0.0: 
        raise ValueError('Implement neutrinos in write_genic_files')
    if cosmo.omk > 0.0: 
        raise ValueError('Implement curvature in write_genic_files')

    h = cosmo.H0/100.0
    box_hkpc=box_Mpc*h*1000.0

    filename=simdir+'/paramfile.genic'
    genic_file = open(filename,"w")

    # main simulation settings (options)
    genic_file.write("Ngrid = %d \n" % Ngrid)
    genic_file.write("BoxSize = %f \n" % box_hkpc)
    genic_file.write("Redshift = %f \n" % z_ini)
    genic_file.write("Seed = %d \n" % seed)
    if paired:
        genic_file.write("InvertPhase = 1 \n")

    # main simulation settings (default)
    genic_file.write("OutputDir = "+simdir+"/output \n")
    genic_file.write("FileBase = IC \n")
    genic_file.write("ProduceGas = 1 \n")
    genic_file.write("RadiationOn = 1 \n")
    genic_file.write("MakeGlassGas = 0 \n")
    genic_file.write("DifferentTransferFunctions = 1 \n")
    genic_file.write("ScaleDepVelocity = 1 \n")
    genic_file.write("UnitaryAmplitude = 1 \n")
    genic_file.write("FileWithInputSpectrum = "+simdir+"/matterpow.dat \n")
    genic_file.write("FileWithTransferFunction = "+simdir+"/transfer.dat \n")
            
    # cosmological parameters
    Oc=cosmo.omch2/h**2
    Ob=cosmo.ombh2/h**2
    Om=Oc+Ob
    OL=1.0-Om
    genic_file.write("Omega0 = %f \n" % Om)
    genic_file.write("OmegaLambda = %f \n" % OL)
    genic_file.write("OmegaBaryon = %f \n" % Ob)
    genic_file.write("HubbleParam = %f \n" % h)

    # for now massless neutrinos
    genic_file.write("MNue = 0.0 \n")
    genic_file.write("MNum = 0.0 \n")
    genic_file.write("MNut = 0.0 \n")

    # primordial power spectrum
    genic_file.write("Sigma8 = -1 \n")
    genic_file.write("InputPowerRedshift = -1 \n")
    genic_file.write("WhichSpectrum = 2 \n")
    genic_file.write("PrimordialIndex = %f \n" % cosmo.InitPower.ns)
    genic_file.write("PrimordialAmp = %.6e \n" % cosmo.InitPower.As)
    genic_file.write("PrimordialRunning = %.6e \n" % cosmo.InitPower.nrun)

    # maxim memory required per node (in DiRAC)
    genic_file.write("MaxMemSizePerNode=231168 \n")    ## For skylake-himem
    #genic_file.write("MaxMemSizePerNode=115584 \n")    ## For skylake

    # not quite sure if needed...
    genic_file.write("UnitLength_in_cm = 3.085678e21 \n")
    genic_file.write("UnitMass_in_g = 1.989e43 \n")
    genic_file.write("UnitVelocity_in_cm_per_s = 1e5 \n")

    genic_file.close()


def get_output_list(zs):
    """Given list of (sorted) redshifts, return string with scale factors.
        Does not include the last redshfit (will be given as TimeMax)"""
    # make sure all redshifts are sorted from high to low redshift
    assert np.any(np.diff(zs)>=0) == False, 'zs not sorted'+str(zs)
    output_list=str(1.0/(1+zs[0]))
    for z in zs[1:-1]:
        output_list+=', '+str(1.0/(1+z))
    return output_list


def write_treecool_file(simdir,z_mid_HI_reion):
    """Write a treecool file for a given reionization history"""

    fname=simdir+'/treecool.txt'
    UVB.generate_treecool_file(output_file=fname,z_mid_HI_reion=z_mid_HI_reion)
    return

def write_gadget_file(simdir,cosmo,heat_amp=1.0,heat_slo=1.0,Ngrid=256,
                zs=[49.0,9.0,8.0,7.0,6.0,5.0,4.5,4.0,3.5,3.0,2.5,2.0]):
    """Write a MP-Gadget file for a given cosmology"""

    # make sure they are not asking what we can not deliver
    if cosmo.omnuh2 > 0.0: 
        raise ValueError('Implement neutrinos in write_genic_files')
    if cosmo.omk > 0.0: 
        raise ValueError('Implement curvature in write_genic_files')

    Nmesh=2*Ngrid

    filename=simdir+'/paramfile.gadget'
    gadget_file = open(filename,"w")

    # main simulation settings (options)
    gadget_file.write("Nmesh = %d \n" % Nmesh)
    gadget_file.write("TreeCoolFile = "+simdir+"/treecool.txt \n")
    gadget_file.write("InitCondFile = "+simdir+"/output/IC \n")
    gadget_file.write("OutputDir = "+simdir+"/output \n")
    # find list of outputs (except last one) 
    output_list=get_output_list(zs)
    gadget_file.write('OutputList = "'+output_list+'" \n')
    gadget_file.write("TimeMax = "+str(1.0/(1+min(zs)))+" \n")

    # main simulation settings (default)
    gadget_file.write("SnapshotWithFOF = 0 \n")
    gadget_file.write("TimeLimitCPU = 430000 \n")
    gadget_file.write("MaxMemSizePerNode=231168 \n")    ## For skylake-himem
    #gadget_file.write("MaxMemSizePerNode=115584 \n")    ## For skylake
    gadget_file.write("CoolingOn = 1 \n")
    gadget_file.write("StarformationOn = 1 \n")
    gadget_file.write("RadiationOn = 1 \n")
    gadget_file.write("HydroOn = 1 \n")
    gadget_file.write("WindOn = 0 \n")
    gadget_file.write("StarformationCriterion = density \n")
    gadget_file.write("DensityKernelType = cubic \n")
    gadget_file.write("DensityIndependentSphOn = 0 \n")
    gadget_file.write("InitGasTemp = 270. \n")
    gadget_file.write("MinGasTemp = 100 \n")
    gadget_file.write("PartAllocFactor = 2 \n")
    gadget_file.write("BlackHoleOn=0 \n")
    gadget_file.write("CritPhysDensity = 0 \n")
    gadget_file.write("CritOverDensity = 1000 \n")
    gadget_file.write("QuickLymanAlphaProbability = 1 \n")
    gadget_file.write("WindModel = nowind \n")

    # cosmological parameters
    h = cosmo.H0/100.0
    Oc=cosmo.omch2/h**2
    Ob=cosmo.ombh2/h**2
    Om=Oc+Ob
    OL=1.0-Om
    gadget_file.write("Omega0 = %f \n" % Om)
    gadget_file.write("OmegaLambda = %f \n" % OL)
    gadget_file.write("OmegaBaryon = %f \n" % Ob)
    gadget_file.write("HubbleParam = %f \n" % h)
    # massless neutrinos
    gadget_file.write("MassiveNuLinRespOn = 0 \n")
    gadget_file.write("MNue = 0.0 \n")
    gadget_file.write("MNum = 0.0 \n")
    gadget_file.write("MNut = 0.0 \n")

    # thermal history parameters
    gadget_file.write("HeliumHeatOn = 1 \n")
    gadget_file.write("HeliumHeatAmp = %f \n" % heat_amp)
    gadget_file.write("HeliumHeatExp = %f \n" % heat_slo)

    gadget_file.close()

    # return list of redshifts (including last output)
    return zs


def write_cube_json_file(simdir,param_space,cube):
    """Write a JSON file with meta data associated to the whole cube."""
    
    # store parameter space
    cube_info={'param_space':param_space}

    # store number of samples in cube
    nsamples=len(cube)
    cube_info['nsamples']=nsamples

    # store actual samples
    samples={}
    for i in range(nsamples):
        samples[str(i)]=list(cube[i])
    cube_info['samples']=samples

    filename=simdir+'/latin_hypercube.json'
    json_file = open(filename,"w")
    json.dump(cube_info,json_file)
    json_file.close()


def write_sim_json_file(simdir,param_space,cosmo_sim,zs):
    """Write a JSON file with meta data associated to this simulation pair."""

    filename=simdir+'/parameter.json'

    json_info={}

    # copy pivot point in parameterization (should be same in all sims)
    json_info['kp_Mpc'] = param_space.kp_Mpc

    # write linear power in each snapshot
    json_info['zs']=list(zs)
    linP_zs=fit_linP.get_linP_Mpc_zs(cosmo_sim,zs,param_space.kp_Mpc,
            include_f_p=True)
    json_info['linP_zs']=list(linP_zs)

    json_file = open(filename,"w")
    json.dump(json_info,json_file)
    json_file.close()

    return linP_zs
