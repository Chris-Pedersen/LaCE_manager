"""Write GenIC / Gadget configuration file from a given cosmology."""

import numpy as np
import os
import json

def mkdir_if_not_exists(dirname,verbose=False):
    """Check if directory exists, and create otherwise."""
    if os.path.exists(dirname):
        if not os.path.isdir(dirname):
            raise ValueError(dirname+' exists but it is not a directory')
        if verbose:
        	print(dirname,'exists')
    else:
        if verbose:
        	print('mkdir',dirname)
        os.mkdir(dirname)


def write_genic_file(simdir,cosmo,Ngrid=256,box_Mpc=90,z_ini=99,
        seed=123,paired=False):
    """Write a GenIC file for a given cosmology"""

    # make sure they are not asking what we can not deliver
    if cosmo.omnuh2 > 0.0: 
        raise ValueError('Implement neutrinos in write_genic_files')
    if cosmo.omk > 0.0: 
        raise ValueError('Implement curvature in write_genic_files')

    h = cosmo.H0/100.0
    box_hMpc=box_Mpc*h

    filename=simdir+'/paramfile.genic'
    genic_file = open(filename,"w")

    # main simulation settings (options)
    genic_file.write("Ngrid = %d \n" % Ngrid)
    genic_file.write("BoxSize = %f \n" % box_hMpc)
    genic_file.write("Redshift = %f \n" % z_ini)
    genic_file.write("Seed = %d \n" % seed)
    if paired:
        genic_file.write("InvertedPhase = 1 \n")

    # main simulation settings (default)
    genic_file.write("OutputDir = output \n")
    genic_file.write("FileBase = IC \n")
    genic_file.write("ProduceGas = 1 \n")
    genic_file.write("RadiationOn = 1 \n")
    genic_file.write("DifferentTransferFunctions = 1 \n")
    genic_file.write("ScaleDepVelocity = 1 \n")
    genic_file.write("UnitaryAmplitude = 1 \n")
    genic_file.write("FileWithInputSpectrum = matterpow.dat \n")
    genic_file.write("FileWithTransferFunction = transfer.dat \n")
            
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

    # not quite sure if needed...
    genic_file.write("UnitLength_in_cm = 3.085678e21 \n")
    genic_file.write("UnitMass_in_g = 1.989e43 \n")
    genic_file.write("UnitVelocity_in_cm_per_s = 1e5 \n")

    genic_file.close()


def write_gadget_file(simdir,cosmo,mu_He=1.0,Ngrid=256):
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
    gadget_file.write("TreeCoolFile = ../test_sim/TREECOOL_P18.txt \n")
    gadget_file.write("OutputList = \"0.1,0.2,0.25,0.3,0.325\" \n")

    # main simulation settings (default)
    gadget_file.write("InitCondFile = output/IC \n")
    gadget_file.write("OutputDir = output \n")
    gadget_file.write("SnapshotFileBase = snap \n")
    gadget_file.write("TimeLimitCPU = 430000 \n")
    gadget_file.write("TimeMax = 0.3333333 \n")
    gadget_file.write("CoolingOn = 1 \n")
    gadget_file.write("StarformationOn = 1 \n")
    gadget_file.write("RadiationOn = 1 \n")
    gadget_file.write("HydroOn = 1 \n")
    gadget_file.write("WindOn = 0 \n")
    gadget_file.write("StarformationCriterion = density \n")
    gadget_file.write("DensityKernelType = cubic \n")
    gadget_file.write("InitGasTemp = 270. \n")
    gadget_file.write("MinGasTemp = 100 \n")
    gadget_file.write("PartAllocFactor = 4 \n")
    gadget_file.write("BufferSize = 100 \n")
    gadget_file.write("BlackHoleOn=0 \n")
    gadget_file.write("CritPhysDensity = 0 \n")
    gadget_file.write("CritOverDensity = 1000 \n")
    gadget_file.write("QuickLymanAlphaProbability = 1 \n")
    gadget_file.write("SnapshotWithFOF = 1 \n")
    gadget_file.write("FOFHaloLinkingLength = 0.2 \n")
    gadget_file.write("FOFHaloMinLength = 32 \n")
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
    gadget_file.write("LinearTransferFunction = transfer.dat \n")
    # massless neutrinos
    gadget_file.write("MassiveNuLinRespOn = 0 \n")
    gadget_file.write("MNue = 0.0 \n")
    gadget_file.write("MNum = 0.0 \n")
    gadget_file.write("MNut = 0.0 \n")

    # thermal history parameters
    gadget_file.write("HeliumHeatOn = 1 \n")
    gadget_file.write("HeliumHeatAmp = %f \n" % mu_He)
    #gadget_file.write("HydrogenHeatAmp = %f \n" % mu_H)

    gadget_file.close()


def write_cube_json_file(simdir,param_space):
    """Write a JSON file with meta data associated to the whole cube."""
    
    filename=simdir+'/latin_hypercube.json'
    json_file = open(filename,"w")
    json.dump(param_space,json_file)
    json_file.close()


def write_sim_json_file(simdir,param_space,sim_params,linP_model):
    """Write a JSON file with meta data associated to this simulation pair."""
    
    filename=simdir+'/parameter.json'

    json_info={}

    # copy pivot point in parameterization (should be same in all sims)
    json_info['z_star']=linP_model.z_star
    assert linP_model.k_units is 'Mpc', 'linP_model not in Mpc units'
    json_info['kp_Mpc'] = linP_model.kp

    # copy values of parameters for this particular simulation
    for key,param in param_space.items():
        ip=param['ip']
        json_info['target_'+key]=sim_params[ip]

    # copy also linear power parameters actually fitted 
    json_info['fit_f_star']=linP_model.get_f_star()
    json_info['fit_g_star']=linP_model.get_g_star()
    json_info['fit_Delta2_star']=linP_model.get_Delta2_star()
    json_info['fit_n_star']=linP_model.get_n_star()
    json_info['fit_alpha_star']=linP_model.get_alpha_star()

    json_file = open(filename,"w")
    json.dump(json_info,json_file)
    json_file.close()

