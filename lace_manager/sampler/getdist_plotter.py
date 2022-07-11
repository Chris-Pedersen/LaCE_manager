# read emcee chains and get them ready to plot with getdist
import numpy as np
from getdist import MCSamples
from lace_manager.sampler import emcee_sampler


# for each parameter name, figure out LaTeX label
param_latex_dict={"Delta2_star":"\Delta^2_\star",
            "n_star":"n_\star",
            "alpha_star":"\\alpha_\star",
            "g_star":"g_\star",
            "f_star":"f_\star",
            "ln_tau_0":"\mathrm{ln} \\tau_0",
            "ln_tau_1":"\mathrm{ln} \\tau_1",
            "ln_sigT_kms_0":"\mathrm{ln} \\sigma^T_0",
            "ln_sigT_kms_1":"\mathrm{ln} \\sigma^T_1",
            "ln_gamma_0":"\mathrm{ln} \\gamma_0",
            "ln_gamma_1":"\mathrm{ln} \\gamma_1",
            "ln_kF_0":"\mathrm{ln} k^F_0",
            "ln_kF_1":"\mathrm{ln} k^F_1",
            "H0":"H_0",
            "mnu":"\\Sigma m_{\\nu}",
            "As":"A_s",
            "ns":"n_s",
            "nrun":"\\alpha_s",
            "ombh2":"\omega_b",
            "omch2":"\omega_c",
            "cosmomc_theta":"\theta_{MC}"
            }


def read_chain_for_getdist(rootdir,subfolder,chain_num,label,
            delta_lnprob_cut=50,
            ignore_rows=0.2,
            smooth_scale=0.2):
    print('will read chain for',label,rootdir,subfolder,chain_num)
    run={'chain_num':chain_num,'label':label}
    sampler=emcee_sampler.EmceeSampler(read_chain_file=chain_num,
                rootdir=rootdir,subfolder=subfolder,
                train_when_reading=False,
                ignore_grid_when_reading=True)
    run['sampler']=sampler

    print('figure out free parameters for',label)
    param_names=[param.name for param in sampler.like.free_params]
    vary_H0=('H0' in param_names)
    if vary_H0:
        blob_names=['Delta2_star','n_star','alpha_star','f_star','g_star']
    else:
        blob_names=['Delta2_star','n_star','alpha_star','f_star','g_star','H0']
    param_names+=blob_names
    run['param_names']=param_names
    print(label,param_names)
    run['param_labels']=[param_latex_dict[par] for par in param_names]

    # read value of free and derived parameters
    free_values,lnprob,blobs=sampler.get_chain(cube=False,
            delta_lnprob_cut=delta_lnprob_cut)
    blob_values=np.array([blobs[key] for key in blob_names]).transpose()
    run['values']=np.hstack([free_values,blob_values])

    # figure out range of allowed values
    ranges={}
    for par in sampler.like.free_params:
        ranges[par.name]=[par.min_value,par.max_value]

    # setup getdist object
    samples=MCSamples(samples=run['values'],label=run['label'],
                      names=run['param_names'],
                      labels=run['param_labels'],
                      ranges=ranges,
                      settings={'ignore_rows':ignore_rows,
                                'mult_bias_correction_order':0,
                                'smooth_scale_2D':smooth_scale,
                                'smooth_scale_1D':smooth_scale})
    run['samples']=samples
    print(label,samples.numrows)

    return run

