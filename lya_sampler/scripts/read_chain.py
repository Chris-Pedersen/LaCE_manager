import numpy as np
import configargparse
# our own modules
import linear_emulator
import gp_emulator
import data_PD2013
import likelihood
import emcee_sampler


# get options from command line
parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, 
        help='config file path')
parser.add_argument('--chain_filename', type=str, default=None,
        help='Read the chain from this file (no extension)', required=True)
parser.add_argument('--free_parameters', type=str, default='ln_tau_0, ln_tau_1',
        help='Comma-separated string of free parameters to use', required=True)
parser.add_argument('--max_arxiv_size', type=int, default=3000, 
        help='Maximum number of models to train emulators', required=False)
parser.add_argument('--undersample_z', type=int, default=1, required=False,
        help='Undersample simulation snapshots, to make lighter emulators')
parser.add_argument('--emu_type', type=str, default='polyGP',
                help='Type of emulator to use (polyGP, kGP or linear)', required=False)
parser.add_argument('--verbose', action='store_true', 
        help='Print runtime information',required=False)

args = parser.parse_args()

print('--- print options from parser ---')
print(args)
print("----------")
print(parser.format_help())
print("----------")
print(parser.format_values())
print("----------")

verbose=args.verbose

# read P1D measurement
data=data_PD2013.P1D_PD2013(blind_data=True)
zs=data.z

# read emulator
basedir='../../p1d_emulator/sim_suites/emulator_512_17052019/'
p1d_label='p1d'
skewers_label='Ns100_wM0.05'
if args.emu_type=='polyGP':
    if verbose: print('use polyGP emulator')
    # do not emulate growth or running (for now)
    #paramList=["mF","Delta2_p","n_p","sigT_Mpc","gamma","kF_Mpc"]
    paramList=None
    emu=gp_emulator.PolyfitGPEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z,
                    paramList=paramList,kmax_Mpc=5,train=True)
elif args.emu_type=='kGP':
    if verbose: print('use kGP emulator')
    # do not emulate growth or running (for now)
    #paramList=["mF","Delta2_p","n_p","sigT_Mpc","gamma","kF_Mpc"]
    paramList=None
    emu=gp_emulator.GPEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z,
                    paramList=paramList,kmax_Mpc=5,train=True)
elif args.emu_type=='linear':
    if verbose: print('use linear emulator')
    emu=linear_emulator.LinearEmulator(basedir,p1d_label,skewers_label,
                    max_arxiv_size=args.max_arxiv_size,
                    undersample_z=args.undersample_z)
else:
    raise ValueError('wrong emulator type '+args.emu_type)

# specify free parameters in likelihood (make sure there are no empty spaces)
free_parameters=[par.strip() for par in args.free_parameters.split(',')]
if verbose: print('input free parameters',free_parameters)

# read chain from file
sampler = emcee_sampler.EmceeSampler(emulator=emu,
            free_parameters=free_parameters,read_chain_file=args.chain_filename)

for p in sampler.like.free_params:
    print(p.name,p.value)

# plot results
sampler.plot_corner(cube=True)
sampler.plot_corner(cube=False)

chain,lnprob=sampler.get_chain()
ind = np.unravel_index(np.argmax(lnprob, axis=None), lnprob.shape)
best_fit=chain[ind]
sampler.like.plot_p1d(values=best_fit,plot_every_iz=2)
