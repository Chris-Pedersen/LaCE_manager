""" Measures temperature-density relation in all snapshots and writes to file"""

import numpy as np
import argparse
# our modules below
import temperature_density
import json

# get options from command line
parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, help='Base simulation directory (crashes if it does not exists)',required=True)
parser.add_argument('--tdr_filename', type=str, default='default_tdr',help='Filename for TDR information',required=False)
parser.add_argument('--tdr_basedir', type=str, help='Base directory where the TDR file will be stored',required=False)
parser.add_argument('--zmax', type=float, default=20.0, help='Compute temperature-density relation only for z < zmax',required=False)
parser.add_argument('--verbose', action='store_true', help='Print runtime information',required=False)
args = parser.parse_args()

basedir=args.basedir
tdr_filename=args.tdr_filename

# compute TDR for all snapshots
thermal_info=temperature_density.compute_TDR(basedir=basedir,zmax=args.zmax)
if args.verbose:
    print('got temperature-density relations')
    print(thermal_info)

if args.tdr_basedir:
    tdr_basedir=args.tdr_basedir
    if not os.path.exists(tdr_basedir):
        raise ValueError(tdr_basedir+' does not exist')
else:
    tdr_basedir=basedir

filename=tdr_basedir+'/'+tdr_filename+'.json'
if args.verbose:
    print('will print TDR to',filename)
json_file = open(filename,"w")
json.dump(thermal_info,json_file)
json_file.close()

print('DONE')
