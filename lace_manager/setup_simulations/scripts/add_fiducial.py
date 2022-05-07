import json


'''
Script to add a fiducial model at the centre of the
Latin hypercube to the simulation suite.
'''

## Read original Latin hypercube
lh="/home/chris/Projects/lace_manager/p1d_emulator/sim_suites/Australia20"
with open(lh+"/latin_hypercube.json") as json_file:
    cube=json.load(json_file)

## Ordered list
parlist=['Delta2_star','n_star','heat_amp','heat_slo','z_rei']

central_values=[]

## Append central value
for param in parlist:
    midpoint=0.5*(cube["param_space"][param]["max_val"]-cube["param_space"][param]["min_val"])+cube["param_space"][param]["min_val"]
    print("Mid value for %s is %.2f" % (param,midpoint))
    central_values.append(midpoint)

print(central_values)

## Append number of samples and 
cube["nsamples"]=cube["nsamples"]+1
cube["samples"][str(cube["nsamples"]-1)]=central_values ## -1 as we index from 0

## Now write this
with open(lh+"/latin_hypercube_fid.json", "w") as outfile:
    json.dump(cube, outfile)

