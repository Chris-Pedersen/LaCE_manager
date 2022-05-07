#!/bin/bash

source /data/desi/software/env/common/bin/activate

echo "args: " $@

python -u /nfs/pic.es/user/a/afontrib/Projects/LaCE/lace/postprocess/single_sim_scripts/extract_tdr_skewers.py $@

echo "after script"

deactivate
