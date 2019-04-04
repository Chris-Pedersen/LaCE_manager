# start by running setup_simulation_suite.py
python /home/dc-font1/Codes/LyaCosmoParams/setup_simulations/scripts/setup_simulation_suite.py -c simulation_suite.config --basedir /home/dc-font1/rds/rds-dirac-dp096/tests_andreu/test_6 > info_setup_test_6 &

# for each simulation, generate matterpow and transfer files with CLASS (from the simulation folder), and generate submission scripts
python /home/dc-font1/Codes/LyaCosmoParams/setup_simulations/scripts/submit_simulations_darwin.py --basedir /home/dc-font1/rds/rds-dirac-dp096/tests_andreu/test_6 --verbose > info_scripts_test_6 &


