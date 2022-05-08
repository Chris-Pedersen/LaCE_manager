The postprocessings are done in 4 steps. To postprocess all snapshots for an individual simulation, in the `/scripts` folder we run:

1. `python3 run_genpk_single.py`
2. `python3 compute_pressure_single.py`
3. `python3 run_skewers_single.py`
4. `pytohn3 run_p1d_single.py`

**BEFORE RUNNING!! Checklist:**

1. Each of these scripts has a variable called `pair_dir` which should be set to the target simulation pair directory (i.e. `/path/to/simulations/central_sim` in the case of the central simulation).
2. Each stage should be completed before running the next step, as many steps require the previous script to have finished running (not true for all but better to just wait to be safe).
3. It's also worth storing the output of each of these commands as a `.log` file
