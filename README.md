# Deep Anticipatory Networks

This repository contains implementation of the following paper: 

Yash Satsangi, Sungsu Lim, Shimon Whiteson, Frans A. Oliehoek, and Martha White. 2020. Maximizing Information Gain via Prediction Rewards. In Proc. of the 19th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2020), Auckland, New Zealand, May 9–13, 2020, IFAAMAS

# Sensor Selection

### Run training

`python3 main.py --result_dir OUTPUT_DIR --agent_json jsonfiles/AGENT_TYPE.json --index IDX` 

OUTPUT_DIR: indicates the directory in where the output will be saved.

AGENT_TYPE: should be one of the following -- 'dan', 'dan_coverage', 'coverage', 'random_policy'

IDX: indicates setting and run number based on the agent jsonfile

### Run multi-person tests

`python3 main_multitest.py --result_dir OUTPUT_DIR --model_dir INPUT_DIR --agent_json jsonfiles/AGENT_TYPE.json --index IDX --num_runs N`

### Plot scripts

`python3 plot_scripts/plot_comparison.py --result_dir INPUT_DIR`

`python3 plot_scripts/plot_multiperson_test.py --result_dir INPUT_DIR`
