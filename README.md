# DeepAnticipatoryNetworks

## Sensor Selection Experiments

### Run training

`python3 main.py --result_dir OUTPUT_DIR --agent_json jsonfiles/AGENT_TYPE.json --index IDX` 

OUTPUT_DIR: indicates the directory in where the output will be saved.

AGENT_TYPE: should be one of the following -- 'dan', 'dan_coverage', 'coverage', 'randomAction'

IDX: indicates setting and run number based on the agent jsonfile

### Run multi-person tests

`python3 main_multitest.py --result_dir OUTPUT_DIR --model_dir INPUT_DIR --agent_json jsonfiles/AGENT_TYPE.json --index IDX --num_runs N`

### Plot scripts

`python3 plot_scripts/plot_comparison.py --result_dir INPUT_DIR`

`python3 plot_scripts/plot_multiperson_test.py --result_dir INPUT_DIR`
