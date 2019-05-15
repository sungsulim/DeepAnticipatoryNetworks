# DeepAnticipatoryNetworks


### Running the Code
from Master branch:

`python3 main.py --result_dir NAME_OF_TEST --agent_type AGENT_TYPE --random_seed INTEGER_SEED` 

from Sweep branch:

`python3 main.py --result_dir NAME_OF_TEST --agent_json jsonfiles/AGENT_TYPE.json --index INTEGER`


NAME_OF_TEST: indicates the directory in ./results/ where the output will be saved.

AGENT_TYPE: Should be one of the following -- 'dan', 'dan_coverage', 'coverage', 'randomAction'
