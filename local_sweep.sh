#!/bin/bash

AGENT_NAME=$1

#source /Users/maclapuser/Documents/DeepAnticipatoryNetworks/venv/bin/activate
source venv/bin/activate

start_idx=$2
increment=$3
end_idx=$4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for i in $(seq ${start_idx} ${increment} ${end_idx})
do
  echo Running...$i
  python3 main.py --result_dir validation_sweep --agent_json jsonfiles/$AGENT_NAME.json --index $i
done
