#!/bin/bash

echo "Starting hydra_train.py test"

export HYDRA_FULL_ERROR=1

# The PROJECT_ROOT is where we are running this script from
PROJECT_ROOT=$(pwd) 
echo "Running from Project Root: $PROJECT_ROOT"
echo "Running 4 experiments"

# Execute the script by specifying its path from the root
# The working directory remains the project root, so Hydra will correctly find ./conf
uv run ${PROJECT_ROOT}/src/astra/pipelines/hydra_train.py 

echo "Script complete"