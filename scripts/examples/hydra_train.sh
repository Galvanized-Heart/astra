#!/bin/bash

echo "Starting hydra_train.py test"

export HYDRA_FULL_ERROR=1

# The PROJECT_ROOT is where we are running this script from
PROJECT_ROOT=$(pwd) 
echo "Running from Project Root: $PROJECT_ROOT"
echo "Running 4 experiments"

# Execute the script by specifying its path from the root
# The working directory remains the project root, so Hydra will correctly find ./conf
uv run ${PROJECT_ROOT}/src/astra/pipelines/hydra_train.py experiment_mode=multi_task/advanced \
    data=hpo \
    trainer.epochs=2 \
    architecture.params.conv_out_dim=64 \
    architecture.params.dropout=0.11083906416651734 \
    architecture.params.hid_dim=256 \
    architecture.params.kernal_1=3 \
    architecture.params.kernal_2=3 \
    architecture.params.last_hid=32 \
    model.lightning_module.loss_weights.w_kcat_logit=2 \
    model.lightning_module.loss_weights.w_km_logit=2 \
    model.lightning_module.loss_weights.w_ki_logit=2 \
    model.lightning_module.lr=0.0002

    #--multirun \
    #multi_task/advanced,multi_task/basic,multi_task/direct,single_task/kcat_only,single_task/ki_only,single_task/km_only

echo "Script complete"