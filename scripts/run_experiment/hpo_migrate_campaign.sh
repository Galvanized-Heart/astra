#!/bin/bash

# --- Configuration ---
PROJECT_NAME="astra"
ENTITY_NAME="lmse-university-of-toronto"

OLD_STATE_FILE="sweep_state.csv.bak"
NEW_STATE_FILE="sweep_state.csv"
# --- End Configuration ---

if [ ! -f "$OLD_STATE_FILE" ]; then
    echo "ERROR: Backup state file '$OLD_STATE_FILE' not found." >&2
    exit 1
fi

echo "### Migrating & Seeding HPO Campaign (Using ALL runs) ###"
echo "Reading from: $OLD_STATE_FILE"
echo "Writing to:   $NEW_STATE_FILE"
echo ""

# Create the header for the new state file
echo "architecture,experiment_mode,sweep_id,completed_counts,total_counts" > "$NEW_STATE_FILE"

# Read the old state file line by line, skipping the header
tail -n +2 "$OLD_STATE_FILE" | while IFS=, read -r arch mode old_sweep_id completed_counts total_counts; do
    
    echo "--- Migrating sweep for arch='${arch}', mode='${mode}' ---"
    echo "    Old Sweep ID: ${old_sweep_id}"

    # ** THE FIRST FIX IS HERE: Define the variable before using it **
    ARCH_CONFIG_FILE="conf/hpo/${arch}.yaml"
    if [ ! -f "$ARCH_CONFIG_FILE" ]; then
        echo "ERROR: Architecture config file not found: $ARCH_CONFIG_FILE. Skipping." >&2
        continue
    fi

    MODE_SAFE_NAME=$(echo "$mode" | tr '/' '-')
    TEMP_SWEEP_CONFIG="temp_sweep_migrated_${MODE_SAFE_NAME}_${arch}.yaml"

    # 1. CREATE NEW SWEEP CONFIGURATION
    # ** THE SECOND FIX IS HERE: The common parameters block was restored **
    cat <<EOF > "${TEMP_SWEEP_CONFIG}"
program: src/astra/pipelines/wandb_hydra_launcher.py
method: bayes
metric: {name: valid_loss_epoch, goal: minimize}
command:
  - \${env}
  - uv
  - run
  - \${program}
  - src/astra/pipelines/hydra_train.py
  - architecture=${arch}
  - experiment_mode=${mode}
  - \${args}
EOF
    # Use this to append new params in the temp config above  and include `parameters:` key at same level as command
    #grep -v "^parameters:" "${ARCH_CONFIG_FILE}" >> "${TEMP_SWEEP_CONFIG}"

    # Use this if all params are defined by conf/hpo/arch.yaml
    cat "${ARCH_CONFIG_FILE}" >> "${TEMP_SWEEP_CONFIG}"

    # 2. CREATE THE NEW SWEEP ON W&B
    SWEEP_NAME="hpo-seq1-${MODE_SAFE_NAME}-${arch}-v2"
    SWEEP_OUTPUT=$(uv run wandb sweep --name "$SWEEP_NAME" --project "$PROJECT_NAME" --entity "$ENTITY_NAME" "${TEMP_SWEEP_CONFIG}" 2>&1)
    NEW_SWEEP_ID_FULL_PATH=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | awk '{print $NF}')

    if [ -z "$NEW_SWEEP_ID_FULL_PATH" ]; then
        echo "ERROR: Failed to create new sweep for ${arch}/${mode}" >&2
        echo "--- W&B CLI OUTPUT ---"
        echo "$SWEEP_OUTPUT"
        echo "------------------------"
        rm "${TEMP_SWEEP_CONFIG}"
        continue
    fi
    rm "${TEMP_SWEEP_CONFIG}"
    echo "    New Sweep ID: ${NEW_SWEEP_ID_FULL_PATH}"

    # 3. MIGRATE HISTORY / SEED THE NEW SWEEP
    #uv run python scripts/run_experiment/seed_sweep.py \
    #    --project "$PROJECT_NAME" \
    #    --entity "$ENTITY_NAME" \
    #    --old_sweep_id "$old_sweep_id" \
    #    --new_sweep_id "$NEW_SWEEP_ID_FULL_PATH"

    # 4. WRITE THE NEW SWEEP INFO TO THE STATE FILE
    echo "${arch},${mode},${NEW_SWEEP_ID_FULL_PATH},0,${total_counts}" >> "$NEW_STATE_FILE"

    # 5. STOP THE OLD SWEEP
    echo "    Stopping old sweep..."
    wandb sweep --stop "$old_sweep_id" > /dev/null 2>&1
    echo "    Done."
    echo ""
done

echo "### Campaign Migration & Seeding Successful! ###"
echo "You can now start your agents using the new sweeps."