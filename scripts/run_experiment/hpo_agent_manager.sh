#!/bin/bash
set -e

# ==============================================================================
# HPO Campaign Manager for Sequential, 1-Agent-per-Wave Sweeps
#
# Modes:
#   - init:   Run once to create all sweeps and the state file.
#   - run:    Submit as a Slurm job to run ONE trial for each active sweep.
#   - status: Check the current progress of all sweeps.
# ==============================================================================

# --- Core Configuration ---
PROJ_DIR="/gpfs/fs0/scratch/m/mahadeva/maxkirby/astra"
STATE_FILE="sweep_state.csv"
PROJECT_NAME="astra"
TOTAL_TRIALS_PER_SWEEP=30

cd ${PROJ_DIR}

# --- Experiment Definitions ---
ARCHITECTURES=(
  "linear" "cpi_pred_conv" "cpi_pred_self_attn" "cpi_pred_cross_attn"
)
EXPERIMENT_MODES=(
  "multi_task/advanced" "multi_task/basic" "multi_task/direct"
  "single_task/kcat_only" "single_task/ki_only" "single_task/km_only"
)

# ==============================================================================
#   MODE 1: INITIALIZE THE CAMPAIGN
# ==============================================================================
function init_campaign() {
    if [ -f "$STATE_FILE" ]; then
        echo "ERROR: State file '$STATE_FILE' already exists." >&2
        echo "Delete it manually if you want to start a new campaign." >&2
        exit 1
    fi

    echo "### Initializing HPO Campaign ###"
    echo "Creating state file: $STATE_FILE"
    echo "architecture,experiment_mode,sweep_id,completed_counts,total_counts" > "$STATE_FILE"

    for mode in "${EXPERIMENT_MODES[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
            echo "--- Creating sweep for arch='${arch}', mode='${mode}' ---"
            MODE_SAFE_NAME=$(echo $mode | tr '/' '-')
            
            TEMP_SWEEP_CONFIG="temp_sweep_${MODE_SAFE_NAME}_${arch}.yaml"
            cat <<EOF > ${TEMP_SWEEP_CONFIG}
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
            cat "conf/hpo/${arch}.yaml" >> ${TEMP_SWEEP_CONFIG}
            
            SWEEP_NAME="hpo-seq1-${MODE_SAFE_NAME}-${arch}"
            SWEEP_OUTPUT=$(wandb sweep --name "$SWEEP_NAME" --project "$PROJECT_NAME" "${TEMP_SWEEP_CONFIG}" 2>&1)
            SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | awk '{print $NF}')

            if [ -z "$SWEEP_ID" ]; then
                echo "ERROR: Failed to create sweep for ${arch}/${mode}" >&2
                exit 1
            fi

            echo "${arch},${mode},${SWEEP_ID},0,${TOTAL_TRIALS_PER_SWEEP}" >> "$STATE_FILE"
            rm ${TEMP_SWEEP_CONFIG}
        done
    done
    echo "### Campaign Initialized Successfully! ###"
}

# ==============================================================================
#   MODE 2: RUN ONE WAVE OF HPO
# ==============================================================================
function run_campaign() {
    # Accept the wave_count from the command line, defaulting to 1.
    local wave_count=${1:-1}

    if ! [[ "$wave_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: Count must be a positive integer. Got '$wave_count'." >&2; exit 1
    fi
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: State file '$STATE_FILE' not found. Did you run 'init'?" >&2
        exit 1
    fi

    echo "### Running a wave of up to ${wave_count} trial(s) per active sweep. ###"
    JOB_IDS_IN_WAVE=()
    
    declare -A counts_to_launch

    # Read the state file and launch one job for each pending sweep
    # We use a process substitution to pipe into the while loop
    while IFS=, read -r arch mode sweep_id completed total; do
        if (( completed >= total )); then
            continue # Skip completed sweeps
        fi

        # Logic to calculate the correct number of trials for this specific agent.
        local counts_still_needed=$(( total - completed ))
        local counts_this_wave=$(( wave_count < counts_still_needed ? wave_count : counts_still_needed ))

        # Create safe names for the job name.
        local safe_arch_name=$(echo "$arch" | tr '/' '-')
        local safe_mode_name=$(echo "$mode" | tr '/' '-')

        echo "LAUNCHING: ${arch}/${mode} | Agent for ${counts_this_wave} trial(s). (${completed}/${total} completed)"
        
        # The command to run is just a single wandb agent. Hydra is NOT used here
        # because we are only launching one run, not a multirun.
        # The 'command' block in the sweep config handles calling hydra_train.py.
        # Our slurm launcher will wrap this with 'uv run'.
        JOB_ID=$(sbatch \
            --job-name="hpo-${safe_arch_name}-${safe_mode_name}" \
            --output="slurm_logs/agent_worker-%j.out" \
            --gpus-per-node=1 \
            --time=16:00:00 \
            scripts/run_experiment/hpo_agent_runner.sh "${sweep_id}" "${counts_this_wave}" | awk '{print $4}')

        if [ -z "$JOB_ID" ]; then echo "FATAL: sbatch command failed for ${arch}/${mode}. Aborting."; exit 1; fi

        JOB_IDS_IN_WAVE+=($JOB_ID)

        # Store the launched count for later.
        counts_to_launch["$sweep_id"]=$counts_this_wave

    done < <(tail -n +2 "$STATE_FILE")

    if [ ${#JOB_IDS_IN_WAVE[@]} -eq 0 ]; then
        echo "All sweeps are complete. Nothing to run."
        exit 0
    fi

    # Wait for all jobs in this wave to finish
    echo "--- Waiting for ${#JOB_IDS_IN_WAVE[@]} jobs to complete... ---"
    JOB_IDS_STR=$(IFS=,; echo "${JOB_IDS_IN_WAVE[*]}")
    while squeue -j ${JOB_IDS_STR} 2>/dev/null | grep -q -E "$(IFS=|; echo "${JOB_IDS_IN_WAVE[*]}")"; do
        sleep 60
        echo -n "."
    done
    echo -e "\n--- Wave Complete. Updating state file. ---"

    # Update the state file atomically
    TMP_STATE_FILE=$(mktemp)
    (
        head -n 1 "$STATE_FILE" # Copy header
        # Read the old state and write the new state
        tail -n +2 "$STATE_FILE" | while IFS=, read -r arch mode sweep_id completed total; do
            if (( completed < total )); then
                completed=$(( completed + 1 ))
            fi
            echo "${arch},${mode},${sweep_id},${completed},${total}"
        done
    ) > "$TMP_STATE_FILE"
    
    mv "$TMP_STATE_FILE" "$STATE_FILE"
    echo "### State file updated. This 'run' job is now complete. ###"
}

# ==============================================================================
#   MODE 3: DISPLAY CAMPAIGN STATUS
# ==============================================================================
function show_status() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: State file '$STATE_FILE' not found. Did you run 'init'?" >&2
        exit 1
    fi
    echo "--- HPO Campaign Status ---"
    column -t -s, "$STATE_FILE"
}

# ==============================================================================
#   MAIN SCRIPT LOGIC (COMMAND DISPATCHER)
# ==============================================================================
MODE=$1
if [ -z "$MODE" ]; then
    echo "Usage: $0 {init|run|status}" >&2
    exit 1
fi

case $MODE in
    init) init_campaign ;;
    run) run_campaign "$2" ;;
    status) show_status ;;
    *)
        echo "Invalid mode: '$MODE'" >&2
        echo "Usage: $0 {init|run|status}" >&2
        exit 1
        ;;
esac