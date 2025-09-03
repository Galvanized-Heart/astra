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
            SWEEP_OUTPUT=$(wandb sweep --name "$SWEEP_NAME" --project "$PROJECT_NAME" --entity "lmse-university-of-toronto" "${TEMP_SWEEP_CONFIG}" 2>&1)
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
#   MODE 2: RUN ONE WAVE OF HPO (FIRE-AND-FORGET VERSION)
# ==============================================================================
function run_campaign() {
    local wave_count=${1:-1}

    if ! [[ "$wave_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: Count must be a positive integer. Got '$wave_count'." >&2; exit 1
    fi
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: State file '$STATE_FILE' not found. Did you run 'init'?" >&2
        exit 1
    fi

    echo "### Submitting a wave of up to ${wave_count} trial(s) per active sweep. ###"
    mkdir -p slurm_logs

    # This file will track the jobs submitted in this wave for the reconcile command.
    local wave_log_file="last_wave.log"
    # Start with a clean slate for this wave
    > "$wave_log_file"
    echo "job_id,sweep_id,trials_launched" > "$wave_log_file"

    while IFS=, read -r arch mode sweep_id completed total; do
        if (( completed >= total )); then continue; fi

        local counts_still_needed=$(( total - completed ))
        local counts_this_wave=$(( wave_count < counts_still_needed ? wave_count : counts_still_needed ))
        local safe_arch_name=$(echo "$arch" | tr '/' '-')
        local safe_mode_name=$(echo "$mode" | tr '/' '-')
        local log_out_path="slurm_logs/agent_worker-%j.out"
        local log_err_path="slurm_logs/agent_worker-%j.err"

        echo "SUBMITTING: ${arch}/${mode} | Agent for ${counts_this_wave} trial(s)."
        
        JOB_ID=$(sbatch \
        --job-name="hpo-${safe_arch_name}-${safe_mode_name}" \
        --output="$log_out_path" \
        --error="$log_err_path" \
        --gpus-per-node=1 \
        --time=10:00:00 \
        --exclusive \
        scripts/run_experiment/hpo_agent_runner.sh "${sweep_id}" "${counts_this_wave}" | awk '{print $4}')

        if [ -z "$JOB_ID" ] || ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            echo "FATAL: sbatch command failed for ${arch}/${mode}. Aborting."; exit 1;
        fi

        # CRITICAL: Log the job mapping for the reconcile command to use later.
        echo "${JOB_ID},${sweep_id},${counts_this_wave}" >> "$wave_log_file"

    done < <(tail -n +2 "$STATE_FILE")

    echo "All jobs for this wave have been submitted. Track progress with 'squeue'."
    echo "Once finished, run '$0 reconcile' to verify results and update state."
}

# ==============================================================================
#   MODE 3: RECONCILE AND VERIFY THE LAST WAVE
# ==============================================================================
function reconcile_campaign() {
    local wave_log_file="last_wave.log"
    if [ ! -f "$wave_log_file" ]; then
        echo "ERROR: Log file '$wave_log_file' not found. Did you run 'run' first?" >&2
        exit 1
    fi

    echo "### Reconciling jobs from '$wave_log_file' ###"

    # Read job data from the log file
    declare -A JOB_TO_SWEEP
    declare -A JOB_TO_COUNT
    JOB_IDS_IN_WAVE=()
    while IFS=, read -r job_id sweep_id trials_launched; do
        JOB_IDS_IN_WAVE+=($job_id)
        JOB_TO_SWEEP["$job_id"]="$sweep_id"
        JOB_TO_COUNT["$job_id"]="$trials_launched"
    done < <(tail -n +2 "$wave_log_file")

    echo "Waiting for ${#JOB_IDS_IN_WAVE[@]} jobs to complete before verification..."
    for job_id in "${JOB_IDS_IN_WAVE[@]}"; do
        while scontrol show job "$job_id" &>/dev/null; do
            echo "Job ${job_id} is still running or in queue. Waiting 60s..."
            sleep 60
        done
    done
    
    echo -e "\nAll jobs complete. Verifying job success..."

    declare -A successful_counts
    local success_sentinel="Agent worker finished successfully"

    for job_id in "${JOB_IDS_IN_WAVE[@]}"; do
        local log_file="slurm_logs/agent_worker-${job_id}.out"
        local job_state=$(sacct -j "${job_id}" -n -o State --noheader | head -n 1 | awk '{print $1}')
        local exit_code=$(sacct -j "${job_id}.batch" -n -o ExitCode --noheader | head -n 1)
        local sweep_id=${JOB_TO_SWEEP[$job_id]}
        local trials_launched=${JOB_TO_COUNT[$job_id]}
        
        # The robust triple-check
        if [[ "$job_state" == "COMPLETED" ]] && [[ "$exit_code" == "0:0" ]] && [ -f "$log_file" ] && grep -q "$success_sentinel" "$log_file"; then
            echo "VERIFIED: Job ${job_id} for sweep ${sweep_id} was successful."
            successful_counts["$sweep_id"]=$(( ${successful_counts["$sweep_id"]:-0} + trials_launched ))
        else
            echo "FAILURE: Job ${job_id} for sweep ${sweep_id} did NOT verify."
            echo "         - Slurm State: '${job_state}' | Exit Code: '${exit_code}'"
        fi
    done

    echo "### Updating state file with verified results... ###"
    TMP_STATE_FILE=$(mktemp)
    (
        head -n 1 "$STATE_FILE"
        tail -n +2 "$STATE_FILE" | while IFS=, read -r arch mode sweep_id completed total; do
            if [[ -v successful_counts["$sweep_id"] ]]; then
                completed=$(( completed + ${successful_counts["$sweep_id"]} ))
            fi
            echo "${arch},${mode},${sweep_id},${completed},${total}"
        done
    ) > "$TMP_STATE_FILE"
    
    mv "$TMP_STATE_FILE" "$STATE_FILE"
    # Archive the log file to prevent re-reconciling
    mv "$wave_log_file" "${wave_log_file}.$(date +%Y%m%d-%H%M%S).processed"
    echo "### Reconciliation complete. State file is updated. ###"
}


# ==============================================================================
#   MODE 4: DISPLAY CAMPAIGN STATUS
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
    echo "Usage: $0 {init|run|status|reconcile}" >&2
    exit 1
fi

case $MODE in
    init) init_campaign ;;
    run) run_campaign "$2" ;;
    reconcile) reconcile_campaign ;;
    status) show_status ;;
    *)
        echo "Invalid mode: '$MODE'" >&2
        echo "Usage: $0 {init|run|status|reconcile}" >&2
        exit 1
        ;;
esac