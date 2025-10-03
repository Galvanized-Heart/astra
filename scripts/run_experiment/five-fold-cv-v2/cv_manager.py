import json
import subprocess
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional



# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CV_RUN_CONFIGS_FILE = os.path.join(SCRIPT_DIR, "cv_run_configs.json")
LAST_SUBMITTED_INDEX_FILE = os.path.join(SCRIPT_DIR, "last_submitted_index.txt")
SBATCH_TEMPLATE_SCRIPT = os.path.join(SCRIPT_DIR, "cv_runner.sh")
SUBMISSION_BATCH_SIZE = 1



# --- Hash for job naming ---
def get_config_hash(config: Dict[str, Any]) -> str:
    """Convert dict to a sorted JSON string to ensure consistent hashing."""
    sorted_json = json.dumps(config, sort_keys=True)
    return hashlib.md5(sorted_json.encode('utf-8')).hexdigest()



# --- Index Persistence ---
def load_last_submitted_index() -> int:
    """Loads the index of the last config that was submitted."""
    if os.path.exists(LAST_SUBMITTED_INDEX_FILE):
        with open(LAST_SUBMITTED_INDEX_FILE, 'r') as f:
            try:
                index = int(f.read().strip())
                print(f"Loaded last submitted index: {index}")
                return index
            except ValueError:
                print(f"Warning: Invalid content in {LAST_SUBMITTED_INDEX_FILE}. Starting from 0.")
                return 0
    else:
        print(f"No {LAST_SUBMITTED_INDEX_FILE} found. Starting from index 0.")
        return 0

def save_last_submitted_index(index: int):
    """Saves the index of the last config that was submitted."""
    with open(LAST_SUBMITTED_INDEX_FILE, 'w') as f:
        f.write(str(index))
    # print(f"Saved last submitted index {index} to {LAST_SUBMITTED_INDEX_FILE}")



# --- Slurm Command Builder ---
def build_sbatch_command(config: Dict[str, Any], config_hash: str) -> str:
    """Constructs the sbatch command with Hydra overrides and metadata."""
    if not os.path.exists(SBATCH_TEMPLATE_SCRIPT):
        raise FileNotFoundError(f"SBATCH template script not found: {SBATCH_TEMPLATE_SCRIPT}")

    hydra_overrides = []
    for key, value in config.items():
        # Hydra typically expects `key=value`. If value contains spaces or special chars, it should be quoted.
        # json.dumps handles quoting for strings and provides correct representation for other types.
        escaped_value = json.dumps(value) 
        # Remove outer quotes if value is a simple string, number, boolean, or null, so Hydra parses it directly.
        # Hydra handles `key="string with spaces"` fine, but `key=123` is preferred over `key="123"`.
        if isinstance(value, (int, float, bool)) or value is None:
             escaped_value = str(value).lower() if isinstance(value, bool) else str(value)
        elif isinstance(value, str) and ' ' not in value and '$' not in value and '{' not in value: # Simple string
             escaped_value = value
        
        # We need to ensure Hydra will parse this correctly. For example, if 'wandb.group'
        # is a string with spaces, passing `wandb.group="my group"` is fine.
        # json.dumps usually wraps strings in double quotes, which Hydra handles.
        hydra_overrides.append(f"{key}={escaped_value}")
            
    # Use config.get for safe access to optional keys
    job_name = config.get('wandb.group', 'cv_job') + f"-{config_hash[:6]}" # Append short hash for uniqueness

    command_parts = [
        "sbatch",
        f"--job-name={job_name}",
        "--output=slurm_logs/cv_worker-%j.out",
        "--error=slurm_logs/agent_worker-%j.err",
        "--gpus-per-node=1",
        "--time=12:00:00",
        SBATCH_TEMPLATE_SCRIPT,
        *hydra_overrides, 
    ]
    return " ".join(command_parts)



def main():
    print(f"--- Starting CV Job Batch Submitter ({datetime.now().isoformat()}) ---")
    
    # Load all configs
    if not os.path.exists(CV_RUN_CONFIGS_FILE):
        print(f"Error: Config file not found at {CV_RUN_CONFIGS_FILE}")
        return

    with open(CV_RUN_CONFIGS_FILE, 'r') as f:
        all_configs: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(all_configs)} configurations from {CV_RUN_CONFIGS_FILE}")

    # Load starting index
    current_index = load_last_submitted_index()

    if current_index >= len(all_configs):
        print("All configurations have already been submitted. Exiting.")
        return

    # Determine the range of jobs to submit in this batch
    start_index = current_index
    end_index = min(start_index + SUBMISSION_BATCH_SIZE, len(all_configs))

    print(f"Preparing to submit jobs from index {start_index} to {end_index - 1} (total configs: {len(all_configs)}).")

    submitted_count = 0
    for i in range(start_index, end_index):
        current_config = all_configs[i]
        current_config_hash = get_config_hash(current_config) # Get hash for job naming

        try:
            sbatch_cmd = build_sbatch_command(current_config, current_config_hash)
            print(f"  Submitting job {i+1}/{len(all_configs)} (Rank {current_config.get('wandb.group', '').split('-top')[-1]}): {sbatch_cmd[:200]}...") # Truncate for display
            
            # Execute sbatch command
            result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                job_id = output.split("Submitted batch job ")[1].split(' ')[0]
                print(f"  --> Successfully submitted SLURM job ID: {job_id}")
                submitted_count += 1
            else:
                print(f"  Error submitting job for config index {i}: Unexpected sbatch output: {output}")
                print(f"  Stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"  Error submitting job for config index {i}: {e}")
            print(f"  Stderr: {e.stderr}")
        except Exception as e:
            print(f"  An unexpected error occurred for config index {i}: {e}")
        
        current_index = i + 1 # Update index after each successful or attempted submission

    # Save the new last_submitted_index
    save_last_submitted_index(current_index)

    print(f"\n--- Batch submission finished ---")
    print(f"Submitted {submitted_count} new jobs in this batch.")
    print(f"Next batch will start from index {current_index}.")
    if current_index >= len(all_configs):
        print("All configurations have now been submitted.")

if __name__ == "__main__":
    main()