import json
import subprocess
import os
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional



# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CV_RUN_CONFIGS_FILE = os.path.join(SCRIPT_DIR, "uncertainty_cv_configs.json")
LAST_SUBMITTED_INDEX_FILE = os.path.join(SCRIPT_DIR, "last_submitted_index_uncertainty.txt")
SBATCH_TEMPLATE_SCRIPT = os.path.join(SCRIPT_DIR, "cv_runner.sh")
SUBMISSION_BATCH_SIZE = 30
EXCLUDE_NODES = "fc10512" # Do not trust fc10512


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



# --- Slurm Command Builder ---
def build_sbatch_command(config: Dict[str, Any], config_hash: str) -> str:
    """Constructs the sbatch command with Hydra overrides and metadata."""
    if not os.path.exists(SBATCH_TEMPLATE_SCRIPT):
        raise FileNotFoundError(f"SBATCH template script not found: {SBATCH_TEMPLATE_SCRIPT}")

    hydra_overrides = build_hydra_overrides(config)
    job_name = config.get('wandb.group', 'cv_job') + f"-{config_hash[:6]}"

    command_parts = [
        "sbatch",
        f"--job-name={job_name}",
        f"--exclude={EXCLUDE_NODES}",
        SBATCH_TEMPLATE_SCRIPT,
        *hydra_overrides,
    ]
    return " ".join(command_parts)


def build_hydra_overrides(config: Dict[str, Any]) -> List[str]:
    hydra_overrides = []
    for key, value in config.items():
        if isinstance(value, list):
            items = ",".join(
                str(v).lower() if isinstance(v, bool)
                else str(v) if isinstance(v, (int, float)) or v is None
                else v
                for v in value
            )
            escaped_value = f"[{items}]"
        elif isinstance(value, bool) or value is None:
            escaped_value = str(value).lower() if isinstance(value, bool) else str(value)
        elif isinstance(value, (int, float)):
            escaped_value = str(value)
        elif isinstance(value, str) and ' ' not in value and '$' not in value and '{' not in value:
            escaped_value = value
        else:
            escaped_value = json.dumps(value)

        override = f"{key}={escaped_value}"
        if any(c in escaped_value for c in "()[]{}"):
            override = f'"{override}"'

        hydra_overrides.append(override)
    return hydra_overrides


def build_local_command(config: Dict[str, Any]) -> str:
    """Constructs a local uv run command, mirroring what cv_runner.sh does."""
    hydra_overrides = build_hydra_overrides(config)
    command_parts = [
        "uv", "run",
        "src/astra/pipelines/hydra_train.py",
        *hydra_overrides,
    ]
    return " ".join(command_parts)


def run_local(config: Dict[str, Any], config_index: int, dry_run: bool = False):
    """Runs a single config locally, optionally as a dry-run (print only)."""
    local_cmd = build_local_command(config)
    config_hash = get_config_hash(config)

    print(f"\n{'='*60}")
    print(f"LOCAL RUN — Config index {config_index} (hash: {config_hash[:6]})")
    print(f"{'='*60}")
    print(f"Command:\n  {local_cmd}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution. Command printed above.")
        return

    # Mirror the env setup from cv_runner.sh
    env = os.environ.copy()
    scratch_cache = ".cache"
    env.update({
        "SCRATCH_CACHE_DIR":  scratch_cache,
        "WANDB_DATA_DIR":     f"{scratch_cache}/wandb-data",
        "WANDB_CACHE_DIR":    f"{scratch_cache}/wandb",
        "WANDB_CONFIG_DIR":   f"{scratch_cache}/wandb-config",
        "WANDB_DIR":          f"{scratch_cache}/wandb-logs",
        "HF_HOME":            f"{scratch_cache}/huggingface",
        "HYDRA_FULL_ERROR":   "1",
        "HF_HUB_OFFLINE":     "1",
    })

    # Create cache dirs (mirrors mkdir -p in cv_runner.sh)
    for d in ["wandb-data", "wandb", "wandb-config", "wandb-logs", "huggingface"]:
        os.makedirs(os.path.join(scratch_cache, d), exist_ok=True)

    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Working directory: {os.getcwd()}\n")

    result = subprocess.run(local_cmd, shell=True, env=env)

    print(f"\nEnd time: {datetime.now().isoformat()}")
    if result.returncode != 0:
        print(f"[WARNING] Process exited with return code {result.returncode}")
    else:
        print("[SUCCESS] Local run completed.")


def main():
    parser = argparse.ArgumentParser(description="Submit CV jobs to SLURM or run locally for testing.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run a single config locally instead of submitting via sbatch.",
    )
    parser.add_argument(
        "--local-index",
        type=int,
        default=None,
        help="Index of the config to run locally (default: uses last submitted index). Implies --local.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the local command without executing it. Implies --local.",
    )
    args = parser.parse_args()

    # --dry-run and --local-index imply --local
    if args.dry_run or args.local_index is not None:
        args.local = True

    print(f"--- Starting CV Job Batch Submitter ({datetime.now().isoformat()}) ---")

    # Load all configs
    if not os.path.exists(CV_RUN_CONFIGS_FILE):
        print(f"Error: Config file not found at {CV_RUN_CONFIGS_FILE}")
        return

    with open(CV_RUN_CONFIGS_FILE, 'r') as f:
        all_configs: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(all_configs)} configurations from {CV_RUN_CONFIGS_FILE}")

    # --- LOCAL / DRY-RUN path ---
    if args.local:
        index = args.local_index if args.local_index is not None else load_last_submitted_index()
        if index >= len(all_configs):
            print(f"Error: config index {index} is out of range (0–{len(all_configs)-1}).")
            return
        run_local(all_configs[index], config_index=index, dry_run=args.dry_run)
        return

    # --- Normal SLURM submission path ---
    current_index = load_last_submitted_index()

    if current_index >= len(all_configs):
        print("All configurations have already been submitted. Exiting.")
        return

    start_index = current_index
    end_index = min(start_index + SUBMISSION_BATCH_SIZE, len(all_configs))

    print(f"Preparing to submit jobs from index {start_index} to {end_index} (total configs: {len(all_configs)}).")

    submitted_count = 0
    for i in range(start_index, end_index):
        current_config = all_configs[i]
        current_config_hash = get_config_hash(current_config)

        try:
            sbatch_cmd = build_sbatch_command(current_config, current_config_hash)
            print(f"  Submitting job {i+1}/{len(all_configs)} (Rank {current_config.get('wandb.group', '').split('-top')[-1]}): {sbatch_cmd[:200]}...")

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

        current_index = i + 1

    save_last_submitted_index(current_index)

    print(f"\n--- Batch submission finished ---")
    print(f"Submitted {submitted_count} new jobs in this batch.")
    print(f"Next batch will start from index {current_index}.")
    if current_index >= len(all_configs):
        print("All configurations have now been submitted.")

if __name__ == "__main__":
    main()