# FILE: scripts/maintenance/find_missing_runs.py
import wandb
import json
from tqdm import tqdm
from astra.constants import PROJECT_ROOT

ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"
MASTER_TAG = "final-cv"

def main():
    """
    Compares the master run list with the actual runs logged to wandb
    to identify any missing runs that failed or were not submitted.
    """
    
    # 1. Load the Ground Truth (our master list)
    master_list_path = PROJECT_ROOT / "scripts/run_experiment/five-fold-cv/master_run_list.json"
    try:
        with open(master_list_path, 'r') as f:
            expected_runs_data = json.load(f)
        print(f"Loaded {len(expected_runs_data)} expected run definitions from the master list.")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{master_list_path}'. Aborting.")
        return

    # 2. Fetch the Reality (what's on wandb)
    print(f"Fetching all runs with tag '{MASTER_TAG}' from project '{ENTITY}/{PROJECT}'...")
    api = wandb.Api()
    actual_runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": MASTER_TAG})
    print(f"Found {len(actual_runs)} actual runs on wandb.")

    # 3. Create Unique Identifiers for both sets
    
    # Expected runs
    expected_run_ids = set()
    for run_def in expected_runs_data:
        # The unique ID is a combination of the group and the specific fold name
        unique_id = f"{run_def['group_name']}_{run_def['run_name']}"
        expected_run_ids.add(unique_id)
        
    # Actual runs
    actual_run_ids = set()
    for run in tqdm(actual_runs, desc="Processing actual runs"):
        unique_id = f"{run.group}_{run.name.split('_')[0]}" # We split to remove the model/timestamp part for comparison
        # Let's refine this to be more robust, in case a run name is just 'fold-0'
        # The run name in the JSON is just 'fold-X', but the actual run name has model/timestamp.
        # The most reliable unique key is (group, fold_index).
        
    # --- A MORE ROBUST WAY ---
    # We will build a set of tuples: (group_name, fold_index)
    
    # Expected runs
    expected_set = set()
    for run_def in expected_runs_data:
        key = (run_def['group_name'], run_def['fold_index'])
        expected_set.add(key)
        
    # Actual runs
    actual_set = set()
    for run in tqdm(actual_runs, desc="Processing actual runs"):
        # The fold index is one of the tags, e.g., 'fold_0'
        fold_index = -1
        for tag in run.tags:
            if tag.startswith("fold_"):
                try:
                    fold_index = int(tag.split('_')[1])
                    break
                except (ValueError, IndexError):
                    continue # Ignore malformed tags
        
        if fold_index != -1:
            key = (run.group, fold_index)
            actual_set.add(key)

    # 4. Compare the sets to find what's missing
    missing_run_keys = expected_set - actual_set
    
    if not missing_run_keys:
        print("\nSUCCESS: All 600 runs are present and accounted for!")
        return
        
    # 5. Report the missing runs
    print(f"\n--- Found {len(missing_run_keys)} Missing Runs ---")
    
    missing_run_definitions = []
    for run_def in expected_runs_data:
        key = (run_def['group_name'], run_def['fold_index'])
        if key in missing_run_keys:
            missing_run_definitions.append(run_def)

    # Sort for clarity
    missing_run_definitions.sort(key=lambda x: x['run_id'])

    for missing_run in missing_run_definitions:
        print(f"  - Missing Run ID: {missing_run['run_id']}, Group: {missing_run['group_name']}, Fold: {missing_run['fold_index']}")

    # Create a list of just the missing run IDs to make resubmission easy
    missing_ids = [run['run_id'] for run in missing_run_definitions]
    print("\n--- To resubmit these specific jobs, use the following SLURM command ---")
    print(f"sbatch --array={','.join(map(str, missing_ids))} jobs/five-fold-cv-sweep/submit_cv_array_job.sbatch")


if __name__ == "__main__":
    main()