# FILE: debug_project.py
# Purpose: A small utility to inspect a single wandb project and debug filtering issues.

import wandb
import argparse
import pprint

def inspect_project(entity, project_name):
    """Connects to a W&B project and prints detailed info for each run."""
    
    print(f"\n{'='*20} Inspecting Project: {project_name} {'='*20}")
    
    api = wandb.Api()
    
    try:
        runs = api.runs(f"{entity}/{project_name}")
    except wandb.errors.CommError:
        print(f"--> ERROR: Project not found at '{entity}/{project_name}'")
        return

    print(f"Found {len(runs)} total runs. Iterating through them...\n")
    
    found_valid_runs = 0
    for i, run in enumerate(runs):
        print(f"--- Run #{i+1}: '{run.name}' (State: {run.state}) ---")
        
        # Print the entire summary dictionary to check for correct metric keys
        print("  Run Summary:")
        pprint.pprint(run.summary, indent=4)
        
        # Let's try to check a potential metric and its type
        metric_to_check = None
        for key in run.summary.keys():
            # Find the first key that looks like a Pearson score
            if "Pearson" in key:
                metric_to_check = key
                break
        
        if metric_to_check:
            metric_value = run.summary.get(metric_to_check)
            print(f"  -> Found potential metric '{metric_to_check}'")
            print(f"     Value: {metric_value}")
            print(f"     Type: {type(metric_value)}")
            if isinstance(metric_value, (int, float)):
                print("     ✅ Recognized as a valid number.")
                found_valid_runs += 1
            else:
                print("     ❌ NOT recognized as a standard number by `isinstance` check.")
        else:
            print("  -> No Pearson metric found in summary keys.")
            
        print("-" * 50)
        
    print(f"\nInspection complete. Found {found_valid_runs} runs that would pass the script's `isinstance` check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a W&B project for debugging.")
    parser.add_argument("--project", type=str, required=True, help="The name of the W&B project (e.g., astra-linear-single-task-kcat-hpo).")
    parser.add_argument("--entity", type=str, default="lmse-university-of-toronto", help="Your W&B entity.")
    args = parser.parse_args()
    
    inspect_project(args.entity, args.project)