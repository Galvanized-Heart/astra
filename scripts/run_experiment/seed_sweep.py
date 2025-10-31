# File: seed_sweep.py
import wandb
import argparse
import os # <-- 1. Import the 'os' module

def seed_new_sweep(project, entity, old_sweep_id, new_sweep_id, top_k=0, metric_name='valid_loss_epoch', goal='minimize'):
    """
    Seeds a new sweep with runs from an old sweep.
    If top_k > 0, seeds with the top K best performing runs.
    If top_k is 0 (default), seeds with ALL runs.
    """
    print(f"    -> Seeding new sweep '{new_sweep_id}' from old sweep '{old_sweep_id}'...")
    api = wandb.Api()

    try:
        old_sweep = api.sweep(old_sweep_id)
    except wandb.errors.CommError:
        print(f"    ERROR: Could not find old sweep: {old_sweep_id}")
        return

    runs_to_process = []
    if top_k > 0:
        is_minimize = goal == 'minimize'
        sort_reverse = not is_minimize
        sorted_runs = sorted(
            old_sweep.runs,
            key=lambda run: run.summary.get(metric_name, float('inf') if is_minimize else float('-inf')),
            reverse=sort_reverse
        )
        runs_to_process = sorted_runs[:top_k]
        print(f"    -> Found {len(old_sweep.runs)} total runs. Seeding with top {len(runs_to_process)}.")
    else:
        runs_to_process = old_sweep.runs
        print(f"    -> Seeding with all {len(runs_to_process)} runs.")

    if not runs_to_process:
        print("    -> No completed runs found in the old sweep. Nothing to seed.")
        return

    for i, old_run in enumerate(runs_to_process):
        final_metric = old_run.summary.get(metric_name)
        if old_run.state != "finished" or final_metric is None:
            print(f"      ({i+1}/{len(runs_to_process)}) Skipping run {old_run.name} (state: {old_run.state}, metric missing: {final_metric is None}).")
            continue
            
        print(f"      ({i+1}/{len(runs_to_process)}) Migrating history from run: {old_run.name}")
        
        # ** THE FIX IS HERE **
        # 2. Set the environment variable to the ID of the NEW sweep
        os.environ["WANDB_SWEEP_ID"] = new_sweep_id
        
        # 3. Call wandb.init() WITHOUT the 'sweep' argument
        seeder_run = wandb.init(
            project=project,
            entity=entity,
            job_type="seeding",
            name=f"seed_from_{old_run.name}",
            tags=["seeding_run"]
        )
        seeder_run.config.update(old_run.config, allow_val_change=True)
        seeder_run.log({metric_name: final_metric})
        seeder_run.finish()

        # Clean up the environment variable for the next loop iteration (good practice)
        del os.environ["WANDB_SWEEP_ID"]

    print(f"    -> Seeding complete for sweep '{new_sweep_id}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed a new W&B sweep from the runs of an old one.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--old_sweep_id", required=True)
    parser.add_argument("--new_sweep_id", required=True)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--metric_name", default="valid_loss_epoch")
    parser.add_argument("--goal", default="minimize", choices=["minimize", "maximize"])
    
    args = parser.parse_args()

    # The environment variable needs ONLY the ID, not the full path
    new_sweep_id_only = args.new_sweep_id.split('/')[-1]

    seed_new_sweep(
        args.project, 
        args.entity, 
        args.old_sweep_id, 
        new_sweep_id_only, # Pass just the short ID to the function
        args.top_k, 
        args.metric_name,
        args.goal
    )