import wandb
from tqdm import tqdm

ENTITY = "lmse-university-of-toronto"
PROJECT = "astra"
MASTER_TAG = "final-cv" # The tag shared by all runs in this experiment
TARGET_TAG = "multi_task" # We only want to modify multi-task runs
TAG_TO_ADD = "target_KI" # The tag we need to add

def main():
    """
    Scans for completed runs and adds a specific tag if they match certain criteria.
    Specifically, it finds 'multi_task' runs and adds the 'target_KI' tag to them,
    as they were all selected based on Ki performance in the first wave.
    """
    api = wandb.Api()
    
    # 1. Find all runs that are part of our big CV experiment.
    print(f"Fetching all runs with tag '{MASTER_TAG}' from project '{ENTITY}/{PROJECT}'...")
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": MASTER_TAG})
    print(f"Found {len(runs)} runs to inspect.")
    
    runs_to_update = []
    
    # 2. Inspect each run to see if it needs updating.
    for run in tqdm(runs, desc="Inspecting runs"):
        current_tags = set(run.tags)
        
        # Check if this is a multi-task run AND if it's missing our new tag
        if TARGET_TAG in current_tags and TAG_TO_ADD not in current_tags:
            # It's a multi-task run that needs to be updated.
            runs_to_update.append(run)

    if not runs_to_update:
        print(f"\nNo runs found that require the '{TAG_TO_ADD}' tag. The operation is complete or was not needed.")
        return

    # 3. Ask for user confirmation before making any changes.
    print(f"\nFound {len(runs_to_update)} runs that match the criteria and are missing the '{TAG_TO_ADD}' tag.")
    print("Example runs that will be updated:")
    for run in runs_to_update[:5]:
        print(f"  - Group: {run.group}, Name: {run.name}")
        
    user_input = input(f"\nDo you want to proceed with adding the '{TAG_TO_ADD}' tag to these {len(runs_to_update)} runs? (yes/no): ")
    
    if user_input.lower() != 'yes':
        print("Operation cancelled by user.")
        return

    # 4. Update the runs.
    print("\nApplying tags...")
    for run in tqdm(runs_to_update, desc="Updating runs"):
        run.tags.append(TAG_TO_ADD)
        run.update() # This saves the changes to the wandb server.
        
    print(f"\nSuccessfully added the '{TAG_TO_ADD}' tag to {len(runs_to_update)} runs.")

if __name__ == "__main__":
    main()