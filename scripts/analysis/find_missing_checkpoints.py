import wandb
from collections import defaultdict

def main():
    entity = "lmse-university-of-toronto"
    project = "astra"
    
    # The groups we know are missing a checkpoint
    target_groups = [
        "LinearBaselineModel-all-advanced-valid/kcat_Pearson-top0",
        "Uncertainty-Direct-Linear",
        "Uncertainty-Advanced-Linear"
    ]
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={
        "tags": "5fcv",
        "state": "finished"
    })
    
    # Track which folds have checkpoints
    found_folds = defaultdict(set)
    
    for run in runs:
        if str(run.group) not in target_groups:
            continue
            
        ckpt = run.summary.get('last_local_checkpoint_path') or run.summary.get('best_local_checkpoint_path')
        if ckpt:
            # Find the fold tag
            for t in run.tags:
                if 'fold' in t.lower():
                    found_folds[str(run.group)].add(t)
                    break

    # Print the missing folds
    expected_folds = {"fold_0", "fold_1", "fold_2", "fold_3", "fold_4"}
    # Note: sometimes your tags are "fold0_split", so we check the number
    expected_nums = {"0", "1", "2", "3", "4"}
    
    print("\n--- Missing Folds Analysis ---")
    for group in target_groups:
        folds = found_folds[group]
        found_nums = set([f.replace('fold_','').replace('fold','').replace('_split','') for f in folds])
        missing_nums = expected_nums - found_nums
        
        print(f"\nGroup: {group}")
        print(f"Found folds: {sorted(list(found_nums))}")
        if missing_nums:
            print(f"❌ MISSING FOLDS: {sorted(list(missing_nums))}")
        else:
            print("✅ All 5 folds have checkpoints!")

if __name__ == "__main__":
    main()