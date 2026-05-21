import wandb
from collections import defaultdict
import numpy as np

def main():
    entity = "lmse-university-of-toronto"
    project = "astra"
    
    print("Fetching '5fcv' runs from W&B to analyze group structures...")
    api = wandb.Api()
    
    # Fetch all finished 5fcv runs
    runs = api.runs(f"{entity}/{project}", filters={
        "tags": "5fcv",
        "state": "finished"
    })
    
    # We will aggregate by group
    groups = defaultdict(lambda: {
        'folds': set(), 
        'kcat_p': [], 'km_p': [], 'ki_p': [], 
        'has_ckpt': 0, 'total': 0
    })
    
    for run in runs:
        group_name = str(run.group)
        groups[group_name]['total'] += 1
        
        # Find fold number
        fold = "unknown"
        for t in run.tags:
            if 'fold' in t.lower(): fold = t; break
        groups[group_name]['folds'].add(fold)
        
        # Check metrics
        kcat = run.summary.get('valid/kcat_Pearson')
        km = run.summary.get('valid/KM_Pearson')
        ki = run.summary.get('valid/Ki_Pearson')
        
        if kcat is not None: groups[group_name]['kcat_p'].append(kcat)
        if km is not None: groups[group_name]['km_p'].append(km)
        if ki is not None: groups[group_name]['ki_p'].append(ki)
            
        # Check checkpoint
        ckpt = run.summary.get('last_local_checkpoint_path') or run.summary.get('best_local_checkpoint_path')
        if ckpt: groups[group_name]['has_ckpt'] += 1

    # Print Table
    print("\n" + "="*120)
    print(f"{'Group Name':<60} | {'Folds':<5} | {'Ckpts':<5} | {'kcat (r)':<8} | {'KM (r)':<8} | {'Ki (r)':<8}")
    print("="*120)
    
    # Sort groups alphabetically for easier reading
    for group in sorted(groups.keys()):
        data = groups[group]
        
        kcat_m = f"{np.mean(data['kcat_p']):.3f}" if data['kcat_p'] else "N/A"
        km_m = f"{np.mean(data['km_p']):.3f}" if data['km_p'] else "N/A"
        ki_m = f"{np.mean(data['ki_p']):.3f}" if data['ki_p'] else "N/A"
        
        fold_count = len(data['folds'])
        ckpt_count = data['has_ckpt']
        
        print(f"{group:<60} | {fold_count:<5} | {ckpt_count:<5} | {kcat_m:<8} | {km_m:<8} | {ki_m:<8}")
    
    print("="*120)
    print(f"Total Unique Groups Found: {len(groups)}")

if __name__ == "__main__":
    main()