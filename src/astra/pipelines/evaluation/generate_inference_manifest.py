import wandb
import json
from pathlib import Path
from collections import defaultdict

# The FULL MATRIX of Champions
WHITELIST_GROUPS = [
    # ---- LINEAR ARCHITECTURE ----
    # Single Task
    "LinearBaselineModel-kcat-none-valid/kcat_Pearson-top0",
    "LinearBaselineModel-KM-none-valid/KM_Pearson-top0",
    "LinearBaselineModel-Ki-none-valid/Ki_Pearson-top0",
    # MT Manual
    "LinearBaselineModel-all-none-valid/kcat_Pearson-top0",     # Direct
    "LinearBaselineModel-all-basic-valid/kcat_Pearson-top0",    # Basic
    "LinearBaselineModel-all-advanced-valid/kcat_Pearson-top0", # Advanced
    # MT Uncertainty
    "Uncertainty-Direct-Linear",                                # Direct
    "Uncertainty-Basic-Linear",                                 # Basic
    "Uncertainty-Advanced-Linear",                              # Advanced

    # ---- SELF-ATTENTION ARCHITECTURE ----
    # Single Task
    "CpiPredSelfAttnModel-kcat-none-valid/kcat_Pearson-top0",
    "CpiPredSelfAttnModel-KM-none-valid/KM_Pearson-top0",
    "CpiPredSelfAttnModel-Ki-none-valid/Ki_Pearson-top0",
    # MT Manual
    "CpiPredSelfAttnModel-all-none-valid/Ki_Pearson-top0",      # Direct (Ki selected for best balance)
    "CpiPredSelfAttnModel-all-basic-valid/KM_Pearson-top0",     # Basic (KM selected for best balance)
    "CpiPredSelfAttnModel-all-advanced-valid/kcat_Pearson-top0",# Advanced (kcat selected for best balance)
    # MT Uncertainty
    "Uncertainty-Direct-SelfAttn",                              # Direct
    "Uncertainty-Basic-SelfAttn",                               # Basic
    "Uncertainty-Advanced-SelfAttn"                             # Advanced
]

def main():
    entity = "lmse-university-of-toronto"
    project = "astra"
    
    print(f"Fetching full matrix from {entity}/{project}...")
    api = wandb.Api()
    
    runs = api.runs(f"{entity}/{project}", filters={
        "tags": "5fcv",
        "state": "finished"
    })
    
    grouped_runs = defaultdict(list)
    
    for run in runs:
        if str(run.group) not in WHITELIST_GROUPS:
            continue
            
        fold = "unknown"
        for t in run.tags:
            if 'fold' in t.lower(): fold = t; break
                
        key = f"{run.group}::{fold}"
        grouped_runs[key].append(run)
    
    manifest = []
    
    for key, run_list in grouped_runs.items():
        run_list.sort(key=lambda r: r.created_at, reverse=True)
        best_run = run_list[0]
        
        ckpt_path = best_run.summary.get("last_local_checkpoint_path") or best_run.summary.get("best_local_checkpoint_path")
        if not ckpt_path: continue
            
        # Parse clean metadata
        arch = "Linear" if "Linear" in best_run.group else "SelfAttn"
        
        # Determine Recomposition type
        recomp = "Direct"
        if "Basic" in best_run.group or "basic" in best_run.group: recomp = "Basic"
        if "Advanced" in best_run.group or "advanced" in best_run.group: recomp = "Advanced"

        # Determine Training Mode
        if "Uncertainty" in best_run.group:
            mode = f"MT-Uncert-{recomp}"
        elif "all" in best_run.group:
            mode = f"MT-Manual-{recomp}"
        else:
            target = "kcat" if "-kcat-" in best_run.group else "KM" if "-KM-" in best_run.group else "Ki"
            mode = f"ST-{target}"
            
        manifest.append({
            "run_id": best_run.id,
            "run_name": best_run.name,
            "group": best_run.group,
            "fold": key.split("::")[1],
            "arch_clean": arch,
            "mode_clean": mode,
            "ckpt_path": ckpt_path,
            "config": {k: v for k, v in best_run.config.items() if not k.startswith('_')}
        })
        
    out_path = Path("results/inference_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✅ Saved {len(manifest)} runs to {out_path}")
    print("Expected: 90 runs (18 groups * 5 folds).")

if __name__ == "__main__":
    main()