import sys
import yaml
from pathlib import Path

def generate_fold_config(base_config_path: Path, fold_id: int, output_dir: Path):
    """
    Reads a base YAML config, updates it for a specific CV fold,
    and writes a new config file.

    Args:
        base_config_path (Path): Path to the source config.yaml.
        fold_id (int): The cross-validation fold number (e.g., 0, 1, 2...).
        output_dir (Path): Directory to save the new config file.

    Returns:
        Path: The path to the newly created config file.
    """
    # Load the base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Modify the configuration ---
    # Update the training and validation paths based on the fold ID
    train_path = f"data/cv_splits_pangenomic/cv_folds_balanced/fold_{fold_id}_train.csv"
    valid_path = f"data/cv_splits_pangenomic/cv_folds_balanced/fold_{fold_id}_valid.csv"

    config['data']['train_path'] = train_path
    config['data']['valid_path'] = valid_path
    
    # Optional: Give a unique run name for this fold for better tracking in wandb
    config['run_name'] = f"fold_{fold_id}_training"

    # --- Save the new configuration ---
    output_dir.mkdir(parents=True, exist_ok=True)
    new_config_path = output_dir / f"fold_{fold_id}_config.yaml"

    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    # Print the path of the new config file to standard output
    # This is how the main job script will know which file to use.
    print(new_config_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepare_config.py <base_config_path> <fold_id> <output_dir>")
        sys.exit(1)

    base_config_path = Path(sys.argv[1])
    fold_id = int(sys.argv[2])
    output_dir = Path(sys.argv[3])

    generate_fold_config(base_config_path, fold_id, output_dir)