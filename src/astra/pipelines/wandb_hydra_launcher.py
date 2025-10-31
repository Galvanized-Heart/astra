import sys
import subprocess

def main():
    """
    This script acts as a bridge between wandb agent and a hydra script,
    using `uv run` for execution.
    
    WandB agent passes arguments like:
    --architecture.params.dim_1=32
    
    This script reformats them for Hydra:
    architecture.params.dim_1=32
    
    And then calls the actual hydra training script using `uv run`.
    """
    
    if len(sys.argv) < 2:
        print("Usage: python wandb_hydra_launcher.py <path_to_hydra_script> [wandb_args...]")
        sys.exit(1)
        
    target_script = sys.argv[1]
    wandb_args = sys.argv[2:]
    
    # --- MODIFICATION FOR UV ---
    # Build the command for the hydra script using `uv run`.
    # The '--' is a best practice: it tells `uv run` to stop parsing its own
    # options and treat everything that follows as an argument for the script.
    command = ["uv", "run", "--", target_script]
    
    # Reformat arguments from wandb's '--key=value' to hydra's 'key=value'
    hydra_overrides = []
    for arg in wandb_args:
        if arg.startswith('--'):
            hydra_overrides.append(arg[2:]) # Strip the leading '--'
        else:
            hydra_overrides.append(arg)
            
    command.extend(hydra_overrides)
    
    print(f"--- W&B Hydra Launcher (using uv) ---")
    print(f"Original argv received by launcher: {sys.argv}")
    print(f"Executing final command: {' '.join(command)}")
    
    # Execute the hydra script with the correctly formatted arguments
    result = subprocess.run(command, check=False)
    
    # Propagate the exit code from the hydra script
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()