# run_sweep_optuna.py
import yaml
import optuna
import wandb
import math
from optuna.integration import PyTorchLightningPruningCallback
from astra.constants import PROJECT_ROOT
from astra.pipelines.run_train import run_training_engine

def softmax(logits: list, scale) -> list:
    """
    Calculates softmax in pure Python to avoid importing torch at the top level.
    """
    # Subtract max for numerical stability (prevents overflow with large logits)
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    softmax_probs = [e / sum_exps for e in exps]
    
    return softmax_probs

def objective(trial: optuna.trial.Trial):
        # Use a single W&B run to group all trials of this study
    # This will create runs like "trial-0", "trial-1", etc. inside one main W&B run.
    # OR, a more common pattern is to have one W&B run PER trial. Let's do that.
    
    # Initialize a new W&B run for each Optuna trial
    run = wandb.init(
        project="astra", 
        config=trial.params,  # Log Optuna's suggested params to W&B
        reinit=True           # Allows wandb.init to be called in a loop
    )
    
    with run:
        # --- Load and Merge Config ---
        base_config_path = PROJECT_ROOT / "configs" / "hpo" / "hpo_base_config.yaml"
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- Suggest Hyperparameters ---
        # Suggest weights for the three tasks: kcat, KM, Ki
        # We suggest "logits" - raw, unconstrained numbers.
        w_kcat_logit = trial.suggest_float("w_kcat_logit", 0.0, 1.0)
        w_km_logit = trial.suggest_float("w_km_logit", 0.0, 1.0)
        w_ki_logit = trial.suggest_float("w_ki_logit", 0.0, 1.0)
        
        # We use the custom softmax function to avoid importing torch before run_engine is called (necessary to maintain determinism)
        logits = [w_kcat_logit, w_km_logit, w_ki_logit]
        final_weights = softmax(logits)

        # Inject the final weights into the config dictionary
        config['model']['lightning_module']['loss_function']['params']['weights'] = final_weights.tolist() # tolist() is necessary

        # Data params
        config['data']['batch_size'] = trial.suggest_categorical("batch_size", [32, 64])
        
        # Model architecture params
        arch_params = config['model']['architecture']['params']
        arch_params['hid_dim'] = trial.suggest_categorical("hid_dim", [32, 64, 128, 256, 512])
        arch_params['dropout'] = trial.suggest_float("dropout", 0.0, 0.5)
        arch_params['kernal_1'] = trial.suggest_categorical("kernal_1", [3, 5, 7, 9, 11])
        arch_params['conv_out_dim'] = trial.suggest_categorical("conv_out_dim", [32, 64, 128, 256, 512])
        arch_params['kernal_2'] = trial.suggest_categorical("kernal_2", [3, 5, 7, 9, 11])
        arch_params['last_hid'] = trial.suggest_categorical("last_hid", [32, 64, 128, 256, 512])

        # Lightning module params
        lm_params = config['model']['lightning_module']
        lm_params['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # Log scale search

        # --- Add Optuna Pruning Callback ---
        # This is the magic for early-stopping bad trials.
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="valid_loss_epoch")
        
        # We need to inject this callback into the trainer config
        config['trainer']['callbacks']['pruning'] = pruning_callback

        # ... run training engine and return the metric ...
        metric = run_training_engine(config)
        return metric

# 2. Main script execution
if __name__ == "__main__":
    # Create the Optuna study
    # The 'sampler' is the search algorithm (TPE is a great default)
    # The 'pruner' decides whether to stop trials early
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    # Start the optimization process
    # n_trials is your "budget"
    study.optimize(objective, n_trials=50)

    # Print the results
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
