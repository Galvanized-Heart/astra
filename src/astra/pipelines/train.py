# DEPRECATED!

from pathlib import Path

import torch
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from astra.model.lightning_models import AstraModule
from astra.data_processing.datamodules import AstraDataModule
from astra.data_processing.configs.registry import (
    MODEL_REGISTRY, OPTIMIZER_REGISTRY, LOSS_FN_REGISTRY,
    FEATURIZER_REGISTRY, SCHEDULER_REGISTRY
)



def train(config_dict: dict = None):
    """
    Runs full training loop for Astra.
    
    Args:
        config_dict (dict): A dictionary containing the complete configuration
                            for the training run.
    """
    # Verify config exists
    if config_dict is None:
        raise ValueError("Configuration dictionary must exist, it cannot be None.")

    # Establish seeded run
    seed = config_dict.get('seed')
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # Instantiate Featurizers
    p_featurizer_cfg = config_dict['featurizers']['protein']
    l_featurizer_cfg = config_dict['featurizers']['ligand']
    
    # TODO: Make a more robust system for setting device for featurizers (should probably check whether GPU exists and use whatever config says)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Does this need to be set or will config tell us?
    if p_featurizer_cfg['name'] == 'ESMFeaturizer': # Might need more robust logic to determine if featurizer's device needs to be set
        p_featurizer_cfg['params']['device'] = device
        
    protein_featurizer = FEATURIZER_REGISTRY[p_featurizer_cfg['name']](**p_featurizer_cfg['params'])
    ligand_featurizer = FEATURIZER_REGISTRY[l_featurizer_cfg['name']](**l_featurizer_cfg['params'])

    # Instantiate model architecture config
    model_arch_cfg = config_dict['model']['architecture']
    model_params = model_arch_cfg.get('params', {})

    # Calculate the expected output dimension from the data config
    data_cfg = config_dict['data']
    # Default to 3 if not specified
    target_columns = data_cfg.get('target_columns', ["kcat", "KM", "Ki"])
    expected_out_dim = len(target_columns)
    # Check if user specified out_dim in the model's parameters
    user_out_dim = model_params.get('out_dim')
    if user_out_dim is None:
        # Set unspecified output_dim automatically
        print(f"INFO: 'out_dim' not specified in model params. Automatically setting to {expected_out_dim} based on target_columns.")
        model_params['out_dim'] = expected_out_dim
    elif user_out_dim != expected_out_dim:
        # Raise error for mismatch between user config and model expectations
        raise ValueError(
            f"Configuration mismatch: The model's specified 'out_dim' ({user_out_dim}) "
            f"does not match the number of 'target_columns' ({expected_out_dim}). "
            f"Please set 'out_dim: {expected_out_dim}' in your config's model parameters or leave it empty."
        )

    # Instantiate DataModule
    datamodule = AstraDataModule(
        data_paths={'train': Path(data_cfg['train_path']), "valid": Path(data_cfg['valid_path'])},
        protein_featurizer=protein_featurizer,
        ligand_featurizer=ligand_featurizer,
        batch_size=data_cfg['batch_size']
    )

    # Instantiate Model Architecture
    model_params['protein_spec'] = datamodule.protein_feature_spec
    model_params['ligand_spec'] = datamodule.ligand_feature_spec
    model_architecture = MODEL_REGISTRY[model_arch_cfg['name']](**model_params)

    # Instantiate Lightning Module
    lightning_cfg = config_dict['model']['lightning_module']
    optimizer_class = OPTIMIZER_REGISTRY[lightning_cfg['optimizer']]

    loss_fn_name = lightning_cfg.get('loss_function')
    expected_out_dim = len(config_dict['data'].get('target_columns', ["kcat", "KM", "Ki"]))

    if loss_fn_name is None:
        # If user didn't specify, infer a default
        if expected_out_dim > 1:
            loss_fn_name = "MaskedMSELoss"
            print(f"INFO: 'loss_function' not specified. Defaulting to '{loss_fn_name}' for multi-target regression.")
        else:
            loss_fn_name = "MSELoss"
            print(f"INFO: 'loss_function' not specified. Defaulting to '{loss_fn_name}' for single-target regression.")
    else:
        # If user DID specify, check if it makes sense and warn them if not.
        import warnings
        if expected_out_dim > 1 and loss_fn_name == "MSELoss":
            warnings.warn(
                f"You specified 'loss_function: MSELoss' for a multi-target problem ({expected_out_dim} targets). "
                "This will not handle NaN values in targets. Consider using 'MaskedMSELoss' instead."
            )
        if expected_out_dim == 1 and loss_fn_name == "MaskedMSELoss":
            warnings.warn(
                f"You specified 'loss_function: MaskedMSELoss' for a single-target problem. "
                "This is unnecessary overhead. Consider using 'MSELoss' for efficiency."
            )

    loss_func = LOSS_FN_REGISTRY.get(loss_fn_name)()

    scheduler_class = None
    scheduler_kwargs = {}
    if 'lr_scheduler' in lightning_cfg and lightning_cfg['lr_scheduler']:
        scheduler_cfg = lightning_cfg['lr_scheduler']
        scheduler_class = SCHEDULER_REGISTRY.get(scheduler_cfg['name'])
        scheduler_kwargs = scheduler_cfg.get('params', {})
    
    model = AstraModule(
        model=model_architecture,
        lr=lightning_cfg['lr'],
        loss_func=loss_func,
        optimizer_class=optimizer_class,
        lr_scheduler_class=scheduler_class,
        lr_scheduler_kwargs=scheduler_kwargs
    )

    # Instantiate Logger
    wandb_logger = WandbLogger(
        name=config_dict['run_name'],
        project=config_dict.get('project_name', 'astra'),
        entity=config_dict.get('entity', 'lmse-university-of-toronto'),
        log_model="all",
        config_dict=config_dict
    )
    
    # Instatiate Callbacks
    cb_cfg = config_dict['trainer']['callbacks']['checkpoint']
    checkpoint_callback = ModelCheckpoint(
        monitor=cb_cfg['monitor'],
        dirpath="checkpoints/",
        filename=f"{config_dict['run_name']}-{{epoch:02d}}-{{{cb_cfg['monitor']}:.2f}}",
        save_top_k=cb_cfg['save_top_k'],
        mode=cb_cfg['mode'],
        save_last=True
    )

    # Instantiate Trainer
    trainer_cfg = config_dict['trainer']
    trainer = L.Trainer(
        max_epochs=trainer_cfg['epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=(seed is not None),
        accelerator=trainer_cfg.get('device', 'auto')
    )

    # NOTE: setting deterministic=True sets torch.use_deterministic_algorithms(True), but does not set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":4096:8" or ":16:8", causing a RunTimeError.
    # You must set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":4096:8" prior to running train.py to ensure proper behaviour. If you run into an OutOfMemoryError, please set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":16:8" instead.
    
    # Run trainer
    trainer.fit(model, datamodule)