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
        config (dict): A dictionary containing the complete configuration
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

    # Instantiate DataModule
    data_cfg = config_dict['data']
    datamodule = AstraDataModule(
        data_paths={'train': Path(data_cfg['train_path']), "valid": Path(data_cfg['valid_path'])},
        protein_featurizer=protein_featurizer,
        ligand_featurizer=ligand_featurizer,
        batch_size=data_cfg['batch_size']
    )

    # Instantiate Model Architecture
    model_arch_cfg = config_dict['model']['architecture']
    model_params = model_arch_cfg.get('params', {})
    model_params['protein_spec'] = datamodule.protein_feature_spec
    model_params['ligand_spec'] = datamodule.ligand_feature_spec
    model_architecture = MODEL_REGISTRY[model_arch_cfg['name']](**model_params)

    # Instantiate Lightning Module
    lightning_cfg = config_dict['model']['lightning_module']
    loss_func = LOSS_FN_REGISTRY[lightning_cfg['loss_function']]()
    optimizer_class = OPTIMIZER_REGISTRY[lightning_cfg['optimizer']]
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








































    if seed is not None:
        L.seed_everything(seed, workers=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instatiate feturizers
    protein_featurizer = ESMFeaturizer(model_name="facebook/esm2_t6_8M_UR50D", device=device)
    ligand_featurizer = MorganFeaturizer(radius=2, fp_size=2048)

    # Add data paths to dict
    data_paths = {'train': Path(train_path), "valid": Path(valid_path)}

    # Instatiate DataModule
    datamodule = AstraDataModule(data_paths, protein_featurizer, ligand_featurizer, batch_size) 

    model_architecture = DummyModel()
    loss_func = MaskedMSELoss()

    # Instatiate Module
    model = AstraModule(
        model=model_architecture,
        lr=1e-3,
        loss_func=loss_func,
        optimizer_class=torch.optim.AdamW
    )

    # Instantiate WandbLogger
    wandb_logger = WandbLogger(
        name="train-pipeline-test1",     # Name of this specific run (we can change this)
        project="astra",                 # The project to log to
        entity="lmse-university-of-toronto", # Your team entity
        log_model="all"                  # Log model checkpoints as W&B Artifacts
    )

    # Instatiate ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss_epoch",      # Metric to monitor (this needs to match validation_step() inside AstraModule())
        dirpath="checkpoints/",          # Directory to save checkpoints
        filename="sample-model-{epoch:02d}-{valid_loss:.2f}", # Checkpoint file name
        save_top_k=1,                    # Save the best k models
        mode="min",                      # 'min' for loss, 'max' for accuracy
        save_last=True,                  # Also save the last checkpoint
    )

    # Instantiate Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,             # Use W&B logger
        callbacks=[checkpoint_callback], # Add the checkpoint callback
        deterministic=True if seed is not None else False # Ensure deterministic behaviour
    )
    # NOTE: setting deterministic=True sets torch.use_deterministic_algorithms(True), but does not set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":4096:8" or ":16:8", causing a RunTimeError.
    # You must set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":4096:8" prior to running train.py to ensure proper behaviour. If you run into an OutOfMemoryError, please set os.environ["CUBLAS_WORKSPACE_config_dict"] = ":16:8" instead.

    # Run training loop
    trainer.fit(model, datamodule)