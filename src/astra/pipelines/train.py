from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer
from astra.data_processing.datamodules import AstraDataModule
from astra.model.lightning_models import AstraModule
from astra.model.models.dummy_model import DummyModel
from astra.model.loss.masked_mse_loss import MaskedMSELoss


def train(train_path: str, valid_path: str, epochs: int = 10, batch_size: int = 32, seed: int = None, device: str = None):
    """
    Runs full training loop for Astra.
    
    Args:
        train_path (str): The CSV path to the training data.
        valid_path (str): The CSV path to the validation data.
        batch_size (int): The size of the batches used during training (and featurizing?).
        device (str): The device to use during training.
    """
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
    # NOTE: setting deterministic=True sets torch.use_deterministic_algorithms(True), but does not set os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" or ":16:8", causing a RunTimeError.
    # You must set os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" prior to running train.py to ensure proper behaviour. If you run into an OutOfMemoryError, please set os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" instead.

    # Run training loop
    trainer.fit(model, datamodule)