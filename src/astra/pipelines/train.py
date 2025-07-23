import torch

from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer
from astra.data_processing.datamodules import AstraDataModule


def train(train_path: str, valid_path: str, batch_size: int = 32, device: str = None):
    """
    Runs full training loop for Astra.
    
    Args:
        train_path (str): The CSV path to the training data.
        valid_path (str): The CSV path to the validation data.
        batch_size (int): The size of the batches used during training (and featurizing?).
        device (str): The device to use during training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instatiate feturizers
    protein_featurizer = ESMFeaturizer(model_name="facebook/esm2_t6_8M_UR50D", device=device)
    ligand_featurizer = MorganFeaturizer(radius=2, fp_size=2048)

    # Add data paths to dict
    data_paths = {'train': train_path, "valid": valid_path}

    # Create DataModule
    datamodule = AstraDataModule(data_paths, protein_featurizer, ligand_featurizer, batch_size) 

    # Create Module


    # Trainer()
    # Checkpoint and W&B callbacks
    pass