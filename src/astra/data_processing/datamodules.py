from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig

import lightning as L
from torch.utils.data import DataLoader

from astra.data_processing.datasets import ProteinLigandDataset
from astra.data_processing.manifests.manifest import create_manifests, generate_manifest_hash
from astra.data_processing.featurizers import Featurizer
from astra.constants import PROJECT_ROOT



class AstraDataModule(L.LightningDataModule):
    """DataModule for Astra."""
    def __init__(self, 
                 data_cfg: DictConfig,
                 protein_featurizer_cfg: DictConfig,
                 ligand_featurizer_cfg: DictConfig,
                 protein_featurizer: Featurizer,
                 ligand_featurizer: Featurizer,
                 batch_size: int = 32,
                 featurizer_batch_size: int = 32,
            ):
        """
        Meant to instantiate states for `torch.utils.data.Dataset` classes.

        It appears that this is intended to intake paths to the data and that
        the `Dataset` class handles the loading via `__getitem__()` for featurizing. 
        Also, it would take configs for training, I suppose.
        """

        super().__init__()
        self.save_hyperparameters(ignore=['protein_featurizer', 'ligand_featurizer']) # Ignore unpicklable objects

        self.protein_feature_spec = protein_featurizer.feature_spec
        self.ligand_feature_spec = ligand_featurizer.feature_spec

        # Extract necessary info from configs
        split_files = {'train': str(data_cfg.train_path), 'valid': str(data_cfg.valid_path)}
        target_columns = data_cfg.target_columns
        
        # Generate manifest hash for unique data preparation
        config_hash = generate_manifest_hash(
            split_files=split_files,
            target_columns=target_columns,
            protein_featurizer_cfg=protein_featurizer_cfg,
            ligand_featurizer_cfg=ligand_featurizer_cfg
        )
        
        manifest_dir = PROJECT_ROOT / "data" / "manifest" / config_hash
        print(f"INFO: Using manifest directory: {manifest_dir}")
        
        # Check if manifests already exist
        expected_manifests = {name: manifest_dir / f"manifest_{name}.csv" for name in split_files.keys()}
        all_manifests_exist = all(p.exists() for p in expected_manifests.values())

        if all_manifests_exist:
            # Set cached manifest files
            print(f"INFO: Found cached manifests in {manifest_dir}. Skipping generation.")
            manifest_files = expected_manifests
        else:
            # Generate and cache manifest files
            print(f"INFO: Manifests not found. Generating new ones in {manifest_dir}.")
            manifest_files = create_manifests(
                split_files=split_files, 
                target_columns=target_columns, 
                output_dir=manifest_dir,
                protein_featurizer=protein_featurizer, 
                ligand_featurizer=ligand_featurizer,
                batch_size=featurizer_batch_size
            )

        # Set file paths if they exist, else set to None
        self.train_path = manifest_files.get("train")
        self.valid_path = manifest_files.get("valid")
        self.test_path = manifest_files.get("test")

        # Set configs
        self.batch_size = batch_size
        self.target_transform = data_cfg.target_transform

        # TODO: Train/Inference configs (possibly split function or seperate train/valid sets in config)


    def prepare_data(self):
        """
        Can include download, tokenization, and featurization logic, but is not required.

        TODO: Add create_manifests to this section.

        - Note: Lightning recommends not assigning states here (e.g. self.x = y).
        """
        pass

    def setup(self, stage: str):
        """
        When `Trainer()` calls `fit`, `validate`, `test`, `predict` attributes, these setup the data prior to calling `*_dataloader()`. 
        """
        # TODO: Splitting needs to be optional so users can provide their own training splits (possibly in config)
            # NOTE: For now, splits will be premade and user will provide the paths for the split
        common_params = {
            "target_columns": self.hparams.target_columns,
            "target_transform": self.hparams.target_transform
        }

        if stage == "fit":
            print("Setting up training dataset...")
            self.train_dataset = ProteinLigandDataset(self.train_path, **common_params)
            print("Training set complete.")

        if stage == "validate" or stage == "fit":
            print("Setting up validation dataset...")
            self.valid_dataset = ProteinLigandDataset(self.valid_path, **common_params)
            print("Validation set complete.")

        # Test is meant to provide metrics
        if stage == "test":
            # TODO: get self.test_dataset
            pass

        # Predict is meant to only provide prediction output
        if stage == "predict":
            # TODO: get self.predict_dataset
            pass

    def train_dataloader(self):
        """Called by `Trainer().fit()`."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Called by `Trainer().fit()` and `Trainer().validate()`."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Called by `Trainer().test`."""
        print("AstraDataModule.test_dataloader() is not yet implemented!")
        pass
        #return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        """Called by `Trainer().predict`."""
        print("AstraDataModule.predict_dataloader() is not yet implemented!")
        pass
        #return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)