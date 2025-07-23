from pathlib import Path
from typing import Dict

import lightning as L
from torch.utils.data import DataLoader

from astra.data_processing.datasets import ProteinLigandDataset
from astra.data_processing.featurize.manifest import create_feature_manifest
from astra.constants import PROJECT_ROOT


class AstraDataModule(L.LightningDataModule):
    """DataModule for Astra."""
    def __init__(self, data_paths: Dict[str, str] = None, batch_size: int = 32):
        """
        Meant to instantiate states for `torch.utils.data.Dataset` classes.

        It appears that this is intended to intake paths to the data and that
        the `Dataset` class handles the loading via `__getitem__()` for featurizing. 
        Also, it would take configs for training, I suppose.
        """
        super().__init__()

        # Create manifest features
        manifest_files = create_feature_manifest(data_paths, PROJECT_ROOT/"data"/"manifest")

        # Set file paths if they exist
        self.train_path = manifest_files.get("train")
        self.valid_path = manifest_files.get("valid")
        self.test_path = manifest_files.get("test")

        # Set configs
        self.batch_size = batch_size

        # TODO: Train/Inference configs (possibly split function or seperate train/valid sets in config)


    def prepare_data(self):
        """
        Can include download, tokenization, and featurization logic, but is not required.

        - Note: Lightning recommends not assigning states here (e.g. self.x = y).
        """
        pass

    def setup(self, stage: str):
        """
        When `Trainer()` calls `fit`, `validate`, `test`, `predict` attributes, these setup the data prior to calling `*_dataloader()`. 
        """
        # TODO: Splitting needs to be optional so users can provide their own training splits (possibly in config)
            # NOTE: For now, splits will be premade and user will provide the paths for the split

        if stage == "fit":
            print("Setting up training dataset...")
            self.train_dataset = ProteinLigandDataset(self.train_path)
            print("Training set complete.")

        if stage == "validate" or stage == "fit":
            print("Setting up validation dataset...")
            self.valid_dataset = ProteinLigandDataset(self.valid_path)
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
        print("AstraDataModule.test_dataloader() is not yet implenmented!")
        pass
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        """Called by `Trainer().predict`."""
        print("AstraDataModule.predict_dataloader() is not yet implenmented!")
        pass
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)