from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from astra.model.layer.function_wrapper_layer import FunctionWrapperLayer

class AstraModule(L.LightningModule):
    def __init__(self, model: nn.Module, recomposition_func: Optional[Callable] = None, lr: float = 1e-3 ):
        """
        A flexible LightningModule that can optionally apply a final
        transformation function to the base model's output.

        Args:
            model (nn.Module): The core model. It can either output the final
                                    parameters or intermediate rates.
            recomposition_func (callable, optional): A function to transform
                                    the model's output. If None, the output
                                    is returned directly. Defaults to None.
            lr (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.model = model
        self.recomposition_func = recomposition_func
        self.lr = lr
        self.optimizer = torch.optim.Adam
        self.loss_func = F.mse_loss

    def forward(self, x):
        """
        The forward pass with conditional logic.
        """
        x = self.model(x)
        if self.recomposition_func is not None:
            x = self.recomposition_func(x)
        return x
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x) # Forward pass inside AstraModule
        loss = self.loss_func(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        val_loss = self.loss_func(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        test_loss = self.loss_func(x_hat, x)
        self.log("test_loss", test_loss)
    
# Example usage
""" 
trainer = L.Trainer()
trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_set)) 
"""


import pytorch_lightning as pl
import xgboost as xgb
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

class XGBoostLightning(pl.LightningModule):
    def __init__(self, xgb_params: dict):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model is instatiated in on_train_epoch_end(), default to None here
        self.model = None

        # Lists to store batch data for training
        self._train_x = []
        self._train_y = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        Note: The input x should be a tensor, but the model expects a numpy array.
        """
        # Make sure model is trained
        if self.model is None:
            raise RuntimeError("The XGBoost model has not been trained yet. Call trainer.fit() first.")
            
        predictions = self.model.predict(x.cpu().numpy())
        return torch.from_numpy(predictions).to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Collect the data from each batch.
        """
        x, y = batch
        self._train_x.append(x.cpu().numpy())
        self._train_y.append(y.cpu().numpy())

        # Not using LightningModules's optimizer, so return None
        return None 

    def on_train_epoch_end(self):
        """
        Train XGBoost in on_train_epoch_end hook (after entire training loop).
        """
        print("\nTraining XGBoost model")
        
        # Concatenate all collected data
        X_train = np.concatenate(self._train_x)
        y_train = np.concatenate(self._train_y)

        # Instantiate the XGBoost model with the given parameters
        self.model = xgb.XGBRegressor(**self.hparams.xgb_params)

        # Train model
        self.model.fit(X_train, y_train)
        print("XGBoost model training finished\n")

        # Clear the collected data to free up memory
        self._train_x.clear()
        self._train_y.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Ensure the model is trained
        if self.model is None:
            return

        preds_np = self.model.predict(x.cpu().numpy())
        y_np = y.cpu().numpy()
        mse = mean_squared_error(y_np, preds_np)        
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        return mse

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Ensure the model is trained
        if self.model is None:
            return

        preds_np = self.model.predict(x.cpu().numpy())
        y_np = y.cpu().numpy()
        mse = mean_squared_error(y_np, preds_np)        
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        return mse
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        For inference without labels or logging.
        """
        x, _ = batch # Assume batch has (x, y), might be just x
        return self(x)

    def configure_optimizers(self):
        # No optimizer required for XGBoost
        return None