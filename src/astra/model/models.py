from collections.abc import Callable

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import mean_squared_error


class AstraModule(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim = None, lr: float = 1e-3, loss_func = None, recomposition_func: Callable = None):
        """
        A flexible LightningModule that can optionally apply a final
        transformation function to the base model's output.

        Args:
            model (nn.Module): The core model. It can either output the final
                                    parameters or intermediate rates.
            optimizer (torch.optim): A PyTorch optimizer used to train the model.
            lr (float): The learning rate for the optimizer.
            loss_func: A PyTorch loss function used to compute model performance 
                       and update gradients during training.
            recomposition_func (callable, optional): A function to transform
                        the model's output. If None, the output
                        is returned directly. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam
        self.lr = lr
        self.loss_func = F.mse_loss
        self.recomposition_func = recomposition_func
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def _shared_step(self, batch):
        # Can use this to reduce replicate code in training_step() and validation_step()

        # Get inputs from batch
        x_prot = batch['protein_embedding']
        x_lig = batch['ligand_embedding']

        # Get targets from batch
        y = batch['targets']
        
        # Make predictions
        output = self.model(x_prot, x_lig)

        # Compute kinetic recomposition
        if self.recomposition_func:
            y_hat = self.recomposition_func(output)
        else:
            y_hat = output
        
        # Return loss
        return self.loss_func(y_hat, y)

    def training_step(self, batch, batch_idx):
        """Functionality for training loop."""
        # Compute shared step
        loss = self._shared_step(batch)

        # Log results
        self.log('train/loss_step', loss, prog_bar=True) # Default on_step=True, on_epoch=False

        # Optionally accumulate per epoch metrics for on_training_epoch_end()
        #self.train_metric.update(logits, y)
        # NOTE: This requires a torchmetrics to be set in __init__()

        # Alternatively set on_step=False for easy to compute metrics
        #self.log('train/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Return loss for optimizer
        return loss

    def on_training_epoch_end(self):
        # Can be used to log accumulated metrics from training_step()
        #epoch_metric = self.train_metric.compute()
        #self.log("train/metric_epoch", epoch_metric)
        #self.train_metric.reset()
        pass
    
    def validation_step(self, batch, batch_idx):
        """Functionality for prediction validation."""
        # Compute shared step
        loss = self._shared_step(batch)

        # Log results
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self, outputs):
        # Can be used to log accumulated metrics from validation_step()
        # NOTE: outputs is a list of dicts from output from validation_step()
        pass

# Example usage
""" 
model = AstraModule()
trainer = L.Trainer()
trainer.fit(model, datamodule)
trainer.test(model, dataloaders=DataLoader(test_set)) 
"""


class XGBoostLightning(L.LightningModule):
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

    def configure_optimizers(self):
        # No optimizer required for XGBoost
        return None

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
    
# Example usage
"""
trainer = L.Trainer(
    max_epochs=1, # Only need 1 epoch to collect all data and train the model
    accelerator='gpu',
    devices=1 # Number of GPUs you want to run it on
)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mse',
    'n_estimators': 150,
    'learning_rate': 0.05,
    'max_depth': 4,
    'tree_method': 'gpu_hist',
    'seed': 42
}
xgb_lightning_model = XGBoostRegressionLightning(xgb_params=xgb_params)
trainer.fit(xgb_lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(xgb_lightning_model, dataloaders=test_loader)

TODO: Consider how to use multiple regression with XGBoost 
- sklearn has MultiOutputRegressor wrapper that creates ensemble
"""