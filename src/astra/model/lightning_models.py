from collections.abc import Callable
from typing import Optional, Type, List, Dict, Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils

import torchmetrics
from torchmetrics.collections import MetricCollection
from torchmetrics.regression import (
    MeanSquaredError, 
    MeanAbsoluteError, 
    R2Score, 
    PearsonCorrCoef, 
    SpearmanCorrCoef,
    KendallRankCorrCoef
)

from astra.model.optim.cagrad import CAGrad



class AstraModule(L.LightningModule):
    def __init__(self, 
                model: nn.Module, 
                lr: float,
                loss_func: nn.Module, 
                optimizer_class: Type[torch.optim.Optimizer],
                target_columns: List[str], 
                recomposition_func: Optional[Callable] = None,
                log_transform_active: bool = False,
                lr_scheduler_class: Optional[Type] = None, 
                lr_scheduler_kwargs: Optional[dict] = None,
                mtl_strategy: str = "manual",
                mtl_optimizer: Optional[str] = None,
                mtl_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            ):
        """
        A flexible LightningModule that can optionally apply a final
        transformation function to the base model's output.

        Args:
            model (nn.Module): The core model architecture.
            lr (float): The learning rate.
            loss_func (nn.Module): An instance of the loss function to use.
                Defaults to a MaskedMSELoss that handles NaNs.
            optimizer_class (Type[torch.optim.Optimizer]): The class of the optimizer to use.
                Example: torch.optim.Adam, torch.optim.SGD.
            target_columns (List[str]): A list of the names of the output parameters,
                e.g., ['kcat', 'KM', 'Ki']. Used for logging.
            recomposition_func (Optional[Callable]): A function to transform model output.
            lr_scheduler_class (Optional[Type]): The class of the LR lr_scheduler to use.
                Example: torch.optim.lr_scheduler.ReduceLROnPlateau.
            lr_scheduler_kwargs (Optional[dict]): A dictionary of arguments for the lr_scheduler.
                Must include a 'monitor' key for lr_schedulers that require one.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'recomposition_func', 'loss_func', 'target_columns'])

        self.model = model
        self.loss_func = loss_func
        self.recomposition_func = recomposition_func
        self.log_transform_active = log_transform_active

        self.optimizer_class = optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        self.target_columns = target_columns

        self.mtl_strategy = mtl_strategy

        self.mtl_optimizer = mtl_optimizer
        self.mtl_optimizer_kwargs = mtl_optimizer_kwargs or {}

        if self.mtl_optimizer is not None:
            self.automatic_optimization = False

        if self.mtl_optimizer == 'cagrad':
            self.mtl_solver = CAGrad(**self.mtl_optimizer_kwargs)

        # Create collections for metrics
        self.train_metrics = nn.ModuleDict()
        self.valid_metrics = nn.ModuleDict()
        
        # Track each target parameter
        for param_name in self.target_columns:
            metrics_for_param = MetricCollection({
                #'MSE': MeanSquaredError(),
                'RMSE': MeanSquaredError(squared=False),
                'MAE': MeanAbsoluteError(),
                'Pearson': PearsonCorrCoef(),
                'R2': R2Score(),
                #'Spearman': SpearmanCorrCoef(),
                #'Kendall': KendallRankCorrCoef()
            })
            
            # Create separate collections for train and valid, with a parameter-specific prefix
            self.train_metrics[param_name] = metrics_for_param.clone(prefix=f'train/{param_name}_')
            self.valid_metrics[param_name] = metrics_for_param.clone(prefix=f'valid/{param_name}_')
    

    def configure_optimizers(self):
        """Instantiate the optimizer and, optionally, the lr_scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.hparams.lr)
        
        if self.lr_scheduler_class is None:
            return optimizer

        scheduler_params = self.lr_scheduler_kwargs.copy()
        monitor_metric = scheduler_params.pop("monitor", "valid_loss_epoch")
        lr_scheduler = self.lr_scheduler_class(optimizer, **scheduler_params)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    

    def _shared_step(self, batch):
        """A flexible shared step for training and validation."""
        # Can use this to reduce replicate code in training_step() and validation_step()

        # Pop out targets
        y = batch.pop("targets")

        # Make predictions
        output = self.model(**batch)
        # TODO: In the future, refactor `ProteinLigandDataset` to pass exact keys for `forward` in each model (see below)
        """
        Using this will make models more modular and remove the 'Get inputs from batch' code above.
        batch = {
            "protein_embedding": protein_features["embedding"],
            "protein_mask": protein_features["attention_mask"], 
            "ligand_embedding": ligand_features["embedding"],
            "ligand_mask": ligand_features["attention_mask"], 
            "targets": targets
        }
        """

        # Compute kinetic recomposition
        if self.recomposition_func:
            y_hat = self.recomposition_func(rates=output)
        else:
            y_hat = output

        # Calculate loss
        loss = self.loss_func(y_hat, y)

        # Return loss, predictions, and targets
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        """Functionality for training loop."""
        # If not using CAGrad, run the original logic
        if self.mtl_optimizer is None:
            # Compute shared step
            loss, y_hat, y = self._shared_step(batch)
            
            # Log loss
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True) 

            if hasattr(self.loss_func, 'log_variance'):
                # Map log_variance tensor target columns (kcat, KM, Ki) for logging
                for i, param_name in enumerate(self.target_columns):
                    # Log exp(-s)
                    precision = torch.exp(-self.loss_func.log_variance[i])
                    self.log(f'loss_weights/precision_{param_name}', precision, on_step=False, on_epoch=True)
                    # log raw log_variance parameter
                    self.log(f'loss_weights/log_var_{param_name}', self.loss_func.log_variance[i], on_step=False, on_epoch=True)


            # Update the metrics for each target parameter
            for i, param_name in enumerate(self.target_columns):
                preds_i = y_hat[:, i]
                targets_i = y[:, i]

                # Create mask for the current parameter
                mask = ~torch.isnan(targets_i)

                # Update metrics only with non-NaN values
                if mask.sum() > 0:
                    self.train_metrics[param_name].update(preds_i[mask], targets_i[mask])

            # Return loss for optimizer        
            return loss

        ### CAGRAD MANUAL OPTIMIZATION LOGIC ###
        opt = self.optimizers()

        # 1. Forward pass to get predictions
        y = batch.pop("targets")
        output = self.model(**batch)
        y_hat = self.recomposition_func(output) if self.recomposition_func else output
        
        # 2. Get individual task losses (both loss functions now support this)
        task_losses_dict = self.loss_func(y_hat, y, return_individual_losses=True)

        # 3. Compute gradients for each task
        task_grads = []
        valid_task_indices = sorted(task_losses_dict.keys())
        num_tasks = len(valid_task_indices)
        
        # Get only parameters that require gradients
        params = [p for p in self.parameters() if p.requires_grad]

        opt.zero_grad()

        for i in range(num_tasks):
            # Isolate the gradient for the current task
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            task_key = valid_task_indices[i]
            
            self.manual_backward(task_losses_dict[task_key], retain_graph=(i < num_tasks - 1)) 
            
            grad_vec = utils.parameters_to_vector([p.grad.clone() for p in params if p.grad is not None])
            task_grads.append(grad_vec)

        if task_grads:
            # 4. Compute CAGrad mathematical update
            update_vector = self.mtl_solver(task_grads)
            
            # 5. Overwrite the .grad attributes with the CAGrad direction
            utils.vector_to_parameters(update_vector, [p.grad for p in params if p.grad is not None])
            
            # 6. Step the Lightning optimizer
            opt.step()


        # 6. Step learning rate scheduler
        sch = self.lr_schedulers()
        if sch is not None:
             monitored_metric = self.trainer.callback_metrics.get("valid_loss_epoch")
             if monitored_metric is not None:
                 sch.step(monitored_metric)
             else:
                 sch.step()

        overall_loss = torch.mean(torch.stack(list(task_losses_dict.values()))) if task_losses_dict else 0
        self.log('train_loss', overall_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update metrics (this is now done outside the loss calculation)
        for i, param_name in enumerate(self.target_columns):
            preds_i = y_hat[:, i]
            targets_i = y[:, i]
            mask = ~torch.isnan(targets_i)
            if mask.sum() > 0:
                self.train_metrics[param_name].update(preds_i[mask], targets_i[mask])

    def validation_step(self, batch, batch_idx):
        """Functionality for prediction validation."""
        # Compute shared step
        loss, y_hat, y = self._shared_step(batch)

        # Log loss
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update the metric state
        for i, param_name in enumerate(self.target_columns):
            preds_i = y_hat[:, i]
            targets_i = y[:, i]
            
            mask = ~torch.isnan(targets_i)
            if mask.sum() > 0:
                self.valid_metrics[param_name].update(preds_i[mask], targets_i[mask])
        
        # Return predictions for PredictionSaver callback
        return {'preds': y_hat, 'targets': y}

    def on_training_epoch_end(self):
        """Functionality for logging training metrics at every epoch."""
        all_metrics = {}
        for param_name in self.target_columns:
            # Compute all metrics for the current parameter
            param_metrics = self.train_metrics[param_name].compute()
            all_metrics.update(param_metrics)
            # Reset for the next epoch
            self.train_metrics[param_name].reset()
        
        self.log_dict(all_metrics, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        """Functionality for logging training metrics at every epoch."""
        all_metrics = {}
        for param_name in self.target_columns:
            # Compute all metrics for the current parameter
            param_metrics = self.valid_metrics[param_name].compute()
            all_metrics.update(param_metrics)
            # Reset for the next epoch
            self.valid_metrics[param_name].reset()

        self.log_dict(all_metrics, on_step=False, on_epoch=True)

# Example usage
""" 
model = AstraModule()
trainer = L.Trainer()
trainer.fit(model, datamodule)
trainer.test(model, dataloaders=DataLoader(test_set)) 
"""