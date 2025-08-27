import lightning as L
import torch
import pandas as pd
from pathlib import Path
import numpy as np

# Try to import wandb, but don't make it a hard requirement
try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    wandb = None
    WandbLogger = None

class PredictionSaver(L.Callback):
    def __init__(self, target_columns, split_tag):
        super().__init__()
        self.target_columns = target_columns
        self.split_tag = split_tag
        self._val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # outputs is {'preds': y_hat, 'targets': y} from validation_step
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only perform saving/logging on the main process (rank 0) in DDP
        if not trainer.is_global_zero:
            return

        # Check if this epoch is the best one so far
        if self._is_best_epoch(trainer):
            save_path = self._construct_save_path(trainer)

            # 1. Gather all predictions and targets
            preds = torch.cat([x['preds'] for x in self._val_outputs]).cpu().numpy()
            targets = torch.cat([x['targets'] for x in self._val_outputs]).cpu().numpy()

            # 2. Create a DataFrame
            pred_cols = [f"{col}_pred" for col in self.target_columns]
            results_df = pd.DataFrame(
                data=np.hstack([targets, preds]),
                columns=self.target_columns + pred_cols
            )

            # Save results
            results_df.to_csv(save_path, index=False)
            
            # --- THIS IS THE MODIFIED PART ---
            # 4. Log the local file path to the W&B run summary
            self._log_path_to_wandb_summary(trainer, save_path)

        # Clear stored outputs for the next validation epoch
        self._val_outputs.clear()

    def _construct_save_path(self, trainer: L.Trainer) -> Path:
        """Constructs a descriptive, unique path for the predictions CSV."""
        # Base directory for all predictions
        base_dir = "predictions"
        
        # Get the unique run name from the logger
        run_name = trainer.logger.experiment.name

        # Create the specific directory for this run
        save_dir = Path(base_dir) / self.split_tag / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filename
        filename = f"best_model_preds.csv"
        
        return save_dir / filename

    def _is_best_epoch(self, trainer: L.Trainer) -> bool:
        """Checks if the current epoch is the best one according to the checkpoint callback."""
        checkpoint_callback = trainer.checkpoint_callback
        return (
            checkpoint_callback.best_model_score is not None and
            checkpoint_callback.current_score == checkpoint_callback.best_model_score
        )
    
    def _log_path_to_wandb_summary(self, trainer: L.Trainer, file_path: Path):
        """Logs the absolute path of the prediction file to the W&B run's summary."""
        # It's best practice to resolve to an absolute path for unambiguous reference
        absolute_path = file_path.resolve()

        if wandb and isinstance(trainer.logger, WandbLogger):
            # Access the summary dictionary and add our custom key
            trainer.logger.experiment.log({"best_predictions_path": str(absolute_path)})
        else:
            print("WARNING: W&B logger not found. Skipping prediction path logging.")