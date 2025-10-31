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
    def __init__(self, target_columns, split_tag: str):
        super().__init__()
        self.target_columns = target_columns
        self.split_tag = split_tag
        self._val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
            
        # Determine if we need to save for "best" or "last"
        is_best = self._is_best_epoch(trainer)
        # Check if it's the last validation epoch (current_epoch is 0-indexed)
        is_last = trainer.current_epoch == trainer.max_epochs - 1

        # Save the 'best' predictions if this is the best epoch so far.
        if is_best:
            print(f"\nINFO: New best model found. Saving 'best' predictions...")
            self._save_and_log(trainer, "best")

        # Save the 'last' predictions if this is the final epoch.
        # This will run even if the last epoch is also the best, creating a separate file.
        if is_last:
            print(f"\nINFO: Final epoch reached. Saving 'last' predictions...")
            self._save_and_log(trainer, "last")

        # Clear the stored outputs for the next validation epoch
        self._val_outputs.clear()

    def _save_and_log(self, trainer: L.Trainer, save_type: str):
        """
        A helper method to handle the logic of gathering, saving, and logging predictions.
        
        Args:
            trainer (L.Trainer): The trainer object.
            save_type (str): The type of save, either 'best' or 'last'.
        """
        # 1. Construct the path with a filename based on the save_type
        base_dir = "predictions"
        run_name = trainer.logger.experiment.name
        save_dir = Path(base_dir) / self.split_tag / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{save_type}_model_preds.csv"
        
        # 2. Gather predictions and targets
        preds = torch.cat([x['preds'] for x in self._val_outputs]).cpu().numpy()
        targets = torch.cat([x['targets'] for x in self._val_outputs]).cpu().numpy()

        # 3. Create and save the DataFrame
        pred_cols = [f"{col}_pred" for col in self.target_columns]
        results_df = pd.DataFrame(
            data=np.hstack([targets, preds]),
            columns=self.target_columns + pred_cols
        )
        results_df.to_csv(save_path, index=False)
        
        # 4. Log rich metadata to W&B
        absolute_path = save_path.resolve()
        print(f"INFO: Saved '{save_type}' predictions to: {absolute_path}")

        if wandb and isinstance(trainer.logger, WandbLogger):
            summary_data = {
                f"{save_type}_predictions_path": str(absolute_path)
            }
            
            trainer.logger.experiment.log(summary_data)
            
            print(f"INFO: Logged '{save_type}' prediction metadata to W&B.")

    def _is_best_epoch(self, trainer: L.Trainer) -> bool:
        checkpoint_callback = trainer.checkpoint_callback
        return (
            checkpoint_callback.best_model_score is not None and
            checkpoint_callback.current_score == checkpoint_callback.best_model_score
        )