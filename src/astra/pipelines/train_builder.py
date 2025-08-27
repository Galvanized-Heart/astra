import torch
import warnings
import copy
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

# --- PyTorch Lightning and W&B ---
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from astra.model.callbacks.predsaver import PredictionSaver

# --- Astra Imports ---
from astra.data_processing.datamodules import AstraDataModule
from astra.model.lightning_models import AstraModule
from astra.data_processing.configs.registry import (
    MODEL_REGISTRY, OPTIMIZER_REGISTRY, LOSS_FN_REGISTRY, FEATURIZER_REGISTRY,
    SCHEDULER_REGISTRY, RECOMPOSITION_REGISTRY, RECOMP_INPUT_DIMS
)
from astra.data_processing.configs.config_schema import FullConfig, ValidationError


class PipelineBuilder:
    """
    Constructs the entire training pipeline from a configuration dictionary.

    The builder validates and resolves the configuration upon instantiation by calling
    an internal helper method, ensuring the object is always in a valid state.
    """

    def __init__(self, config_dict: DictConfig):
        """
        Initializes the builder by creating a fully validated and resolved configuration.

        Args:
            config_dict (DictConfig): The raw config object from Hydra.
        """
        print("INFO: Initializing PipelineBuilder...")
        # The config is now a DictConfig object, not a dict
        self.final_config: FullConfig = self._resolve_and_validate_config(config_dict)
        print("INFO: Configuration resolved and validated successfully!")

        # Initialize other attributes to be populated by the build methods.
        self.protein_featurizer = None
        self.ligand_featurizer = None
        self.datamodule = None
        self.model_architecture = None
        self.model = None
        self.trainer = None

    def _resolve_and_validate_config(self, config_dict: DictConfig) -> FullConfig:
        """
        Takes a raw config dict, applies defaults and validation logic, and returns
        a validated Pydantic FullConfig object.

        Args:
            config_dict (DictConfig): The raw config object from Hydra.

        Returns:
            FullConfig: A validated and resolved Pydantic configuration object.
        
        Raises:
            ValueError: If the configuration is logically inconsistent.
            pydantic.ValidationError: If the configuration violates the defined schema.
        """
        resolved_config = config_dict.copy()

        # Temporarily disable struct mode to allow dynamic modifications
        OmegaConf.set_struct(resolved_config, False)

        # Resolve `target_columns` and establish lightning_module config (`lm_cfg`)
        data_cfg = resolved_config.setdefault('data', {})
        target_columns = data_cfg.setdefault('target_columns', ["kcat", "KM", "Ki"])
        lm_cfg = resolved_config['model']['lightning_module']

        # Normalize the recomposition_func from a potentially empty dict to None
        recomp_func_raw = lm_cfg.get('recomposition_func')
        if isinstance(recomp_func_raw, dict) and not recomp_func_raw:
            print("INFO: 'recomposition_func' was an empty dict. Coercing to 'None'.")
            lm_cfg['recomposition_func'] = None

        # Normalize the loss_function part of the config to always be a dictionary
        loss_config_raw = lm_cfg.get('loss_function', {})
        if isinstance(loss_config_raw, str):
            loss_config_dict = {'name': loss_config_raw, 'params': {}}
        elif loss_config_raw is None:
            loss_config_dict = {}
        else:
            loss_config_dict = loss_config_raw
        
        # Ensure a 'params' key exists within the loss function config
        loss_config_dict.setdefault('params', {})

        # Unconditionally determine the correct loss function name based on the number of targets
        correct_loss_name = "MaskedMSELoss" if len(target_columns) > 1 else "MSELoss"
        
        # If the current name is different, update it and print a helpful message.
        if loss_config_dict.get('name') != correct_loss_name:
             print(f"INFO: Overriding loss function. Setting name to '{correct_loss_name}' based on {len(target_columns)} target column(s).")
             loss_config_dict['name'] = correct_loss_name

        # Clean up incompatible parameters. This logic is now guaranteed to work correctly.
        if loss_config_dict['name'] == 'MSELoss' and 'weights' in loss_config_dict['params']:
            print("INFO: 'weights' parameter is not applicable for 'MSELoss'. Removing from config.")
            del loss_config_dict['params']['weights']

        # Update the main config with the fully resolved and cleaned loss function dictionary
        lm_cfg['loss_function'] = loss_config_dict

        # Resolve the model's `out_dim`
        arch_params = resolved_config['model']['architecture'].setdefault('params', {})
        recomp_func_name = lm_cfg.get('recomposition_func')

        expected_out_dim, source = (
            (RECOMP_INPUT_DIMS[recomp_func_name], f"recomposition function '{recomp_func_name}'")
            if recomp_func_name else (len(target_columns), "number of target_columns")
        )
        
        user_out_dim = arch_params.get('out_dim')
        if user_out_dim is None:
            arch_params['out_dim'] = expected_out_dim
            print(f"INFO: Set 'out_dim' to {expected_out_dim} based on {source}.")
        elif user_out_dim != expected_out_dim:
            raise ValueError(f"Config Mismatch: Model 'out_dim' ({user_out_dim}) must be {expected_out_dim} to match {source}.")

        # Re-enable struct mode for safety
        OmegaConf.set_struct(resolved_config, True)

        # Validate corrected configuration with Pydantic
        try:
            # Convert the OmegaConf object to a primitive python dict for Pydantic.
            # `resolve=True` ensures all interpolations like `${...}` are resolved to their values.
            final_dict = OmegaConf.to_container(resolved_config, resolve=True, throw_on_missing=True)
            return FullConfig(**final_dict)
        except ValidationError as e:
            print("\nERROR: The final, resolved configuration is invalid!")
            # Also print the resolved config to help debug
            print("--- Final Resolved Config ---")
            print(OmegaConf.to_yaml(resolved_config))
            print("-----------------------------")
            raise e

    def build_featurizers(self):
        print("INFO: Building featurizers...")
        p_cfg = self.final_config.featurizers.protein
        l_cfg = self.final_config.featurizers.ligand

        # This logic could be improved with a `requires_device` flag in the config schema.
        # For now, this pragmatic approach works for ESM.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if p_cfg.name == 'ESMFeaturizer':
            p_cfg.params['device'] = device
            

        self.protein_featurizer = FEATURIZER_REGISTRY[p_cfg.name](**p_cfg.params)
        self.ligand_featurizer = FEATURIZER_REGISTRY[l_cfg.name](**l_cfg.params)
        return self

    def build_datamodule(self):
        print("INFO: Building datamodule...")
        data_cfg = self.final_config.data
        self.datamodule = AstraDataModule(
            data_paths={'train': str(data_cfg.train_path), 'valid': str(data_cfg.valid_path)},
            protein_featurizer=self.protein_featurizer,
            ligand_featurizer=self.ligand_featurizer,
            batch_size=data_cfg.batch_size,
            featurizer_batch_size=data_cfg.featurizer_batch_size,
            target_columns=data_cfg.target_columns,
            target_transform=data_cfg.target_transform
        )
        return self

    def build_model_architecture(self):
        print("INFO: Building model architecture...")
        arch_cfg = self.final_config.model.architecture
        model_params = arch_cfg.params.copy()

        # Add the dynamically generated featurizer specs
        model_params['protein_emb_dim'] = self.datamodule.protein_feature_spec
        model_params['ligand_emb_dim'] = self.datamodule.ligand_feature_spec
        
        self.model_architecture = MODEL_REGISTRY[arch_cfg.name](**model_params)
        return self

    def build_lightning_module(self):
        print("INFO: Building Lightning module...")
        lm_cfg = self.final_config.model.lightning_module
        
        loss_cfg = lm_cfg.loss_function
        loss_func = LOSS_FN_REGISTRY[loss_cfg.name](**loss_cfg.params)
        recomp_func = RECOMPOSITION_REGISTRY.get(lm_cfg.recomposition_func)
        optimizer_class = OPTIMIZER_REGISTRY[lm_cfg.optimizer]

        scheduler_class, scheduler_kwargs = None, {}
        if lm_cfg.lr_scheduler:
            scheduler_class = SCHEDULER_REGISTRY[lm_cfg.lr_scheduler.name]
            scheduler_kwargs = lm_cfg.lr_scheduler.params or {}

        target_columns = self.final_config.data.target_columns

        log_transform_active = (self.final_config.data.target_transform == "log10")
        if log_transform_active:
            print("INFO: Log10 transformation is active. Recomposition will operate in log space.")

        self.model = AstraModule(
            model=self.model_architecture,
            lr=lm_cfg.lr,
            loss_func=loss_func,
            optimizer_class=optimizer_class,
            target_columns=target_columns,
            recomposition_func=recomp_func,
            log_transform_active=log_transform_active,
            lr_scheduler_class=scheduler_class,
            lr_scheduler_kwargs=scheduler_kwargs
        )
        return self

    def build_trainer(self, extra_callbacks: list = None):
        print("INFO: Building Trainer...")
        trainer_cfg = self.final_config.trainer

        # Get wandb from configs
        wandb_cfg = self.final_config.wandb or {}
        user_provided_name = self.final_config.run_name
        model_name = self.final_config.model.architecture.name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if user_provided_name:
            # Case 1: A user provides a 'run_name' in the config (for standalone runs).
            final_run_name = f"{user_provided_name}_{model_name}_{timestamp}"
            print(f"INFO: Using user-provided name to generate run name: {final_run_name}")
        elif wandb_cfg.get("name"):
            # Case 2: The orchestrator provides a name in the 'wandb' dict.
            base_name = wandb_cfg.get("name")
            final_run_name = f"{base_name}_{model_name}_{timestamp}"
            print(f"INFO: Using orchestrator-provided name to generate run name: {final_run_name}")
        else:
            # Case 3: No name was provided at all.
            final_run_name = f"{model_name}_{timestamp}"
            print(f"INFO: No run name provided. Generated default name: {final_run_name}")

        raw_tags = self.final_config.tags or []
        final_tags = []
        for tag in raw_tags:
            if tag is None:
                continue
            if isinstance(tag, list) or isinstance(tag, tuple):
                # If the tag is a list (like target_columns), extend the final list
                final_tags.extend([str(t) for t in tag])
            else:
                # Otherwise, just append the tag
                final_tags.append(str(tag))

        # Remove duplicates and sort for consistency
        final_tags = sorted(list(set(final_tags)))
        print(f"INFO: Final W&B tags for this run: {final_tags}")

        group_name = wandb_cfg.get("group")

        split_tag = "untagged"  # A safe default if no split tag is found
        for tag in final_tags:
            if tag.endswith("_split"):
                split_tag = tag
                break  # Found the first one, stop searching
        print(f"INFO: Using split tag '{split_tag}' for predictions directory.")

        # Instantiate wandb logger
        wandb_logger = WandbLogger(
            name=final_run_name,
            project=self.final_config.project_name,
            entity="lmse-university-of-toronto",
            group=group_name,
            tags=final_tags,
            log_model=False,
            config=self.final_config.dict()
        )

        if group_name:
            checkpoint_dir = f"checkpoints/{group_name}/{final_run_name}"
        else:
            checkpoint_dir = f"checkpoints/{final_run_name}"
        print(f"INFO: Checkpoints will be saved in: {checkpoint_dir}")

        # Instantiate model checkpoint callback
        cb_cfg = trainer_cfg.callbacks.checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=cb_cfg.monitor,
            dirpath=checkpoint_dir,
            filename=f"{{epoch:02d}}-{{{cb_cfg.monitor}:.2f}}",
            save_top_k=cb_cfg.save_top_k,
            mode=cb_cfg.mode,
            save_last=True
        )

        # Instantiate prediction saver callback
        prediction_saver_callback = PredictionSaver(target_columns=self.final_config.data.target_columns, split_tag=split_tag)

        callbacks = [checkpoint_callback, prediction_saver_callback]

        # If any extra_callbacks were passed, add them.
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
            print(f"INFO: Added {len(extra_callbacks)} extra callbacks.")

        self.trainer = L.Trainer(
            max_epochs=trainer_cfg.epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=(self.final_config.seed is not None),
            accelerator=trainer_cfg.device
        )
        return self

    def run(self, extra_callbacks: list = None):
        """Builds all components in order and starts the training process."""
        try:
            seed = self.final_config.seed
            if seed is not None:
                L.seed_everything(seed, workers=True)

            self.build_featurizers()
            self.build_datamodule()
            self.build_model_architecture()
            self.build_lightning_module()
            self.build_trainer(extra_callbacks=extra_callbacks)
            
            print("\n--- LAUNCHING TRAINING ---")
            self.trainer.fit(self.model, self.datamodule)

            # Get checkpoint callback
            checkpoint_callback = None
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    checkpoint_callback = cb
                    break

            # Upload path to best model checkpoint to wandb
            if checkpoint_callback and checkpoint_callback.best_model_path:
                best_path = checkpoint_callback.best_model_path
                print(f"INFO: Best model saved locally at: {best_path}")
                
                # Log the path as text metadata to the run's summary
                self.trainer.logger.experiment.summary["best_local_checkpoint_path"] = best_path

            # Return metric
            final_metric = self.trainer.callback_metrics.get(self.final_config.trainer.callbacks.checkpoint.monitor)
            if final_metric is not None:
                return final_metric.item()
            else:
                print(f"WARNING: Monitored metric '{self.final_config.trainer.callbacks.checkpoint.monitor}' not found in trainer.callback_metrics.")
                return float('inf')
            
        finally:
            # Ensure that the W&B run is properly finished and the process is cleaned up
            if self.trainer and self.trainer.logger:
                print("INFO: Finalizing W&B run...")
                # The logger is a WandbLogger instance, its experiment is the run object
                self.trainer.logger.experiment.finish()
                print("INFO: W&B run finalized.")