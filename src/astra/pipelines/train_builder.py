import torch
import warnings
import copy
from pathlib import Path
from datetime import datetime

# --- PyTorch Lightning and W&B ---
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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

    def __init__(self, config_dict: dict):
        """
        Initializes the builder by creating a fully validated and resolved configuration.

        Args:
            config_dict (dict): The raw dictionary loaded directly from the YAML file.
        """
        print("INFO: Initializing PipelineBuilder...")
        # Correct and validate config provided by user
        self.final_config: FullConfig = self._resolve_and_validate_config(config_dict)
        print("INFO: Configuration resolved and validated successfully!")

        # Initialize other attributes to be populated by the build methods.
        self.protein_featurizer = None
        self.ligand_featurizer = None
        self.datamodule = None
        self.model_architecture = None
        self.model = None
        self.trainer = None

    def _resolve_and_validate_config(self, config_dict: dict) -> FullConfig:
        """
        Takes a raw config dict, applies defaults and validation logic, and returns
        a validated Pydantic FullConfig object.

        Args:
            config_dict (dict): The raw user-provided configuration.

        Returns:
            FullConfig: A validated and resolved Pydantic configuration object.
        
        Raises:
            ValueError: If the configuration is logically inconsistent.
            pydantic.ValidationError: If the configuration violates the defined schema.
        """
        resolved_config = copy.deepcopy(config_dict)

        # Resolve `target_columns` and `loss_function`
        data_cfg = resolved_config.setdefault('data', {})
        target_columns = data_cfg.setdefault('target_columns', ["kcat", "KM", "Ki"])
        lm_cfg = resolved_config['model']['lightning_module']
        if 'loss_function' not in lm_cfg or lm_cfg['loss_function'] is None:
            lm_cfg['loss_function'] = "MaskedMSELoss" if len(target_columns) > 1 else "MSELoss"
            print(f"INFO: Defaulted 'loss_function' to '{lm_cfg['loss_function']}'.")
        
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

        # Validate corrected configuration with Pydantic
        try:
            return FullConfig(**resolved_config)
        except ValidationError as e:
            print("\nERROR: The final, resolved configuration is invalid!")
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
            data_paths={'train': data_cfg.train_path, 'valid': data_cfg.valid_path},
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

        self.model = AstraModule(
            model=self.model_architecture,
            lr=lm_cfg.lr,
            loss_func=loss_func,
            optimizer_class=optimizer_class,
            target_columns=target_columns,
            recomposition_func=recomp_func,
            lr_scheduler_class=scheduler_class,
            lr_scheduler_kwargs=scheduler_kwargs
        )
        return self

    def build_trainer(self, extra_callbacks: list = None):
        print("INFO: Building Trainer...")
        trainer_cfg = self.final_config.trainer

        run_name_prefix = self.final_config.run_name
        if run_name_prefix is None:
            # Create descriptive run name if None is provided
            # Format: ModelName_YYYYMMDD-HHMMSS
            model_name = self.final_config.model.architecture.name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name_prefix = f"{model_name}_{timestamp}"
            print(f"INFO: run_name not provided. Generated run name: {run_name_prefix}")

        wandb_logger = WandbLogger(
            name=run_name_prefix,  # Use wandb generated or user provided name
            project=self.final_config.project_name,
            entity="lmse-university-of-toronto",
            log_model="all",
            config=self.final_config.dict() # Log the final, resolved config
        )

        checkpoint_dir = f"checkpoints/{run_name_prefix}"
        print(f"INFO: Checkpoints will be saved in: {checkpoint_dir}")

        cb_cfg = trainer_cfg.callbacks.checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=cb_cfg.monitor,
            dirpath=checkpoint_dir,
            filename=f"{{epoch:02d}}-{{{cb_cfg.monitor}:.2f}}",
            save_top_k=cb_cfg.save_top_k,
            mode=cb_cfg.mode,
            save_last=True
        )

        callbacks = [checkpoint_callback]

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

        # Return metric for Optuna
        final_metric = self.trainer.callback_metrics.get(self.final_config.trainer.callbacks.checkpoint.monitor)
        if final_metric is not None:
            return final_metric.item()
        else:
            print(f"WARNING: Monitored metric '{self.final_config.trainer.callbacks.checkpoint.monitor}' not found in trainer.callback_metrics.")
            return float('inf')