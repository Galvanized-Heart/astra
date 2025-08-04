import torch
import warnings
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from astra.data_processing.datamodules import AstraDataModule
from astra.model.lightning_models import AstraModule
from astra.data_processing.configs.registry import (
    MODEL_REGISTRY, OPTIMIZER_REGISTRY, LOSS_FN_REGISTRY, FEATURIZER_REGISTRY,
    SCHEDULER_REGISTRY, RECOMPOSITION_REGISTRY, RECOMP_INPUT_DIMS
)

class PipelineBuilder:
    """Builds the entire training pipeline from a validated Pydantic config object."""

    def __init__(self, config):
        """
        Args:
            config: A validated FullConfig Pydantic object.
        """
        self.config = config
        self.protein_featurizer = None
        self.ligand_featurizer = None
        self.datamodule = None
        self.model_architecture = None
        self.model = None

    def build_featurizers(self):
        print("INFO: Building featurizers...")
        p_cfg = self.config.featurizers.protein
        l_cfg = self.config.featurizers.ligand

        # TODO: Make more robust device logic for featurizers!
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if p_cfg.name == 'ESMFeaturizer': # This could be improved with a config flag or featurizer could have flag inside the class
            p_cfg.params['device'] = device

        self.protein_featurizer = FEATURIZER_REGISTRY[p_cfg.name](**p_cfg.params)
        self.ligand_featurizer = FEATURIZER_REGISTRY[l_cfg.name](**l_cfg.params)
        return self

    def build_datamodule(self):
        print("INFO: Building datamodule...")
        data_cfg = self.config.data
        self.datamodule = AstraDataModule(
            data_paths={'train': data_cfg.train_path, 'valid': data_cfg.valid_path},
            protein_featurizer=self.protein_featurizer,
            ligand_featurizer=self.ligand_featurizer,
            batch_size=data_cfg.batch_size,
            target_columns=data_cfg.target_columns,
            target_transform=data_cfg.target_transform
        )
        return self

    def build_model_architecture(self):
        print("INFO: Building model architecture...")
        arch_cfg = self.config.model.architecture
        model_params = arch_cfg.params.copy() # Use a copy to avoid modifying config

        # --- NEW, ROBUST out_dim VALIDATION ---
        recomp_func_name = self.config.model.lightning_module.recomposition_func
        
        if recomp_func_name:
            # Case 1: Recomposition is used. Model output must match its input needs.
            expected_out_dim = RECOMP_INPUT_DIMS.get(recomp_func_name)
            if expected_out_dim is None:
                raise ValueError(f"Recomposition function '{recomp_func_name}' has no defined input dimension in RECOMP_INPUT_DIMS.")
            validation_source = f"recomposition function '{recomp_func_name}'"
        else:
            # Case 2: No recomposition. Model output must match the number of targets.
            expected_out_dim = len(self.config.data.target_columns)
            validation_source = "number of target_columns"
            
        user_out_dim = model_params.get('out_dim')

        if user_out_dim is None:
            print(f"INFO: 'out_dim' not specified. Setting to {expected_out_dim} based on {validation_source}.")
            model_params['out_dim'] = expected_out_dim
        elif user_out_dim != expected_out_dim:
            raise ValueError(
                f"Configuration Mismatch: Model 'out_dim' ({user_out_dim}) is incorrect. "
                f"It must be {expected_out_dim} to match the {validation_source}. Please update your config."
            )

        model_params['protein_spec'] = self.datamodule.protein_feature_spec
        model_params['ligand_spec'] = self.datamodule.ligand_feature_spec
        self.model_architecture = MODEL_REGISTRY[arch_cfg.name](**model_params)
        return self

    def build_lightning_module(self):
        print("INFO: Building Lightning module...")
        lm_cfg = self.config.model.lightning_module
        
        # Loss function validation
        loss_fn_name = lm_cfg.loss_function
        n_targets = len(self.config.data.target_columns)
        if n_targets > 1 and loss_fn_name == "MSELoss":
             warnings.warn(f"Using 'MSELoss' for multi-target ({n_targets}) problem. Consider 'MaskedMSELoss'.")
        
        loss_func = LOSS_FN_REGISTRY[loss_fn_name]()

        recomp_func = RECOMPOSITION_REGISTRY.get(lm_cfg.recomposition_func)

        scheduler_class, scheduler_kwargs = None, {}
        if lm_cfg.lr_scheduler:
            scheduler_class = SCHEDULER_REGISTRY[lm_cfg.lr_scheduler.name]
            scheduler_kwargs = lm_cfg.lr_scheduler.params or {}

        self.model = AstraModule(
            model=self.model_architecture,
            lr=lm_cfg.lr,
            loss_func=loss_func,
            recomposition_func=recomp_func,
            optimizer_class=OPTIMIZER_REGISTRY[lm_cfg.optimizer],
            lr_scheduler_class=scheduler_class,
            lr_scheduler_kwargs=scheduler_kwargs
        )
        return self

    def build_trainer(self):
        print("INFO: Building Trainer...")
        trainer_cfg = self.config.trainer
        
        wandb_logger = WandbLogger(
            name=self.config.run_name,
            project=self.config.project_name,
            entity="lmse-university-of-toronto",
            log_model="all",
            config=self.config.dict() # Pass the validated config as a dict
        )
        
        cb_cfg = trainer_cfg.callbacks.checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=cb_cfg.monitor,
            dirpath="checkpoints/",
            filename=f"{self.config.run_name}-{{epoch:02d}}-{{{cb_cfg.monitor}:.2f}}",
            save_top_k=cb_cfg.save_top_k,
            mode=cb_cfg.mode,
            save_last=True
        )

        self.trainer = L.Trainer(
            max_epochs=trainer_cfg.epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            deterministic=(self.config.seed is not None),
            accelerator=trainer_cfg.device
        )
        return self

    def run(self):
        """Build all components and run the training loop."""
        self.build_featurizers()
        self.build_datamodule()
        self.build_model_architecture()
        self.build_lightning_module()
        self.build_trainer()
        
        print("\n--- LAUNCHING TRAINING ---")
        self.trainer.fit(self.model, self.datamodule)