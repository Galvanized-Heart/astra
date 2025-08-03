# astra/utils/registry.py
import torch
from astra.model.models.dummy_model import DummyModel
# Import your other models here as you create them
# from astra.model.models.real_model import RealModel
from astra.model.loss.masked_mse_loss import MaskedMSELoss
from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer

MODEL_REGISTRY = {
    "DummyModel": DummyModel,
     
}

OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

LOSS_FN_REGISTRY = {
    "MaskedMSELoss": MaskedMSELoss,
    "MSELoss": torch.nn.MSELoss,
}

FEATURIZER_REGISTRY = {
    "ESMFeaturizer": ESMFeaturizer,
    "MorganFeaturizer": MorganFeaturizer,
}

SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    # Add other schedulers as needed
}