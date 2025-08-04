import torch

from astra.model.models.dummy_model import DummyModel
from astra.model.loss.masked_mse_loss import MaskedMSELoss
from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer

MODEL_REGISTRY = {
    "DummyModel": DummyModel,
     #"CpiPredConvModel": CpiPredConvModel,
     #"CpiPredSelfAttnModel": CpiPredSelfAttnModel,
     #"CpiPredCrossAtnnModel": CpiPredCrossAtnnModel,
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
}