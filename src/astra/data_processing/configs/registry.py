import torch

from astra.model.models import DummyModel, CpiPredConvModel, CpiPredSelfAttnModel, CpiPredCrossAttnModel
from astra.model.loss.masked_mse_loss import MaskedMSELoss
from astra.data_processing.featurizers import ESMFeaturizer, MorganFeaturizer
from astra.model.modules.kinetics import elemtary_to_michaelis_menten_basic, elemtary_to_michaelis_menten_advanced

# Models
MODEL_REGISTRY = {
    "DummyModel": DummyModel,
    "CpiPredConvModel": CpiPredConvModel,
    "CpiPredSelfAttnModel": CpiPredSelfAttnModel,
    "CpiPredCrossAttnModel": CpiPredCrossAttnModel,
}

# Optimizers
OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

# Loss functions
LOSS_FN_REGISTRY = {
    "MaskedMSELoss": MaskedMSELoss,
    "MSELoss": torch.nn.MSELoss,
}

# Featurizers
FEATURIZER_REGISTRY = {
    "ESMFeaturizer": ESMFeaturizer,
    "MorganFeaturizer": MorganFeaturizer,
}

# Kinetic ecomposition functions
RECOMPOSITION_REGISTRY = {
    "BasicRecomp": elemtary_to_michaelis_menten_basic,
    "AdvancedRecomp": elemtary_to_michaelis_menten_advanced,
}

RECOMP_INPUT_DIMS = {
    "BasicRecomp": 3,
    "AdvancedRecomp": 5,
}

# Learning rate schedulers
SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}