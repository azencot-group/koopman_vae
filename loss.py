from dataclasses import dataclass
from torch import Tensor

@dataclass
class ModelLoss:
    """
    Dataclass for the different losses.
    """
    sum_loss_weighted: Tensor
    reconstruction_loss: Tensor
    x_pred_loss: Tensor
    z_pred_loss: Tensor
    spectral_loss: Tensor
