from dataclasses import dataclass

@dataclass
class ModelLoss:
    """
    Dataclass for the different losses.
    """
    sum_loss_weighted: float
    reconstruction_loss: float
    kl_divergence: float
    x_pred_loss: float
    z_pred_loss: float
    spectral_loss: float
