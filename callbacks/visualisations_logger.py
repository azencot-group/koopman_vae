import neptune
import pytorch_lightning as pl
import numpy as np
from lightning.fabric.utilities import rank_zero_only
from matplotlib import pyplot as plt

from experiments.disentenglement_swaps import swap
from utils.general_utils import reorder, imshow_seqeunce

from model import KoopmanVAE


class VisualisationsLoggerCallback(pl.Callback):
    """
    This module logs several visualisation images:
    1. Spectral image.
    2. Swap figure.
    3. Sampling using a prior.
    4. Reconstruction of z_pred.
    5. Reconstruction of z_post.

    This module should be called only in validation mode without grads as it transfers data through the model.
    Therefore, the preferred way using it is during the validation phase.
    Specifically, it will be called every validations_interval validation epochs.

    For example, if the validations_interval is 10, and the validation is called every 5 training epochs, the logging
    will be done every 5*10=50 epochs.

    This module can be run in a distributed manner, all the calculations will be done in the zero rank.
    """

    # The indices to swap in order to create the swap image.
    __SWAP_INDICES = [0, 1]

    # The index in the batch of the reconstructed image.
    __RECONSTRUCTION_INDEX = __SWAP_INDICES[0]

    # The name of the visualizations dir in Neptune.
    __VISUALISATION_DIR = "visualisations"

    # The name of the spectral figures' series.
    __SPECTRAL_FIGURES_SERIES_NAME = "spectral"

    # The name of the swap figures' series.
    __SWAP_FIGURES_SERIES_NAME = "swap"

    # The name of the reconstruction figures' series.
    __RECONSTRUCTION_FIGURES_SERIES_NAME = "reconstruction"

    # The name of the sampled figures' series.
    __SAMPLED_FIGURES_SERIES_NAME = "spectral"

    def __init__(self, validations_interval: int):
        super().__init__()

        # Save the validations_interval.
        self._validations_interval = validations_interval

        # Initialize the validations counter.
        self._validation_epochs_counter = 0

    @classmethod
    @rank_zero_only
    def log_figure_to_series(cls, trainer: pl.Trainer, series_name: str, figure: plt.Figure):
        trainer.logger.experiment[f'{cls.__VISUALISATION_DIR}/{series_name}'].append(
            value=figure,
            name=f'{series_name}-epoch-{trainer.current_epoch}')

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, model: KoopmanVAE) -> None:
        # Increase the counter.
        self._validation_epochs_counter += 1

        # Check whether the log should be run.
        if ((self._validation_epochs_counter % self._validations_interval == 0) and
                not model.koopman_layer.is_matrix_singular):
            # Get the validation dataloader.
            val_loader = trainer.val_dataloaders

            # Get a new batch from the validation loader.
            x = reorder(next(iter(val_loader))['images']).to(model.device)

            # Transfer the data through the model.
            outputs = model(x)
            if outputs is None:
                trainer.should_stop = True
                return

            # Unpack the outputs.
            (z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, z_post_koopman, z_post_dropout,
             Ct, koopman_recon_x, recon_x) = outputs

            # Get the Spectral image and the swap plots.
            spectral_fig, swap_fig = swap(model, x, z_post, Ct, self.__SWAP_INDICES, model.koopman_layer.static_size)

            # Get the reconstruction plots.
            titles = ['Original image:', 'Reconstructed image koopman:', 'Original image:',
                      'Reconstructed image normal:']
            reconstructions_fig = imshow_seqeunce(
                [[x[self.__RECONSTRUCTION_INDEX]], [koopman_recon_x[self.__RECONSTRUCTION_INDEX]],
                 [x[self.__RECONSTRUCTION_INDEX]], [recon_x[self.__RECONSTRUCTION_INDEX]]],
                titles=np.asarray([titles]).T)

            # Sample an example from the model.
            sampled_examples = model.sample_x(batch_size=4, random_sampling=True)
            titles = ['Sampling example #1:', 'Sampling example #2:', 'Sampling example #3:',
                      'Sampling example #4:']
            samples_fig = imshow_seqeunce([[sampled_examples[0]], [sampled_examples[1]], [sampled_examples[2]],
                                           [sampled_examples[3]]],
                                          titles=np.asarray([titles]).T)

            # Log the spectral figure.
            self.log_figure_to_series(trainer, self.__SPECTRAL_FIGURES_SERIES_NAME, spectral_fig)

            # Log the swap figure.
            self.log_figure_to_series(trainer, self.__SWAP_FIGURES_SERIES_NAME, swap_fig)

            # Log the reconstruction figure.
            self.log_figure_to_series(trainer, self.__RECONSTRUCTION_FIGURES_SERIES_NAME, reconstructions_fig)

            # Log the sampled exampled figures.
            self.log_figure_to_series(trainer, self.__SAMPLED_FIGURES_SERIES_NAME, samples_fig)
