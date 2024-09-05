from lightning.pytorch import Callback, Trainer
from lightning.fabric.utilities import rank_zero_only
from multifactor_metrics.latent_space_automatic_explorer import extract_latent_code, get_mappings
from utils.koopman_utils import intervention_based_metrics, consistency_metrics, \
    predictor_based_metrics
from utils.general_utils import load_data_for_explore_and_test

from model import KoopmanVAE


class MultifactorMetricsLogger(Callback):
    """
    Calculate and log the Multi-factor metrics.
    """

    def __init__(self, should_log_files: bool = False):
        super().__init__()

        # Save whether this
        self.should_log_files = should_log_files

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, model: KoopmanVAE) -> None:
        # Load the data for the exploration and the test.
        x, labels, val_loader = load_data_for_explore_and_test(model.device, model.dataset_dir_path)

        # Extract the latent codes from the model and check the output.
        ZL = extract_latent_code(model, x)
        if ZL is None:
            model.trainer.should_stop = True
            return

        # Get the mappings of the labels to subset of indices.
        map_label_to_idx = get_mappings(ZL, labels, model.multifactor_exploration_type,
                                        model.multifactor_classifier_type)

        # Calculate the metrics and log them.
        intervention_based_metrics(model, model.multifactor_classifier, val_loader, map_label_to_idx,
                                   model.multifactor_classifier.LABEL_TO_NAME_DICT,
                                   verbose=False,
                                   should_log_files=self.should_log_files)
        consistency_metrics(model, model.multifactor_classifier, val_loader, map_label_to_idx,
                            model.multifactor_classifier.LABEL_TO_NAME_DICT, verbose=False,
                            should_log_files=self.should_log_files)
        predictor_based_metrics(model, ZL, labels, map_label_to_idx, model.multifactor_classifier.LABEL_TO_NAME_DICT,
                                model.multifactor_dci_classifier_type, verbose=False,
                                should_log_files=self.should_log_files)
