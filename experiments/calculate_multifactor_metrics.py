import torch
import sys
import os
from typing import TYPE_CHECKING
from lightning.pytorch import seed_everything

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from input_parser.basic_input_parser import define_basic_args
from datamodule.sprite_datamodule import SpriteDataModule
from callbacks.multifactor_metrics_logger import MultifactorMetricsLogger
from multifactor_classifier import MultifactorSpritesClassifier

from model import KoopmanVAE


def define_args():
    # Define the basic arguments of the model.
    parser = define_basic_args()

    parser.add_argument('--model_path', type=str, default=None, help='ckpt directory')

    return parser


if __name__ == '__main__':
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Create the data module.
    data_module = SpriteDataModule(args.dataset_path, args.batch_size)
    data_module.setup("")
    validation_loader = data_module.val_dataloader()

    # Create the model.
    model = KoopmanVAE.load_from_checkpoint(args.model_path, strict=False)
    model.eval()

    # Initialize the classifier.
    multifactor_classifier = MultifactorSpritesClassifier(args)
    loaded_dict = torch.load(args.multifactor_classifier_path)
    multifactor_classifier.load_state_dict(loaded_dict)
    multifactor_classifier.cuda().eval()

    # Evaluate the metrics.
    MultifactorMetricsLogger.calculate_and_log_metrics(model, multifactor_classifier,
                                                       multifactor_classifier.LABEL_TO_NAME_DICT,
                                                       args.multifactor_dci_classifier_type,
                                                       args.multifactor_exploration_type,
                                                       args.multifactor_classifier_type, verbose=True)
