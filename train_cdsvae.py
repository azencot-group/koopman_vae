import os
import torch
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import NeptuneLogger
from datamodule.sprite_datamodule import SpriteDataModule

from input_parser.basic_input_parser import define_basic_args
from neptune_tags_manager.neptune_tags_manager import NeptuneTagsManager
from experiments.disentenglement_swaps import swap_within_batch
from utils.general_utils import reorder
from model import KoopmanVAE


@rank_zero_only
def create_eigenimage(model, static_size):
    # Get a new validation loader.
    validation_loader = data_module.val_dataloader()

    # Get a new batch from the validation loader.
    x = reorder(next(iter(validation_loader))['images']).to(model.device)

    # Create the eigenimage.
    fig = swap_within_batch(model, x, first_idx=0, second_idx=1, static_size=static_size, plot=False)

    return fig

if __name__ == '__main__':
    # Receive the hyperparameters.
    parser = define_basic_args()
    args = parser.parse_args()

    # Create the name of the model.
    args.model_name = f'KoopmanVAE_Sprite' \
                      f'_epochs={args.epochs}' \
                      f'_lstm={args.lstm}' \
                      f'_conv={args.conv_dim}' \
                      f'_k_dim={args.k_dim}' \
                      f'_conv_out={args.conv_output_dim}' \
                      f'_enc_lstm={args.encoder_lstm_output_dim}' \
                      f'_prior_in={args.prior_lstm_inner_size}' \
                      f'_dec_lstm={args.decoder_lstm_output_size}' \
                      f'_drop={args.dropout}' \
                      f'_stat_num={args.static_size}' \
                      f'_dyn_mode={args.dynamic_mode}' \
                      f'_ball={args.ball_thresh}' \
                      f'_dyn={args.dynamic_thresh}' \
                      f'_eigs={args.eigs_thresh}' \
                      f'_klz={args.weight_kl_z}' \
                      f'_xpred={args.weight_x_pred}' \
                      f'_zpred={args.weight_z_pred}' \
                      f'_spec={args.weight_spectral}' \
 \
        # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Create model.
    model = KoopmanVAE(args)

    # Create the checkpoints.
    current_training_logs_dir = os.path.join(args.models_during_training_dir, args.model_name)
    checkpoint_every_n = ModelCheckpoint(dirpath=current_training_logs_dir,
                                         filename="model-{epoch}",
                                         every_n_epochs=args.save_interval,
                                         save_on_train_epoch_end=True,
                                         save_last=True)
    checkpoint_best_fixed_content_purity = ModelCheckpoint(dirpath=current_training_logs_dir,
                                                           filename="model-{epoch}-content_purity={fixed_content_purity:.3f}",
                                                           auto_insert_metric_name=False,
                                                           save_top_k=args.save_n_val_best,
                                                           monitor="fixed_content_purity",
                                                           mode="max",
                                                           save_last=False)
    checkpoint_best_fixed_action_purity = ModelCheckpoint(dirpath=current_training_logs_dir,
                                                          filename="model-{epoch}-action_purity={fixed_action_purity:.3f}",
                                                          auto_insert_metric_name=False,
                                                          save_top_k=args.save_n_val_best,
                                                          monitor="fixed_action_purity",
                                                          mode="max",
                                                          save_last=False)

    # Check whether there is a checkpoint to resume from.
    last_checkpoint_path = os.path.join(current_training_logs_dir,
                                        checkpoint_every_n.CHECKPOINT_NAME_LAST + checkpoint_every_n.FILE_EXTENSION)
    checkpoint_to_resume = last_checkpoint_path if os.path.isfile(last_checkpoint_path) else None

    # Create the EarlyStopping callback.
    early_stop = EarlyStopping(monitor="val/sum_loss_weighted", patience=args.early_stop_patience, mode="min")

    # Create the logger.
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjg4NDkxMS04N2NhLTRkOTctYjY0My05NDY2OGU0NGJjZGMifQ==",
        project="azencot-group/koopman-vae",
        log_model_checkpoints=False)
    neptune_logger.log_hyperparams(args)

    # Add the appropriate tags to the run.
    NeptuneTagsManager.add_tags(neptune_logger.run, args, is_optuna=False)

    # Train the model.
    data_module = SpriteDataModule(args.dataset_path, args.batch_size)

    trainer = Trainer(max_epochs=args.epochs,
                      check_val_every_n_epoch=args.evl_interval,
                      accelerator='gpu',
                      strategy='ddp',
                      callbacks=[checkpoint_every_n, checkpoint_best_fixed_content_purity, checkpoint_best_fixed_action_purity,
                                 early_stop],
                      logger=neptune_logger,
                      devices=-1,
                      num_nodes=1)
    trainer.fit(model, data_module, ckpt_path=checkpoint_to_resume)

    # Create the eigen-image and upload it.
    fig = create_eigenimage(model, static_size=args.static_size)
    neptune_logger.experiment['eigen-image'].upload(fig)
