import os
import argparse
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import NeptuneLogger
from datamodule.sprite_datamodule import SpriteDataModule
from model import KoopmanVAE


def define_args():
    parser = argparse.ArgumentParser()

    # Training parameters.
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--evl_interval', default=5, type=int, help='evaluate every n epoch')
    parser.add_argument('--save_interval', default=30, type=int, help='save checkpoint n epoch')
    parser.add_argument('--early_stop_patience', default=8, type=int, help='Patience for the early stop.')
    parser.add_argument('--save_n_val_best', default=3, type=int,
                        help='The number of best models in validation to save')
    parser.add_argument('--sche', default='cosine', type=str, help='scheduler')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

    # Technical parameters.
    parser.add_argument('--dataset_path', default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML',
                        type=str, help='dataset to train')
    parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
    project_working_directory = '/cs/cs_groups/azencot_group/inon/koopman_vae'
    parser.add_argument('--models_during_training_dir', default=f'{project_working_directory}/models_during_training',
                        type=str,
                        help='base directory to save the models during the training.')
    parser.add_argument('--checkpoint_dir', default=f'{project_working_directory}/checkpoints', type=str,
                        help='base directory to save the last checkpoint.')
    parser.add_argument('--final_models_dir', default=f'{project_working_directory}/final_models', type=str,
                        help='base directory to save the final models.')

    # Architecture parameters.
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--image_height', default=64, type=int, help='the height / width of the input image to network')
    parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
    parser.add_argument('--lstm', type=str, choices=['encoder', 'decoder', 'both'],
                        default='both',
                        help='Specify the LSTM type: "encoder", "decoder", or "both" (default: "both")')

    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--k_dim', default=40, type=int,
                        help='Dimensionality of the Koopman module.')
    parser.add_argument('--conv_output_dim', default=40, type=int,
                        help='Dimensionality of the output of the encoder\'s convolution.')
    parser.add_argument('--encoder_lstm_output_dim', default=40, type=int,
                        help='Dimensionality of the output of the encoder\'s LSTM.')
    parser.add_argument('--prior_lstm_inner_size', default=40, type=int,
                        help='Dimensionality of the prior LSTM.')
    parser.add_argument('--decoder_lstm_output_size', default=40, type=int,
                        help='Dimensionality of the output of the decoder\'s LSTM.')

    # Koopman layer implementation parameters.
    parser.add_argument('--static_size', type=int, default=7)
    parser.add_argument('--static_mode', type=str, default='ball', choices=['norm', 'real', 'ball'])
    parser.add_argument('--dynamic_mode', type=str, default='real',
                        choices=['strict', 'thresh', 'ball', 'real', 'none'])
    parser.add_argument('--ball_thresh', type=float, default=0.6)  # related to 'ball' dynamic mode
    parser.add_argument('--dynamic_thresh', type=float, default=0.5)  # related to 'thresh', 'real'
    parser.add_argument('--eigs_thresh', type=float, default=.5)  # related to 'norm' static mode loss

    # Loss parameters.
    parser.add_argument('--weight_kl_z', default=5e-5, type=float, help='Weight of KLD between prior and posterior.')
    parser.add_argument('--weight_x_pred', default=0.07, type=float, help='Weight of Koopman matrix leading to right '
                                                                          'decoding.')
    parser.add_argument('--weight_z_pred', default=0.07, type=float, help='Weight of Koopman matrix leading to right '
                                                                          'transformation in time.')
    parser.add_argument('--weight_spectral', default=0.07, type=float, help='Weight of the spectral loss.')

    # Classifier arguments.
    parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
    parser.add_argument('--classifier_path', type=str,
                        default=f'{project_working_directory}/judges/Sprite/sprite_judge.tar',
                        help='Path to the classifier weights.')

    return parser


if __name__ == '__main__':
    # Receive the hyperparameters.
    parser = define_args()
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
    checkpoint_best_models = ModelCheckpoint(dirpath=current_training_logs_dir,
                                             filename="model-{epoch}-{val_loss:.7f}",
                                             save_top_k=args.save_n_val_best,
                                             monitor="val/sum_loss_weighted",
                                             mode="min",
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
        log_model_checkpoints=False,
        tags=["full_train"])
    neptune_logger.log_hyperparams(args)

    # Train the model.
    data_module = SpriteDataModule(args.dataset_path, args.batch_size)
    trainer = Trainer(max_epochs=args.epochs,
                      check_val_every_n_epoch=args.evl_interval,
                      accelerator='gpu',
                      strategy='ddp',
                      callbacks=[checkpoint_every_n, checkpoint_best_models, early_stop],
                      logger=neptune_logger,
                      devices=-1,
                      num_nodes=1)
    trainer.fit(model, data_module, ckpt_path=checkpoint_to_resume)
