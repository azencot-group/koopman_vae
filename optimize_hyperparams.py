import argparse
import torch
import optuna
from functools import partial
from optuna.integration import PyTorchLightningPruningCallback
from optuna import Trial
import neptune
import neptune.integrations.optuna as npt_utils
from lightning.pytorch import Trainer, seed_everything

from datamodule.sprite_datamodule import SpriteDataModule
import train_cdsvae
from model import KoopmanVAE


def define_args():
    # Define the arguments of the model.
    parser = train_cdsvae.define_basic_args()

    parser.add_argument('--pruning', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether bad trials will be pruned.')
    parser.add_argument('--multi-objective', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to do a multi-objective optimization.'
                             ' Such optimization is not supported with pruning.')
    parser.add_argument('--n_trials', default=300, type=int, help='The number of optimization trials.')

    return parser


def objective(args: argparse.Namespace, trial: Trial) -> float:
    # Set seeds to all the randoms.
    seed_everything(args.seed)

    # Set the trial arguments into args.
    args.static_size = trial.suggest_int("static_size", 4, 11)
    args.lstm = trial.suggest_categorical("lstm", ["encoder", "both"])
    args.weight_kl_z = trial.suggest_float("weight_kl_z", 1e-7, 1e-1, log=True)
    args.weight_x_pred = trial.suggest_float("weight_x_pred", 1e-5, 1e0, log=True)
    args.weight_z_pred = trial.suggest_float("weight_z_pred", 1e-5, 1e0, log=True)
    args.weight_spectral = trial.suggest_float("weight_spectral", 1e-5, 1e0, log=True)

    # Create model.
    model = KoopmanVAE(args)

    # Set the pruning callback.
    if args.pruning:
        callbacks = [PyTorchLightningPruningCallback(trial, monitor="fixed_content_purity")]
    else:
        callbacks = None

    # Train the model.
    data_module = SpriteDataModule(args.dataset_path, args.batch_size)
    trainer = Trainer(max_epochs=args.epochs,
                      check_val_every_n_epoch=args.evl_interval,
                      accelerator='gpu',
                      callbacks=callbacks,
                      devices=1,
                      num_nodes=1)
    trainer.fit(model, data_module)

    try:
        if args.multi_objective:
            return trainer.callback_metrics["fixed_content_purity"].item(), \
                   trainer.callback_metrics["fixed_action_purity"].item()
        else:
            return trainer.callback_metrics["fixed_content_purity"].item()
    except KeyError:
        return 0


if __name__ == "__main__":
    # Define the different arguments and parser them.
    parser = define_args()
    args = parser.parse_args()

    # Set the pruner.
    pruner = optuna.pruners.HyperbandPruner() if args.pruning else optuna.pruners.NopPruner()

    # Initialize the neptune run.
    run = neptune.init_run(project="azencot-group/koopman-vae",
                           api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjg4NDkxMS04N2NhLTRkOTctYjY0My05NDY2OGU0NGJjZGMifQ==",
                           tags=["Optuna"]
                           )
    neptune_callback = npt_utils.NeptuneCallback(run)

    # Set the matmul precision in order to save time.
    torch.set_float32_matmul_precision("high")

    # Create the study and add tag if needed.
    if args.multi_objective:
        study = optuna.create_study(directions=["maximize", "maximize"])
        run["sys/tags"].add("Multi-Objective")
    else:
        study = optuna.create_study(direction="maximize", pruner=pruner)

    # Optimize the objective (with a little trick in order to pass args to it.
    objective = partial(objective, args)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[neptune_callback])
