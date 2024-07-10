import utils
from utils import load_dataset
import numpy as np
import os
import argparse
import neptune
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from model import CDSVAE, classifier_Sprite_all
from tqdm import tqdm


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')

    parser.add_argument('--dataset_path', default='/cs/cs_groups/azencot_group/datasets/SPRITES_ICML/datasetICML',
                        type=str, help='dataset to train')
    parser.add_argument('--dataset', default='Sprite', type=str, help='dataset to train')
    parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
    parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
    parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
    parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

    parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
    parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
    parser.add_argument('--z_dim', default=32, type=int, help='dimensionality of z_t')
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')

    parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
    parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--sche', default='cosine', type=str, help='scheduler')

    # Logs paths
    project_working_directory = '/cs/cs_groups/azencot_group/inon/koopman_vae'
    parser.add_argument('--models_during_training_dir', default=f'{project_working_directory}/models_during_training',
                        type=str,
                        help='base directory to save the models during the training.')
    parser.add_argument('--checkpoint_dir', default=f'{project_working_directory}/checkpoints', type=str,
                        help='base directory to save the last checkpoint.')
    parser.add_argument('--final_models_dir', default=f'{project_working_directory}/final_models', type=str,
                        help='base directory to save the final models.')

    return parser


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def train(args, model):
    # --------- training loop ------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        # Log the number of the epoch.
        run['epoch'] = epoch

        if epoch and scheduler is not None:
            scheduler.step()

        # Notify the model about the training mode.
        model.train()

        # Perform the train part of the epoch.
        for i, data in tqdm(enumerate(train_loader)):
            # Reorder the data dimensions as needed.
            x = reorder(data['images']).to(args.device)

            # Zero the gradients of the model.
            model.zero_grad()

            # Pass the data through the model.
            outputs = model(x)

            # Calculate the losses.
            loss, reconstruction_loss, kld_z = model.loss(x, outputs, args.batch_size)

            # Log the losses and the learning rate.
            run['train/lr'].append(args.optimizer.param_groups[0]['lr'])
            run['train/sum_loss_weighted'].append(loss)
            run['train/reconstruction_loss'].append(reconstruction_loss)
            run['train/kld_z'].append(kld_z)

            # Perform backpropagation.
            loss.backward()
            args.optimizer.step()

        # Save the checkpoint.
        save_checkpoint(args.optimizer, model, epoch, args.checkpoint_path)

        # Perform evaluation for the model.
        if epoch % args.evl_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    # Reorder the data dimensions as needed.
                    x = reorder(data['images']).to(args.device)

                    # Pass the data through the model.
                    outputs = model(x)

                    # Calculate the losses.
                    loss, reconstruction_loss, kld_z = model.loss(x, outputs, args.batch_size)

                    # Log the losses.
                    run['test/sum_loss_weighted'].append(loss)
                    run['test/reconstruction_loss'].append(reconstruction_loss)
                    run['test/kld_z'].append(kld_z)

            # Save the net in the middle.
            net2save = model.module if torch.cuda.device_count() > 1 else model
            torch.save(net2save.state_dict(), '%s/model%d.pth' % (args.current_training_logs_dir, epoch))

    # Stop the neptune run.
    run.stop()

    # Save the model.
    net2save = model.module if torch.cuda.device_count() > 1 else model
    torch.save(net2save.state_dict(), os.path.join(args.final_models_dir, args.model_name + ".pth"))


# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def get_batch(train_loader):
    while True:
        for sequence in train_loader:
            yield sequence


def print_log(print_string, log=None, verbose=True):
    if verbose:
        print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()


def create_model(args):
    return CDSVAE(args)


def save_checkpoint(optimizer, model, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        print("Loading Checkpoint from '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Resuming Training From Epoch {}".format(start_epoch))
        return start_epoch
    except:
        print("No Checkpoint Exists At '{}'.Start Fresh Training".format(checkpoint_path))
        return 0


if __name__ == '__main__':
    # Receive the hyperparameters.
    parser = define_args()
    args = parser.parse_args()

    # Initialize neptune.
    run = neptune.init_run(project="azencot-group/koopman-vae",
                           api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjg4NDkxMS04N2NhLTRkOTctYjY0My05NDY2OGU0NGJjZGMifQ==",
                           )

    # Create the name of the model.
    args.model_name = f"CDSVAE_Sprite_epoch-{args.epochs}_bs-{args.batch_size}_decoder={args.decoder}{args.image_width}x{args.image_width}-rnn_size={args.rnn_size}-g_dim={args.g_dim}-z_dim={args.z_dim}-lr={args.lr}-weight:kl_z={args.weight_z}-sche_{args.sche}"

    # Create the path of the checkpoint.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name + ".pth")

    # Create the path of the training logs dir
    args.current_training_logs_dir = os.path.join(args.models_during_training_dir, args.model_name)
    os.makedirs(args.current_training_logs_dir, exist_ok=True)

    # Create the path of the final models dir.
    os.makedirs(args.final_models_dir, exist_ok=True)

    # Log the hyperparameters used and the name.
    run['config/hyperparameters'] = vars(args)
    run['config/model_name'] = args.model_name

    # Set PRNG seed.
    args.device = set_seed_device(args.seed)

    # load data
    train_data, test_data = load_dataset(args)
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=args.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=args.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    # Create model.
    model = create_model(args).to(device=args.device)
    model.apply(utils.init_weights)

    # Set the optimizer.
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Set the scheduler.
    if args.sche == "cosine":
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs+1)//2, eta_min=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(args.optimizer, eta_min=2e-4,
                                                                         T_0=(args.epochs + 1) // 2, T_mult=1)
    elif args.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, step_size=args.epochs // 2, gamma=0.5)
    elif args.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    # Load the model.
    args.start_epoch = load_checkpoint(model, args.optimizer, args.checkpoint_path)

    # Train the model.
    print("number of model parameters: {}".format(sum(param.numel() for param in model.parameters())))
    train(args, model)
