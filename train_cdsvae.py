import utils
from utils import load_dataset
import numpy as np
import os
import argparse
import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from model import CDSVAE, classifier_Sprite_all
from tqdm import tqdm

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs_sprite', type=str, help='base directory to save logs')
    parser.add_argument('--model_dir', default='', type=str, help='model to load or resume')
    parser.add_argument('--data_root', default='./data', type=str, help='root directory for data')
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
    parser.add_argument('--f_dim', default=256, type=int, help='dim of f')
    parser.add_argument('--z_dim', default=32, type=int, help='dimensionality of z_t')
    parser.add_argument('--g_dim', default=128, type=int,
                        help='dimensionality of encoder output vector and decoder input vector')

    parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
    parser.add_argument('--loss_recon', default='L2', type=str, help='reconstruction loss: L1, L2')
    parser.add_argument('--note', default='S3', type=str, help='appx note')
    parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--sche', default='cosine', type=str, help='scheduler')

    return parser


def set_seed_device(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


# --------- training funtions ------------------------------------
def train2(x, label_A, label_D, model, optimizer, args, mode="train"):
    if mode == "train":
        model.zero_grad()

    if isinstance(x, list):
        batch_size = x[0].size(0)  # 128
    else:
        batch_size = x.size(0)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(
        x)  # pred

    if args.loss_recon == 'L2':  # True branch
        l_recon = F.mse_loss(recon_x, x, reduction='sum')
    else:
        l_recon = torch.abs(recon_x - x).sum()

    f_mean = f_mean.view((-1, f_mean.shape[-1]))  # [128, 256]
    f_logvar = f_logvar.view((-1, f_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var = torch.exp(z_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(z_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar +
                            ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size

    loss = l_recon + kld_f * args.weight_f + kld_z * args.weight_z

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in [l_recon, kld_f, kld_z]]


def train(args):

    epoch_loss = Loss()

    # --------- training loop ------------------------------------
    for epoch in range(args.epochs):
        # Log the number of the epoch.
        run['epoch'] = epoch

        if epoch and scheduler is not None:
            scheduler.step()

        # Notify the model about the training mode.
        args.model.train()
        
        # Reset the losses for the following epoch.
        epoch_loss.reset()

        for i, data in tqdm(enumerate(train_loader)):
            # Reorder the data dimensions as needed.
            x = reorder(data['images']).to(args.device)

            # train frame_predictor
            recon, kld_f, kld_z = train(x, label_A, label_D, cdsvae,
                                        optimizer, args)

            lr = optimizer.param_groups[0]['lr']

            # Log the losses and lr.
            run['train/lr'].append(lr)
            run['train/mse'].append(recon.item())
            run['train/kld_f'].append(kld_f.item())
            run['train/kld_z'].append(kld_z.item())

            epoch_loss.update(recon, kld_f, kld_z)
            if i % 100 == 0 and i:
                print_log(
                    '[%02d] recon: %.3f | kld_f: %.3f | kld_z: %.3f | lr: %.5f' % (epoch, recon, kld_f, kld_z, lr),
                    log)

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        print_log('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | lr: %.5f' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2], lr), log)

        if epoch % args.evl_interval == 0 or epoch == args.epochs - 1:
            cdsvae.eval()
            # save the model
            net2save = cdsvae.module if torch.cuda.device_count() > 1 else cdsvae
            torch.save({
                'model': net2save.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (args.log_dir, epoch))

        if epoch == args.epochs - 1 or epoch % 5 == 0:
            val_mse = val_kld_f = val_kld_z = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D = reorder(data['images']), data['A_label'], data['D_label']
                x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

                with torch.no_grad():
                    recon, kld_f, kld_z = train(x, label_A, label_D,
                                                cdsvae, optimizer,
                                                args,
                                                mode="val")

                val_mse += recon
                val_kld_f += kld_f
                val_kld_z += kld_z

            n_batch = len(test_loader)

            # Log the losses.
            run['test/mse'].append(val_mse.item() / n_batch)
            run['test/kld_f'].append(val_kld_f.item() / n_batch)
            run['test/kld_z'].append(val_kld_z.item() / n_batch)

    run.stop()

    # Save the model.
    torch.save(cdsvae.state_dict(), model_path)


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


class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, recon, kld_f, kld_z):
        self.recon.append(recon)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)

    def reset(self):
        self.recon = []
        self.kld_f = []
        self.kld_z = []

    def avg(self):
        return [np.asarray(i).mean() for i in
                [self.recon, self.kld_f, self.kld_z]]


def create_model(args):
    return CDSVAE(args)


def save_checkpoint(args, epoch, checkpoints):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': args.model.state_dict(),
        'optimizer': args.optimizer.state_dict(),
        'losses': args.epoch_losses_test},
        checkpoints)


def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        print("Loading Checkpoint from '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_losses_test = checkpoint['losses']
        print("Resuming Training From Epoch {}".format(start_epoch))
        return start_epoch, epoch_losses_test
    except:
        print("No Checkpoint Exists At '{}'.Start Fresh Training".format(checkpoint_path))
        return 0, []


if __name__ == '__main__':
    # Receive the hyperparameters.
    parser = define_args()
    args = parser.parse_args()

    # Initialize neptune.
    run = neptune.init_run(project="azencot-group/koopman-vae",
                           api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNjg4NDkxMS04N2NhLTRkOTctYjY0My05NDY2OGU0NGJjZGMifQ==",
                           )

    # Create the name of the run.
    args.name = "CDSVAE_Sprite_epoch-{}_bs-{}_decoder={}{}x{}-rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}-weight:kl_f={}-kl_z={}-sche_{}-{}".format(
        args.epochs, args.batch_size, args.decoder, args.image_width, args.image_width, args.rnn_size, args.g_dim,
        args.f_dim,
        args.z_dim, args.lr,
        args.weight_f, args.weight_z, args.loss_recon, args.sche, args.note)
    model_name = args.name + ".pth"
    args.checkpoint_path = os.path.join(args.model_dir_path, model_name)


    # Log the hyperparameters used and the name.
    run['config/hyperparameters'] = vars(args)
    run['config/name'] = args.name

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
    args.model = create_model(args).to(device=args.device)
    args.model.apply(utils.init_weights)

    # Set the optimizer.
    args.optimizer = optim.Adam(args.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

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
    args.start_epoch, args.epoch_losses_test = load_checkpoint(args.model, args.optimizer, args.checkpoint_path)

    # Train the model.
    print("number of model parameters: {}".format(sum(param.numel() for param in args.model.parameters())))
    train(args)
