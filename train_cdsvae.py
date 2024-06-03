import json
import random
import functools
import PIL
import utils
import progressbar
import numpy as np
import os
import argparse
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from mutual_info import logsumexp, log_density, log_importance_weight_matrix

from model import CDSVAE, classifier_Sprite_all

from torch.utils.tensorboard import SummaryWriter
from utils import entropy_Hy, entropy_Hyx, inception_score, KL_divergence

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--log_dir', default='./logs_sprite', type=str, help='base directory to save logs')
parser.add_argument('--model_dir', default='', type=str, help='model to load or resume')
parser.add_argument('--data_root', default='./data', type=str, help='root directory for data')
parser.add_argument('--nEpoch', default=300, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')

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

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

mse_loss = nn.MSELoss().cuda()


# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
# CE_loss = nn.CrossEntropyLoss().cuda()

# --------- training funtions ------------------------------------
def train(x, label_A, label_D, model, optimizer, opt, mode="train"):
    if mode == "train":
        model.zero_grad()

    if isinstance(x, list):
        batch_size = x[0].size(0)  # 128
    else:
        batch_size = x.size(0)

    f_mean, f_logvar, f, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior, recon_x = model(
        x)  # pred

    if opt.loss_recon == 'L2':  # True branch
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

    loss = l_recon + kld_f * opt.weight_f + kld_z * opt.weight_z

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in [l_recon, kld_f, kld_z]]


def main(opt):
    name = 'CDSVAE_Sprite_epoch-{}_bs-{}_decoder={}{}x{}-rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
           '-weight:kl_f={}-kl_z={}-sche_{}-{}'.format(
        opt.nEpoch, opt.batch_size, opt.decoder, opt.image_width, opt.image_width, opt.rnn_size, opt.g_dim, opt.f_dim,
        opt.z_dim, opt.lr,
        opt.weight_f, opt.weight_z, opt.loss_recon, opt.sche, opt.note)

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    log = os.path.join(opt.log_dir, 'log.txt')
    mi_path = os.path.join(opt.log_dir, 'mi.txt')

    summary_dir = os.path.join('./summary/', opt.dataset, name)
    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    print_log("Random Seed: {}".format(opt.seed), log)
    os.makedirs(summary_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_dir)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)

    # control the sequence sample
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    print_log('Running parameters:')
    print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), log)

    # ---------------- optimizers ----------------
    opt.optimizer = optim.Adam
    cdsvae = CDSVAE(opt)

    cdsvae.apply(utils.init_weights)
    optimizer = opt.optimizer(cdsvae.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.sche == "cosine":
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.nEpoch+1)//2, eta_min=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                                         T_0=(opt.nEpoch + 1) // 2, T_mult=1)
    elif opt.sche == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.nEpoch // 2, gamma=0.5)
    elif opt.sche == "const":
        scheduler = None
    else:
        raise ValueError('unknown scheduler')

    if opt.model_dir != '':
        cdsvae = saved_model['cdsvae']

    # --------- transfer to gpu ------------------------------------
    if torch.cuda.device_count() > 1:
        print_log("Let's use {} GPUs!".format(torch.cuda.device_count()), log)
        cdsvae = nn.DataParallel(cdsvae)

    cdsvae = cdsvae.cuda()
    print_log(cdsvae, log)

    classifier = classifier_Sprite_all(opt)
    opt.cls_path = './judges/Sprite/sprite_judge.tar'
    loaded_dict = torch.load(opt.cls_path)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset(opt)
    N, seq_len, dim1, dim2, n_c = train_data.data.shape
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=opt.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=opt.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    test_video_enumerator = get_batch(test_loader)
    opt.dataset_size = len(train_data)

    epoch_loss = Loss()

    # --------- training loop ------------------------------------
    cur_step = 0
    for epoch in range(opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        cdsvae.train()
        epoch_loss.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()
        for i, data in enumerate(train_loader):
            '''
            images : torch.Size([128, 8, 64, 64, 3])
            A_label : torch.Size([128, 4])
            D_label : torch.Size([128])
            OF_label : torch.Size([128, 8, 9])
            mask : torch.Size([128, 8, 9])
            images_pos : torch.Size([128, 8, 64, 64, 3])
            images_neg : torch.Size([128, 8, 64, 64, 3])
            index : torch.Size([128])
            '''

            progress.update(i + 1)
            x, label_A, label_D = reorder(data['images']), data['A_label'], data['D_label']
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            # train frame_predictor
            recon, kld_f, kld_z = train(x, label_A, label_D, cdsvae,
                                        optimizer, opt)

            lr = optimizer.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar("lr", lr, cur_step)
                writer.add_scalar("Train/mse", recon.item(), cur_step)
                writer.add_scalar("Train/kld_f", kld_f.item(), cur_step)
                writer.add_scalar("Train/kld_z", kld_z.item(), cur_step)
                cur_step += 1

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

        if epoch % opt.evl_interval == 0 or epoch == opt.nEpoch - 1:
            cdsvae.eval()
            # save the model
            net2save = cdsvae.module if torch.cuda.device_count() > 1 else cdsvae
            torch.save({
                'model': net2save.state_dict(),
                'optimizer': optimizer.state_dict()},
                '%s/model%d.pth' % (opt.log_dir, epoch))

        if epoch == opt.nEpoch - 1 or epoch % 5 == 0:
            val_mse = val_kld_f = val_kld_z = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D = reorder(data['images']), data['A_label'], data['D_label']
                x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

                with torch.no_grad():
                    recon, kld_f, kld_z = train(x, label_A, label_D,
                                                cdsvae, optimizer,
                                                opt,
                                                mode="val")

                val_mse += recon
                val_kld_f += kld_f
                val_kld_z += kld_z

            n_batch = len(test_loader)
            if writer is not None:
                writer.add_scalar("Val/mse", val_mse.item() / n_batch, epoch)
                writer.add_scalar("Val/kld_f", val_kld_f.item() / n_batch, epoch)
                writer.add_scalar("Val/kld_z", val_kld_z.item() / n_batch, epoch)

    torch.save(cdsvae.state_dict(), "/home/azencot_group/inon/koopman_vae/saved_models/model_no_mi_nEpochs_3.pth")


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


if __name__ == '__main__':
    main(opt)
