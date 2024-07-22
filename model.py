import argparse

import torch
import torch.nn as nn
import numpy as np
import lightning as L

from utils.general_utils import imshow_seqeunce, load_dataset, load_checkpoint, save_checkpoint, reorder, set_seed_device, init_weights, log_losses
from utils.koopman_utils import get_unique_num


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()

        # Set the needed parameters for the encoder.
        self.frames = args.frames
        self.channels = args.channels
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.conv_dim = args.conv_dim
        self.conv_output_dim = args.conv_output_dim
        self.encoder_lstm_output_dim = args.encoder_lstm_output_dim
        self.lstm = args.lstm

        # input is (nc) x 64 x 64
        self.c1 = conv(self.channels, self.conv_dim)
        # state size. (nf) x 32 x 32
        self.c2 = conv(self.conv_dim, self.conv_dim * 2)
        # state size. (self.conv_dim*2) x 16 x 16
        self.c3 = conv(self.conv_dim * 2, self.conv_dim * 4)
        # state size. (self.conv_dim*4) x 8 x 8
        self.c4 = conv(self.conv_dim * 4, self.conv_dim * 8)
        # state size. (self.conv_dim*8) x 4 x 4

        # Declare the LSTM layer if needed.
        if self.lstm in ['encoder', 'both']:
            self.lstm_layer = nn.LSTM(self.conv_output_dim, self.encoder_lstm_output_dim, batch_first=True, bias=True,
                                      bidirectional=False)
            self._output_size = self.encoder_lstm_output_dim
        else:
            self._output_size = self.conv_output_dim

        self.c5 = nn.Sequential(
            nn.Conv2d(self.conv_dim * 8, self.conv_output_dim, 4, 1, 0),
            nn.BatchNorm2d(self.conv_output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Reshape the batch and the timeseries together as they are invariant in convolutions.
        x = x.reshape(-1, self.channels, self.image_height, self.image_width)

        # Pass the data through the encoder.
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4).reshape(-1, self.frames, self.conv_output_dim)

        # LSTM if needed.
        if self.lstm in ['encoder', 'both']:
            h5 = self.lstm_layer(h5)[0]

        # Return b x t x output_size
        return h5

    @property
    def output_size(self):
        return self._output_size


class upconv(nn.Module):
    def __init__(self, nin, nout):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__()

        # Set the needed parameters for the decoder.
        self.frames = args.frames
        self.channels = args.channels
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.conv_dim = args.conv_dim
        self.decoder_lstm_output_size = args.decoder_lstm_output_size
        self.k_dim = args.decoder_lstm_output_size
        self.lstm = args.lstm

        # Declare the LSTM layer and the first convolution layer size.
        if self.lstm in ['decoder', 'both']:
            self.lstm_layer = nn.LSTM(self.k_dim, self.decoder_lstm_output_size, batch_first=True, bias=True,
                                      bidirectional=False)

            first_conv_size = self.decoder_lstm_output_size

        else:
            first_conv_size = self.k_dim

        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(first_conv_size, self.conv_dim * 8, 4, 1, 0),
            nn.BatchNorm2d(self.conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = upconv(self.conv_dim * 8, self.conv_dim * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = upconv(self.conv_dim * 4, self.conv_dim * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = upconv(self.conv_dim * 2, self.conv_dim)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(self.conv_dim, self.channels, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        # LSTM if needed.
        if self.lstm in ['decoder', 'both']:
            x = self.lstm_layer(x.reshape(-1, self.frames, self.k_dim))[0].reshape(-1, self.decoder_lstm_output_size, 1, 1)
        else:
            x = x.reshape(-1, self.k_dim, 1, 1)

        d1 = self.upc1(x)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(-1, self.frames, self.channels, self.image_height, self.image_width)

        return output


class KoopmanLayer(nn.Module):

    def __init__(self, args):
        super(KoopmanLayer, self).__init__()

        # Layer structure.
        self.frames = args.frames
        self.k_dim = args.decoder_lstm_output_size

        # Eigenvalues arguments.
        self.static_size = args.static_size
        self.static_mode = args.static_mode
        self.dynamic_mode = args.dynamic_mode
        self.eigs_tresh_squared = args.eigs_thresh ** 2
        self.ball_thresh = args.ball_thresh

        # loss functions
        self.loss_func = nn.MSELoss()
        self.dynamic_threshold_loss = nn.Threshold(args.dynamic_thresh, 0)

    def forward(self, Z):
        # Z is in b * t x c x 1 x 1
        Zr = Z.squeeze().reshape(-1, self.frames, self.k_dim)

        # Split the latent variable to past and future.
        X, Y = Zr[:, :-1], Zr[:, 1:]

        # Solve the linear system that takes us forward in time.
        Ct = torch.linalg.lstsq(X.reshape(-1, self.k_dim), Y.reshape(-1, self.k_dim)).solution

        # Predict (broadcast) by calculating the forward in time.
        Y2 = X @ Ct
        assert (torch.sum(torch.isnan(Y2)) == 0)

        # Concatenate t0 to the forward in time.
        Z2 = torch.cat((X[:, 0].unsqueeze(dim=1), Y2), dim=1)

        return Z2.reshape(Z.shape), Ct

    def loss(self, dropout_recon_x, koopman_recon_x, z_post_dropout, z_post_koopman, Ct):

        # predict ambient
        x_pred_loss = self.loss_func(dropout_recon_x, koopman_recon_x)

        # predict latent
        z_pred_loss = self.loss_func(z_post_dropout, z_post_koopman)

        # Koopman operator constraints (disentanglement)
        # Compute the eigenvalues.
        D = torch.linalg.eigvals(Ct)

        # Compute the norm of the eigenvalues.
        Dn = torch.real(torch.conj(D) * D)

        # Extract the real part of the eigenvalues.
        Dr = torch.real(D)

        # Compute the distance of each eigenvalue to 1.
        Db = torch.sqrt((Dr - torch.ones(len(Dr)).to(Dr)) ** 2 + torch.imag(D) ** 2)

        # ----- static loss ----- #
        Id, new_static_number = None, None
        if self.static_mode == 'norm':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static_size)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dns = torch.index_select(Dn, 0, Is)
            spectral_static_loss = self.loss_func(Dns, torch.ones(len(Dns)).to(Dns))

        elif self.static_mode == 'real':
            I = torch.argsort(Dr)
            new_static_number = get_unique_num(D, I, self.static_size)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Drs = torch.index_select(Dr, 0, Is)
            spectral_static_loss = self.loss_func(Drs, torch.ones(len(Drs)).to(Drs))

        elif self.static_mode == 'ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static_size)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            spectral_static_loss = self.loss_func(Dbs, torch.zeros(len(Dbs)).to(Dbs))

        elif self.static_mode == 'space_ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static_size)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            # spectral_static_loss = torch.mean(self.sp_b_thresh(Dbs))

        elif self.static_mode == 'none':
            spectral_static_loss = torch.zeros(1).to(x_pred_loss)

        if self.dynamic_mode == 'strict':
            Dnd = torch.index_select(Dn, 0, Id)
            spectral_dynamic_loss = self.loss_func(Dnd, self.eigs_tresh_squared * torch.ones(len(Dnd)).to(Dnd))

        elif self.dynamic_mode == 'thresh' and self.static_mode == 'none':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static_size)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dnd = torch.index_select(Dn, 0, Id)
            spectral_dynamic_loss = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_mode == 'thresh':
            Dnd = torch.index_select(Dn, 0, Id)
            spectral_dynamic_loss = torch.mean(self.dynamic_threshold_loss(Dnd))

        elif self.dynamic_mode == 'ball':
            Dbd = torch.index_select(Db, 0, Id)
            spectral_dynamic_loss = torch.mean(
                (Dbd < self.ball_thresh).float() * ((torch.ones(len(Dbd))).to(Dbd) * 2 - Dbd))

        elif self.dynamic_mode == 'real':
            Drd = torch.index_select(Dr, 0, Id)
            spectral_dynamic_loss = torch.mean(self.dynamic_threshold_loss(Drd))

        if self.dynamic_mode == 'none':
            spectral_loss = spectral_static_loss
        else:
            spectral_loss = spectral_static_loss + spectral_dynamic_loss

        return x_pred_loss, z_pred_loss, spectral_loss


class KoopmanVAE(L.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(KoopmanVAE, self).__init__()

        # Net structure.
        self.channels = args.channels  # frame feature
        self.k_dim = args.decoder_lstm_output_size
        self.prior_lstm_inner_size = args.prior_lstm_inner_size
        self.frames = args.frames

        # Frame encoder and decoder
        self.encoder = encoder(args)
        self.decoder = decoder(args)

        # Dropout and Koopman layers
        self.drop = torch.nn.Dropout(args.dropout)
        self.koopman_layer = KoopmanLayer(args)

        # Prior of the dynamics is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.k_dim, args.prior_lstm_inner_size)
        self.z_prior_lstm_ly2 = nn.LSTMCell(args.prior_lstm_inner_size, args.prior_lstm_inner_size)

        self.z_prior_mean = nn.Linear(args.prior_lstm_inner_size, self.k_dim)
        self.z_prior_logvar = nn.Linear(args.prior_lstm_inner_size, self.k_dim)

        # Each timestep is for each z so no reshaping and feature mixing
        self.z_mean = nn.Linear(self.encoder.output_size, self.k_dim)
        self.z_logvar = nn.Linear(self.encoder.output_size, self.k_dim)

        # The loss function.
        self.loss_func = nn.MSELoss()

        # The loss weights.
        # The loss of the reconstruction is implicitly set to 1.0, the rest of the losses are relative to it.
        self.kld_z_weight = args.weight_kl_z
        self.x_pred_weight = args.weight_x_pred
        self.z_pred_weight = args.weight_z_pred
        self.spectral_weight = args.weight_spectral

        # Args for general use.
        self.sche = args.sche
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.run = args.run

    def configure_optimizers(self):
        # Initialize the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Set the scheduler.
        if self.sche == "cosine":
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs+1)//2, eta_min=2e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4,
                                                                             T_0=(self.epochs + 1) // 2, T_mult=1)
        elif self.sche == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs // 2, gamma=0.5)
        elif self.sche == "const":
            scheduler = None
        else:
            raise ValueError('unknown scheduler')

        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        # Reorder the data dimensions as needed.
        x = reorder(batch['images'])

        # Pass the data through the model.
        outputs = self(x)

        # Calculate the losses.
        loss, losses = self.loss(x, outputs, self.batch_size)

        # Log the different losses and the number of the epoch.
        log_losses(self.run, loss, losses, test=False)
        self.run['epoch'] = self.current_epoch

        return loss


    def validation_step(self, batch, batch_idx):
        # Reorder the data dimensions as needed.
        x = reorder(batch['images'])

        # Pass the data through the model.
        outputs = self(x)

        # Calculate the losses.
        loss, losses = self.loss(x, outputs, self.batch_size)

        # Log the losses to Neptune.
        log_losses(self.run, loss, losses, test=True)

        # Log the loss for monitor.
        self.log("val_loss", loss)


    def loss(self, x, outputs, batch_size):
        # Unpack the outputs.
        z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, z_post_koopman, z_post_dropout, Ct, koopman_recon_x, dropout_recon_x = outputs

        # Calculate the reconstruction loss.
        reconstruction_loss = self.loss_func(dropout_recon_x, x)

        # Calculate the kld_z and normalize it by the batch size.
        z_post_var = torch.exp(z_logvar_post)  # [128, 8, 32]
        z_prior_var = torch.exp(z_logvar_prior)  # [128, 8, 32]
        kld_z = 0.5 * torch.sum(z_logvar_prior - z_logvar_post +
                                ((z_post_var + torch.pow(z_mean_post - z_mean_prior, 2)) / z_prior_var) - 1)
        kld_z /= batch_size

        x_pred_loss, z_pred_loss, spectral_loss = self.koopman_layer.loss(dropout_recon_x, koopman_recon_x,
                                                                          z_post_dropout, z_post_koopman, Ct)

        # Calculate the loss.
        # The weight of the reconstruction loss is implicitly 1.
        # The rest of the weights are relative to it.
        loss = reconstruction_loss + \
               self.kld_z_weight * kld_z + \
               self.x_pred_weight * x_pred_loss + \
               self.z_pred_weight * z_pred_loss + \
               self.spectral_weight * spectral_loss

        return loss, (reconstruction_loss, kld_z, x_pred_loss, z_pred_loss, spectral_loss)

    def encode_and_sample_post(self, x):
        # Encode the input.
        z = self.encoder(x)

        # pass to one direction rnn
        z_mean = self.z_mean(z)
        z_logvar = self.z_logvar(z)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)

        return z_mean, z_logvar, z_post

    def forward(self, x):
        # Get the posterior.
        z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # Get the prior.
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=self.training)

        # Pass the posterior through the Koopman module and the Dropout layer.
        z_post_koopman, Ct = self.koopman_layer(z_post)
        z_post_dropout = self.drop(z_post)

        # Reconstruct the data.
        koopman_recon_x = self.decoder(z_post_koopman)
        dropout_recon_x = self.decoder(z_post_dropout)

        return z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, z_post_koopman, z_post_dropout, Ct, koopman_recon_x, dropout_recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_fixed_motion_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).to(x), torch.zeros(f_logvar.shape).to(x),
                                      random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.k_dim).to(self.device)
        h_t_ly1 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        c_t_ly1 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        h_t_ly2 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        c_t_ly2 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
                # z_t = z_post[:,i,:]
            z_t = z_prior
        return z_means, z_logvars, z_out

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.k_dim).to(self.device)
        h_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size).to(self.device)
        c_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size).to(self.device)
        h_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size).to(self.device)
        c_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size).to(self.device)

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:, i, :]
        return z_means, z_logvars, z_out

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.k_dim).to(self.device)
        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        c_t_ly1 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        h_t_ly2 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        c_t_ly2 = torch.zeros(batch_size, self.self.prior_lstm_inner_size).to(self.device)
        for _ in range(self.frames):
            # h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def decode(self, Z):
        X_dec = self.decoder(Z)

        return X_dec

    def swap(self, x, first_idx, second_idx, plot=True):
        s_mean, s_logvar, s, d_mean_post, d_logvar_post, d = self.encode_and_sample_post(x)
        s1, d1 = s_mean[first_idx][None, :].expand(self.frames, s.shape[-1]), d_mean_post[0]
        s2, d2 = s_mean[second_idx][None, :].expand(self.frames, s.shape[-1]), d_mean_post[1]

        s1d2 = torch.cat((d2, s1), dim=1)
        s2d1 = torch.cat((d1, s2), dim=1)

        sd = torch.stack([s1d2, s2d1])

        recon_s1_d2 = self.decoder(s1d2[None, :, :])
        recon_s2_d1 = self.decoder(s2d1[None, :, :])

        # visualize
        if plot:
            titles = ['S{}'.format(first_idx), 'S{}'.format(second_idx), 'S{}d{}s'.format(second_idx, first_idx),
                      'S{}d{}s'.format(first_idx, second_idx)]
            imshow_seqeunce([[x[first_idx]], [x[second_idx]], [recon_s2_d1.squeeze()], [recon_s1_d2.squeeze()]],
                            plot=plot, titles=np.asarray([titles]).T, figsize=(50, 10), fontsize=50)

        return recon_s1_d2, recon_s2_d1


class classifier_Sprite_all(nn.Module):
    def __init__(self, args):
        super(classifier_Sprite_all, self).__init__()
        self.k_dim = args.decoder_lstm_output_size  # frame feature
        self.channels = args.channels  # frame feature
        self.hidden_dim = args.rnn_size
        self.frames = args.frames
        from model import encoder
        self.encoder = encoder(self.k_dim, self.channels)
        self.bilstm = nn.LSTM(self.k_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_top = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_pant = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_hair = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_action = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 9))

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_action(lstm_out_f), self.cls_skin(lstm_out_f), self.cls_pant(lstm_out_f), \
               self.cls_top(lstm_out_f), self.cls_hair(lstm_out_f)
