import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import lightning as L

from classifier import classifier_Sprite_all
from utils.general_utils import reorder, t_to_np, calculate_metrics, dataclass_to_dict, init_weights
from utils.koopman_utils import get_unique_num, static_dynamic_split, get_sorted_indices
from loss import ModelLoss


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
    def __init__(self, args: argparse.Namespace):
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
    def __init__(self, args: argparse.Namespace):
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
            x = self.lstm_layer(x.reshape(-1, self.frames, self.k_dim))[0].reshape(-1, self.decoder_lstm_output_size, 1,
                                                                                   1)
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

    def __init__(self, args: argparse.Namespace):
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

        # If the calculation is not stable - return None as the latent variable and the koopman matrix.
        if torch.sum(torch.isnan(Y2)) != 0:
            return None, None

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
        Db = torch.sqrt((Dr - torch.ones(len(Dr), device=Dr.device)) ** 2 + torch.imag(D) ** 2)

        # ----- static loss ----- #
        Id, new_static_number = None, None
        if self.static_mode == 'norm':
            I = torch.argsort(Dn)
            new_static_number = get_unique_num(D, I, self.static_size)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Dns = torch.index_select(Dn, 0, Is)
            spectral_static_loss = self.loss_func(Dns, torch.ones(len(Dns), device=Dns.device))

        elif self.static_mode == 'real':
            I = torch.argsort(Dr)
            new_static_number = get_unique_num(D, I, self.static_size)
            Is, Id = I[-new_static_number:], I[:-new_static_number]
            Drs = torch.index_select(Dr, 0, Is)
            spectral_static_loss = self.loss_func(Drs, torch.ones(len(Drs), device=Drs.device))

        elif self.static_mode == 'ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static_size)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            spectral_static_loss = self.loss_func(Dbs, torch.zeros(len(Dbs), device=Dbs.device))

        elif self.static_mode == 'space_ball':
            I = torch.argsort(Db)
            # we need to pick the first indexes from I and not the last
            new_static_number = get_unique_num(D, torch.flip(I, dims=[0]), self.static_size)
            Is, Id = I[:new_static_number], I[new_static_number:]
            Dbs = torch.index_select(Db, 0, Is)
            # spectral_static_loss = torch.mean(self.sp_b_thresh(Dbs))

        elif self.static_mode == 'none':
            spectral_static_loss = torch.zeros(1, device=x_pred_loss.device)

        if self.dynamic_mode == 'strict':
            Dnd = torch.index_select(Dn, 0, Id)
            spectral_dynamic_loss = self.loss_func(Dnd,
                                                   self.eigs_tresh_squared * torch.ones(len(Dnd), device=Dnd.device))

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
                (Dbd < self.ball_thresh).float() * ((torch.ones(len(Dbd), device=Dbd.device)) * 2 - Dbd))

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
        self.save_hyperparameters("args")

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

        # The classifier. It is "excluded" from the model since we don't want to learn it.
        self.classifier = classifier_Sprite_all(args)
        loaded_dict = torch.load(args.classifier_path)
        self.classifier.load_state_dict(loaded_dict['state_dict'])
        self.exclude_classifier_from_model()

        # Whether the evaluation is by prior sampling.
        self.prior_sampling = args.prior_sampling

        # Init the model's weights.
        self.apply(init_weights)

    def exclude_classifier_from_model(self):
        # Insert it to evaluation mode.
        self.classifier.eval()

        # Mark that its parameters don't need gradient.
        for param in self.classifier.parameters():
            param.requires_grad = False

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

    def log_dataclass(self, dataclass, key_prefix="", val=False, on_epoch=None, on_step=None):
        # Set the name of the directory to log to.
        directory = 'val' if val else 'train'

        # Get the data in the form of a dictionary.
        data_dict = dataclass_to_dict(dataclass)

        # Add the prefix to each of the dict keys.
        data_dict = {f"{directory}/{key_prefix}{key}": value for key, value in data_dict.items()}

        # Log the dictionary.
        self.log_dict(data_dict, on_epoch=on_epoch, on_step=on_step, sync_dist=True)

    def training_step(self, batch, batch_idx):
        # Get the data of the batch and reorder the images.
        x = reorder(batch['images'])

        # Pass the data through the model.
        outputs = self(x)
        if outputs is None:
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=x.device).requires_grad_()

        # Calculate the losses.
        model_losses = self.loss(x, outputs, self.batch_size)

        # Gather the losses.
        model_losses = self.gather_dataclass(model_losses, is_tensors=True)

        # Log the different losses.
        self.log_dataclass(model_losses, val=False, on_epoch=True, on_step=False)

        # Log the epoch number.
        self.log('epoch', self.current_epoch, on_epoch=True, on_step=False, sync_dist=True)

        return model_losses.sum_loss_weighted

    def validation_step(self, batch, batch_idx):
        # Receive the data and make it fit to dataloader.
        x, label_A, label_D = batch['images'], batch['A_label'][:, 0], batch['D_label'][:, 0]

        # Reorder the received x.
        x = reorder(x)

        # Pass the data through the model.
        outputs = self(x)

        # Calculate the losses.
        model_losses = self.loss(x, outputs, self.batch_size)

        # Log the losses.
        self.log_dataclass(model_losses, val=True, on_epoch=True, on_step=False)

    def gather_dataclass(self, instance, is_tensors=False):
        # Convert dataclass to dictionary.
        instance_dict = dataclass_to_dict(instance)

        # Convert dictionary to tensor for gathering.
        if is_tensors:
            # Preserve gradients.
            instance_tensor = torch.stack(list(instance_dict.values()))
        else:
            instance_tensor = torch.tensor(list(instance_dict.values()), device=self.device)

        # Reshape the instance tensor to be a line vector.
        instance_tensor = instance_tensor.reshape(1, -1)

        # Gather tensors across devices
        gathered_tensors = self.all_gather(instance_tensor, sync_grads=is_tensors)
        gathered_tensors = gathered_tensors.squeeze(dim=1)

        # Convert gathered tensors back to dictionary
        gathered_dict = {key: gathered_tensors[:, i].mean() for i, key in enumerate(instance_dict.keys())}

        # Convert dictionary back to dataclass
        return instance.__class__(**gathered_dict)

    def calculate_val_metrics_and_log(self, fixed: str):
        # Calculate the metrics.
        metrics, _ = calculate_metrics(self, self.classifier, self.trainer.val_dataloaders, fixed=fixed)

        # Gather all the metrics in a distributed run.
        metrics = self.gather_dataclass(metrics, is_tensors=False)

        # Log the metrics.
        self.log_dataclass(metrics, key_prefix=f"fixed_{fixed}_", val=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self) -> None:
        # Calculate and log the fixed content metrics.
        self.calculate_val_metrics_and_log(fixed="content")

        # Calculate and log the fixed action metrics.
        self.calculate_val_metrics_and_log(fixed="action")

    def forward_fixed_element_for_classification_prior_sampled(self, x, fixed_content, pick_type='norm',
                                                               static_size=None):
        # Set the static size if it was not set.
        if static_size is None:
            static_size = self.koopman_layer.static_size

        # Get the posterior.
        _, _, z_post = self.encode_and_sample_post(x)

        # Pass the posterior through the Koopman module and the Dropout layer.
        _, Ct = self.koopman_layer(z_post)
        z_post = self.drop(z_post)
        z_original_shape = z_post.shape

        # Get the batch size and time series size
        bsz, fsz = x.shape[0:2]

        # Transfer the tensors to ndarrays.
        z_post = t_to_np(z_post.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # Receive the static and dynamic indices.
        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, static_size)

        # Sample z from the prior distribution.
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z(bsz, random_sampling=True)

        # Convert the sampled to ndarray.
        z_sampled = t_to_np(z_prior)

        # Project the original and prior on the eigenvectors plane.
        z_orig_projected = z_post @ V
        z_sampled_projected = z_sampled @ V

        # Calculate the static and dynamic parts of the zs.
        z_orig_dynamic, z_orig_static = z_orig_projected[:, :, Id] @ U[Id], z_orig_projected[:, :, Is] @ U[Is]
        z_sampled_dynamic, z_sampled_static = z_sampled_projected[:, :, Id] @ U[Id], z_sampled_projected[:, :, Is] @ U[
            Is]

        # Switch the content/motion according to the flag.
        if fixed_content:
            swapped_z = torch.from_numpy(np.real(z_orig_static + z_sampled_dynamic)).to(self.device)
        else:
            swapped_z = torch.from_numpy(np.real(z_orig_dynamic + z_sampled_static)).to(self.device)

        # Reconstruct the sampled X.
        recon_x_sample = self.decoder(swapped_z.reshape(z_original_shape))

        # Reconstruct the original X.
        z_post = torch.from_numpy(z_post).to(self.device)
        recon_x = self.decoder(z_post.reshape(z_original_shape))

        return recon_x_sample, recon_x

    def forward_fixed_element_for_classification_skd_sampled(self, X, fixed_content, pick_type='real', duplicate=False,
                                                             static_size=None):
        # Set the static size if it was not set.
        if static_size is None:
            static_size = self.koopman_layer.static_size

        # ----- X.shape: b x t x c x w x h ------
        # Get the posterior.
        z_mean, z_logvar, z_post = self.encode_and_sample_post(X)

        # Pass the posterior through the Koopman module and the Dropout layer.
        Z2, Ct = self.koopman_layer(z_post)

        # swap a single pair in batch
        bsz, fsz = X.shape[0:2]

        # swap contents of samples in indices
        Z = t_to_np(z_post.reshape(bsz, fsz, -1))
        C = t_to_np(Ct)

        # eig
        D, V = np.linalg.eig(C)
        U = np.linalg.inv(V)

        # static/dynamic split
        I = get_sorted_indices(D, pick_type)
        Id, Is = static_dynamic_split(D, I, pick_type, self.koopman_layer.static_size)

        convex_size = 2

        Js = [np.random.permutation(bsz) for _ in range(convex_size)]  # convex_size permutations
        # J = np.random.permutation(bsz)              # bsz
        # J2 = np.random.permutation(bsz)

        A = np.random.rand(bsz, convex_size)  # bsz x 2
        A = A / np.sum(A, axis=1)[:, None]

        Zp = Z @ V

        # prev code
        # Zp1 = [a * z for a, z in zip(A[:, 0], Zp[J2])]
        # Zp2 = [a * z for a, z in zip(A[:, 1], Zp[J])]

        # bsz x time x feats
        # Zpc = np.array(Zp1) + np.array(Zp2)

        # Edit

        import functools
        Zpi = [np.array([a * z for a, z in zip(A[:, c], Zp[j])]) for c, j in enumerate(Js)]
        Zpc = functools.reduce(lambda a, b: a + b, Zpi)

        Zp2 = copy.deepcopy(Zp)

        if fixed_content:
            # Swap the dynamic info.
            Zp2[:, :, Id] = Zpc[:, :, Id]

        else:
            # Swap the static info.
            if duplicate:
                Zp2[:, :, Is] = np.repeat(np.expand_dims(np.mean(Zpc[:, :, Is], axis=1), axis=1), 8, axis=1)
            else:
                Zp2[:, :, Is] = Zpc[:, :, Is]

        Z2 = np.real(Zp2 @ U)

        X2_dec = self.decoder(torch.from_numpy(Z2).to(self.device))
        X_dec = self.decoder(torch.from_numpy(Z).to(self.device))

        return X2_dec, X_dec

    def forward_fixed_content_for_classification(self, x, static_size=None):
        if self.prior_sampling:
            return self.forward_fixed_element_for_classification_prior_sampled(x, fixed_content=True,
                                                                               static_size=static_size)

        return self.forward_fixed_element_for_classification_skd_sampled(x, fixed_content=True, static_size=static_size)

    def forward_fixed_motion_for_classification(self, x, static_size=None):
        if self.prior_sampling:
            return self.forward_fixed_element_for_classification_prior_sampled(x, fixed_content=False,
                                                                               static_size=static_size)

        return self.forward_fixed_element_for_classification_skd_sampled(x, fixed_content=False,
                                                                         static_size=static_size)

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

        return ModelLoss(sum_loss_weighted=loss,
                         reconstruction_loss=reconstruction_loss,
                         kl_divergence=kld_z,
                         x_pred_loss=x_pred_loss,
                         z_pred_loss=z_pred_loss,
                         spectral_loss=spectral_loss)

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
        if Ct is None:
            self.trainer.should_stop = True
            return None

        z_post_dropout = self.drop(z_post)

        # Reconstruct the data.
        koopman_recon_x = self.decoder(z_post_koopman)
        dropout_recon_x = self.decoder(z_post_dropout)

        return z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, z_post_koopman, z_post_dropout, Ct, koopman_recon_x, dropout_recon_x

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

        z_t = torch.zeros(batch_size, self.k_dim, device=self.device)
        h_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        h_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)

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

        z_t = torch.zeros(batch_size, self.k_dim, device=self.device)
        h_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        h_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)

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
        z_t = torch.zeros(batch_size, self.k_dim, device=self.device)

        # z_mean_t = torch.zeros(batch_size, self.z_dim)
        # z_logvar_t = torch.zeros(batch_size, self.z_dim)
        h_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly1 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        h_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
        c_t_ly2 = torch.zeros(batch_size, self.prior_lstm_inner_size, device=self.device)
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
