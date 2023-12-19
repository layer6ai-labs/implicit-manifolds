import torch
import torch.nn as nn

from .model import Model


class Autoencoder(Model):
    """A simple autoencoder.

    Args:
        encoder: a Map defining the encoder
        decoder: a Map defining the decoder
        device: the device on which the computations will be performed (all networks will be moved
            to this device)
    """

    def __init__(self, encoder, decoder, device):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def train_step(self, optim, batch, clip_norm=10.):
        """Run a single training step.

        Args:
            optim: an optimizer for the parameters of `self.encoder` and `self.decoder`
            batch: batch of data
            clip_norm: the maximum norm to which gradient will be clipped
        """
        self.encoder.train()
        self.decoder.train()

        batch = batch.to(self.device)
        optim.zero_grad()

        encoding = self.encoder(batch)
        recon = self.decoder(encoding)

        loss = (recon - batch).square().mean()
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), clip_norm)
        optim.step()

        self.encoder.eval()
        self.decoder.eval()

        return {
            "loss": loss.detach().cpu().tolist(),
        }

    def reconstruct(self, batch):
        latent = self.encoder(batch)
        return self.decoder(latent)


class VariationalAutoencoder(Autoencoder):
    """A variational autoencoder.

    Args:
        encoder: a Map defining the encoder (map to space of dimension 2*codom_dim)
        decoder: a Map defining the decoder
        device: the device on which the computations will be performed (all networks will be moved
            to this device)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_encoder = self.encoder
        self.encoder = lambda x: self.full_encoder(x).chunk(2, dim=1)[0]

    def train_step(self, optim, batch, beta=0.1, clip_norm=10.):
        """Run a single training step.

        Args:
            optim: an optimizer for the parameters of `self.encoder` and `self.decoder`
            batch: batch of data
            clip_norm: the maximum norm to which gradient will be clipped
            beta: a weighting on the KL term
        """
        self.full_encoder.train()
        self.decoder.train()

        batch = batch.to(self.device)
        optim.zero_grad()

        mu_z, log_sigma_z = self.full_encoder(batch).chunk(2, dim=1)
        z = torch.randn_like(mu_z)*torch.exp(log_sigma_z) + mu_z

        # Note decoder variance is fixed
        recon = (self.decoder(z) - batch).square().mean()
        kl = (z.square() - log_sigma_z).mean()

        loss = recon + beta * kl
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(self.full_encoder.parameters(), clip_norm)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), clip_norm)
        optim.step()

        self.full_encoder.eval()
        self.decoder.eval()

        return {
            "loss": loss.detach().cpu().tolist(),
            "kl": kl.detach().cpu().tolist(),
        }
