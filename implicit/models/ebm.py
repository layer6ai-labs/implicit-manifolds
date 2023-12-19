import random

import numpy as np
import torch
import torch.nn as nn
from functorch import jacfwd, vjp, vmap
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from tqdm import tqdm

from .autoencoder import Autoencoder, VariationalAutoencoder
from .mcmc import ConstrainedLangevinMC, LangevinMC
from .model import Model


class EBM(Model):
    """An energy-based model.

    Provides sampling and training functionality.

    Args:
        energy: the energy function with which to sample
        lims: a 2-tuple of either floats or tensors defining the boundaries of the data
        device: the device on which the computations will be performed (all networks will be moved
            to this device)
        buffer_size: the number of samples to store in the buffer
        noise_kwargs: keyword arguments to be passed to self.sample_noise
    """

    def __init__(self, energy, lims, device, buffer_size=10000, **noise_kwargs):
        self.device = device
        self.energy = energy.to(device)
        self.mcmc = LangevinMC(energy)

        if hasattr(energy, "dom_shape"):
            self.dom_shape = self.energy.dom_shape
        elif hasattr(energy, "dom_dim"):
            self.dom_shape = (self.energy.dom_dim,)
        else:
            raise ValueError("Provided energy must have attribute for domain dimension or shape")

        self.lims = lims
        self._init_buffer(buffer_size=buffer_size, **noise_kwargs)

    @property
    def lims(self):
        return self._lims

    @lims.setter
    def lims(self, value):
        try:
            iter(value[0])
        except TypeError: # Broadcast lims into input shape if input is not a Tensor
            value = (torch.full(self.dom_shape, value[0]), torch.full(self.dom_shape, value[1]))
        self._lims = value

    @property
    def _buffer_module(self):
        """Need to specify where to store buffer: subclasses may not use self.energy"""
        return self.energy

    @property
    def buffer(self):
        """Buffer is actually stored in the energy module"""
        return self._buffer_module.buffer

    @buffer.setter
    def buffer(self, value):
        if not hasattr(self._buffer_module, "buffer"):
            self._buffer_module.register_buffer("buffer", value)
        else:
            self._buffer_module.buffer = value

    @property
    def buffer_size(self):
        return len(self.buffer)

    def _init_buffer(self, buffer_size=10000, **noise_kwargs):
        self.buffer = self.sample_noise(buffer_size, **noise_kwargs).cpu() # Buffer is stored on CPU

    def _clamp(self, x):
        lims = self.lims[0][None,...].to(x.device), self.lims[1][None,...].to(x.device)
        return torch.maximum(torch.minimum(x, lims[1]), lims[0])

    def partition(self, sample_num=5000):
        # Estimate partition function
        with torch.no_grad():
            uniform_noise = self.sample_noise(sample_num)
            unnorm_probs = torch.exp(-self.energy(uniform_noise))
            latent_volume = torch.prod(self.lims[1] - self.lims[0])
            return torch.mean(unnorm_probs) * latent_volume

    def sample_noise(self, size):
        """Samples noise to initialize new MCMC samples.

        Args:
            size: the number of samples to return

        Returns:
            A tensor of noise
        """
        noise = Uniform(*self.lims).sample((size,))[:,...].to(self.device)
        noise = self._clamp(noise)
        return noise

    def sample(self, size, buffer_frac=1., update_buffer=False, noise_kwargs={}, mc_kwargs={}):
        """Sample from this energy-based model.

        Args:
            size: the number of samples to return
            buffer_frac: the percentage of points to initialize from the sample buffer
            update_buffer: whether to update the sample buffer (should be true during training)
            noise_kwargs: dictionary of arguments to be passed to self.sample_noise
            mc_kwargs: dictionary of additional arguments to be passed to self.mcmc.sample_chain

        Returns:
            A tensor of samples
        """
        if (buffer_frac > 0 or update_buffer) and not hasattr(self, "buffer"):
            raise ValueError("No buffer available, but `buffer_frac` is greater than 0.")

        num_buffer = np.random.binomial(size, buffer_frac)
        num_rand = size - num_buffer

        if num_rand > 0:
            rand_samples = self.sample_noise(num_rand, **noise_kwargs)
        else:
            rand_samples = torch.empty(0, *self.dom_shape).to(self.device)

        if num_buffer > 0:
            buffer_samples = torch.stack(random.choices(self.buffer, k=num_buffer)).to(self.device)
        else:
            buffer_samples = torch.empty(0, *self.dom_shape).to(self.device)

        init_samples = torch.cat((rand_samples, buffer_samples))
        init_samples = init_samples[torch.randperm(size)] # Shuffle tensors
        samples = self.mcmc.sample_chain(init_samples, **mc_kwargs)
        samples = self._clamp(samples)

        if update_buffer:
            self.buffer = torch.cat((samples.cpu(), self.buffer))[:self.buffer_size]

        return samples.detach()

    def train_step(self, optim, batch, neg_weight=1., beta=1., clip_norm=1., **sample_kwargs):
        """Train this EBM using contrastive divergence.

        Args:
            optim: an optimizer for the parameters of `self.energy`
            batch: batch of data
            beta: the coefficient for the regularizer
            clip_norm: the maximum norm to which gradient will be clipped
            sample_kwargs: additional arguments to be passed into `self.sample`
        """
        self.energy.train()

        optim.zero_grad()

        pos_samples = batch.to(self.device)
        neg_samples = self.sample(len(batch), update_buffer=True, **sample_kwargs)

        pos = self.energy(pos_samples)
        neg = self.energy(neg_samples) * neg_weight

        cd_loss = (pos-neg).mean()
        reg_loss = (pos.square() + neg.square()).mean()

        loss = cd_loss + beta * reg_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.energy.parameters(), clip_norm)
        optim.step()

        self.energy.eval()

        return {
            "loss": loss.detach().cpu().tolist(),
            "scale_loss": reg_loss.detach().cpu().tolist(),
        }


class ConstrainedEBM(EBM):
    """A constrained energy-based model.

    Provides sampling and training functionality.

    Args:
        manifold: the ImplicitManifold to which the EBM should be constrained
        energy: the energy function with which to sample
        lims: a 2-tuple of either floats or tensors defining the boundaries of the data
        device: the device on which the computations will be performed (all networks will be moved
            to this device)
        kwargs: keyword arguments to be passed to EBM.__init__
    """

    def __init__(self, manifold, energy, lims, device, **kwargs):
        self.manifold = manifold
        super().__init__(energy, lims=lims, device=device, **kwargs)

        self.init_mcmc = self.mcmc # Store simple MCMC method from parent class
        self.mcmc = ConstrainedLangevinMC(manifold.mdf, energy) # Use constrained MC for sampling

    def sample_noise(self, size, batch_size=512, **project_kwargs):
        """Samples noise, projected to the manifold.

        This is used to initialize new samples before MCMC.

        Args:
            size: the number of samples to return
            ambient_sample: if True, sample from ambient energy before projecting to manifold
            batch_size: batch size for generating samples from manifold
            project_kwargs: keyword arguments to be passed into self.manifold.project

        Returns:
            A tensor of noise projected to the manifold
        """
        noise = []
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size-i)
            ambient_noise = self.manifold.sample(sample_size, mc_kwargs={"n_steps": 0}) # Warning: uses buffer for now
            manifold_noise = self.manifold.project(ambient_noise, **project_kwargs)
            manifold_noise = self._clamp(manifold_noise)
            noise.append(manifold_noise)
        return torch.cat(noise)


class PushforwardEBM:
    """An autoencoder with an EBM in the latent space.

    Provides sampling and training functionality.

    Args:
        encoder: a Map defining the encoder
        decoder: a Map defining the decoder
        energy: the energy function for the latent space
        lims: a 2-tuple of either floats or tensors defining the boundaries of the data
        device: the device on which the computations will be performed (all networks will be moved
            to this device)
        kwargs: keyword arguments to be passed to EBM.__init__
    """

    def __init__(self, autoencoder, energy, device, infer_lims=True, **kwargs):
        self.device = device
        self.autoencoder = autoencoder
        self.energy = energy.to(device)
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        self.ebm = EBM(energy, lims=(-10., 10.), device=device, **kwargs)
        self.infer_lims = infer_lims

    def sample(self, *args, **kwargs):
        latent_sample = self.ebm.sample(*args, **kwargs)
        with torch.no_grad():
            return self.decoder(latent_sample)

    def prob(self, x=None, z=None, sample_num=5000):
        """Computes the probability density in n-dimensional ambient space."""
        ambient_energy = self.ambient_energy(x, z)
        partition = self.ebm.partition(sample_num=sample_num)
        return torch.exp(-ambient_energy)/partition

    def ambient_energy(self, x=None, z=None):
        """Computes the energy in n-dimensional ambient space."""
        assert x is not None or z is not None
        if z is None:
            z = self.encoder(x)

        # Compute change-of-volume expression to go from latent to ambient energy
        single_z_decoder = lambda z: self.decoder(z[None, ...]).squeeze()
        jac = vmap(jacfwd(single_z_decoder))(z)
        jac = jac.flatten(start_dim=1, end_dim=-2) # Convert Jacobian to 2D matrix
        jtj = torch.bmm(jac.transpose(1, 2), jac)
        cholesky_factor = torch.linalg.cholesky(jtj)
        cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=1, dim2=2)
        half_log_det_jtj = torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)

        return self.energy(z) + half_log_det_jtj

    def train_ae(self, *args, **kwargs):
        """Train the autoencoder component of the model."""
        self.autoencoder.train(*args, **kwargs)

    def train_ebm(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train(self, optim, dataloader, **kwargs):
        """Fit the EBM component of the model.

        Args:
            optim: an optimizer for the parameters of `self.energy`
            dataloader: iterable from which to load training batches
            kwargs: keyword arguments to be passed into `ebm.train`
        """

        encoder = self.encoder
        device = self.device

        # Create latent dataloader
        class LatentLoader:
            def __iter__(self):
                for batch in dataloader:
                    if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                        batch, _ = batch
                    batch = batch.to(device)
                    yield encoder(batch)

        latent_loader = LatentLoader()

        if self.infer_lims: # Update lims for EBM
            latent_sample = next(iter(latent_loader))
            min_input = latent_sample[0]
            max_input = latent_sample[0]

            for latent_batch in latent_loader:
                for latent in latent_batch:
                    with torch.no_grad():
                        min_input = torch.minimum(min_input, latent)
                        max_input = torch.maximum(max_input, latent)

            pad = 0.5 * (max_input - min_input) # Pad lims on each side
            self.ebm.lims = (min_input - pad, max_input + pad)

        return self.ebm.train(optim, latent_loader, **kwargs)


class ImplicitManifold(EBM):
    """A manifold represented implicitly as the zero set of some smooth map

    Args:
        mdf: a manifold-defining function (the manifold is given by M = {x: mdf(x) = 0})
        lims: a 2-tuple of either floats or tensors defining the boundaries of the data
        device: the device on which the computations will be performed (the mdf will be moved
            to this device)
        energy_norm: the norm applied to the MDF to create an energy, which is used to generate
            samples ("l1", "l2", or "l2-squared")
        kwargs: keyword arguments to be passed to EBM.__init__
    """
    def __init__(self, mdf, lims, device, energy_norm="l2", **kwargs):
        self.energy = self.MDFEnergy(mdf, norm_type=energy_norm)
        self.mdf = mdf.to(device)
        self.device = device

        EBM.__init__(self, self.energy, lims, device, **kwargs)

    @property
    def _buffer_module(self):
        """Store buffer in the MDF instead of the energy"""
        return self.mdf

    def train_step(self, optim, batch, mu=1., sv_min=0.01, sv_max=None, beta=1., neg_weight=0.,
                   pos_norm=None, neg_norm=None, clip_norm=None, clip_value=None, **sample_kwargs):
        """Fit the implicit manifold to data.

        Args:
            optim: an optimizer for the parameters of `self.mdf`
            batch: a batch of data
            mu: the coefficient for the MDF rank-regularization terms
            sv_min: the minimum singular value targeted by rank-regularization
            sv_max: the maximum singular value targeted by rank-regularization
            beta: the coefficient for scale regularizer (only applies when neg_weight > 0.)
            neg_weight: the weight applied to the negative sample regularization term;
                if positive, training will be similar to an EBM
            pos_norm: the MDF norm used to minimize positive samples ("l1", "l2", or "l2-squared")
            neg_norm: the MDF norm used to maximize negative samples ("l1", "l2", or "l2-squared")
            clip_norm: the maximum norm to which gradient will be clipped
            clip_value: the maximum value to which gradient entries will be clipped
            sample_kwargs: additional arguments to be passed into `self.sample`
        """
        self.mdf.train()

        optim.zero_grad()
        pos_samples = batch.to(self.device)

        # Regularize the unit vjp to be between sv_min and sv_max on positive samples
        v = torch.randn(pos_samples.shape[0], self.mdf.codom_dim).to(self.device)
        out, vjp_fn = vjp(self.mdf, pos_samples)
        vec_jac_prod = vjp_fn(v)[0]

        vjp_norm = torch.linalg.vector_norm(vec_jac_prod.flatten(start_dim=1), dim=1)
        v_norm = torch.linalg.vector_norm(v, dim=1)
        sv_min_reg = F.relu(sv_min - vjp_norm/v_norm).square().mean()
        if sv_max is not None:
            sv_max_reg = F.relu(vjp_norm/v_norm - sv_max).square().mean()
        else:
            sv_max_reg = torch.zeros_like(sv_min_reg)

        # Compute contrastive divergence loss
        pos = self.energy.norm(out, norm_type=pos_norm)
        if neg_weight > 0:
            neg_samples = self.sample(len(batch), update_buffer=True, **sample_kwargs)
            neg = self.energy(neg_samples, norm_type=neg_norm)
        else:
            neg = torch.zeros_like(pos)

        pos_mean = pos.mean()
        neg_mean = neg.mean()
        scale_reg = neg.square().mean() # Penalize unstably-large negative values

        # Sum all losses and take a gradient step
        loss = pos_mean - neg_weight*neg_mean + mu*(sv_min_reg + sv_max_reg) + beta*scale_reg
        loss.backward()

        if clip_norm is not None:
            nn.utils.clip_grad_norm_(self.mdf.parameters(), clip_norm)
        if clip_value is not None:
            nn.utils.clip_grad_value_(self.mdf.parameters(), clip_value)

        optim.step()

        self.mdf.eval()

        return {
            "loss": loss.detach().cpu().tolist(),
            "pos": pos_mean.detach().cpu().tolist(),
            "neg": neg_mean.detach().cpu().tolist(),
            "sv_min": sv_min_reg.detach().cpu().tolist(),
            "sv_max": sv_max_reg.detach().cpu().tolist(),
            "scale": scale_reg.detach().cpu().tolist(),
        }

    def project(self, x, opt_cls="LBFGS", opt_steps=100, **opt_kwargs):
        """Projects `x` onto the manifold.

        Minimizes `||mdf(x)||^2` initialized from `x` and returns the result.
        """
        x = x.detach().clone()
        x.requires_grad = True

        opt_cls = getattr(torch.optim, opt_cls)
        zero_opt = opt_cls([x], **opt_kwargs)

        for epoch in range(opt_steps):
            def closure():
                zero_opt.zero_grad()
                out = self.mdf(x)
                loss = out.square().mean()
                loss.backward()
                return loss

            loss = zero_opt.step(closure)

        return x.detach()

    class MDFEnergy(nn.Module):
        def __init__(self, mdf, norm_type="l2"):
            """An energy derived from the norm of an MDF."""
            super().__init__()
            self.mdf = mdf
            self.norm_type = norm_type

            if hasattr(self.mdf, "dom_shape"):
                self.dom_shape = self.mdf.dom_shape
            elif hasattr(self.mdf, "dom_dim"):
                self.dom_shape = (self.mdf.dom_dim,)
            else:
                raise ValueError("Provided mdf must have attribute for domain dimension or shape")

            if hasattr(self.mdf, "codom_dim"):
                self.codom_dim = self.mdf.codom_dim
            else:
                raise ValueError("Provided mdf must have attribute for codomain dimension")

        def norm(self, x, norm_type=None):
            if norm_type is None:
                norm_type = self.norm_type

            if norm_type == "l1":
                energy =  x.abs().sum(dim=1)
            elif norm_type == "l2":
                energy = torch.linalg.vector_norm(x, dim=1)
            elif norm_type == "l2-squared":
                energy = x.square().sum(dim=1)
            else:
                raise ValueError(f"Unrecognized norm type {norm_type}")

            return energy

        def forward(self, x, norm_type=None):
            out = self.mdf(x)
            energy = self.norm(out, norm_type=norm_type)
            return energy
