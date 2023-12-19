from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_fid import fid_score
from tqdm import tqdm

from implicit.models.mcmc import ConstrainedLangevinMC


class Callback(ABC):
    def __init__(self, *, dataset=None, loader=None, ood_dataset=None, ood_loader=None, writer=None,
                 model=None, network=None, optimizer=None, output_dir=None, model_type=None,
                 freq=1, freq_epochs=None):

        assert freq or freq_epochs, "Must specify a callback frequency in steps or epochs"
        if freq_epochs is not None:
            self.freq = len(loader) * freq_epochs
        else:
            self.freq = freq

        self.dataset = dataset
        self.loader = loader
        self.ood_dataset = ood_dataset
        self.ood_loader = ood_loader
        self.writer = writer
        self.model = model
        self.network = network
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.model_type = model_type

    def __call__(self, *, global_step, **kwargs):
        if global_step % self.freq == 0:
            self.call(global_step=global_step, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        pass

    def epoch_num(self, global_step):
        return global_step // len(self.loader)


class SaveModel(Callback):
    def call(self, global_step, **kwargs):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
        }, self.output_dir / f"{self.model_type}-checkpoint-{global_step:07d}.pt")


class SampleToTensorBoard(Callback):
    def __init__(self, sample_kwargs, *, num_images=64, tag=None, clamp_bounds=(0., 1.), **kwargs):
        super().__init__(**kwargs)
        self.sample_kwargs = sample_kwargs
        self.num_images = num_images
        self.tag = f"{self.model_type}/samples" if tag is None else tag
        self.clamp_bounds = clamp_bounds

    def call(self, *, global_step, **kwargs):
        images = []

        while len(images) < self.num_images:
            images.append(self.model.sample(**self.sample_kwargs).cpu())

        images = torch.cat(images)[:self.num_images]
        images = images.clamp(*self.clamp_bounds)
        self.writer.add_images(self.tag, images, global_step=global_step)


class SaveStatsToTensorBoard(Callback):
    def __init__(self, *, tag_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.tag_prefix = f"{self.model_type}/" if tag_prefix is None else tag_prefix

    def call(self, *, global_step, stats, **kwargs):
        for key, val in stats.items():
            self.writer.add_scalar(self.tag_prefix + key, val, global_step=global_step)


class UpdateProgressBarStats(Callback):
    """Update the tqdm progress bar with stats."""
    def call(self, global_step, stats, pbar, **kwargs):
        epoch_num = self.epoch_num(global_step)
        pbar_msg = f"[E{epoch_num:3d}] " + " | ".join(f"{k}: {v:.4f}" for k, v in stats.items())
        pbar.set_description(pbar_msg)


class OODHistogram(Callback):
    """Save a histogram of MDF norms on in-distribution and out-of-distribution datasets"""
    def __init__(self, *, tag=None, **kwargs):
        super().__init__(**kwargs)
        self.tag = f"{self.model_type}/ood" if tag is None else tag

    def call(self, *, global_step, **kwargs):
        mdf = self.network
        device = self.model.device

        # Compute scores
        id_scores = []
        for batch in self.loader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch
            batch = batch.to(device)
            id_scores.append(self.ood_score(batch))

            break

        ood_scores = []
        for batch in self.ood_loader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch
            batch = batch.to(device)
            ood_scores.append(self.ood_score(batch))

            break

        id_scores = torch.cat(id_scores).numpy()
        ood_scores = torch.cat(ood_scores).numpy()

        # Create figure (TODO: auto-detect plotting args or make configurable)
        fig, ax  = plt.subplots(1, 1)
        ax.hist([id_scores, ood_scores], label=["ID", "OOD"], alpha=0.7, bins=20)
        ax.legend()

        self.writer.add_figure(self.tag, fig, global_step=global_step)

    @abstractmethod
    def ood_score(self, batch):
        pass


class ManifoldOODHistogram(OODHistogram):
    def ood_score(self, batch):
        with torch.no_grad():
            return torch.linalg.vector_norm(self.network(batch), dim=1).cpu()


class EnergyOODHistogram(OODHistogram):
    def ood_score(self, batch):
        with torch.no_grad():
            if hasattr(self.model, "ambient_energy"): # Model is a pushforward EBM
                energy = self.model.ambient_energy(batch)
            else:
                energy = self.model.energy(batch)
        return energy.cpu().squeeze()


class SampleManifoldTangents(Callback):
    """Visualize random tangents of an MDF"""
    def __init__(self, *, num_examples=8, num_samples=8, tag=None, clamp_bounds=(0., 1.), **kwargs):
        super().__init__(**kwargs)
        self.tag = f"{self.model_type}/tangents" if tag is None else tag
        self.clamp_bounds = clamp_bounds

        # Save a persistent batch to check tangent space directions
        for batch in self.loader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch

            self.batch = batch[:num_examples].to(self.model.device)
            break

        # Create dummy CLD instance to sample momenta (Gaussian samples from the tangent space)
        energy = lambda x: torch.zeros(x.shape[0], 1)
        self.mcmc = ConstrainedLangevinMC(self.network, energy)
        self.num_samples = num_samples

    def call(self, *, global_step, **kwargs):
        tangents = [self.mcmc.sample_momentum(self.batch) for _ in range(self.num_samples)]
        images = torch.cat((self.batch, *tangents)).clamp(*self.clamp_bounds)
        self.writer.add_images(self.tag, images, global_step=global_step)


class SampleReconstructions(Callback):
    """Visualze reconstructions"""
    def __init__(self, *, num_examples=8, tag=None, clamp_bounds=(0., 1.), **kwargs):
        super().__init__(**kwargs)
        self.tag = f"{self.model_type}/reconstructions" if tag is None else tag
        self.clamp_bounds = clamp_bounds

        # Save a persistent batch to check tangent space directions
        for batch in self.loader:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch

            self.batch = batch[:num_examples].to(self.model.device)
            break

    def call(self, *, global_step, **kwargs):
        encoding = self.model.encoder(self.batch)
        recon = self.model.decoder(encoding).clamp(*self.clamp_bounds)
        images = torch.cat((self.batch, recon))
        self.writer.add_images(self.tag, images, global_step=global_step)


class EvaluateFID(Callback):
    def __init__(self, sample_kwargs, *, num_images=50000, batch_size=64, tag=None,
                 clamp_bounds=(0., 1.), **kwargs):
        super().__init__(**kwargs)
        self.sample_kwargs = sample_kwargs
        self.num_images = num_images
        self.batch_size = batch_size
        self.tag = f"{self.model_type}/fid" if tag is None else tag
        self.clamp_bounds = clamp_bounds

        self.inception = fid_score.InceptionV3().to(self.model.device)
        self.inception.eval()

        self.groundtruth_mu, self.groundtruth_sigma = self.compute_stats(self.loader)

    def fake_loader(self):
        for i in range(0, self.num_images, self.batch_size):
            if self.num_images - i < self.batch_size:
                batch_size = self.num_images - i
            else:
                batch_size = self.batch_size

            yield self.model.sample(batch_size, **self.sample_kwargs).clamp(*self.clamp_bounds)

    def compute_stats(self, loader):
        features = []

        for batch in tqdm(loader, "Computing inception features"):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch
            batch = batch.to(self.model.device)

              # Convert grayscale to RGB
            if batch.ndim == 3:
                batch.unsqueeze_(1)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch_features = self.inception(batch)[0].squeeze().cpu().numpy()

            features.append(batch_features)

        features = np.concatenate(features)

        # Compute stats
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma


    def call(self, *, global_step, **kwargs):
        fake_mu, fake_sigma = self.compute_stats(self.fake_loader())

        fid = fid_score.calculate_frechet_distance(
            self.groundtruth_mu, self.groundtruth_sigma, fake_mu, fake_sigma)

        self.writer.add_scalar(self.tag, fid, global_step=global_step)
