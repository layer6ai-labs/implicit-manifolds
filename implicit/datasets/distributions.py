from abc import ABC, abstractmethod

import numpy as np
import pyro
import torch
import torch.distributions as dist


class ManifoldDistribution(ABC):
    """A probability distribution supported on a lower-dimensional submanifold of Euclidean space"""

    @abstractmethod
    def sample(self, n):
        pass

    @abstractmethod
    def prob(self, x):
        """The probability density of the distribution at x"""
        pass

    @abstractmethod
    def manifold_points(self, count):
        """A tuple of tensors of nonrandom, ordered points representing the manifold

        Each entry of the tuple is a tensor of points corresponding to one component of the manifold
        """
        pass


class MixtureDistribution(ManifoldDistribution):
    """A mixture of multiple manifold-supported distributions"""

    def __init__(self, components, weights):
        self.components = components
        self.weights = torch.Tensor(weights)
        self.cat = dist.Categorical(self.weights)

    def sample(self, n):
        sample_comps_nums = self.cat.sample((n,))[:,None]
        data = [self.components[comp_num].sample(1) for comp_num in sample_comps_nums]
        return torch.cat(data)

    def prob(self, x):
        probs = torch.stack([comp.prob(x) for comp in self.components], dim=-1)
        return probs @ self.weights

    def manifold_points(self, count=500):
        assert count % len(self.components) == 0
        count_per_comp = count // len(self.components)
        return tuple(points for comp in self.components
                     for points in comp.manifold_points(count=count_per_comp))


class VonMises(ManifoldDistribution):
    def __init__(self, loc=0., concentration=2., radius=1., centre=(0, 0)):
        loc = torch.Tensor([loc])
        concentration = torch.Tensor([concentration])
        self.radius = torch.Tensor([radius])
        self.centre = torch.Tensor(centre)
        self.gt = dist.VonMises(loc, concentration)

    def _transform(self, thetas):
        return self._polar_to_euclidean(thetas) * self.radius + self.centre

    def sample(self, n):
        thetas = self.gt.sample((n,))
        data = self._transform(thetas)
        return data

    def prob(self, x):
        thetas = self._euclidean_to_polar((x - self.centre)/self.radius)
        log_probs = self.gt.log_prob(thetas).squeeze()
        return torch.exp(log_probs) * self.on_manifold(x)

    def on_manifold(self, x):
        return torch.isclose(torch.linalg.norm(x - self.centre, dim=-1),
                             torch.ones(x.shape[:-1]) * self.radius)

    def manifold_points(self, count=500):
        thetas = torch.linspace(-np.pi, np.pi, count)[:,None]
        points = self._transform(thetas)
        return points,

    @staticmethod
    def _polar_to_euclidean(theta):
        return torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)

    @staticmethod
    def _euclidean_to_polar(xy):
        return torch.arctan2(xy[:,1], xy[:,0])[:,None]


class VonMisesMixture(MixtureDistribution):
    def __init__(self, locs=(0., np.pi), concentrations=(2., 2.), radii=(1., 1.),
                 centres=((-1.8, 0), (1.8, 0)), weights=(0.5, 0.5)):
        assert len(locs) == len(concentrations) == len(radii) == len(centres) == len(weights)
        components = [VonMises(loc, conc, rad, centre)
                      for loc, conc, rad, centre in zip(locs, concentrations, radii, centres)]
        super().__init__(components, weights)


class ProjectedNormal(ManifoldDistribution):
    """ A Gaussian distribution projected to an m-sphere embedded in (m+1)-dimensions.

    Based on the ProjectedNormal distribution of Pyro.

    Args:
        concentration: an (m+1)-dimensional vector defining the mean of the Gaussian relative to
            the sphere's centre
        centre: the centre of the sphere
    """

    def __init__(self, concentration, centre=(0., 0., 0.)):
        self.concentration = torch.Tensor(concentration)
        self.pyro_dist = pyro.distributions.ProjectedNormal(self.concentration)
        self.centre = torch.Tensor(centre)

    def sample(self, size):
        return self.pyro_dist.rsample(sample_shape=(size,)) + self.centre

    def prob(self, x):
        log_prob = self.pyro_dist.log_prob(x - self.centre)
        return  torch.exp(log_prob) * self.on_manifold(x)

    def on_manifold(self, x):
        return torch.isclose(torch.linalg.norm(x - self.centre, dim=-1),
                             torch.ones(x.shape[:-1]))

    def manifold_points(self, count=100):
        theta = torch.linspace(0, np.pi, count) # Polar angle
        phi = torch.linspace(0, 2*np.pi, count) # Azimuthal angle

        # Create the sphere surface in xyz coordinates
        x = torch.outer(torch.cos(phi), torch.sin(theta)) + self.centre[0]
        y = torch.outer(torch.sin(phi), torch.sin(theta)) + self.centre[1]
        z = torch.outer(torch.ones_like(phi), torch.cos(theta)) + self.centre[2]

        return torch.stack((x, y, z), dim=2),


class TorusDistribution(ManifoldDistribution):

    @staticmethod
    def angles_to_euclidean(angles, R, r):
        x = (R + r * torch.cos(angles[..., 1])) * torch.cos(angles[..., 0])
        y = (R + r * torch.cos(angles[..., 1])) * torch.sin(angles[..., 0])
        z = r * torch.sin(angles[..., 1])
        return torch.stack((x, y, z), dim=-1)

    @staticmethod
    def euclidean_to_angles(xyz, R):
        phi = torch.arctan2(xyz[..., 1], xyz[..., 0])[..., None]
        psi = torch.arctan2(xyz[..., 2],
            (torch.pow(torch.pow(xyz[..., 0], 2) + torch.pow(xyz[..., 1], 2), 0.5) - R)
        )[..., None]
        return torch.cat((phi, psi), dim=-1)

    @staticmethod
    def uniform_torus_sample(count=100, R=3, r=2, centre=(0., 0., 0.)):
        successes = 0
        torus_points = np.zeros((count, 3))
        while successes < count:
            unif = np.random.uniform(size=3)
            theta = 2 * np.pi * unif[0]
            phi = 2 * np.pi * unif[1]
            W = (R + r * np.cos(theta)) / (R + r)
            # Rejection sampling
            if W >= unif[2]:
                torus_points[successes] = [
                    (R + r * np.cos(theta)) * np.cos(phi),
                    (R + r * np.cos(theta)) * np.sin(phi),
                    r * np.sin(theta)
                ]
                successes += 1

        return torch.from_numpy((torus_points + centre).astype(np.float32))


class SineBivariateVonMises(TorusDistribution):
    """ A bivariate normal distribution on a torus S^1 x S^1.

    Based on the SineBivariateVonMises distribution of Pyro.

    Args:
        location: a 2 element vector of the means of the bivariate distribution
        concentration: a 2 element vector of the concentrations of the bivariate distribution
        correlation: a scalar correlation between the two directions
        R: the major radius - distance from center of torus to center of tube
        r: the minor radius - radius of tube
        centre: the centre of the torus
    """

    def __init__(self, location, concentration, correlation, R=3, r=2, centre=(0., 0., 0.)):
        self.location = torch.Tensor(location)
        self.concentration = torch.Tensor(concentration)
        self.correlation = torch.Tensor(correlation)
        self.pyro_dist = pyro.distributions.SineBivariateVonMises(self.location[0],
                                                                  self.location[1],
                                                                  self.concentration[0],
                                                                  self.concentration[1],
                                                                  correlation=self.correlation,
                                                                  validate_args=True)
        self.R = torch.tensor(R)
        self.r = torch.tensor(r)
        self.centre = torch.Tensor(centre)

    def sample(self, size):
        angles = self.pyro_dist.sample(sample_shape=(size,))[:,0,:]
        return  self.angles_to_euclidean(angles, self.R, self.r) + self.centre

    def prob(self, x):
        angles = self.euclidean_to_angles(x - self.centre, self.R)
        log_prob = self.pyro_dist.log_prob(angles)
        return  torch.exp(log_prob) * self.on_manifold(x)

    def on_manifold(self, x):
        # Use implicit equation defining torus (sqrt(x^2 + y^2) - R)^2 +z^2 - r^2 = 0
        x_c = x - self.centre
        torus_eqn = (torch.pow(torch.pow(torch.pow(x_c[..., 0], 2) + torch.pow(x_c[..., 1], 2), 0.5)
                     - self.R, 2) + torch.pow(x_c[..., 2], 2) - torch.pow(self.r, 2))
        return torch.isclose(torus_eqn, torch.zeros(x.shape[:-1]), atol=5e-05)

    def manifold_points(self, count=100):
        phi = torch.linspace(0, 2*np.pi, count) # Angle of rotation around torus' axis
        psi = torch.linspace(0, 2*np.pi, count) # Angle around tube of torus

        # Create the torus surface in xyz coordinates
        x = torch.outer(self.R + self.r * torch.cos(psi), torch.cos(phi)) + self.centre[0]
        y = torch.outer(self.R + self.r * torch.cos(psi), torch.sin(phi)) + self.centre[1]
        z = torch.outer(self.r * torch.sin(psi), torch.ones_like(phi)) + self.centre[2]

        return torch.stack((x, y, z), dim=2),
