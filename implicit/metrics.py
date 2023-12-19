from itertools import product

import numpy as np
import ot
import torch


def distance_to_implicit_manifold(point, manifold, opt_steps=100, manifold_weight=1e10):
    """Estimate distance to nearest point on implicit manifold"""
    nearest_point = point.detach().clone()
    nearest_point.requires_grad = True
    opt = torch.optim.LBFGS([nearest_point], line_search_fn='strong_wolfe')
    norm_dims = tuple(range(1, point.ndim))

    # Optimize `nearest_point` to become the nearest point on manifold
    for epoch in range(opt_steps):
        def closure():
            opt.zero_grad()

            manifold_loss = torch.linalg.vector_norm(manifold.mdf(nearest_point), dim=1).square().mean()
            proximity_loss = torch.linalg.vector_norm(nearest_point - point, dim=norm_dims).square().mean()
            loss = manifold_weight*manifold_loss + proximity_loss

            loss.backward()
            return loss

        loss = opt.step(closure)

    nearest_point = nearest_point.detach()
    return torch.linalg.vector_norm(nearest_point - point, dim=norm_dims)


def distance_to_pushforward_manifold(point, autoencoder, opt_steps=100):
    """Estimate distance to nearest point on pushforward manifold"""
    nearest_latent = autoencoder.encoder(point.detach()).detach().clone()
    nearest_latent.requires_grad = True
    opt = torch.optim.LBFGS([nearest_latent], line_search_fn='strong_wolfe')
    norm_dims = tuple(range(1, point.ndim))

    # Optimize `nearest_latent` to correspond to the nearest point on manifold
    for _ in range(opt_steps):
        def closure():
            opt.zero_grad()
            loss = torch.linalg.vector_norm(
                autoencoder.decoder(nearest_latent) - point, dim=norm_dims).square().mean()
            #loss = (autoencoder.decoder(nearest_latent) - point).abs().mean()
            loss.backward()
            return loss

        loss = opt.step(closure)

    nearest_latent = nearest_latent.detach()

    with torch.no_grad():
        return torch.linalg.vector_norm(autoencoder.decoder(nearest_latent) - point,
                                        dim=norm_dims)


def wasserstein_between_discretized_densities(
        manifold1, densities1, manifold2, densities2, lims, granularity=100, metric="sqeuclidean"):
    """Compute a discretized Wasserstein distance between two distributions.

    Takes from each distribution a collection of points that roughly define the manifold,
    and the corresponding density at each point.
    """

    manifold1, manifold2 = np.array(manifold1), np.array(manifold2)
    densities1, densities2 = np.array(densities1), np.array(densities2)
    data_dim = len(lims)
    axes_ticks = [np.linspace(ax_lims[0], ax_lims[1], granularity) for ax_lims in lims]

    # Grids representing the near and far corners of rectangles partitioning the space
    rects_close = np.moveaxis(
        np.array(np.meshgrid(*[ax_ticks[:-1] for ax_ticks in axes_ticks])), (0, 1), (-1, 1))
    rects_far = np.moveaxis(
        np.array(np.meshgrid(*[ax_ticks[1:] for ax_ticks in axes_ticks])), (0, 1), (-1, 1))

    def create_hist_from_points(manifold, densities):
        bin_sums = np.zeros(rects_close.shape[:-1]) # To be populated
        bin_counts = np.zeros(rects_close.shape[:-1]) # To be populated

        for point_id, point in enumerate(manifold):
            rects_contain_point = np.all((rects_close <= point) & (point < rects_far), axis=-1)
            assert rects_contain_point.sum() <= 1, "At most one rectangle should contain the point"

            bin_sums[rects_contain_point] += densities[point_id]
            bin_counts[rects_contain_point] += 1

        hist = np.divide(
            bin_sums, bin_counts, out=np.zeros_like(bin_sums), where=bin_counts!=0)
        hist /= hist.sum()
        return hist

    # Discretize points into a histogram
    hist1 = create_hist_from_points(manifold1, densities1)
    hist2 = create_hist_from_points(manifold2, densities2)

    # points (hist1.shape[0]*hist1.shape[1], data_dim) should contain the
    # location of the bottom-left corner of every square in the histogram
    points = rects_close.reshape(-1, data_dim)

    # M needs to contain the distances between every pair of cells
    M = ot.dist(points, points, metric=metric)
    M /= M.max()

    a, b = hist1.flatten(), hist2.flatten()
    return ot.emd2(a, b, M)


def wasserstein_discretized_density_to_points(
        manifold1, densities1, datapoints2, lims, granularity=100, metric="sqeuclidean"):
    """Compute a discretized Wasserstein distance between a density and a set of points

    Takes from the former distribution a collection of points that roughly define the manifold,
    and the corresponding density at each point.
    """

    manifold1 = np.array(manifold1)
    densities1 = np.array(densities1)
    datapoints2 = np.array(datapoints2)
    data_dim = len(lims)
    axes_ticks = [np.linspace(ax_lims[0], ax_lims[1], granularity) for ax_lims in lims]

    # Grids representing the near and far corners of rectangles partitioning the space
    rects_close = np.moveaxis(
        np.array(np.meshgrid(*[ax_ticks[:-1] for ax_ticks in axes_ticks])), (0, 1), (-1, 1))
    rects_far = np.moveaxis(
        np.array(np.meshgrid(*[ax_ticks[1:] for ax_ticks in axes_ticks])), (0, 1), (-1, 1))

    def create_hist_from_points(manifold, densities):
        bin_sums = np.zeros(rects_close.shape[:-1]) # To be populated
        bin_counts = np.zeros(rects_close.shape[:-1]) # To be populated

        for point_id, point in enumerate(manifold):
            rects_contain_point = np.all((rects_close <= point) & (point < rects_far), axis=-1)
            assert rects_contain_point.sum() <= 1, "At most one rectangle should contain the point"

            bin_sums[rects_contain_point] += densities[point_id]
            bin_counts[rects_contain_point] += 1

        hist = np.divide(
            bin_sums, bin_counts, out=np.zeros_like(bin_sums), where=bin_counts!=0)
        hist /= hist.sum()
        return hist

    # Discretize points into a histogram
    hist1 = create_hist_from_points(manifold1, densities1)
    # points (hist1.shape[0]*hist1.shape[1], data_dim) should contain the
    # location of the centre of every square in the histogram
    disc_points1 = ((rects_close + rects_far)/2).reshape(-1, data_dim)

    # M needs to contain the distances between every pair of cells
    M = ot.dist(disc_points1, datapoints2, metric=metric)
    M /= M.max()

    a = hist1.flatten()
    b = np.full(len(datapoints2), 1/len(datapoints2))
    return ot.emd2(a, b, M, numItermax=1000000)
