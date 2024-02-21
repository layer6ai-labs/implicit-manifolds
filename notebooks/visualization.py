import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.collections import LineCollection
from torchmcubes import marching_cubes
from torchvision import transforms
from IPython.display import display


CMAP = cm.GnBu
CMAP_R = cm.GnBu_r # Reversed
COLOURS = [CMAP(i/4.) for i in range(5)]
SURFACE_COLOUR = COLOURS[2]
POINT_COLOUR = COLOURS[4]


to_pil = transforms.ToPILImage()


def _create_ax(position=(1, 1, 1), projection=None, computed_zorder=True):
    fig = plt.figure(figsize=(12, 12))
    if projection:
        return fig.add_subplot(*position, projection=projection, computed_zorder=computed_zorder)
    else:
        return fig.add_subplot(*position, projection=projection)


def display_images(images, n_cols=4):
    """Displays a batch of images in tensor format"""
    images = images.cpu().detach()
    images.clamp_(0., 1.)

    fig = plt.figure(figsize=(n_cols * 3, len(images) * 3 // n_cols))
    for i, img in enumerate(images):
        ax = fig.add_subplot(math.ceil(len(images) / n_cols), n_cols, i+1)
        ax.axis('off')
        if img.shape[0] == 1:
            plt.imshow(to_pil(img), cmap='gray', interpolation='none', aspect='auto')
        else:
            plt.imshow(to_pil(img), interpolation='nearest', aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)

    # Need to explicitly display and get an id if we want to dynamically update it
    display_id = random.randint(0, 100000)
    display(fig, display_id=display_id)

    return fig, display_id


def plot_2d_points(points, s=None, c=None, darken=0.0, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), ax=None,
                   text="", text_x=-1.25, text_y=1.1, fontsize=40):
    if c is None:
        c = [COLOURS[-1]]
    else:
        c = CMAP(c*1.3 + darken)

    ax = _create_ax() if ax is None else ax

    ax.scatter(points[:,0], points[:,1], s=s, c=c)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    plt.xticks([])
    plt.yticks([])
    plt.text(s=text, x=text_x, y=text_y, fontsize=fontsize)


def plot_3d_points(points, c=None, darken=0.0, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), zlims=(-1.5, 1.5),
                   ax=None, elev=None, azim=None, equal_aspect=False):
    if c is None:
        c = [COLOURS[-1]]
    else:
        c = CMAP(c + darken)

    ax = _create_ax(projection='3d') if ax is None else ax
    ax.scatter(points[...,0], points[...,1], points[...,2], c=c)
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_zlim(*zlims)
    
    if equal_aspect:
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


def plot_surface(pointgrid, facecolors=None, cmap=None, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5),
                 zlims=(-1.5, 1.5), ax=None, elev=None, azim=None, equal_aspect=False,
                 normalize_cols=False, **kwargs):
    ax = _create_ax(projection='3d') if ax is None else ax
    xx, yy, zz = pointgrid.permute(2, 0, 1).numpy()
    
    if cmap is None:
        cmap = CMAP

    if facecolors is not None:
        if normalize_cols:
            facecolors -= facecolors.min()
            facecolors /= facecolors.max()
        ax.plot_surface(
            xx, yy, zz, cstride=1, rstride=1, facecolors=cmap(facecolors), zorder=-1, **kwargs)
    else:
        ax.plot_surface(xx, yy, zz, cstride=1, rstride=1, color=SURFACE_COLOUR, zorder=1, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_zlim(*zlims)
    
    if equal_aspect:
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


def plot_2d_to_1d_func(func, points2d=None, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), zlims=(-1.5, 1.5),
                       s=None, color=SURFACE_COLOUR, lattice_num=100, ax=None, elev=30, azim=-60,
                         raise_data=0.0, computed_zorder=True):
    x_lattice = np.linspace(*xlims, lattice_num)
    y_lattice = np.linspace(*ylims, lattice_num)

    xx, yy = np.meshgrid(x_lattice, y_lattice)
    x = torch.Tensor(np.stack((xx, yy), axis=2).reshape(-1, 2))

    with torch.no_grad():
        z = func(x).reshape(lattice_num, lattice_num).numpy()

    ax = _create_ax(projection='3d', computed_zorder=computed_zorder) if ax is None else ax
    
    ax.plot_surface(xx, yy, z, color=color)
    if points2d is not None:
        if raise_data:
            if isinstance(points2d, torch.Tensor):
                points2d = points2d.numpy()
            num_points = points2d.shape[0]
            points2d = np.c_[points2d, np.zeros(num_points)]
            points2d[:, 2] += raise_data
        cols = []
        for column in points2d.T:
            cols.append(column)
        if s:
            ax.scatter(*cols, s=s, c=POINT_COLOUR)
        else:
            ax.scatter(*cols, c=POINT_COLOUR)    
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_zlim(*zlims)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


def plot_func_heatmap(func, points2d=None, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), stepsize=0.01,
                      text=None, text_x=12, text_y=30, fontsize=44, ax=None):
    x_lattice = torch.arange(*xlims, stepsize)
    y_lattice = torch.arange(*ylims, stepsize)

    x = torch.stack(torch.meshgrid(x_lattice, y_lattice, indexing="xy"), dim=2)

    with torch.no_grad():
        values = func(x).squeeze().numpy()

    ax = _create_ax() if ax is None else ax
    ax.imshow(values, cmap=CMAP)

    if points2d is not None:
        ax.scatter(points2d[:,0], points2d[:,1], c=POINT_COLOUR)

    plt.xticks([])
    plt.yticks([])

    if text is not None:
        plt.text(s=text, x=text_x, y=text_y, bbox=dict(fill=False, linewidth=0), fontsize=fontsize)


def plot_1d_to_2d_func_range(func, density_func=None, dom_lims=(-10, 10), xlims=(-1.5, 1.5),
                             ylims=(-1.5, 1.5), lattice_num=500, ax=None):
    lattice = torch.linspace(*dom_lims, lattice_num)[:, None]

    with torch.no_grad():
        segments = func(lattice).numpy()


    lc = LineCollection([segments], cmap=CMAP)
    lc.set_linewidth(6.)

    ax = _create_ax() if ax is None else ax
    ax.add_collection(lc)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    plt.axis("off")


def plot_curve(points, c=None, xlims=(-1.5, 1.5), ylims=(-1.5, 1.5), linewidth=8.,
               ax=None):
    segments = torch.stack((points[:-1], points[1:]), dim=1)

    lc = LineCollection(segments, cmap=CMAP, norm=plt.Normalize(0., 0.95))
    lc.set_linewidth(linewidth)
    if c is not None:
        lc.set_array(c)

    ax = _create_ax() if ax is None else ax
    ax.add_collection(lc)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    plt.axis("off")


def plot_mdf_surface(mdf, col_func, xlims=(-1.3, 1.3), ylims=(-1.3, 1.3), zlims=(-1., 1.), res=128,
                     ax=None, elev=None, azim=None, equal_aspect=False, cmap=None):
    x = torch.linspace(*xlims, res)
    y = torch.linspace(*ylims, res)
    z = torch.linspace(*zlims, res)
    mins, maxs = torch.Tensor([xlims, ylims, zlims]).T

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="xy")

    with torch.no_grad():
        values = mdf(torch.stack((xx, yy, zz), dim=3))

    verts, faces = marching_cubes(values.squeeze(), 0.01)
    verts = verts[:, [1, 2, 0]] # Marching cubes permutes the indices; permute them back
    verts = verts / res * (maxs - mins) + mins

    with torch.no_grad():
        cols = col_func(verts)[faces[:, 0].numpy()]

    ax = _create_ax(projection='3d') if ax is None else ax
    ax.view_init(elev=elev, azim=azim)
    if cmap is None:
        cmap=CMAP
    polyc = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, cmap=cmap, zorder=2)
    polyc.set_array(cols)

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_zlim(*zlims)
    
    if equal_aspect:
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
