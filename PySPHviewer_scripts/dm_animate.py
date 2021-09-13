import matplotlib as ml

ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import sys
from guppy import hpy; h = hpy()
import os
from swiftsimio import load
import unyt
import gc


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i
                    in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def get_normalised_image(img, vmin=None, vmax=None):
    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)

    return img


def getimage(data, poss, masses, hsml, num, cmap, vmin, vmax):
    print('There are', poss.shape[0], 'dark matter particles in the region')

    # Set up particle objects
    P = sph.Particles(poss, mass=masses, hsml=hsml)

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = 3840
    i['ysize'] = 2160
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    print("Image limits:", np.min(img), np.max(img))

    img = ndimage.gaussian_filter(img, sigma=(3, 3), order=0)

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))

    return rgb, R.get_extent()


def single_frame(num, nframes):
    snap = "%04d" % num

    # Define path
    path = "/cosma/home/dp004/dc-rope1/cosma7/SWIFT/" \
           "hydro_1380_ani/data/ani_hydro_" + snap + ".hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print("Boxsize:", boxsize)

    # Define centre
    cent = np.array([11.76119931, 3.95795609, 1.26561173])

    # Define targets
    targets = [[0, 0, 0]]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same',
                            'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same',
                             'same']
    anchors['r'] = [boxsize.value + 4, 'same', 'same', 'same', 'same', 'same',
                    'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same',
                       'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same',
                         'same']

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    poss = data.dark_matter.coordinates.value
    masses = data.dark_matter.masses.value * 10 ** 10
    poss -= cent
    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    hsmls = data.dark_matter.softenings.value

    mean_den = np.sum(masses) / boxsize ** 3

    vmax = 12.7
    vmin = 1

    print(np.log10(200 * mean_den),
          np.log10(1000 * mean_den),
          np.log10(8 * 200 * mean_den),
          np.log10(5000 * mean_den))

    print(np.log10(200 * mean_den) / 7,
          np.log10(1000 * mean_den) / 7,
          np.log10(8 * 200 * mean_den) / 7,
          np.log10(5000 * mean_den) / 7)

    hex_list = ["#000000", "#6c1c55", "#7e2e84", "#ba4051",
                "#f6511d", "#ffb400", "#f7ec59", "#fbf6ac"]
    float_list = [0, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1]

    cmap = get_continuous_cmap(hex_list, float_list=float_list)

    # Get images
    rgb_output, ang_extent = getimage(cam_data, poss, masses, hsmls,
                                      num, cmap, vmin, vmax)

    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print(ang_extent, extent)

    dpi = rgb_output.shape[0] / 2
    print(dpi, rgb_output.shape)
    fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=ang_extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=1, color="w")

    ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w', clip_on=False,
            transform=ax.transAxes)

    ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)

    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])

    print(left, right,
          (right[0] - left[0]) / (ang_extent[1] - ang_extent[0]), dist)

    ax.text(0.1, 0.055, "%.2f cMpc" % dist,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='center', fontsize=1, color="w")

    plt.margins(0, 0)

    fig.savefig('../plots/Ani/GasStars_flythrough_' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)


if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6, nframes=1380)
        gc.collect()
