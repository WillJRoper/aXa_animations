import matplotlib as ml

ml.use('Agg')
import numpy as np
from sphviewer.tools import camera_tools
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import sys
from swiftsimio import load
from images import getimage
import cmasher as cmr
import os


def single_frame(num, nframes, res):

    # Define path
    path = "/cosma8/data/dp004/dc-husk1/SWIFT/disk/extremeres/output_0028.hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print("Boxsize:", boxsize)

    # Define centre
    cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define targets
    targets = [[0, 0, 0]]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same',
                            'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same',
                             'same']
    anchors['r'] = [boxsize.value + 8, 'same', 'same', 'same', 'same', 'same',
                    'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same',
                       'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same',
                         'same']

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    poss = data.gas.coordinates.value
    masses = data.gas.masses.value * 10 ** 10
    dm_masses = data.dark_matter.masses.value * 10 ** 10

    poss -= cent
    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    hsmls = data.gas.smoothing_lengths.value

    # Fix broken properties
    if dm_masses.max() == 0:
        return

    mean_den = np.sum(dm_masses) / boxsize ** 3

    vmax, vmin = np.log10(100 * mean_den), 7

    print("Norm:", vmin, vmax)

    cmap = cmr.chroma

    # Get images
    rgb_output, ang_extent = getimage(cam_data, poss, masses, hsmls,
                                      num, cmap, vmin, vmax, res)

    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print("Extents:", ang_extent, extent)

    dpi = rgb_output.shape[0] / 2
    print("DPI, Output Shape:", dpi, rgb_output.shape)
    fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=ang_extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=1, color="w")

    # ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w', clip_on=False,
    #         transform=ax.transAxes)

    # ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
    #         transform=ax.transAxes)
    # ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
    #         transform=ax.transAxes)

    # axis_to_data = ax.transAxes + ax.transData.inverted()
    # left = axis_to_data.transform((0.05, 0.075))
    # right = axis_to_data.transform((0.15, 0.075))
    # dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])

    # ax.text(0.1, 0.055, "%.2f cMpc" % dist,
    #         transform=ax.transAxes, verticalalignment="top",
    #         horizontalalignment='center', fontsize=1, color="w")

    plt.margins(0, 0)

    fig.savefig('plots/COLIBRE_Galaxy_frame' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)


res = (4320, 7680)
    
if int(sys.argv[2]) > 0:
    snap = "%05d" % int(sys.argv[1])
    if os.path.isfile(
            'plots/COLIBRE_Galaxy_frame' + snap + '.png'):
        print("File exists")
    else:
        single_frame(int(sys.argv[1]), nframes=1800, res=res)
else:
    single_frame(int(sys.argv[1]), nframes=1800, res=res)
