import matplotlib as ml

ml.use('Agg')
import numpy as np
from sphviewer.tools import camera_tools
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import sys
from swiftsimio import load
from utilities import get_normalised_image
from images import getimage


def single_frame(num, nframes, res):
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

    vmax, vmin = 14, 6

    print("Norm:", vmin, vmax)

    try:
        poss = data.stars.coordinates.value
        masses = data.stars.masses.value * 10 ** 10
        poss -= cent
        poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
        poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

        hsmls = data.stars.smoothing_lengths.value

        i = 0
        while hsmls.max() == 0:

            new_snap = "%04d" % (num + i)

            # Define path
            path = "/cosma/home/dp004/dc-rope1/cosma7/SWIFT/" \
                   "hydro_1380_ani/data/ani_hydro_" + new_snap + ".hdf5"

            data = load(path)
            hsmls = data.stars.smoothing_lengths.value

            if hsmls.size != masses.size:
                hsmls = hsmls[:masses.size]
            i += 1

        # print("Cmap Limits")
        # print("------------------------------------------")
        #
        # print(np.log10(200 * mean_den),
        #       np.log10(1000 * mean_den),
        #       np.log10(1600 * mean_den),
        #       np.log10(2000 * mean_den),
        #       np.log10(3000 * mean_den),
        #       np.log10(4000 * mean_den))
        #
        # print(np.log10(200 * mean_den) / vmax,
        #       np.log10(1000 * mean_den) / vmax,
        #       np.log10(1600 * mean_den) / vmax,
        #       np.log10(2000 * mean_den) / vmax,
        #       np.log10(3000 * mean_den) / vmax,
        #       np.log10(4000 * mean_den) / vmax)
        #
        # print("------------------------------------------")

        # hex_list = ["#000000", "#03045e", "#0077b6",
        #             "#48cae4", "#caf0f8", "#ffffff"]
        # float_list = [0,
        #               np.log10(mean_den) / vmax,
        #               np.log10(200 * mean_den) / vmax,
        #               np.log10(1600 * mean_den) / vmax,
        #               np.log10(2000 * mean_den) / vmax,
        #               1.0]
        #
        # cmap = get_continuous_cmap(hex_list, float_list=float_list)
        cmap = ml.cm.get_cmap('Greys_r')

        # Get images
        rgb_output, ang_extent = getimage(cam_data, poss, masses, hsmls,
                                          num, cmap, vmin, vmax, res)

    except AttributeError as e:
        print(e)
        cmap = ml.cm.get_cmap('Greys_r')
        rgb_output = cmap(np.zeros(res))
        ang_extent = [-45.00001, 45.00001, -25.312506, 25.312506]

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

    ax.text(0.1, 0.055, "%.2f cMpc" % dist,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='center', fontsize=1, color="w")

    plt.margins(0, 0)

    fig.savefig('../plots/Ani/Stars/Stars_Cube_' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)


res = (2160, 3840)
single_frame(int(sys.argv[1]), nframes=1380, res=res)
