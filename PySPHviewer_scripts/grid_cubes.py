import os
import sys

import cmasher as cmr
import matplotlib as ml
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from images import getimage, getimage_weighted
from sphviewer.tools import camera_tools
import h5py
from utilities import get_continuous_cmap

ml.use('Agg')


def single_frame(num, nframes, res):
    snap = "%04d" % num

    # Define path
    path = "/cosma/home/dp004/dc-rope1/cosma7/SWIFT/" \
           "hydro_1380_ani/data/ani_hydro_" + snap + ".hdf5"

    snap = "%05d" % num

    hdf = h5py.File(path, "r")

    boxsize = hdf["Header"].attrs["BoxSize"][0]
    z = hdf["Header"].attrs["Redshift"]

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
    anchors['r'] = [boxsize + 8, 'same', 'same', 'same', 'same', 'same',
                    'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same',
                       'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same',
                         'same']

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    dm_poss = hdf["/PartType1/Coordinates"][:, :]
    gas_poss = hdf["/PartType0/Coordinates"][:, :]
    try:
        star_poss = hdf["/PartType4/Coordinates"][:, :]
    except KeyError as e:
        print(e)
        star_poss = np.array([[]])

    dm_masses = hdf["/PartType1/Masses"][:] * 10 ** 10
    gas_masses = hdf["/PartType0/Masses"][:] * 10 ** 10
    try:
        star_masses = hdf["/PartType4/Masses"][:] * 10 ** 10
    except KeyError as e:
        print(e)
        star_masses = np.array([])

    dm_hsmls = hdf["/PartType1/Softenings"][:]
    gas_hsmls = hdf["/PartType0/SmoothingLengths"][:]
    try:
        star_hsmls = hdf["/PartType4/SmoothingLengths"][:]
    except KeyError as e:
        print(e)
        star_hsmls = np.array([])

    gas_temps = hdf["/PartType0/Temperatures"][:]

    hdf.close()

    # Fix broken properties
    i = 0
    while dm_masses.max() == 0:
        print("No masses", snap, i)
        new_snap = "%04d" % (num + i)

        # Define path
        path = "/cosma/home/dp004/dc-rope1/cosma7/SWIFT/" \
               "hydro_1380_ani/data/ani_hydro_" + new_snap + ".hdf5"

        hdf = h5py.File(path, "r")
        dm_masses = hdf["/PartType1/Masses"][:] * 10 ** 10
        hdf.close()
        i += 1

    if star_hsmls.max() == 0:
        return

    i = 0
    while gas_temps.max() == 0:
        print("No temps", snap, i)

        new_snap = "%04d" % (num + i)

        # Define path
        path = "/cosma/home/dp004/dc-rope1/cosma7/SWIFT/" \
               "hydro_1380_ani/data/ani_hydro_" + new_snap + ".hdf5"

        hdf = h5py.File(path, "r")
        gas_temps = hdf["/PartType0/Temperatures"][:]
        hdf.close()

        if gas_temps.size != gas_masses.size:
            gas_temps = gas_temps[:gas_masses.size]
        i += 1

    dm_poss -= cent
    dm_poss[np.where(dm_poss > boxsize / 2)] -= boxsize
    dm_poss[np.where(dm_poss < - boxsize / 2)] += boxsize
    gas_poss -= cent
    gas_poss[np.where(gas_poss > boxsize / 2)] -= boxsize
    gas_poss[np.where(gas_poss < - boxsize / 2)] += boxsize
    if star_masses.size > 0:
        star_poss -= cent
        star_poss[np.where(star_poss > boxsize / 2)] -= boxsize
        star_poss[np.where(star_poss < - boxsize / 2)] += boxsize

    mean_den = np.sum(dm_masses) / boxsize ** 3
    print(boxsize, np.sum(dm_masses), mean_den)

    dm_vmax, dm_vmin = np.log10(10000 * mean_den), 4
    gas_vmax, gas_vmin = np.log10(100 * mean_den), 7
    gas_temp_vmax, gas_temp_vmin = 7, 1
    star_vmax, star_vmin = 14, 5

    dm_hex_list = ["#000000", "#03045e", "#0077b6",
                   "#48cae4", "#caf0f8", "#ffffff"]
    gast_hex_list = ["#000000", "#184e77", "#0d108f",
                     "#a81979", "#c78c16", "#eef743",
                     "#ffffff"]

    dm_float_list = [0,
                     np.log10(mean_den) / dm_vmax,
                     np.log10(200 * mean_den) / dm_vmax,
                     np.log10(1600 * mean_den) / dm_vmax,
                     np.log10(2000 * mean_den) / dm_vmax,
                     1.0]
    gast_float_list = [0,
                       3 / gas_temp_vmax,
                       4 / gas_temp_vmax,
                       4.5 / gas_temp_vmax,
                       5 / gas_temp_vmax,
                       6.5 / gas_temp_vmax,
                       7 / gas_temp_vmax]

    print("Mean density cmap Values")
    print("------------------------------------------")

    print(np.log10(200 * mean_den),
          np.log10(1000 * mean_den),
          np.log10(1600 * mean_den),
          np.log10(2000 * mean_den),
          np.log10(3000 * mean_den),
          np.log10(4000 * mean_den))

    print(np.log10(200 * mean_den) / dm_vmax,
          np.log10(1000 * mean_den) / dm_vmax,
          np.log10(1600 * mean_den) / dm_vmax,
          np.log10(2000 * mean_den) / dm_vmax,
          np.log10(3000 * mean_den) / dm_vmax,
          np.log10(4000 * mean_den) / dm_vmax)

    print("Gas temp cmap Limits")
    print("------------------------------------------")

    print(gast_float_list)

    print("------------------------------------------")

    dm_cmap = cmr.eclipse
    gas_cmap = cmr.chroma
    gast_cmap = get_continuous_cmap(gast_hex_list, float_list=gast_float_list)
    star_cmap = ml.cm.get_cmap('Greys_r')

    # Get images
    DM_output, ang_extent = getimage(cam_data, dm_poss, dm_masses, dm_hsmls,
                                     num, dm_cmap, dm_vmin, dm_vmax,
                                     (int(res[0] / 2), int(res[1] / 2)))

    gas_output, ang_extent = getimage(cam_data, gas_poss, gas_masses,
                                      gas_hsmls, num, gas_cmap,
                                      gas_vmin, gas_vmax,
                                      (int(res[0] / 2), int(res[1] / 2)))

    gast_output, ang_extent = getimage_weighted(cam_data, gas_poss, gas_masses,
                                                gas_temps, gas_hsmls, num,
                                                gast_cmap,
                                                gas_temp_vmin, gas_temp_vmax,
                                                (int(res[0] / 2),
                                                 int(res[1] / 2)))

    if star_masses.size > 0:
        print(star_masses.size)
        star_output, ang_extent = getimage(cam_data, star_poss, star_masses,
                                           star_hsmls, num, star_cmap,
                                           star_vmin, star_vmax,
                                           (int(res[0] / 2), int(res[1] / 2)))
    else:
        print("No Stars")
        star_output = star_cmap(np.zeros((int(res[0] / 2), int(res[1] / 2))))

    if DM_output.sum() == np.nan or gas_output.sum() == np.nan \
            or gast_output.sum() == np.nan or star_output.sum() == np.nan:
        print("Found NaNs in one of the images")
        return

    rgb_output = np.zeros((res[0], res[1], 4))
    rgb_output[DM_output.shape[0]:, : DM_output.shape[1], :] = DM_output
    rgb_output[star_output.shape[0]:, star_output.shape[1]:, :] = star_output
    rgb_output[: gas_output.shape[0], : gas_output.shape[1], :] = gas_output
    rgb_output[: gast_output.shape[0], gast_output.shape[1]:, :] = gast_output

    rgb_output[0, :] = np.nan
    rgb_output[DM_output.shape[0], :] = np.nan
    rgb_output[-1, :] = np.nan
    rgb_output[:, 0] = np.nan
    rgb_output[:, DM_output.shape[1]] = np.nan
    rgb_output[:, -1] = np.nan

    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print("Extents:", ang_extent, extent)

    dpi = rgb_output.shape[0] / 2
    print("DPI, Resolution:", dpi, rgb_output.shape)
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

    fig.savefig('../plots/Ani/Grid/Grid_Cubes_' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)


if int(sys.argv[2]) > 0:
    snap = "%05d" % int(sys.argv[1])
    if os.path.isfile('../plots/Ani/Grid/Grid_Cubes_' + snap + '.png'):
        print("File exists")
    else:
        res = (2160, 3840)
        single_frame(int(sys.argv[1]), nframes=1380, res=res)
else:
    res = (2160, 3840)
    single_frame(int(sys.argv[1]), nframes=1380, res=res)
