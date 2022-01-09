#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml

ml.use('Agg')
import numpy as np
from sphviewer.tools import camera_tools
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.cosmology import Planck13 as cosmo
import sys
import h5py
from images import get_mono_image
import cmasher as cmr
import utilities
import os
import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def single_frame(num, nframes, size, rank, comm):
    # Define MPI message tags
    tags = utilities.enum('READY', 'DONE', 'EXIT', 'START')

    snap = "0020"

    # Define path
    path = "/cosma8/data/dp004/jlvc76/FLAMINGO/ScienceRuns/L2800N5040/" \
           "HYDRO_FIDUCIAL/snapshots/flamingo_" + snap \
           + "/flamingo_" + snap + ".hdf5"

    frame = "%05d" % num

    # Open HDF5 file
    hdf = h5py.File(path, "r")

    # Get metadata
    boxsize = hdf["Header"].attrs["BoxSize"][0]
    z = hdf["Header"].attrs["Redshift"]
    nparts = hdf["Header"].attrs["NumPart_Total"][1]
    pmass = hdf["/PartType1/Masses"][0] * 10 ** 10
    ncell_dimens = hdf["Cells/Meta-data"].attrs["dimension"]
    ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
    cell_width = hdf["Cells/Meta-data"].attrs["size"]
    tot_mass = nparts * pmass

    # Define the simulation's "resolution"
    pix_res = hdf["/PartType1/Softenings"][0]

    res = (2 * np.int32(np.floor(cell_width / pix_res)),
           2 * np.int32(np.floor(cell_width / pix_res)))

    if rank == 0:
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Npart:", nparts)
        print("Number of cells:", ncells)
        print("Cell width:", cell_width)
        print("Cell Pixel resolution:", np.int32(np.floor(cell_width / pix_res)))
        print("Cell Pixel resolution (with padding):",
              res)
        print("Full image resolution:", np.int32(np.floor(boxsize / pix_res)))

    # Define centre
    true_cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define the camera's position
    cam_pos = np.array([boxsize / 2, boxsize / 2, - boxsize])

    # Define targets
    targets = [[0, 0, 0]]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same',
                            'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same',
                             'same']
    anchors['r'] = ["infinity", 'same', 'same',
                    'same', 'same', 'same',
                    'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same',
                       'same']
    anchors['extent'] = [2 * cell_width, 'same', 'same', 'same', 'same',
                         'same', 'same',
                         'same']

    if rank == 0:

        ncell = -1

        # Master process executes code below
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:

            data = comm.recv(source=MPI.ANY_SOURCE,
                             tag=MPI.ANY_TAG,
                             status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:

                ncell += 1

                # Worker is ready, so send it a task
                if ncell < ncells:

                    comm.send(ncell, dest=source, tag=tags.START)

                else:

                    # There are no tasks left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.EXIT:

                closed_workers += 1

    else:

        results = []

        while True:

            comm.send(None, dest=0, tag=tags.READY)
            my_cell = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                # Retrieve the offset and counts
                my_offset = hdf["/Cells/OffsetsInFile/PartType1"][my_cell]
                my_count = hdf["/Cells/Counts/PartType1"][my_cell]
                my_cent = hdf["/Cells/Centres"][my_cell, :]

                poss = hdf["/PartType1/Coordinates"][
                       my_offset:my_offset + my_count, :]
                masses = hdf["/PartType1/Masses"][
                         my_offset:my_offset + my_count] * 10 ** 10
                poss -= my_cent

                hsmls = hdf["/PartType1/Softenings"][
                        my_offset:my_offset + my_count]

                # Compute camera radial distance to cell
                cam_sep = cam_pos - my_cent
                cam_dist = np.sqrt(cam_sep[0] ** 2
                                   + cam_sep[1] ** 2
                                   + cam_sep[2] ** 2)

                # Define anchors dict for camera parameters
                anchors['r'] = ["infinity", 'same', 'same',
                                'same', 'same', 'same',
                                'same', 'same']

                # Define the camera trajectory
                cam_data = camera_tools.get_camera_trajectory(targets, anchors)

                mean_den = tot_mass / boxsize ** 3

                vmax, vmin = 8, 0

                print("Norm:", vmin, "-", vmax)

                cmap = cmr.eclipse

                # Get images
                img, ang_extent = get_mono_image(cam_data, poss, masses, hsmls,
                                                 num, res)

                print(img)

                norm = Normalize(vmin=vmin, vmax=vmax)

                rgb_output = cmap(norm(img))

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
                               labeltop=False, labelright=False,
                               labelbottom=False)

                ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
                        transform=ax.transAxes, verticalalignment="top",
                        horizontalalignment='right', fontsize=1, color="w")

                ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w',
                        clip_on=False,
                        transform=ax.transAxes)

                ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w',
                        clip_on=False,
                        transform=ax.transAxes)
                ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w',
                        clip_on=False,
                        transform=ax.transAxes)

                axis_to_data = ax.transAxes + ax.transData.inverted()
                left = axis_to_data.transform((0.05, 0.075))
                right = axis_to_data.transform((0.15, 0.075))
                dist = extent[1] * (right[0] - left[0]) / (
                        ang_extent[1] - ang_extent[0])

                ax.text(0.1, 0.055, "%.2f cMpc" % dist,
                        transform=ax.transAxes, verticalalignment="top",
                        horizontalalignment='center', fontsize=1, color="w")

                plt.margins(0, 0)

                fig.savefig('../plots/Ani/DM/Flamingo_DM_' + frame
                            + '_' + str(my_cell) + '.png',
                            bbox_inches='tight',
                            pad_inches=0)

                plt.close(fig)

                results.append((my_cell, my_cent, img))

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    # collected_img = comm.gather(img, root=0)
    #
    # if rank == 0:
    #
    #     final_img = np.zeros_like(img)
    #
    #     for rk, i_img in enumerate(collected_img):
    #         final_img += i_img
    #
    #     norm = Normalize(vmin=vmin, vmax=vmax)
    #
    #     rgb_output = cmap(norm(final_img))
    #
    #     i = cam_data[num]
    #     extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
    #               0, 2 * np.tan(ang_extent[-1]) * i['r']]
    #     print("Extents:", ang_extent, extent)
    #
    #     dpi = rgb_output.shape[0] / 2
    #     print("DPI, Output Shape:", dpi, rgb_output.shape)
    #     fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
    #     ax = fig.add_subplot(111)
    #
    #     ax.imshow(rgb_output, extent=ang_extent, origin='lower')
    #     ax.tick_params(axis='both', left=False, top=False, right=False,
    #                    bottom=False, labelleft=False,
    #                    labeltop=False, labelright=False, labelbottom=False)
    #
    #     ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
    #             transform=ax.transAxes, verticalalignment="top",
    #             horizontalalignment='right', fontsize=1, color="w")
    #
    #     ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w', clip_on=False,
    #             transform=ax.transAxes)
    #
    #     ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
    #             transform=ax.transAxes)
    #     ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
    #             transform=ax.transAxes)
    #
    #     axis_to_data = ax.transAxes + ax.transData.inverted()
    #     left = axis_to_data.transform((0.05, 0.075))
    #     right = axis_to_data.transform((0.15, 0.075))
    #     dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])
    #
    #     ax.text(0.1, 0.055, "%.2f cMpc" % dist,
    #             transform=ax.transAxes, verticalalignment="top",
    #             horizontalalignment='center', fontsize=1, color="w")
    #
    #     plt.margins(0, 0)
    #
    #     fig.savefig('../plots/Ani/DM/Flamingo_DM_' + frame + '.png',
    #                 bbox_inches='tight',
    #                 pad_inches=0)
    #
    #     plt.close(fig)


nframes = 1000
if int(sys.argv[2]) > 0:
    frame = "%05d" % int(sys.argv[1])
    if os.path.isfile('../plots/Ani/DM/Flamingo_DM_' + frame + '.png'):
        print("File exists")
    else:
        single_frame(int(sys.argv[1]), nframes=nframes,
                     size=size, rank=rank, comm=comm)
else:
    single_frame(int(sys.argv[1]), nframes=nframes,
                 size=size, rank=rank, comm=comm)
