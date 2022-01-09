#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml

ml.use('Agg')
import numpy as np
from get_images import make_spline_img_cart
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from astropy.cosmology import Planck13 as cosmo
import sys
import h5py
# from images import get_mono_image
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

    npix_per_2cells = 2 * np.int32(np.floor(cell_width / pix_res))
    for i in range(3):
        if npix_per_2cells[i] % 2 != 0:
            npix_per_2cells[i] += 1
    res = (npix_per_2cells[0], npix_per_2cells[1])
    full_image_res = (int(ncells**(1/3) * res[0] // 2),
                      int(ncells**(1/3) * res[1] // 2))

    # Define width and height
    w, h = 2 * cell_width[1], 2 * cell_width[0]

    mean_den = tot_mass / boxsize ** 3

    vmax, vmin = 10000 * mean_den, mean_den

    cmap = cmr.eclipse

    if rank == 0:
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Npart:", nparts)
        print("Number of cells:", ncells)
        print("Cell width:", cell_width)
        print("Cell Pixel resolution:", np.int32(np.floor(cell_width[0]
                                                          / pix_res)))
        print("Cell Pixel resolution (with padding):",
              res)
        print("Full image resolution:", full_image_res)

    # Define centre
    true_cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define the camera's position
    cam_pos = np.array([boxsize / 2, boxsize / 2, - boxsize])

    if rank == 0:

        results = []

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

                if my_count > 0:

                    poss = hdf["/PartType1/Coordinates"][
                           my_offset:my_offset + my_count, :]
                    masses = hdf["/PartType1/Masses"][
                             my_offset:my_offset + my_count] * 10 ** 10
                    poss -= my_cent

                    hsmls = hdf["/PartType1/Softenings"][
                            my_offset:my_offset + my_count]

                    # Compute camera radial distance to cell
                    cam_sep = cam_pos - my_cent - true_cent
                    cam_dist = np.sqrt(cam_sep[0] ** 2
                                       + cam_sep[1] ** 2
                                       + cam_sep[2] ** 2)

                    # # Define anchors dict for camera parameters
                    # anchors['r'] = ["infinity", 'same', 'same',
                    #                 'same', 'same', 'same',
                    #                 'same', 'same']
                    #
                    # # Define the camera trajectory
                    # cam_data = camera_tools.get_camera_trajectory(targets,
                    #                                               anchors)

                    # Get images
                    img = make_spline_img_cart(poss, res, w, h, masses, hsmls)
                    print("Image limits:", img.min(), img.max())

                    # if img.max() > 0:
                    #
                    #     norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
                    #
                    #     rgb_output = cmap(norm(img))
                    #
                    #     # print("Extents:", ang_extent, extent)
                    #
                    #     dpi = rgb_output.shape[0] / 2
                    #     # print("DPI, Output Shape:", dpi, rgb_output.shape)
                    #     fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
                    #     ax = fig.add_subplot(111)
                    #
                    #     ax.imshow(rgb_output, extent=[-h/2, h/2, -w/2, w/2],
                    #               origin='lower')
                    #     ax.tick_params(axis='both', left=False, top=False, right=False,
                    #                    bottom=False, labelleft=False,
                    #                    labeltop=False, labelright=False,
                    #                    labelbottom=False)
                    #
                    #     # ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
                    #     #         transform=ax.transAxes, verticalalignment="top",
                    #     #         horizontalalignment='right', fontsize=1, color="w")
                    #     #
                    #     # ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w',
                    #     #         clip_on=False,
                    #     #         transform=ax.transAxes)
                    #     #
                    #     # ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w',
                    #     #         clip_on=False,
                    #     #         transform=ax.transAxes)
                    #     # ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w',
                    #     #         clip_on=False,
                    #     #         transform=ax.transAxes)
                    #
                    #     plt.margins(0, 0)
                    #
                    #     fig.savefig('../plots/Ani/DM/Flamingo_DM_' + frame
                    #                 + '_' + str(my_cell) + '.png',
                    #                 bbox_inches='tight',
                    #                 pad_inches=0)
                    #
                    #     plt.close(fig)

                    results.append((my_cell, my_cent, img))

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    results_list = comm.gather(results, root=0)

    if rank == 0:

        final_img = np.zeros_like(full_image_res)

        for res in results_list:
            for tup in res:

                cell = tup[0]
                cent = tup[1]
                img = tup[2]

                i = int(cent[0] / pix_res)
                j = int(cent[1] / pix_res)

                dimens = img.shape

                ilow = i - (dimens[0] // 2)
                ihigh = i + (dimens[0] // 2)
                jlow = j - (dimens[1] // 2)
                jhigh = j + (dimens[1] // 2)

                if ilow < 0:
                    img = img[abs(ilow):, :]
                    ilow = 0
                if jlow < 0:
                    img = img[:, abs(jlow):]
                    jlow = 0
                if ihigh >= full_image_res[1]:
                    img = img[:ihigh - full_image_res[1] - 1, :]
                    ihigh = full_image_res[1] - 1
                if jhigh >= full_image_res[0]:
                    img = img[:, :jhigh - full_image_res[0] - 1]
                    jhigh = full_image_res[0] - 1

                print(ihigh - ilow, jhigh - jlow, img.shape)

                final_img[ilow: ihigh, jlow: jhigh] += img

        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        rgb_output = cmap(norm(final_img))

        dpi = rgb_output.shape[0] / 2
        print("DPI, Output Shape:", dpi, rgb_output.shape)
        fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
        ax = fig.add_subplot(111)

        ax.imshow(rgb_output, origin='lower')
        ax.tick_params(axis='both', left=False, top=False, right=False,
                       bottom=False, labelleft=False,
                       labeltop=False, labelright=False, labelbottom=False)

        # ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
        #         transform=ax.transAxes, verticalalignment="top",
        #         horizontalalignment='right', fontsize=1, color="w")
        #
        # ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w', clip_on=False,
        #         transform=ax.transAxes)
        #
        # ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
        #         transform=ax.transAxes)
        # ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
        #         transform=ax.transAxes)
        #
        # axis_to_data = ax.transAxes + ax.transData.inverted()
        # left = axis_to_data.transform((0.05, 0.075))
        # right = axis_to_data.transform((0.15, 0.075))
        # dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])

        # ax.text(0.1, 0.055, "%.2f cMpc" % dist,
        #         transform=ax.transAxes, verticalalignment="top",
        #         horizontalalignment='center', fontsize=1, color="w")

        plt.margins(0, 0)

        fig.savefig('../plots/Ani/DM/Flamingo_DM_' + frame + '.png',
                    bbox_inches='tight',
                    pad_inches=0)

        plt.close(fig)


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
