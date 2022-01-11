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

    snaps = [str(i).zfill(4) for i in range(0, 30)]
    snap = snaps[num]

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
    cdim = hdf["Cells/Meta-data"].attrs["dimension"]
    ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
    cell_width = hdf["Cells/Meta-data"].attrs["size"]
    tot_mass = nparts * pmass

    # Define the simulation's "resolution"
    pix_res = hdf["/PartType1/Softenings"][0]

    npix_per_cell = np.int32(np.floor(cell_width / pix_res))
    npix_per_cell_with_pad = npix_per_cell + 200
    for i in range(3):
        if npix_per_cell_with_pad[i] % 2 != 0:
            npix_per_cell_with_pad[i] += 1
    res = (npix_per_cell_with_pad[0], npix_per_cell_with_pad[1])
    full_image_res = (int(ncells**(1/3) * npix_per_cell[0]) + 500,
                      int(ncells**(1/3) * npix_per_cell[1]) + 500)

    # Set up the final image for each rank
    rank_final_img = np.zeros(full_image_res)

    # Define width and height
    w, h = 2 * cell_width[1], 2 * cell_width[0]

    mean_den = tot_mass / boxsize ** 3

    vmax, vmin = 5000 * mean_den, mean_den

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
        print("Vmin - Vmax:", np.log10(vmin), "-", np.log10(vmax))

    # Define centre
    true_cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define the camera's position
    cam_pos = np.array([boxsize / 2, boxsize / 2, - boxsize])

    # Get cells for this rank
    rank_cells = np.linspace(0, ncells, size + 1, dtype=int)
    my_cells = np.arange(rank_cells[rank], rank_cells[rank + 1], 1, dtype=int)

    print("Rank:", rank)
    print("My Ncells:", my_cells.size)

    for my_cell in my_cells:

        i = my_cell / (cdim[1] * cdim[2])
        j = (my_cell / cdim[2]) % cdim[1]
        k = my_cell % cdim[2]

        if k > 1:
            continue

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

            # Get images
            img = make_spline_img_cart(poss, res, w, h, masses, hsmls)

            ilow = int((my_cent[0] - (cell_width[0] / 2) - 100 * pix_res) / pix_res) + 100
            jlow = int((my_cent[1] - (cell_width[1] / 2) - 100 * pix_res) / pix_res) + 100

            dimens = img.shape

            ihigh = ilow + dimens[0]
            jhigh = jlow + dimens[1]

            if ilow < 0:
                img = img[100:, :]
                ilow = 0
                print("ilow<0", ilow, img.shape)
            if jlow < 0:
                img = img[:, 100:]
                jlow = 0
                print("jlow<0", jlow, img.shape)
            if ihigh >= full_image_res[1]:
                img = img[:res[1] - 100, :]
                print("ihigh>size", ihigh, img.shape)
                ihigh = full_image_res[1] - 1
                print("ihigh>size", ihigh, img.shape)
            if jhigh >= full_image_res[0]:
                img = img[:, :res[0] - 100]
                print("jhigh>size", jhigh, img.shape)
                jhigh = full_image_res[0] - 1
                print("jhigh>size", jhigh, img.shape)

            rank_final_img[ilow: ihigh, jlow: jhigh] += img

    hdf.close()

    out_hdf = h5py.File("logs/out_" + str(num) + "_" + str(rank) + ".hdf5",
                        "w")

    out_hdf.create_dataset("Img", data=rank_final_img,
                           shape=rank_final_img.shape)

    out_hdf.close()

    comm.Barrier()

    if rank == 0:

        final_img = np.zeros(full_image_res)

        for rank in range(0, size):
            out_hdf = h5py.File("logs/out_" + str(num)
                                + "_" + str(rank) + ".hdf5", "r")

            img = out_hdf["Img"][...]

            final_img += img

            out_hdf.close()

        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        rgb_output = cmap(norm(final_img))

        dpi = 2**16 - 1
        print("DPI, Output Shape:", dpi, rgb_output.shape)
        fig = plt.figure(figsize=(2, 2), dpi=dpi)
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
