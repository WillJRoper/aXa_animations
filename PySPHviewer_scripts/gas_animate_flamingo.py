#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml

ml.use('Agg')
import numpy as np
from get_images import make_spline_img_3d
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
import scipy.sparse as sp
from scipy.spatial import cKDTree

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

    # Resolution modification for debugging
    mod = 1

    # Get metadata
    boxsize = hdf["Header"].attrs["BoxSize"][0]
    z = hdf["Header"].attrs["Redshift"]
    nparts = hdf["Header"].attrs["NumPart_Total"][1]
    pmass = hdf["/PartType0/Masses"][0] * 10 ** 10
    cdim = hdf["Cells/Meta-data"].attrs["dimension"]
    ncells = hdf["/Cells/Meta-data"].attrs["nr_cells"]
    cell_width = hdf["Cells/Meta-data"].attrs["size"]
    tot_mass = nparts * pmass

    print(hdf["/PartType0/SmoothingLengths"].max())

    # Define the simulation's "resolution"
    soft = hdf["/PartType1/Softenings"][0]
    pix_res = soft * mod

    # Define padding
    pad_pix = 24
    pad_mpc = pad_pix * pix_res

    # Define (half) the kth dimension of spline smoothing array in Mpc
    k_dim = soft * 20
    k_res = int(np.ceil(k_dim / pix_res))
    k_dim = k_res * pix_res

    npix_per_cell = np.int32(cell_width / pix_res)
    npix_per_cell_with_pad = npix_per_cell + pad_pix
    res = (npix_per_cell_with_pad[0], npix_per_cell_with_pad[1], k_res)
    full_image_res = (int(ncells**(1/3) * npix_per_cell[0]),
                      int(ncells**(1/3) * npix_per_cell[1]))

    # Set up the final image for each rank
    rank_final_img = np.zeros(full_image_res, dtype=np.float32)

    mean_den = tot_mass / boxsize ** 3

    vmax, vmin = 10**8, 10**1

    cmap = plt.get_cmap('magma')

    if rank == 0:
        print("Boxsize:", boxsize)
        print("Redshift:", z)
        print("Npart:", nparts)
        print("Number of cells:", ncells)
        print("Cell width:", cell_width)
        print("Cell Pixel resolution:", np.int32(np.floor(cell_width[0]
                                                          / pix_res)))
        print("Cell Pixel resolution (with padding):", res)
        print("Full image resolution:", full_image_res)
        print("Vmin - Vmax:", np.log10(vmin), "-", np.log10(vmax))

    # Define centre
    true_cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define the camera's position
    cam_pos = np.array([boxsize / 2, boxsize / 2, - boxsize])

    # Get cells for this rank
    all_cells = []
    i_s = []
    j_s = []
    k_s = []
    for i in range(cdim[0]):
        for j in range(cdim[1]):
            for k in range(3):

                cell = (k + cdim[2] * (j + cdim[1] * i))
                all_cells.append(cell)
                i_s.append(i)
                j_s.append(j)
                k_s.append(k)

    rank_cells = np.linspace(0, len(all_cells), size + 1, dtype=int)
    my_cells = all_cells[rank_cells[rank]: rank_cells[rank + 1]]
    my_i_s = i_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_j_s = j_s[rank_cells[rank]: rank_cells[rank + 1]]
    my_k_s = k_s[rank_cells[rank]: rank_cells[rank + 1]]

    print("Rank:", rank)
    print("My Ncells:", len(my_cells))

    # Define range and extent for the images
    imgrange = ((-(pad_mpc / 2), cell_width + (pad_mpc / 2)),
                (-(pad_mpc / 2), cell_width + (pad_mpc / 2)),
                (-k_dim / 2, k_dim / 2))
    imgextent = [-(pad_mpc / 2), cell_width + (pad_mpc / 2),
                 -(pad_mpc / 2), cell_width + (pad_mpc / 2)]

    for i, j, k, my_cell in zip(my_i_s, my_j_s, my_k_s, my_cells):

        # Retrieve the offset and counts
        my_offset = hdf["/Cells/OffsetsInFile/PartType0"][my_cell]
        my_count = hdf["/Cells/Counts/PartType0"][my_cell]
        my_cent = hdf["/Cells/Centres"][my_cell, :]

        # Define the edges of this cell with pad region
        my_edges = np.array([(i * cell_width[0]),
                             (j * cell_width[1]),
                             (k * cell_width[2])])

        if my_count > 0:

            # Get particle data
            ini_poss = hdf["/PartType0/Coordinates"][
                   my_offset:my_offset + my_count, :]
            masses = hdf["/PartType0/Masses"][
                     my_offset:my_offset + my_count] * 10 ** 10
            temps = hdf["PartType0/Temperatures"][
                     my_offset:my_offset + my_count]
            hsmls = hdf["/PartType0/SmoothingLengths"][
                    my_offset:my_offset + my_count]

            # Shift particle positions to this cell with pad region
            poss = ini_poss - my_edges + (pad_mpc / 2)

            # Remove particles too far from the cell
            xokinds = np.logical_and(poss[:, 0] < cell_width[0] + (pad_mpc / 2),
                                     poss[:, 0] > - (pad_mpc / 2))
            yokinds = np.logical_and(poss[:, 1] < cell_width[1] + (pad_mpc / 2),
                                     poss[:, 1] > - (pad_mpc / 2))
            okinds = np.logical_and(xokinds, yokinds)
            poss = poss[okinds, :]
            masses = masses[okinds]
            temps = masses[okinds]
            hsmls = hsmls[okinds]

            # Compute camera radial distance to cell
            cam_sep = cam_pos - my_cent - true_cent
            cam_dist = np.sqrt(cam_sep[0] ** 2
                               + cam_sep[1] ** 2
                               + cam_sep[2] ** 2)

            # Get images
            if poss.shape[0] > 0:
                mimg = make_spline_img_3d(poss, res, pad_mpc, masses,
                                         hsmls, pix_res)
                tmimg = make_spline_img_3d(poss, res, pad_mpc, temps * masses,
                                         hsmls, pix_res)

                img = tmimg / mimg
                
                dimens = img.shape

                # Get the indices for this cell edge
                ilow = int((my_edges[0] - (pad_mpc / 2)) / pix_res)
                jlow = int((my_edges[1] - (pad_mpc / 2)) / pix_res)
                ihigh = ilow + dimens[0]
                jhigh = jlow + dimens[1]

                # If we are not at the edges we don't need any wrapping
                # and can just assign the grid at once
                if (i != 0 and i < cdim[0] - 1
                        and j != 0 and j < cdim[0] - 1):
                    rank_final_img[ilow: ihigh, jlow: jhigh] += img

                else:  # we must wrap

                    # Define indices ranges
                    irange = np.arange(ilow, ihigh, 1, dtype=int)
                    jrange = np.arange(jlow, jhigh, 1, dtype=int)

                    # To allow for wrapping we need to assign pix by pix ( :( )
                    for i_img, i_full in enumerate(irange):
                        for j_img, j_full in enumerate(jrange):
                            rank_final_img[i_full % rank_final_img.shape[0],
                                           j_full % rank_final_img.shape[1]] += img[i_img, j_img]

    hdf.close()

    # Convert image to a spare matrix for efficient saving
    sparse_img = sp.coo_matrix(rank_final_img)
    sp.save_npz("logs/img_" + str(num) + "_" + str(rank) + ".npz",
                sparse_img)

    comm.Barrier()

    if rank == 0:

        final_img = np.zeros(full_image_res, dtype=np.float32)

        for rk in range(0, size):

            sparse_rank_img = sp.load_npz("logs/img_" + str(num)
                                          + "_" + str(rk) + ".npz")

            os.remove("logs/img_" + str(num) + "_" + str(rk) + ".npz")

            final_img += sparse_rank_img.toarray()

        print("Minimum/Maximum", np.log10(final_img[final_img >0].min()),
              np.log10(final_img.max()))
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        rgb_output = cmap(norm(final_img))

        print(rgb_output.shape, rgb_output.dtype,
              rgb_output.min(), rgb_output.max())

        # im = Image.fromarray(rgb_output, "RGBA")
        # im.save('../plots/Ani/DM/Flamingo_DM_' + frame + '.tiff')

        # cv2.imwrite('../plots/Ani/DM/Flamingo_DM_' + frame + '.jp2',
        #             cv2.cvtColor(rgb_output, cv2.COLOR_RGBA2BGR))

        if rgb_output.shape[0] > 2**14:

            # Compute the number of images to split full projection into
            img_size_i = rgb_output.shape[0]
            img_size_j = rgb_output.shape[1]
            ilims = np.linspace(0, img_size_i, int(img_size_i / 2**13) + 1,
                                dtype=int)
            jlims = np.linspace(0, img_size_j, int(img_size_j / 2 ** 13) + 1,
                                dtype=int)

            for i_ind in range(ilims[:-1].size):
                for j_ind in range(jlims[:-1].size):

                    # Get the subsample image
                    subsample = rgb_output[ilims[i_ind]: ilims[i_ind + 1],
                                jlims[j_ind]: jlims[j_ind + 1], :]

                    dpi = subsample.shape[0]
                    print("DPI, Output Shape:", dpi, subsample.shape,
                          rgb_output.shape)
                    fig = plt.figure(figsize=(1, 1), dpi=dpi)
                    ax = fig.add_subplot(111)

                    ax.imshow(subsample, origin='lower')
                    ax.tick_params(axis='both', left=False, top=False,
                                   right=False,
                                   bottom=False, labelleft=False,
                                   labeltop=False, labelright=False,
                                   labelbottom=False)
                    ax.axis('off')

                    plt.margins(0, 0)

                    fig.savefig('../plots/Ani/Gas_Temp/Flamingo_Gas_Temp_%s_%d_%d.tiff'
                                % (frame, i_ind, j_ind),
                                bbox_inches='tight',
                                pad_inches=0, transparent=True)

                    plt.close(fig)

        else:

            dpi = rgb_output.shape[0]
            print("DPI, Output Shape:", dpi, rgb_output.shape)
            fig = plt.figure(figsize=(1, 1), dpi=dpi)
            ax = fig.add_subplot(111)

            ax.imshow(rgb_output, origin='lower')
            ax.tick_params(axis='both', left=False, top=False, right=False,
                           bottom=False, labelleft=False,
                           labeltop=False, labelright=False, labelbottom=False)
            ax.axis('off')

            plt.margins(0, 0)

            fig.savefig('../plots/Ani/Gas_Temp/Flamingo_Gas_Temp_%s.tiff'
                        % frame,
                        bbox_inches='tight',
                        pad_inches=0, transparent=True)

            plt.close(fig)


nframes = 1000
single_frame(int(sys.argv[1]), nframes=nframes,
             size=size, rank=rank, comm=comm)
