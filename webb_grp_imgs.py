import sys
from synthobs.sed import models
from mpi4py import MPI
import mpi4py
from scipy.spatial import cKDTree
from scipy import signal
import h5py
from astropy.cosmology import Planck13 as cosmo
import seaborn as sns
import warnings
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


# The above has to be imported first
import webbpsf
import flare
import flare.filters

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
nranks = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()   # get MPI status object

sns.set_context("paper")
sns.set_style('whitegrid')


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def DTM_fit(Z, Age):
    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5 / (D0 * Z)
    DTM = D0 + (D1 - D0) * (1. - np.exp(-alpha * (Z ** beta)
                                        * ((Age / (1e3 * tau)) ** gamma)))
    if np.isnan(DTM) or np.isinf(DTM):
        DTM = 0.

    return DTM


def flux(tag, Masses, Ages, Metallicities, MetSurfaceDensities, gasMetallicities,
         kappa=0.0795, BC_fac=1, IMF='Chabrier_300',
         filters=flare.filters.NIRCam_W, Type='Total', log10t_BC=7.):

    # Load the kernel
    kinp = np.load('/cosma7/data/dp004/dc-payy1/my_files/los/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    # Set up output dictionary
    Fnus = {}

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    model.dust_ISM = (
        'simple', {'slope': -1.})  # Define dust curve for ISM
    model.dust_BC = ('simple', {
        'slope': -1.})  # Define dust curve for birth cloud component

    z = float(tag[5:].replace('p', '.'))
    F = flare.filters.add_filters(filters, new_lam=model.lam * (1. + z))

    # Create new Fnu grid for each filter. In units of nJy/M_sol
    model.create_Fnu_grid(F, z, cosmo)

    Mage = np.nansum(Masses * Ages) / np.nansum(Masses)
    Z = np.nanmean(gasMetallicities)

    MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

    if Type == 'Total':
        # Calculate V-band (550nm) optical depth for each star particle
        tauVs_ISM = kappa * MetSurfaceDensities
        tauVs_BC = BC_fac * (Metallicities / 0.01)
        fesc = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = BC_fac * (Metallicities / 0.01)
        fesc = 0.0

    else:
        tauVs_ISM = None
        tauVs_BC = None
        fesc = None
        ValueError(F"Undefined Type {Type}")

    # Calculate rest-frame Luminosity. In units of erg/s/Hz
    for f in filters:
        print("Computing fluxes for %s" % f)
        # Calculate rest-frame flux of each object in nJy
        Fnu = models.generate_Fnu_array(model, F, f, Masses, Ages,
                                        Metallicities, tauVs_ISM, tauVs_BC,
                                        fesc=fesc, log10t_BC=log10t_BC)

        Fnus[f] = Fnu

    return Fnus


def cubic_spline(q):
    k = 21 / (2 * np.pi)

    w = np.zeros_like(q)
    okinds = q <= 1

    w[okinds] = (1 - q[okinds]) ** 4 * (1 + 4 * q[okinds])

    return k * w


def make_spline_img_3d(pos, Ndim, ls, smooth, oversample,
                       fov_arcsec, cent, arc_res, spline_func=cubic_spline,
                       spline_cut_off=1):

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    # Set up particle pointer
    n = 0
    step = 1000

    # Initialise the image array
    img = {f: np.zeros((Ndim, Ndim), dtype=np.float64) for f in ls.keys()}

    if rank == 0:

        # Master process executes code below
        num_workers = nranks - 1
        closed_workers = 0
        while closed_workers < num_workers:

            data = comm.recv(source=MPI.ANY_SOURCE,
                             tag=MPI.ANY_TAG,
                             status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            # Report progress
            if n % 10000 == 0:
                print(n)

            if tag == tags.READY:

                # If there are still particles send some
                if n < pos.shape[0]:
                    comm.send(n, dest=source, tag=tags.START)
                    n += step
                else:
                    # There are no particles left so terminate this process
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.EXIT:

                closed_workers += 1
                print("Finished workers:", closed_workers, "of", num_workers)

    else:

        # Initialise the temporary image array for each particle
        temp_img = np.zeros((Ndim, Ndim), dtype=np.float64)

        # Create a dictionary to cache psfs
        psfs = {}

        # Loop until all particles are done
        while True:

            comm.send(None, dest=0, tag=tags.READY)
            n = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:

                # Define bounds
                low = n
                high = n + step
                if high > pos.shape[0]:
                    high = pos.shape[0]

                # Get this particle's data
                iposs = pos[low: high, :]
                smls = smooth[low: high]

                for (ind, ipos), sml in zip(enumerate(iposs), smls):

                    # How many pixels are in the smoothing length?
                    n_sml = int(np.ceil(sml / arc_res)) + 1

                    # Where is this particle in the image?
                    i = int(np.floor(ipos[0] / arc_res))
                    j = int(np.floor(ipos[1] / arc_res))
                    k = int(np.floor(ipos[2] / arc_res))

                    # Define kernel lower edges
                    i_low = np.max((i - n_sml, 0))
                    j_low = np.max((j - n_sml, 0))
                    k_low = np.max((k - n_sml, 0))

                    # Define kernel upper edges
                    i_high = np.min((i + n_sml, Ndim))
                    j_high = np.min((j + n_sml, Ndim))
                    k_high = np.min((k + n_sml, Ndim))

                    # Define kernel shape
                    k_shape = (i_high - i_low,
                               j_high - j_low,
                               k_high - k_low)

                    # Create array of pixel positions in the kernel
                    dist = np.zeros(k_shape)
                    for ii, iii in enumerate(range(i_low, i_high)):
                        for jj, jjj in enumerate(range(j_low, j_high)):
                            for kk, kkk in enumerate(range(k_low, k_high)):

                                # Compute distance
                                dx = (iii * arc_res) - ipos[0]
                                dy = (jjj * arc_res) - ipos[1]
                                dz = (kkk * arc_res) - ipos[2]
                                d = np.sqrt(dx * dx + dy * dy + dz * dz)

                                # Store the distance
                                dist[ii, jj, kk] = d

                    # Get the kernel
                    w = spline_func(dist / sml)

                    # Place the kernel for this particle within the img
                    kernel = w / sml ** 3
                    norm_kernel = kernel / np.sum(kernel)
                    sum_kernel = np.sum(norm_kernel, axis=-1)

                    # Loop over filters applying this
                    for f in ls.keys():
                        temp_img[i_low: i_high,
                                 j_low: j_high] = ls[f][low + ind] * sum_kernel

                        # Handle oversample for long wavelength channel
                        if f in ["F277W", "F356W", "F444W"]:
                            oversample *= 2

                        # Get central pixel indices
                        # cent_ind = inds[np.argmin(dist)]
                        # i, j = pix_pos[cent_ind, 0], pix_pos[cent_ind, 1]

                        # # Get cached psf
                        # if (i, j) in psfs:
                        #     psf = psfs[(i, j)]
                        # else:

                        #     # Calculate the r and theta for this particle
                        #     ipos -= cent
                        #     r = np.sqrt(ipos[0] ** 2 + ipos[1] ** 2)
                        #     theta = (np.rad2deg(np.arctan(ipos[1] / ipos[2]))
                        #              + 360) % 360

                        #     # Get PSF for this filter
                        #     nc = webbpsf.NIRCam()
                        #     nc.options['source_offset_r'] = r
                        #     nc.options['source_offset_theta'] = theta
                        #     nc.filter = f
                        #     psf = nc.calc_psf(fov_arcsec=fov_arcsec,
                        #                       oversample=oversample)

                        #     # Cache this psf
                        #     psfs[(i, j)] = psf

                        # # Convolve the PSF and include this particle in the image
                        # temp_img = signal.fftconvolve(temp_img, psf[0].data,
                        #                               mode="same")
                        img[f] += temp_img
                        temp_img[:, :] = 0.

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    # Lets let everyone catch up before we continue
    print(rank, "finished")
    comm.Barrier()

    return img


def make_image(reg, snap, width_mpc, width_arc, half_width, npix, oversample,
               arc_res, kpc_res, arcsec_per_kpc_proper, filters,
               rank_plot=False):

    # Set up FLARES regions
    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Open the master file
    hdf = h5py.File(master_base, "r")
    hf = hdf[reg]

    # Get the required data
    gal_smass = np.array(hf[snap + '/Galaxy'].get('M500'),
                         dtype=np.float64)
    cops = np.array(hf[snap + '/Galaxy'].get("COP"),
                    dtype=np.float64).T / (1 + z)
    S_mass_ini = np.array(hf[snap + '/Particle'].get('S_MassInitial'),
                          dtype=np.float64) * 10 ** 10
    S_mass = np.array(hf[snap + '/Particle'].get('S_Mass'),
                      dtype=np.float64) * 10 ** 10
    S_Z = np.array(hf[snap + '/Particle'].get('S_Z_smooth'),
                   dtype=np.float64)
    S_age = np.array(hf[snap + '/Particle'].get('S_Age'),
                     dtype=np.float64) * 1e3
    S_los = np.array(hf[snap + '/Particle'].get('S_los'),
                     dtype=np.float64)
    G_Z = np.array(hf[snap + '/Particle'].get('G_Z_smooth'),
                   dtype=np.float64)
    S_sml = np.array(hf[snap + '/Particle'].get('S_sml'),
                     dtype=np.float64)
    S_coords = np.array(hf[snap + '/Particle'].get('S_Coordinates'),
                        dtype=np.float64).T / (1 + z)
    hdf.close()

    # Get the main target position
    ind = np.argmax(gal_smass)
    target = np.array([3200 / (1 + z) / 2,
                       3200 / (1 + z) / 2,
                       3200 / (1 + z) / 2])
    # target = cops[ind, :]

    # Center positions
    S_coords -= target

    # Excise the region to make an image of
    okinds = np.logical_and(np.abs(S_coords[:, 0]) < half_width,
                            np.abs(S_coords[:, 1]) < half_width)
    okinds = np.logical_and(okinds, np.abs(S_coords[:, 2]) < half_width)
    S_mass_ini = S_mass_ini[okinds]
    S_mass = S_mass[okinds]
    S_Z = S_Z[okinds]
    S_age = S_age[okinds]
    S_los = S_los[okinds]
    S_sml = S_sml[okinds]
    S_coords = S_coords[okinds, :]

    if rank == 0:
        print("There are %d particles in the FOV" % S_mass.size)

    # Shift positions such that they are positive
    S_coords += half_width

    # Convert coordinates into arcseconds
    S_coords *= 10 ** 3 * arcsec_per_kpc_proper
    S_sml *= 10 ** 3 * arcsec_per_kpc_proper
    target_arc = width_arc / 2

    # Get fluxes
    fluxes = flux(snap, S_mass_ini, S_age, S_Z, S_los, G_Z, filters=filters)

    # Define range and extent for the images in arc seconds
    imgrange = ((0, width_arc), (0, width_arc))
    imgextent = [0, width_arc, 0, width_arc]

    # Redefine the z axis to reduce memory uses
    max_sml = np.max(S_sml)
    z_ax_pix = int(np.ceil(max_sml / arc_res)) + 2
    max_sml = z_ax_pix * arc_res

    # Create images
    mono_imgs = make_spline_img_3d(S_coords, npix, fluxes,
                                   S_sml, oversample, width_arc,
                                   target_arc, arc_res)

    # Open file to write out results
    hdf = h5py.File("data/Webb_reg-%s_snap-%s_rank%d.npy"
                    % (reg, snap, rank), "w")

    # Loop over filters saving images
    for f in filters:

        # Get filter code
        fcode = f.split(".")[-1]

        hdf.create_dataset(f, shape=mono_imgs[f].shape,
                           dtype=mono_imgs[f].dtype, data=mono_imgs[f],
                           compression="gzip")

        if rank == 0:
            print("Completed Image for %s" % fcode)

    hdf.close()

    # # Set up RGB image
    # rgb_img = np.zeros((npix, npix, 3))

    # # Populate RGB image
    # for f in filters:

    #     # Get filter code
    #     fcode = f.split(".")[-1]

    #     # Get color for filter
    #     if fcode in ["F356W", "F444W"]:
    #         rgb = 0
    #     elif fcode in ["F200W", "F277W"]:
    #         rgb = 1
    #     elif fcode in ["F090W", "F150W"]:
    #         rgb = 2
    #     else:
    #         print("Failed to assign color for filter %s EXITING..." % fcode)
    #         break

    #     # Assign the image
    #     rgb_img[:, :, rgb] += mono_imgs[f]

    # if rank_plot:

    #     # Set up figure
    #     dpi = rgb_img.shape[0]
    #     fig = plt.figure(figsize=(1, 1), dpi=dpi)
    #     ax = fig.add_subplot(111)
    #     ax.grid(False)

    #     ax.imshow(rgb_img, extent=imgextent, origin='lower')
    #     ax.tick_params(axis='both', left=False, top=False, right=False,
    #                    bottom=False, labelleft=False,
    #                    labeltop=False, labelright=False, labelbottom=False)

    #     plt.margins(0, 0)

    #     fig.savefig('plots/TempWebb_reg-%s_snap-%s_rank%d.png'
    #                 % (reg, snap, rank),
    #                 bbox_inches='tight',
    #                 pad_inches=0)

    #     plt.close(fig)

    comm.Barrier()


# Define region and snapshot
reg = "00"
snap = '010_z005p000'

# Get redshift
z = float(snap.split("z")[-1].replace("p", "."))

# Set up filters
filters = ["JWST.NIRCAM." + f for f in ["F090W", "F150W", "F200W",
                                        "F277W", "F356W", "F444W"]]

# Define the initial image size in Mpc
width = 1.0

# Get the conversion between arcseconds and pkpc at this redshift
arcsec_per_kpc_proper = cosmo.arcsec_per_kpc_proper(z).value

# Set up image resolution
oversample = 2
arc_res = 0.031 / oversample
kpc_res = arc_res / arcsec_per_kpc_proper
npix = int(np.ceil(width * 10 ** 3 / kpc_res))

if rank == 0:
    print("Image resolution is %d" % npix)

# Define the true image size in Mpc
width_mpc = kpc_res * npix / 10 ** 3
width_arc = arc_res * npix
half_width = width_mpc / 2

if rank == 0:
    print("Image FOV is (%.2f, %.2f) arcseconds/(%.2f, %.2f) pMpc"
          % (width_arc, width_arc, width_mpc, width_mpc))

# Define range and extent for the images in arc seconds
imgrange = ((0, width_arc), (0, width_arc))
imgextent = [0, width_arc, 0, width_arc]

if nranks > 1:
    make_image(reg, snap, width_mpc, width_arc, half_width, npix, oversample,
               arc_res, kpc_res, arcsec_per_kpc_proper, filters)

if rank == 0:

    # Are we using an existing file?
    use_file = False
    if len(sys.argv) > 3:
        use_file = bool(sys.argv[3])

    # Apply the PSF?
    use_psf = False
    if len(sys.argv) > 4:
        use_psf = bool(sys.argv[4])

    # Initialise the image array
    img = np.zeros((npix, npix, 3), dtype=np.float64)

    if use_file:

        # Open file to write out results
        hdf = h5py.File("data/Webb_reg-%s_snap-%s.npy"
                        % (reg, snap), "r")

        for f in filters:

            # Get filter code
            fcode = f.split(".")[-1]

            print("Combining filter %s" % f)

            fimg = hdf[f]

            if use_psf:

                # Handle oversample for long wavelength channel
                if fcode in ["F277W", "F356W", "F444W"]:
                    osample = 2 * oversample
                else:
                    osample = oversample

                # Get PSF for this filter
                nc = webbpsf.NIRCam()
                nc.filter = fcode
                psf = nc.calc_psf(fov_arcsec=width_arc,
                                  oversample=osample)

                # Convolve the PSF and include this particle in the image
                fimg = signal.fftconvolve(fimg, psf[0].data, mode="same")

            # Get color for filter
            if fcode in ["F356W", "F444W"]:
                rgb = 0
            elif fcode in ["F200W", "F277W"]:
                rgb = 1
            elif fcode in ["F090W", "F150W"]:
                rgb = 2
            else:
                print("Failed to assign color for filter %s EXITING..." % fcode)
                break

            # Assign the image
            img[:, :, rgb] += fimg

        hdf.close()

    else:

        # Open file to write out results
        hdf = h5py.File("data/Webb_reg-%s_snap-%s.npy"
                        % (reg, snap), "w")

        files = glob.glob("data/*.npy")

        for f in filters:

            fimg = np.zeros((npix, npix), dtype=np.float64)

            # Get filter code
            fcode = f.split(".")[-1]

            # Combine rank images together
            for r, path in enumerate(files):

                print("Combinging image from file %s" % path)

                rank_hdf = h5py.File("data/Webb_reg-%s_snap-%s_rank%d.npy"
                                     % (reg, snap, r), "w")

                rank_img = rank_hdf[f][...]
                fimg += rank_img
                rank_hdf.close()

                hdf.create_dataset(f,
                                   shape=fimg.shape,
                                   dtype=fimg.dtype,
                                   data=fimg,
                                   compression="gzip")

            if use_psf:

                # Handle oversample for long wavelength channel
                if fcode in ["F277W", "F356W", "F444W"]:
                    osample = 2 * oversample
                else:
                    osample = oversample

                # Get PSF for this filter
                nc = webbpsf.NIRCam()
                nc.filter = fcode
                psf = nc.calc_psf(fov_arcsec=width_arc,
                                  oversample=osample)

                # Convolve the PSF and include this particle in the image
                fimg = signal.fftconvolve(fimg, psf[0].data, mode="same")

            # Get color for filter
            if fcode in ["F356W", "F444W"]:
                rgb = 0
            elif fcode in ["F200W", "F277W"]:
                rgb = 1
            elif fcode in ["F090W", "F150W"]:
                rgb = 2
            else:
                print("Failed to assign color for filter %s EXITING..." % fcode)
                break

            # Assign the image
            img[:, :, rgb] += fimg

        hdf.close()

    # Normalise image between 0 and 1
    vmin, vmax = float(sys.argv[1]), float(sys.argv[2])
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    print(img[img > 0].min(), img.max(), np.percentile(img[img > 0], 50),
          np.percentile(img[img > 0], 99.9))
    img = norm(img).data
    print(img.min(), img.max())

    # Set up figure
    dpi = img.shape[0]
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.grid(False)

    ax.imshow(img, extent=imgextent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    plt.margins(0, 0)

    if use_psf:
        out_path = 'plots/Webb_reg-%s_snap-%s_PSF.png' % (reg, snap)
    else:
        out_path = 'plots/Webb_reg-%s_snap-%s.png' % (reg, snap)
    fig.savefig(out_path,
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)
