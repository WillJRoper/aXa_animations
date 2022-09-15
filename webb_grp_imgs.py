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

sns.set_context("paper")
sns.set_style('whitegrid')


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

    # --- create new Fnu grid for each filter. In units of nJy/M_sol
    model.create_Fnu_grid(F, z, cosmo)

    Mage = np.nansum(Masses * Ages) / np.nansum(Masses)
    Z = np.nanmean(gasMetallicities)

    MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

    if Type == 'Total':
        # --- calculate V-band (550nm) optical depth for each star particle
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

    # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    for f in filters:
        print("Computing fluxes for %s" % f)
        # --- calculate rest-frame flux of each object in nJy
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


def make_spline_img_3d(pos, Ndim, tree, ls, smooth, f, oversample,
                       fov_arcsec, cent, spline_func=cubic_spline,
                       spline_cut_off=1):

    # Initialise the image array
    img = np.zeros((Ndim, Ndim), dtype=np.float32)

    # Define x and y positions of pixels
    X, Y, Z = np.meshgrid(np.arange(0, Ndim, 1),
                          np.arange(0, Ndim, 1),
                          np.arange(0, Ndim, 1))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 3), dtype=int)
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()
    pix_pos[:, 2] = Z.ravel()

    # Handle oversample for long wavelength channel
    if f in ["F277W", "F356W", "F444W"]:
        oversample *= 2

    # Split particles over ranks
    rank_bins = np.linspace(0, pos.shape[0], nranks + 1, dtype=int)

    # Loop over particles
    for ipos, l, sml in zip(pos[rank_bins[rank]: rank_bins[rank + 1], :],
                            ls[rank_bins[rank]: rank_bins[rank + 1]],
                            smooth[rank_bins[rank]: rank_bins[rank + 1]]):

        # Create an empty image for this particle
        smooth_img = np.zeros((Ndim, Ndim, Ndim), dtype=np.float32)

        # Query the tree for this particle
        dist, inds = tree.query(ipos, k=pos.shape[0],
                                distance_upper_bound=spline_cut_off * sml)

        if type(dist) is float:
            continue

        okinds = dist < spline_cut_off * sml
        dist = dist[okinds]
        inds = inds[okinds]

        # Get the kernel
        w = spline_func(dist / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)
        smooth_img[pix_pos[inds, 0], pix_pos[inds, 1], pix_pos[
            inds, 2]] += l * norm_kernel

        # Create 2D image
        temp_img = np.sum(smooth_img, axis=-1)

        # Calculate the r and theta for this particle
        ipos -= cent
        r = np.sqrt(ipos[0] ** 2 + ipos[1] ** 2)
        theta = (np.rad2deg(np.arctan(ipos[1] / ipos[2])) + 360) % 360

        # Get PSF for this filter
        nc = webbpsf.NIRCam()
        nc.options['source_offset_r'] = r
        nc.options['source_offset_theta'] = theta
        nc.filter = f
        psf = nc.calc_psf(fov_arcsec=fov_arcsec,
                          oversample=oversample)

        # Convolve the PSF and include this particle in the image
        img += signal.fftconvolve(img, psf[0].data, mode="same")

    return img


def make_image(reg, snap, width_mpc, width_arc, half_width, npix, oversample,
               arc_res, kpc_res, arcsec_per_kpc_proper):

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
    target = cops[ind, :]

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
    target_arc = width_arc / 2

    # Set up filters
    filters = ["JWST.NIRCAM." + f for f in ["F090W", "F150W", "F200W",
                                            "F277W", "F356W", "F444W"]]

    # Get fluxes
    fluxes = flux(snap, S_mass_ini, S_age, S_Z, S_los, G_Z, filters=filters)

    # Define range and extent for the images in arc seconds
    imgrange = ((0, width_arc), (0, width_arc))
    imgextent = [0, width_arc, 0, width_arc]

    # Define x and y positions of pixels
    X, Y, Z = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], npix),
                          np.linspace(imgrange[1][0], imgrange[1][1], npix),
                          np.linspace(imgrange[1][0], imgrange[1][1], npix))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 3))
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()
    pix_pos[:, 2] = Z.ravel()

    # Build KDTree
    tree = cKDTree(pix_pos)

    # Create dictionary to store each filter's image
    mono_imgs = {}

    # Loop over filters creating images
    for f in filters:

        # Get filter code
        fcode = f.split(".")[-1]

        mono_imgs[f] = make_spline_img_3d(S_coords, npix, tree, fluxes[f],
                                          S_sml, fcode,
                                          oversample, width_arc,
                                          target_arc)

        if rank == 0:
            print("Completed Image for %s", fcode)

    # Set up RGB image
    rgb_img = np.zeros((npix, npix, 3))

    # Populate RGB image
    for f in filters:

        # Get filter code
        fcode = f.split(".")[-1]

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
        rgb_img[:, :, rgb] += mono_imgs[f]

    # Set up figure
    dpi = rgb_img.shape[0]
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_img, extent=imgextent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    plt.margins(0, 0)

    fig.savefig('plots/TempWebb_reg-%s_snap-%s_rank%d.png'
                % (reg, snap, rank),
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)

    # Save the array
    np.save("data/Webb_reg-%s_snap-%s_rank%d.npy" % (reg, snap, rank), rgb_img)

    comm.Barrier()


# Define region and snapshot
reg = "00"
snap = '010_z005p000'

# Get redshift
z = float(snap.split("z")[-1].replace("p", "."))

# Define the initial image size in Mpc
width = 0.03

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
               arc_res, kpc_res, arcsec_per_kpc_proper)

if rank == 0:

    # Initialise the image array
    img = np.zeros((npix, npix, 3), dtype=np.float32)

    files = glob.glob("data/*.npy")

    # Combine rank images together
    for r, f in enumerate(files):
        print("Combinging image from rank %d" % r)
        rank_img = np.load(f)
        img += rank_img

    # Set up figure
    dpi = img.shape[0]
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(img, extent=imgextent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    plt.margins(0, 0)

    fig.savefig('plots/Webb_reg-%s_snap-%s.png'
                % (reg, snap),
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)
