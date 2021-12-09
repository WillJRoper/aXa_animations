import argparse
import sys
from glob import glob
import os
os.environ['FLARE'] = '/Users/willroper/Documents/University/' \
                      'Modules/flare/flare'

import matplotlib as ml
import matplotlib.colors as mcolors
import numpy as np
import astropy.constants as const
import astropy.units as u
from scipy.spatial import cKDTree
import flare
from synthobs.sed import models
import time


sys.path.append("/cosma7/data/dp004/dc-rope1/SWIFT/"
                "swiftsim_master/csds/src/.libs/")

# import libcsds as csds


ml.use('Agg')


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Read a logfile and plots some basic properties')

    default_files = "index_*dump"
    default_files = glob(default_files)

    parser.add_argument('files', metavar='filenames', type=str, nargs="*",
                        help='The filenames of the logfiles')
    args = parser.parse_args()
    if len(args.files) == 0:
        args.files = default_files
    return args


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


# def data_from_logger(args, time, part_type, fields=("Coordinates", )):
#
#     print("basename: %s" % args.files)
#
#     # read the csds
#     data = {}
#     for f in args.files:
#         if f.endswith(".dump"):
#             filename = f[:-5]
#         else:
#             raise Exception(
#                 "It seems that you are not providing a logfile (.dump)")
#         with csds.Reader(filename, verbose=0, number_index=10,
#                          restart_init=False, use_cache=True) as reader:
#
#             print(reader.get_list_fields(part_type=[0, 1, 2, 3, 4, 5, 6]))
#
#             # Get boxsize
#             data["Boxsize"] = reader.get_box_size()
#
#             # Check the time limits
#             t0, t1 = reader.get_time_limits()
#             print("Time limits:", t0, t1)
#             if t0 > time > t1:
#                 raise Exception("The time is outside the simulation range")
#
#             # Ensure that the fields are present
#             missing = set(fields).difference(
#                 set(reader.get_list_fields(part_type=part_type)))
#
#             if missing:
#                 raise Exception(
#                     "Fields %s not found in the logfile." % missing)
#
#             # Rewind a bit the time in order to update the particles
#             dt = 1e-3 * (time - t0)
#
#             # Read all the particles
#             out = reader.get_data(
#                 fields=fields, time=time - dt,
#                 filter_by_ids=None, part_type=part_type)
#
#             # Update the particles
#             out = reader.update_particles(fields=fields, time=time)
#
#             # Add the data to a dict and return
#             for key in fields:
#                 data.setdefault(key, []).extend(out[key])
#
#     print(f"Extracted (part_type: {part_type})", [key for key in data.keys()])
#
#     return data


def gas_sed(lams, Ts, pix_res, dists, trans):

    c = const.c
    h = const.h
    kb = const.k_B

    pix_area = (pix_res**2).to(u.sr)

    lams = lams
    Ts = Ts[:, None]

    term1 = 2 * h * c ** 2 / lams ** 5
    exponent = h * c / (kb * Ts * lams)
    term2 = 1 / (np.exp(exponent.value) - 1)

    conv = lams ** 2 / c * u.sr**-1

    SED = (term1 * term2 * conv).to(u.nJy * u.sr**-1)

    lum = (np.trapz(np.multiply(SED, trans), x=lams, axis=1)
           / np.trapz(trans, x=lams)
           * pix_area).to(u.nJy)

    return SED, lum


def read_array(hdf, group_keys, subsample=(None, None)):
    grp = hdf[group_keys]

    print(group_keys, grp.attrs['Expression for physical CGS units'],
          grp.attrs['Conversion factor to physical '
                    'CGS (including cosmological corrections)'])

    if subsample[0] is None and subsample[1] is None:
        return grp[...] * grp.attrs['Conversion factor to physical '
                                    'CGS (including cosmological corrections)']
    elif not subsample[0] is None and subsample[1] is None:
        return grp[subsample[0]:] * grp.attrs['Conversion factor to physical '
                                              'CGS (including cosmological ' \
                                              'corrections)']
    elif subsample[0] is None and not subsample[1] is None:
        return grp[:subsample[1]] * grp.attrs['Conversion factor to physical '
                                              'CGS (including cosmological ' \
                                              'corrections)']
    else:
        return grp[subsample[0]: subsample[1]] * grp.attrs['Conversion ' \
                                                           'factor to ' \
                                                           'physical CGS ' \
                                                           '(including ' \
                                                           'cosmological ' \
                                                           'corrections)']


def lum(model, data, kappa, BC_fac, F, filters, IMF='Chabrier_300',
        Type='Total', log10t_BC=7., extinction='default'):
    kinp = np.load('/Users/willroper/Documents/University/Animations/'
                   'aXa_animations/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    (Masses, Metallicities, Ages, gasMetallicities,
     gasSML, gasMasses, starCoords, gasCoords) = data

    Lums = {f: np.zeros(len(Masses), dtype=np.float64) for f in filters}

    # --- create rest-frame luminosities
    model.create_Lnu_grid(F)  # --- create new L grid for each filter. In units of erg/s/Hz
    print("Made luminosity grid")
    start = time.time()
    MetSurfaceDensities = get_Z_LOS(starCoords, gasCoords,
                                    gasMasses, gasMetallicities,
                                    gasSML, (0, 1, 2),
                                    lkernel, kbins)
    print("Timer:", time.time() - start)

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
        Lnu = models.generate_Lnu_array(model, Masses=Masses, Ages=Ages,
                                        Metallicities=Metallicities,
                                        tauVs_ISM=tauVs_ISM, tauVs_BC=tauVs_BC,
                                        F=F, f=f, fesc=fesc, log10t_BC=log10t_BC)
        Lums[f] = Lnu

    return Lums

def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, dimens, lkernel, kbins):
    """

    Compute the los metal surface density (in g/cm^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """

    conv = (u.solMass / u.Mpc ** 2).to(u.solMass / u.pc ** 2)

    n = s_cood.shape[0]
    Z_los_SD = np.zeros(n)

    # Fixing the observer direction as z-axis. Use make_faceon() for changing the
    # particle orientation to face-on
    xdir, ydir, zdir = dimens

    tree = cKDTree(g_cood[:, (xdir, ydir)])

    max_sml = np.max(g_sml)

    for ii in range(n):
        thisspos = s_cood[ii, :]

        inds = tree.query_ball_point(thisspos[0:2], r=max_sml)
        thisgpos = g_cood[inds, :]
        thisgsml = g_sml[inds]
        thisgZ = g_Z[inds]
        thisgmass = g_mass[inds]

        ok = thisgpos[:, zdir] > thisspos[zdir]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgZ = thisgZ[ok]
        thisgmass = thisgmass[ok]
        x = thisgpos[:, xdir] - thisspos[xdir]
        y = thisgpos[:, ydir] - thisspos[ydir]

        b = np.sqrt(x * x + y * y)
        boverh = b / thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok] * thisgZ[ok] / (
                thisgsml[ok] * thisgsml[
            ok])) * kernel_vals) * conv  # in units of Msun/pc^2

    return Z_los_SD


def get_star_formation_time(SFT, redshift, cosmo):

    SFz = (1/SFT) - 1.
    SFa = cosmo.age(redshift).value - cosmo.age(SFz).value
    return SFa
