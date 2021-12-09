import os

os.environ['FLARE'] = '/Users/willroper/Documents/University/' \
                      'Modules/flare/flare'

import astropy.constants as const
import astropy.units as u
import coord_transform as trans
import get_images as gimg
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utilities as util
from astropy.cosmology import Planck18 as cosmo
from flare import filters as fs
from matplotlib.colors import Normalize
from synthobs.sed import models
from interrogator.sed import dust_curves
import mpi4py
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process

trans_curve_path = "/Users/willroper/Documents/University/Animations/" \
                   "aXa_animations/human_eye_trans_lowres.csv"

df = pd.read_csv(trans_curve_path)

# res = (3840, 2160)
res = (3840, 3840)
fov = (45, 45)

path = "/Users/willroper/Documents/University/Animations/" \
       "aXa_animations/ani_hydro_1379.hdf5"

hdf = h5py.File(path, "r")

U_M = hdf["Units"].attrs['Unit mass in cgs (U_M)']
U_L = hdf["Units"].attrs['Unit length in cgs (U_L)'] * u.cm
U_t = hdf["Units"].attrs['Unit time in cgs (U_t)']
U_T = hdf["Units"].attrs['Unit temperature in cgs (U_T)']

redshift = hdf["Header"].attrs["Redshift"]
npart = hdf["Header"].attrs["NumPart_Total"]
boxsize = hdf["Header"].attrs["BoxSize"] / (1 + redshift) * U_L

gas_rank_bins = np.linspace(0, npart[0], size + 1, dtype=int)
star_rank_bins = np.linspace(0, npart[4], size + 1, dtype=int)
print(str(rank) + ":", "This gas bin", gas_rank_bins[rank],
      gas_rank_bins[rank + 1])
print(str(rank) + ":", "This star bin", star_rank_bins[rank],
      star_rank_bins[rank + 1])
gas_subsample = (gas_rank_bins[rank], gas_rank_bins[rank + 1])
gas_subsample = (None, None)
star_subsample = (star_rank_bins[rank], star_rank_bins[rank + 1])

gpos = util.read_array(hdf, "PartType0/Coordinates",
                       subsample=gas_subsample) * u.cm
gsmls = util.read_array(hdf, "PartType0/SmoothingLengths",
                        subsample=gas_subsample) * u.cm
gvel = util.read_array(hdf, "PartType0/Velocities",
                       subsample=gas_subsample) * u.cm / u.s
gtemp = util.read_array(hdf, "PartType0/Temperatures",
                        subsample=gas_subsample) * u.K
G_Z = util.read_array(hdf, "PartType0/SmoothedMetalMassFractions",
                      subsample=gas_subsample)
G_mass = util.read_array(hdf, "PartType0/Masses",
                         subsample=gas_subsample) * u.g

star_pos = util.read_array(hdf, "PartType4/Coordinates",
                           subsample=star_subsample) * u.cm
star_smls = util.read_array(hdf, "PartType4/SmoothingLengths",
                            subsample=star_subsample) * u.cm
star_vel = util.read_array(hdf, "PartType4/Velocities",
                           subsample=star_subsample) * u.cm / u.s
S_mass_ini = util.read_array(hdf, "PartType4/InitialMasses",
                             subsample=star_subsample) * u.g
S_Z = util.read_array(hdf, "PartType4/SmoothedMetalMassFractions",
                      subsample=star_subsample)
S_mass = util.read_array(hdf, "PartType4/Masses",
                         subsample=star_subsample) * u.g
S_as = util.read_array(hdf, "PartType4/BirthScaleFactors",
                       subsample=star_subsample)
hdf.close()

S_age = util.get_star_formation_time(S_as, redshift, cosmo) * 10**3

# Remove anomalous values
okinds = S_age > 0
star_pos = star_pos[okinds]
star_smls = star_smls[okinds]
star_vel = star_vel[okinds]
S_mass_ini = S_mass_ini[okinds]
S_Z = S_Z[okinds]
S_mass = S_mass[okinds]
S_age = S_age[okinds]

cam_pos = boxsize / 2
cam_pos[2] = -boxsize[2]
w = boxsize[0]
h = boxsize[1]
print("Camera at:", cam_pos)
print("(Width, Height):", (w, h))

# =========================== Gas ===========================

# # Calculate the spherical representation
# spher_pos, proj_pos = trans.cart_project(gpos, cam_pos,
#                                          yaw=0, pitch=0, roll=0,
#                                          width=w, height=h,
#                                          fov=fov)
#
# # Calculate angular smoothing lengths
# gas_proj_poss = trans.smooth_project(gpos, gsmls, cam_pos,
#                                      yaw=0, pitch=0, roll=0,
#                                      width=w, height=h, fov=fov)
# sml_sphere_low, sml_low, sml_sphere_high, sml_high = gas_proj_poss
#
# ang_smls_3d = sml_sphere_high - spher_pos
# ang_smls = np.sqrt(ang_smls_3d[:, 0] ** 2 + ang_smls_3d[:, 1] ** 2)

gdists = np.linalg.norm(gpos - cam_pos, axis=1)
g_rec_vel = (cosmo.H(redshift).to(u.km * u.s ** -1 * u.cm ** -1) * (
        gpos - cam_pos)).to(u.cm * u.s ** -1) + gvel
g_part_redshifts = np.linalg.norm(g_rec_vel, axis=1) / const.c.to(
    u.cm * u.s ** -1)

# ========================== Stars ==========================

# # Calculate the spherical representation
# star_spher_pos, star_proj_pos = trans.cart_project(star_pos, cam_pos,
#                                                    yaw=0, pitch=0, roll=0,
#                                                    width=w, height=h,
#                                                    fov=fov)
#
# # Calculate angular smoothing lengths
# star_proj_poss = trans.smooth_project(star_pos, star_smls, cam_pos,
#                                       yaw=0, pitch=0, roll=0,
#                                       width=w, height=h, fov=fov)
# (star_sml_sphere_low, star_sml_low,
#  star_sml_sphere_high, star_sml_high) = star_proj_poss

star_dists = np.linalg.norm(star_pos - cam_pos, axis=1)
star_rec_vel = (cosmo.H(redshift).to(u.km * u.s ** -1 * u.cm ** -1) * (
        star_pos - cam_pos)).to(u.cm * u.s ** -1) + star_vel
star_part_redshifts = np.linalg.norm(star_rec_vel, axis=1) / const.c.to(
    u.cm * u.s ** -1)

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# H, bin_edges = np.histogram(part_redshifts, bins=100)
#
# ax.plot(bin_edges[1:], H)
#
# ax.set_xlabel("$z$")
# ax.set_ylabel("$N$")
#
# ax.set_yscale("log")
# ax.set_xscale("log")
#
# fig.savefig("particle_redshifts.png", bbox_inches="tight")
#
# plt.close(fig)

pix_res = fov[0] / res[0] * u.degree

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# for i in range(10):
#     ax.plot(lams, gas_sed[i, :])
#
# ax.set_xlabel("$\lambda$ / [nm]")
# ax.set_ylabel("$SED$")
#
# ax.set_yscale("log")
# ax.set_xscale("log")
#
# fig.savefig("gas_SED.png", bbox_inches="tight")
#
# plt.close(fig)

lams = df["Lambda"].to_numpy() * u.nm
transmissions = {col: df[col].to_numpy() for col in ["R", "G", "B"]}
img = np.zeros((res[1], res[0], 3))
img_noproj = np.zeros((res[1], res[0], 3))
print(star_pos.to(u.Mpc).value)
print(gpos.to(u.Mpc).value)

data = (S_mass_ini.to(u.solMass).value, S_Z, S_age, G_Z,
        gsmls.to(u.Mpc).value, G_mass.to(u.solMass).value,
        star_pos.to(u.Mpc).value, gpos.to(u.Mpc).value)

gsmls_3d = np.zeros((gsmls.size, 3))
gsmls_3d[:, 0] = gsmls
gsmls_3d[:, 1] = gsmls
gsmls_3d[:, 2] = gsmls
star_smls_3d = np.zeros((star_smls.size, 3))
star_smls_3d[:, 0] = star_smls
star_smls_3d[:, 1] = star_smls
star_smls_3d[:, 2] = star_smls
IMF = "Chabrier_300"
model = models.define_model(
    F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
model.dust_ISM = ('simple', {'slope': -1})  # Define dust curve for ISM
model.dust_BC = (
    'simple', {'slope': -1})  # Define dust curve for birth cloud component

f = "Human.Eye." + "R"
F = {f: fs.filter(f, new_lam=model.lam, lams=lams.to(u.angstrom).value,
                  trans=transmissions["R"])}
dust_model, dust_model_params = model.dust_ISM
print(getattr(dust_curves, dust_model)(params=dust_model_params).tau(
    F[f].pivwv()))

for i, col in enumerate(["R", "G", "B"]):
    transmissions[col][np.isnan(transmissions[col])] = 0
    f = "Human.Eye." + col
    print(f)
    F = {f: fs.filter(f, new_lam=model.lam, lams=lams.to(u.angstrom).value,
                      trans=transmissions[col])}
    F["filters"] = (f,)

    star_ls = (util.lum(model, data, kappa=0.0795, BC_fac=1, F=F, filters=(f,),
                        IMF='Chabrier_300', Type='Total',
                        log10t_BC=7., extinction='default')[f]
               * u.erg * u.s ** -1 * u.Hz ** -1
               / (4 * np.pi * star_dists ** 2)).to(u.nJy)

    # gas_sed, gas_lums = util.gas_sed(lams, gtemp, pix_res,
    #                                  gdists, transmissions[col])
    # print(col, "Gas B_v:", gas_lums.shape)
    # print("Without redshift (min, max):", (np.min(gas_lums), np.max(gas_lums)))
    # print("With redshift (min, max):",
    #       (np.min(gas_lums * (1 + g_part_redshifts)),
    #        np.max(gas_lums * (1 + g_part_redshifts))))
    print("Stellar Without redshift (min, max):",
          (np.min(star_ls), np.max(star_ls)))
    print("Stellar With redshift (min, max):",
          (np.min(star_ls * (1 + star_part_redshifts)),
           np.max(star_ls * (1 + star_part_redshifts))))

    # Calculate redshifted fluxes
    # ls = gas_lums.value * (1 + g_part_redshifts)
    star_ls *= (1 + star_part_redshifts)

    # img_noproj[:, :, i] = gimg.make_spline_img_cart(gpos.value, Ndim=res,
    #                                                 w=w.value, h=h.value,
    #                                                 ls=ls,
    #                                                 smooth=gsmls.value)

    img_noproj[:, :, i] += gimg.make_spline_img_cart(star_pos.value, Ndim=res,
                                                     w=w.value, h=h.value,
                                                     ls=star_ls,
                                                     smooth=star_smls.value)

    # img[:, :, i] = gimg.make_spline_img(proj_pos, Ndim=res,
    #                                     w=w.value, h=h.value, ls=ls,
    #                                     smooth_low=sml_low,
    #                                     smooth_high=sml_high)
    # img[:, :, i] += gimg.make_spline_img(star_proj_pos, Ndim=res,
    #                                     w=w.value, h=h.value, ls=star_ls,
    #                                     smooth_low=star_sml_low,
    #                                     smooth_high=star_sml_high)

    # img_noproj[:, :, i], _, _ = np.histogram2d(gpos[:, 0].value,
    #                                            gpos[:, 1].value,
    #                                            bins=(res[1], res[0]),
    #                                            weights=ls)
    #
    # img_noproj_temp, _, _ = np.histogram2d(star_pos[:, 0].value,
    #                                            star_pos[:, 1].value,
    #                                            bins=(res[1], res[0]),
    #                                            weights=star_ls)
    # img_noproj[:, :, i] += img_noproj_temp.value

# img = np.sum(comm.gather(img, root=0), axis=0)
collect = comm.gather(img_noproj, root=0)
final_img_noproj = np.zeros((res[1], res[0], 3))
if rank == 0:
    for i in collect:

        final_img_noproj += i
    print(rank, final_img_noproj.shape)

    okinds = final_img_noproj > 0
    norm = Normalize(vmin=np.percentile(final_img_noproj[okinds], 16),
                     vmax=np.percentile(final_img_noproj[okinds], 99),
                     clip=True)

    fig = plt.figure(figsize=(1, 1), dpi=final_img_noproj.shape[1])
    ax = fig.add_subplot(111)

    im = ax.imshow(norm(final_img_noproj), extent=(0, w.to(u.Mpc).value, 0,
                                                   h.to(u.Mpc).value))

    ax.axis("off")

    fig.savefig("test_spread_cart.png", bbox_inches="tight")

    plt.close(fig)

    # norm = Normalize(vmin=np.percentile(img, 16),
    #                  vmax=np.percentile(img, 99), clip=True)
    #
    # fig = plt.figure(figsize=(1, 1), dpi=img.shape[1])
    # ax = fig.add_subplot(111)
    #
    # im = ax.imshow(norm(img), extent=[-fov[0], fov[0], -fov[1], fov[1]])
    #
    # ax.axis("off")
    #
    # fig.savefig("test.png")
