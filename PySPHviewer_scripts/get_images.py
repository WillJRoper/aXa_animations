import numpy as np
import scipy.ndimage as ndimage


def quartic_spline(q):

    k3 = 7 / (478 * np.pi)

    w = np.zeros_like(q)
    okinds1 = q < 1 / 2
    okinds2 = np.logical_and(1 / 2 <= q, q < 3 / 2)
    okinds3 = np.logical_and(3 / 2 <= q, q < 5 / 2)

    w[okinds1] = (5 / 2 - q[okinds1]) ** 4 \
                 - 5 * (3 / 2 - q[okinds1]) ** 4 \
                 + 10 * (1 / 2 - q[okinds1]) ** 4
    w[okinds2] = (5 / 2 - q[okinds2]) ** 4 \
                 - 5 * (3 / 2 - q[okinds2]) ** 4
    w[okinds3] = (5 / 2 - q[okinds3]) ** 4

    return k3 * w


def cubic_spline(q):
    k = 21 / (2 * np.pi)

    w = np.zeros_like(q)
    okinds = q <= 1

    w[okinds] = (1 - q[okinds]) ** 4 * (1 + 4 * q[okinds])

    return k * w


def make_spline_img(part_pos, Ndim, w, h, ls, smooth_low, smooth_high,
                    spline_func=cubic_spline, spline_cut_off=1):

    # Initialise the image array
    smooth_img = np.zeros((Ndim[1], Ndim[0]))

    # Compute pixel width
    w_pix_width = w / Ndim[1]
    h_pix_width = h / Ndim[1]
    n = 0
    for ipos, l, sml_l, sml_h in zip(part_pos, ls, smooth_low, smooth_high):

        i, j = int(ipos[1] / pix_width), int(ipos[0] / pix_width)
        i_low = int(sml_l[1] / pix_width)
        j_low = int(sml_l[0] / pix_width)
        i_high = int(sml_h[1] / pix_width)
        j_high = int(sml_h[0] / pix_width)

        if i < 0 or i > Ndim[1] or j < 0 or j > Ndim[0]:
            continue

        if i_low < 0:
            i_low = 0
        if j_low < 0:
            j_low = 0
        if i_high >= Ndim[1]:
            i_high = Ndim[1] - 1
        if j_high >= Ndim[0]:
            j_high = Ndim[0] - 1

        # NOTE: SMOOTHING LENGTHS ARE WRONG FIX THIS FUTURE WILL

        i_range = np.arange(i_low, i_high + 1, 1)
        j_range = np.arange(j_low, j_high + 1, 1)

        ii, jj = np.meshgrid(j_range - j, i_range - i)

        dists = np.sqrt(ii**2 + jj**2) * pix_width
        sml = np.sqrt((sml_h[0] - ipos[0])**2 + (sml_h[1] - ipos[1])**2)

        # Get the kernel
        w = spline_func(dists / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)
        smooth_img[i_low: i_high + 1, j_low: j_high + 1] += l.value * norm_kernel
        n += 1

    return smooth_img


def make_spline_img_3d(part_pos, Ndim, pad_mpc, ls, smooth, pix_width,
                       spline_func=quartic_spline, spline_cut_off=5/2):

    # Initialise the image array
    smooth_img = np.zeros((Ndim[0], Ndim[1], Ndim[2]))

    max_sml = np.max(smooth)

    kernel_rad = max_sml * 1.5 * spline_cut_off

    ilow = int(-kernel_rad / pix_width)
    ihigh = int(kernel_rad / pix_width)
    jlow = int(-kernel_rad / pix_width)
    jhigh = int(kernel_rad / pix_width)
    klow = -Ndim[2] // 2
    khigh = Ndim[2] // 2

    ipix_range = np.arange(ilow, ihigh + 1, 1, dtype=int)
    jpix_range = np.arange(jlow, jhigh + 1, 1, dtype=int)
    kpix_range = np.arange(klow, khigh, 1, dtype=int)

    ii, jj, kk = np.meshgrid(ipix_range, jpix_range, kpix_range)

    dists = np.sqrt(ii ** 2 + jj ** 2 + kk ** 2) * pix_width

    for ipos, l, sml, in zip(part_pos, ls, smooth):

        # Get the kernel
        w = spline_func(dists / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)

        i_low = int((ipos[0] - kernel_rad) / pix_width)
        j_low = int((ipos[1] - kernel_rad) / pix_width)
        i_high = i_low + norm_kernel.shape[0]
        j_high = j_low + norm_kernel.shape[1]

        smooth_img[i_low: i_high, j_low: j_high, :] += l * norm_kernel

    return np.sum(smooth_img, axis=-1)


def make_spline_img_cart_gas(part_pos, Ndim, w, h, ls, smooth, my_cent,
                            spline_func=quartic_spline, spline_cut_off=5 / 2):

    # Initialise the image array
    smooth_img = np.zeros((Ndim[0], Ndim[1]))

    # Compute pixel width
    pix_width = w / Ndim[1]
    max_sml = np.max(smooth)

    low = int(-(max_sml * 1.5 * spline_cut_off) / pix_width)
    high = int(max_sml * 1.5 * spline_cut_off / pix_width)

    pix_range = np.arange(low, high + 1, 1, dtype=int)

    ii, jj = np.meshgrid(pix_range, pix_range)

    dists = np.sqrt(ii ** 2 + jj ** 2) * pix_width

    for ipos, l, sml, in zip(part_pos, ls, smooth):

        # Get the kernel
        w = spline_func(dists / sml)

        # Place the kernel for this particle within the img
        kernel = w / sml ** 3
        norm_kernel = kernel / np.sum(kernel)

        i, j = int((ipos[1] / pix_width) + Ndim[1] / 2), \
               int((ipos[0] / pix_width) + Ndim[0] / 2)
        i_low = i - (norm_kernel.shape[1] // 2)
        j_low = j - (norm_kernel.shape[0] // 2)
        i_high = i + (norm_kernel.shape[1] // 2)
        j_high = j + (norm_kernel.shape[0] // 2)

        try:
            smooth_img[i_low: i_high + 1, j_low: j_high + 1] += l * norm_kernel
        except ValueError:
            print(my_cent, ipos, ipos + my_cent, i_low, i_high, j_low,
                  j_high, i_high - i_low, j_high - j_low,
                  smooth_img[i_low: i_high + 1, j_low: j_high + 1].shape,
                  norm_kernel.shape)

    # smooth_img = ndimage.gaussian_filter(smooth_img, sigma=(2.5, 2.5), order=0)

    return smooth_img

def make_spline_img_cart_dm(part_pos, Ndim, w, h, ls, smooth, my_cent,
                            spline_func=quartic_spline, spline_cut_off=5 / 2):

    # Initialise the image array
    smooth_img = np.zeros((Ndim[0], Ndim[1]))

    # Compute pixel width
    pix_width = w / Ndim[1]
    sml = smooth[0]

    low = int(-(sml * 1.5 * spline_cut_off) / pix_width)
    high = int(sml * 1.5 * spline_cut_off / pix_width)

    pix_range = np.arange(low, high + 1, 1, dtype=int)

    ii, jj = np.meshgrid(pix_range, pix_range)

    dists = np.sqrt(ii ** 2 + jj ** 2) * pix_width

    # Get the kernel
    w = spline_func(dists / sml)

    # Place the kernel for this particle within the img
    kernel = w / sml ** 3
    norm_kernel = kernel / np.sum(kernel)

    for ipos, l in zip(part_pos, ls):

        i, j = int((ipos[1] / pix_width) + Ndim[1] / 2), \
               int((ipos[0] / pix_width) + Ndim[0] / 2)
        i_low = i - (norm_kernel.shape[1] // 2)
        j_low = j - (norm_kernel.shape[0] // 2)
        i_high = i + (norm_kernel.shape[1] // 2)
        j_high = j + (norm_kernel.shape[0] // 2)

        try:
            smooth_img[i_low: i_high + 1, j_low: j_high + 1] += l * norm_kernel
        except ValueError:
            print(my_cent, ipos, ipos + my_cent, i_low, i_high, j_low,
                  j_high, i_high - i_low, j_high - j_low,
                  smooth_img[i_low: i_high + 1, j_low: j_high + 1].shape,
                  norm_kernel.shape)

    # smooth_img = ndimage.gaussian_filter(smooth_img, sigma=(2.5, 2.5), order=0)

    return smooth_img
