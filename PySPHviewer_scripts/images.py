import matplotlib as ml

ml.use('Agg')
import numpy as np
import sphviewer as sph
import scipy.ndimage as ndimage
from utilities import get_normalised_image
from matplotlib import colors


def getimage(data, poss, masses, hsml, num, cmap, vmin, vmax, res):
    print('There are', poss.shape[0], 'particles in the region')

    # Set up particle objects
    P = sph.Particles(poss, mass=masses, hsml=hsml)

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = res[1]
    i['ysize'] = res[0]
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    print("Image limits:", np.min(img), np.max(img))

    img = ndimage.gaussian_filter(img, sigma=(2.5, 2.5), order=0)

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))

    return rgb, R.get_extent()


def getimage_weighted(data, poss, weight, quant, hsml, num, cmap,
                      vmin, vmax, res, center=None):
    print('There are', poss.shape[0], 'particles in the region')

    # Set up particle objects
    P = sph.Particles(poss, mass=weight, hsml=hsml)
    Pt = sph.Particles(poss, mass=(quant * weight), hsml=hsml)

    # Initialise the scene
    S = sph.Scene(P)
    St = sph.Scene(Pt)

    i = data[num]
    i['xsize'] = res[1]
    i['ysize'] = res[0]
    i['roll'] = 0
    S.update_camera(**i)
    St.update_camera(**i)
    R = sph.Render(S)
    Rt = sph.Render(St)
    # R.set_logscale()
    # Rt.set_logscale()
    imgden = R.get_image()
    imgt = Rt.get_image()
    img = imgt / imgden

    print("Image limits:")
    print("Density:", np.nanmin(imgden), np.nanmax(imgden))
    print("Quantity:", np.nanmin(imgt), np.nanmax(imgt))

    img = ndimage.gaussian_filter(img, sigma=(2.5, 2.5), order=0)

    print("Image:", np.nanmin(img), np.nanmax(img))

    # Convert images to rgb arrays
    if center is None:
        rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))
    else:
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        rgb = cmap(divnorm(img))

    return rgb, R.get_extent()
