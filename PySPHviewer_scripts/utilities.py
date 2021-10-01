import argparse
import sys
from glob import glob

import matplotlib as ml
import matplotlib.colors as mcolors
import numpy as np

sys.path.append("/cosma7/data/dp004/dc-rope1/SWIFT/"
                "swiftsim_master/csds/src/.libs/")

import libcsds as csds


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


def data_from_logger(args, time, part_type, fields=("Coordinates", )):

    print("basename: %s" % args.files)

    # read the csds
    data = {}
    for f in args.files:
        if f.endswith(".dump"):
            filename = f[:-5]
        else:
            raise Exception(
                "It seems that you are not providing a logfile (.dump)")
        with csds.Reader(filename, verbose=0, number_index=10,
                         restart_init=False, use_cache=True) as reader:

            print(reader.get_list_fields(part_type=[0, 1, 2, 3, 4, 5, 6]))

            # Get boxsize
            data["Boxsize"] = reader.get_box_size()

            # Check the time limits
            t0, t1 = reader.get_time_limits()
            print("Time limits:", t0, t1)
            if t0 > time > t1:
                raise Exception("The time is outside the simulation range")

            # Ensure that the fields are present
            missing = set(fields).difference(
                set(reader.get_list_fields(part_type=part_type)))

            if missing:
                raise Exception(
                    "Fields %s not found in the logfile." % missing)

            # Rewind a bit the time in order to update the particles
            dt = 1e-3 * (time - t0)

            # Read all the particles
            out = reader.get_data(
                fields=fields, time=time - dt,
                filter_by_ids=None, part_type=part_type)

            # Update the particles
            out = reader.update_particles(fields=fields, time=time)

            # Add the data to a dict and return
            for key in fields:
                data.setdefault(key, []).extend(out[key])

    print(f"Extracted (part_type: {part_type})", [key for key in data.keys()])

    return data
