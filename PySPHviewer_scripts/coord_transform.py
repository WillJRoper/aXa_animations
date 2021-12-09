import numpy as np


def cart_project(pos, cam_pos, yaw, pitch, roll, width, height, fov=(45, 45)):
    """ Projects cartesian coordinates into spherical and on to the imaging
    plane defined by camera coordinates and orientation. (courtesy of:
    http://code.aldream.net/article/2013-04-13-painter-s-algorithm/)

    """

    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # rot1 = np.array([[1, 0, 0],
    #                  [0, np.cos(yaw), np.sin(yaw)],
    #                  [0, -np.sin(yaw), np.cos(yaw)]])
    # rot2 = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
    #                  [0, 1, 0],
    #                  [np.sin(pitch), 0, np.cos(pitch)]])
    # rot3 = np.array([[np.cos(roll), np.sin(roll), 0],
    #                  [-np.sin(roll), np.cos(roll), 0],
    #                  [0, 0, 1]])

    # Extract the particle positions
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    cx, cy, cz = cam_pos[0], cam_pos[1], cam_pos[2]
    ax, ay, az = x - cx, y - cy, z - cz

    # Calculate the projection elements
    dx = np.cos(pitch) * (np.sin(roll) * ay
                          + np.cos(roll) * ax) - np.sin(pitch) * az
    dy = np.sin(yaw) * (np.cos(pitch) * az + np.sin(pitch)
                        * (np.sin(roll) * ay + np.cos(roll) * ax)) \
         + np.cos(yaw) * (np.cos(roll) * ay - np.sin(roll) * ax)
    dz = np.cos(yaw) * (np.cos(pitch) * az + np.sin(pitch)
                        * (np.sin(roll) * ay + np.cos(roll) * ax)) \
         - np.sin(yaw) * (np.cos(roll) * ay - np.sin(roll) * ax)

    # Combine dx, dy and dz into single coordinate array
    spherical_pos = np.zeros_like(pos)
    spherical_pos[:, 0] = dx
    spherical_pos[:, 1] = dy
    spherical_pos[:, 2] = dz

    # Convert to a projection onto the "screen"
    screen_x = (dx / dz) * (width / 2) / np.tan(np.radians(fov[0])) + (width / 2)
    screen_y = (dy / dz) * (height / 2) / np.tan(np.radians(fov[1])) + (height / 2)

    # Combine screen coordinates into a screen coordinate array
    screen_pos = np.zeros((pos.shape[0], 2))
    screen_pos[:, 0] = screen_x
    screen_pos[:, 1] = screen_y

    return spherical_pos, screen_pos


def smooth_project(pos, smls, cam_pos, yaw, pitch, roll, width, height, fov=(45, 45)):
    """ Projects cartesian coordinates into spherical and on to the imaging
    plane defined by camera coordinates and orientation. (courtesy of:
    http://code.aldream.net/article/2013-04-13-painter-s-algorithm/)

    """

    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    smls_3d = np.zeros_like(pos)
    for i in range(3):
        smls_3d[:, i] = smls

    pos_high = pos + smls_3d
    pos_low = pos - smls_3d

    # Extract the particle positions
    xl, yl, zl = pos_low[:, 0], pos_low[:, 1], pos_low[:, 2]
    xh, yh, zh = pos_high[:, 0], pos_high[:, 1], pos_high[:, 2]
    cx, cy, cz = cam_pos[0], cam_pos[1], cam_pos[2]
    axl, ayl, azl = xl - cx, yl - cy, zl - cz
    axh, ayh, azh = xh - cx, yh - cy, zh - cz

    # Calculate the projection elements
    dxl = np.cos(pitch) * (np.sin(roll) * ayl
                          + np.cos(roll) * axl) - np.sin(pitch) * azl
    dyl = np.sin(yaw) * (np.cos(pitch) * azl + np.sin(pitch)
                        * (np.sin(roll) * ayl + np.cos(roll) * axl)) \
         + np.cos(yaw) * (np.cos(roll) * ayl - np.sin(roll) * axl)
    dzl = np.cos(yaw) * (np.cos(pitch) * azl + np.sin(pitch)
                        * (np.sin(roll) * ayl + np.cos(roll) * axl)) \
         - np.sin(yaw) * (np.cos(roll) * ayl - np.sin(roll) * axl)
    
    dxh = np.cos(pitch) * (np.sin(roll) * ayh
                          + np.cos(roll) * axh) - np.sin(pitch) * azh
    dyh = np.sin(yaw) * (np.cos(pitch) * azh + np.sin(pitch)
                        * (np.sin(roll) * ayh + np.cos(roll) * axh)) \
         + np.cos(yaw) * (np.cos(roll) * ayh - np.sin(roll) * axh)
    dzh = np.cos(yaw) * (np.cos(pitch) * azh + np.sin(pitch)
                        * (np.sin(roll) * ayh + np.cos(roll) * axh)) \
         - np.sin(yaw) * (np.cos(roll) * ayh - np.sin(roll) * axh)

    # Combine dx, dy and dz into single coordinate array
    spherical_pos_low = np.zeros_like(pos)
    spherical_pos_low[:, 0] = dxl
    spherical_pos_low[:, 1] = dyl
    spherical_pos_low[:, 2] = dzl
    spherical_pos_high = np.zeros_like(pos)
    spherical_pos_high[:, 0] = dxh
    spherical_pos_high[:, 1] = dyh
    spherical_pos_high[:, 2] = dzh

    # Convert to a projection onto the "screen"
    screen_x_low = (dxl / dzl) * (width / 2) / np.tan(np.radians(fov[0])) + (
                width / 2)
    screen_y_low = (dyl / dzl) * (height / 2) / np.tan(np.radians(fov[1])) + (
                height / 2)
    screen_x_high = (dxh / dzh) * (width / 2) / np.tan(np.radians(fov[0])) + (
                width / 2)
    screen_y_high = (dyh / dzh) * (height / 2) / np.tan(np.radians(fov[1])) + (
                height / 2)

    # Combine screen coordinates into a screen coordinate array
    screen_pos_low = np.zeros((pos.shape[0], 2))
    screen_pos_low[:, 0] = screen_x_low
    screen_pos_low[:, 1] = screen_y_low
    screen_pos_high = np.zeros((pos.shape[0], 2))
    screen_pos_high[:, 0] = screen_x_high
    screen_pos_high[:, 1] = screen_y_high

    return spherical_pos_low, screen_pos_low, spherical_pos_high, screen_pos_high


