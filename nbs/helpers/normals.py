import numpy as np


# utils functions returning cos and sinus for planar and azimuthal angle
def _cos_az_grid(shape, az):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = cos(i/res_az)


    Parameters
    ----------
    shape: tuple
        image dimension

    res_az: float
        resolution of the azimuthal angle

    Returns
    -------
    cos_grid: ndarray
        Grid containing cos of the azimuthal angles
    """
    cos_az = np.repeat(az, shape[1]).reshape(shape)
    return np.cos(cos_az)


def _sin_az_grid(shape, az):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = sin(i/res_az)


    Parameters
    ----------
    shape: tuple
        image dimension

    res_az: float
        resolution of the azimuthal angle

    Returns
    -------
    sin_grid: ndarray
        Grid containing sin of the azimuthal angles
    """
    sin_az = np.repeat(az, shape[1]).reshape(shape)
    return np.sin(sin_az)


def _sin_pl_grid(shape, res_planar):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = sin(j/res_planar + np.pi)

    Parameters
    ----------
    shape: tuple
        image dimension

    res_planar: float
        resolution of the planar angle

    Returns
    -------
    sin_grid: ndarray
        Grid containing cosinus of the planar angles
    """

    sin_pl = np.tile(np.arange(shape[1]), (shape[0], 1)) / res_planar + (np.pi / 2)
    return np.sin(sin_pl)


def _cos_pl_grid(shape, res_planar):
    """
    Given an input shape n,m this function builds a grid of n,m where the value of grid[i,j] = cos(j/res_planar + np.pi)

    Parameters
    ----------
    shape: tuple
        image dimension

    res_planar: float
        resolution of the planar angle

    Returns
    -------

    cos_grid: ndarray
        Grid containing cos of the planar angles
    """
    cos_pl = np.tile(np.arange(shape[1]), (shape[0], 1)) / res_planar + (np.pi / 2)
    return np.cos(cos_pl)


def shift(key, array, axis=0):
    """
    Utils function that shift elements of a vector

    Parameters
    ----------

    key: int
        number of col/rows to shift

    array: ndarray
        Array/matrix to shift

    axis: {0,1} optional
        0: to shift axis 0 of the array
        1: to shift axis 1 of the array

    Returns
    -------
    shifted_array: ndarray
        Array shifted of key columns/rows
    """

    shape = array.shape
    shifted_array = np.zeros(shape, dtype=array.dtype)
    print(shifted_array.shape)

    if axis == 0:
        if key >= 0:
            shifted_array[key:] = array[:shape[0] - key]
        else:
            shifted_array[:shape[0] + key] = array[-key:]
    elif axis == 1:
        if key >= 0:
            shifted_array[:, key:] = array[:, :shape[1] - key]
        else:
            shifted_array[:, :shape[1] + key] = array[:, -key:]
    else:
        raise ValueError('Method only implemented for shift in the 0 or 1 axis')

    return shifted_array


def shift_on_T2(key, array, axis=0):
    """
    Utils function that shift on a 2 dimensional thorus, i.e. we identify as equals the top and bottom frontier
    of image and the left and right frontier of image.

    Parameters
    ----------
    key: int
        number of col/rows to shift

    array: ndarray
        Array/matrix to shift

    axis: {0,1} optional
        0: to shift axis 0 of the array
        1: to shift axis 1 of the array

    Returns
    -------
    shifted_array: ndarray
        Array shifted of key columns/rows

    """
    shape = array.shape
    if axis == 0:
        # remember positive values of key shift array toward above and negative toward below
        return np.concatenate([array[key % shape[axis]:], array[:key % shape[axis]]], axis=axis)

    if axis == 1:
        # remember positive values of key shift array toward left and negative toward right
        return np.concatenate([array[:, key % shape[axis]:], array[:, :key % shape[axis]]], axis=axis)
    else:
        raise ValueError('Method only implemented for shift in the 0 or 1 axis')


def _azimuthal_rho_derivative(img, res_rho):
    """
    Function that compute the derivative of the rho coordinate along azimuthal axis
    Parameters
    ----------
    img: ndarray
        Input image

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    az_rho_derivative:  ndarray
        image of the derivative of rho coordinate along azimuthal axis

    """
    above = shift_on_T2(-1, img)
    below = shift_on_T2(1, img)
    return (below.astype(np.float64) - above.astype(np.float64)) / (2 * res_rho)


def _planar_rho_derivative(img, res_rho):
    """
    Function that compute the derivative of the rho coordinate along azimuthal axis

    Parameters
    ----------
    img: ndarray
        Input image

    res_rho: float
        Resolution of the rho coordinate

    Returns
    -------
    az_rho_derivative:  ndarray
        image of the derivative of rho coordinate along azimuthal axis

    """
    right = shift_on_T2(-1, img, axis=1)
    left = shift_on_T2(1, img, axis=1)
    return (left.astype(np.float64) - right.astype(np.float64)) / (2 * res_rho)


def azimuthal_derivative(img, az, res_rho, res_planar):
    """

    Parameters
    ----------
    img: ndarray
        input image

    res_rho: float
        Resolution of the rho coordinate

    az: float
        az_centers

    res_planar: float
        Resolution of the planar coordinate

    Returns
    -------
    _az_derivative_: ndarray
        azimuthal derivative of the input img
    """
    # getting az_rho_derivative
    diff_rho = _azimuthal_rho_derivative(img, res_rho)

    nr, nc = img.shape[:2]
    # auxiliary cos and sin grid
    cos_az = _cos_az_grid((nr, nc), az)
    sin_az = _sin_az_grid((nr, nc), az)
    cos_pl = _cos_pl_grid((nr, nc), res_planar)
    sin_pl = _sin_pl_grid((nr, nc), res_planar)

    rho = img.copy()
    rho = rho.astype(np.float64) / res_rho

    # initialize img to return
    _az_derivative_ = np.zeros((nr, nc, 3))

    # derivative along x
    _az_derivative_[:, :, 0] = diff_rho * sin_az * cos_pl + (rho / len(az)) * cos_az * cos_pl
    # derivative along y
    _az_derivative_[:, :, 1] = diff_rho * sin_az * sin_pl + (rho / len(az)) * cos_az * sin_pl
    # derivative along z
    _az_derivative_[:, :, 2] = diff_rho * cos_az - (rho / len(az)) * sin_az

    return _az_derivative_


def planar_derivative(img, az, res_rho, res_planar):
    """

    Parameters
    ----------
    img: ndarray
        input image

    res_rho: float
        Resolution of the rho coordinate

    res_az: float
        Resolution of the azimuthal angle

    res_planar: float
        Resolution of the planar angle

    Returns
    -------
    _pl_derivative_: ndarray
        derivative of the input image along the planar coordinate
    """
    # getting az_rho_derivative
    diff_rho = _planar_rho_derivative(img, res_rho)

    nr, nc = img.shape[:2]
    # auxiliary cos and sin grid
    cos_az = _cos_az_grid((nr, nc), az)
    sin_az = _sin_az_grid((nr, nc), az)
    cos_pl = _cos_pl_grid((nr, nc), res_planar)
    sin_pl = _sin_pl_grid((nr, nc), res_planar)

    rho = img.copy()
    rho = rho.astype(np.float64) / res_rho

    # initialize img to return
    _pl_derivative_ = np.zeros((nr, nc, 3))

    # derivative along x
    _pl_derivative_[:, :, 0] = diff_rho * sin_az * cos_pl - (rho / res_planar) * sin_az * sin_pl
    # derivative along y
    _pl_derivative_[:, :, 1] = diff_rho * sin_az * sin_pl + (rho / res_planar) * sin_az * cos_pl
    # derivative along z
    _pl_derivative_[:, :, 2] = diff_rho * cos_az

    return _pl_derivative_


def estimate_normals_from_spherical_img(img, az, res_rho, res_planar):
    """
    Function that estimates point cloud normals from a spherical img.
    It returns a nr x nc x 3 image such that for each pixel we have the estimated normal

    Parameters
    ----------
    img: ndarray
        input image

    az: float
        azimuthal angles

    res_rho: float
        Resolution of coordinate rho

    res_planar: float
        resolution of coordinate planar


    Returns
    -------
    vr: ndarray
        matrix that at position i,j contains the normal inferred for pixel i,j
    """

    az_deriv = azimuthal_derivative(img, res_rho=res_rho, az=az, res_planar=res_planar)
    pl_deriv = planar_derivative(img, res_rho=res_rho, az=az, res_planar=res_planar)

    vr = np.cross(az_deriv, pl_deriv)
    vr_lenght = np.linalg.norm(vr, axis=2)
    idx = vr_lenght > 0

    for i in range(3):
        vr[idx, i] = np.divide(vr[idx, i], vr_lenght[idx])

    # we remove normals where we do not have information
    vr[img == 0] = [0, 0, 0]

    return vr
