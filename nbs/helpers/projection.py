from abc import ABC, abstractmethod
import numpy as np


class AbstractProjector(ABC):
    """
    Abstract class that represent a general projector
    """
    def __init__(self):
        super(AbstractProjector, self).__init__()

    @abstractmethod
    def project_point(self, point, *args, **kwargs):
        pass

    @abstractmethod
    def get_image_size(self, points):
        pass


class LinearProjector(AbstractProjector):
    """
    Class used to linearly project a point to an image. Can be used to obtain BEV images
    """
    def __init__(self, res_width=5, res_height=5):
        """
        Class constructor
        Parameters
        ----------
        res_width: float
            resolution along width of the image expressed in px/meter

        res_height: float
            resolution along height of the image expressed in px/meter
        """
        self.res_width = res_width
        self.res_height = res_height
        super(LinearProjector, self).__init__()

    def project_point(self, point, *args, **kwargs):
        """
        Function that project a 3D point on a pixel of image
        :param point: 2 dimensional point
        :param args:
        :param kwargs:
        :return:
        """
        if len(point.shape) > 1:  # if there are more than a point to project

            # first coordinate
            i = np.floor(point[:, 1] * self.res_height).astype(int)
            # the second coordinate is associated to the column coordinate of the image
            j = np.floor(point[:, 0] * self.res_width).astype(int)
        else:
            # the first coordinate is associated to the row coordinate of the image
            i = np.floor(point[1] * self.res_height).astype(np.int)
            # the second coordinate is associated to the column coordinate of the image
            j = np.floor(point[0] * self.res_width).astype(np.int)

        return i, j

    def get_image_size(self, points):
        delta_x, delta_y, _ = np.abs(points.max(0) - points.min(0))
        height = np.ceil(delta_y * self.res_height).astype(int)
        width = np.ceil(delta_x * self.res_width).astype(int)

        return height, width


class SphericalProjector(AbstractProjector):
    """
    Class that is used to project a point to a spherical image
    """

    def __init__(self, res_az=100, res_planar=300):
        """

        Parameters
        ----------
        res_az: float
            resolution along azimuthal angle

        res_planar: float
            resolution along planar angle
        """

        self.res_az = res_az

        self.res_planar = res_planar

        super(SphericalProjector, self).__init__()

    @staticmethod
    def _get_spherical_coordinates(p):
        """
        Function that apply spherical transformation to a 3D point in coordinates X,Y,Z
        :param p: 3D point in cartesian coordinates (X,Y,Z)
        :return rho, planar, azimuthal: spherical coordinates
        """
        if len(p.shape) == 1:
            # norm of the point
            rho = np.linalg.norm(p)

            # trivial case
            if rho == 0:
                return 0, 0, 0

            # planar angle
            planar = np.arctan2(p[1], p[0]) + np.pi

            # azimuthal angle
            azimuthal = np.arccos(p[2] / rho)
        else:
            # if p is a vector of points instead of a single point
            # norm of each point in the point list
            rho = np.linalg.norm(p, axis=1)
            # vector of planar angles
            planar = np.arctan2(p[:, 1], p[:, 0]) + np.pi

            azimuthal = np.zeros(len(rho))

            # vector of azimuthal angles
            azimuthal[rho != 0] = np.arccos(p[rho != 0, 2] / rho[rho != 0])

            # fix for the trivial case of x,y,z = (0,0,0)
            if (rho == 0).sum() > 0:
                planar[rho == 0] = 0
                azimuthal[rho == 0] = 0

        return rho, planar, azimuthal

    def project_point(self, point, *args, **kwargs):
        """
        Function that project a point from 3D to a pixel of a spherical image
        :param point:
        :param args:
        :param kwargs:
        :return:
        """

        rho, planar, az = self._get_spherical_coordinates(point)

        i = np.floor(az * self.res_az).astype(np.int)
        j = np.floor(planar * self.res_planar).astype(np.int)

        return i, j

    def get_image_size(self, points):
        """
        Method that computes image size given a point cloud
        :param points:
        :return:
        """

        height = np.ceil(np.pi * self.res_az).astype(int)

        width = np.ceil(2 * np.pi * self.res_planar).astype(int)

        return height, width


class Projection:
    def __init__(self,
                 proj_type,
                 res_planar=-1,
                 res_azimuthal=-1,
                 res_x=-1,
                 res_y=-1):
        """
        Constructor of class Projection

        Parameters
        ----------
        proj_type: {'spherical', 'linear'}, optional
            Type of projection used to project the point cloud over the image.
            Spherical: The point cloud is projected over the unitary sphere. Coordinate of each points are expressed
                using spherical coordinate system, i.e. [X,Y,Z] -> [rho, theta, phi]. Azimuthal and planar angles are
                used to identify the pixels where a point is projected.

            Linear: The point cloud is projected over one of the three main principal plane that are XY, XZ, YZ.

        res_planar: float
            Used only for Spherical projections. This value is the resolution (pixel/radiant)
            to use for the planar angle

        res_azimuthal: float
            Used only for Spherical projection. This value is the resolution (pixel/radiant)
            to use for the azimuthal angle


        res_x: float
            Used only for linear projection. This value is the resolution (pixel/meter) to use for the x-axis.

        res_y: float
            Used only for linear projection. This value is the resolution (pixel/meter) to use for the y-axis.

        """
        self.proj_type = proj_type

        if self.proj_type == 'spherical':
            self.res_planar = res_planar
            self.res_azimuthal = res_azimuthal
            self.projector = self._init_spherical_projector(self.res_azimuthal, self.res_planar)

        elif self.proj_type == 'linear':
            self.res_x = res_x
            self.res_y = res_y
            self.projector = self._init_linear_projector(self.res_x, self.res_y)

        else:
            raise ValueError("Projection Type can be only 'spherical' or 'linear'")

    @staticmethod
    def _init_spherical_projector(res_az, res_planar):
        """
        Function that initialize a spherical projector

        Parameters
        ----------
        res_az: float
            Resolution for the azimuthal angle
        res_planar: float
            Resolution for the planar angle

        Returns
        -------
        SphericalProjector: BaseProjector (object)
            Spherical projector
        """
        return SphericalProjector(res_az=res_az, res_planar=res_planar)

    @staticmethod
    def _init_linear_projector(res_x, res_y):
        """
        Function that initialize a linear projector

        Parameters
        ----------
        res_x: float
            resolution for the x-axis
        res_y:
            resolution for the y-axis

        Returns
        -------
        LinearProjection: BaseProjector (object)
            Linear projector
        """
        return LinearProjector(res_width=res_x, res_height=res_y)

    def project_points_values(self, points, values, res_values=1, dtype=np.uint16, f=max):
        """
        General function that generates a projection image given a point cloud and an array of values.
        At each point in the point cloud must correspond a value in the values array.
        Using the class projector we identify the pixel p[n]=(i[n],j[n]) where the points[n] is projected, and we assign
        values[n] as value to the projection image img[i[n],[j[n]].

        In case of collision of multiple points over the same pixel the function f is used to assign the final value.

        Parameters
        ----------
        points: ndarray
            Nx3 array representing the point cloud

        values: ndarray
            Values to project over the image

        res_values: float
            Resolution to use for the value

        dtype: data-type, optional
            Type of resulting image value. Example np.uint16 or np.float.

        f: function
            Function to use for collisions.

        Returns
        -------
        img: ndarray
            Projection image.

        """
        # Project point and get the pixels where the points are projected
        I, J = self.projector.project_point(points)

        # Nx2 array with pixels coordinates
        IJ = np.c_[I, J]

        # getting uniques pixels and inverse array to rebuild IJ
        unique, inverse = np.unique(IJ, return_inverse=True, axis=0)

        # for each unique pixel
        values_dict = {i: [] for i in range(len(unique))}

        # projection values
        proj_values = values * res_values

        # after this loop the dictionary values_dict contains for each unique pixel the list of values projected over
        # the pixel
        for n in range(len(proj_values)):
            values_dict[inverse[n]].append(proj_values[n])

        # height and width of projection image
        height, width = self.projector.get_image_size(points)

        # initialize projection image
        img = np.zeros((height, width), dtype=dtype)

        # in the following loop we assign the final value to each unique pixel
        for k in values_dict:
            img[unique[k][0], unique[k][1]] = f(values_dict[k]).astype(dtype)

        return img

    def back_project(self, points, img, res_value=1, dtype=np.float):
        """
        Function that back project values in a projection image to point cloud

        Parameters
        ----------
        points: ndarray
            points to use for back projection

        img: ndarray
            Projected image

        res_value: float
            Resolution value

        dtype: data-type, optional
            Type for the back projection values

        Returns
        -------
        values: ndarray
            Back projected values
        """

        # shape of projection image
        nr, nc, nz = np.atleast_3d(img).shape

        # initializing array of values
        values = np.zeros((len(points), nz), dtype=dtype)

        # retrieving pixels coordinates
        i, j = self.projector.project_point(points)

        # we assign values at each point
        for n in range(len(points)):
            values[n] = img[i[n], j[n]] / res_value

        if nz == 1:
            return values.flatten()

        return values
