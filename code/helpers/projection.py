from abc import ABC, abstractmethod
import numpy as np


class AbstractProjector(ABC):
    """
    Abstract class that represent a general projector
    """
    def __init__(self):
        super(AbstractProjector, self).__init__()

    @abstractmethod
    def project_point(self, points):
        pass

    @abstractmethod
    def get_image_size(self):
        pass


class BEVProjector(AbstractProjector):
    """
    Class used project a point cloud on a BEV image
    """
    def __init__(self, height=400, width=200, res=0.1):
        """
        Class constructor

        Parameters
        ----------
        height: int
            img height

        width: int
            img width

        res: float
            spatial resolution
        """
        self.proj_H = height
        self.proj_W = width
        self.res = res
        self.side_range = (-10, 10)
        self.fwd_range = (6, 46)

        super(BEVProjector, self).__init__()

    def set_ranges(self, side_range, fwd_range):
        self.side_range = side_range
        self.fwd_range = fwd_range

    def project_point(self, points, x0=None, y0=None, delta_x=None, delta_y=None):
        """
        Function that project a 3D point on a pixel of image

        Parameters
        ----------
        points: ndarray
            point to project

        x0: float

        y0: float

        delta_x: float

        delta_y: float

        Returns
        -------


        """

        if len(points.shape) < 2:
            points = np.atleast_2d(points)

        # we suppose that we are working with kitti velodyne, thus the reference system of the point cloud is
        #
        #                x
        #                ^
        #                |
        #                |
        #         y <----0 Car
        #
        x = points[:, 0]
        y = points[:, 1]

        # MAPPING
        # Mappings from one point to grid
        # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
        # j_img_mapping = (-y / self.res).astype(np.int32)  # x axis is -y in LIDAR
        # i_img_mapping = (x / self.res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later
        #
        # # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # # floor used to prevent issues with -ve vals rounding upwards
        # j_img_mapping -= int(np.floor(self.side_range[0] / self.res))
        # i_img_mapping -= int(np.floor(self.fwd_range[0] / self.res))

        # Linerize the mappings to 1D
        # lidx = ((-i_img_mapping) % self.proj_H) * self.proj_W + j_img_mapping
        i_img_mapping = (self.fwd_range[1] - x)
        j_img_mapping = (self.side_range[1] - y)

        i_img_mapping = np.floor(i_img_mapping / self.res).astype(int)
        j_img_mapping = np.floor(j_img_mapping / self.res).astype(int)

        lidx = ((i_img_mapping) % self.proj_H) * self.proj_W + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self):
        return self.proj_H, self.proj_W


class SphericalProjector(AbstractProjector):
    """
    Class that is used to project a point to a spherical image
    """

    def __init__(self, height=64, width=1024, fov_up=85.0, fov_down=115.0):
        """
        Parameters
        ----------
        height: int
            img height

        width: int
            img width

        fov_up: float
            up field of view

        fov_down: float
            down field of view
        """
        self.proj_H = height
        self.proj_W = width
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

        super(AbstractProjector, self).__init__()

    def project_point(self, points):
        """
        Function used to project points to the spherical image

        Parameters
        ----------
        points: ndarray
            input point cloud

        Returns
        -------
        lidx: ndarray
            pixel id where each point is projected

        i_img_mapping: ndarray
            pixel row coordinate for each point projected

        j_img_mapping: ndarray
            pixel col coordinate for each point projected
        """
        # number of columns in the image
        # img_width = np.ceil(np.pi * res_planar).astype(int)

        # x coordinates
        x = points[:, 0]
        # y coordinates
        y = points[:, 1]
        # z coordinates
        z = points[:, 2]

        # lengths of points
        rho = np.linalg.norm(points[:, :3], axis=1)

        # we need to invert orientation of y coordinates
        yaw = np.arctan2(-y, x)

        pitch = np.arccos(z / rho)

        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = fov_down - fov_up  # get field of view total in rad

        # get projections in image coords
        i_img_mapping = (pitch - fov_up) / fov  # in [0.0, 1.0]
        j_img_mapping = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

        # scale to image size using angular resolution
        i_img_mapping *= self.proj_H  # in [0.0, H]
        j_img_mapping *= self.proj_W  # in [0.0, W]

        # round and clamp for use as index
        i_img_mapping = np.floor(i_img_mapping)
        i_img_mapping = np.minimum(self.proj_H - 1, i_img_mapping)
        i_img_mapping = np.maximum(0, i_img_mapping).astype(np.int32)  # in [0,H-1]

        j_img_mapping = np.floor(j_img_mapping)
        j_img_mapping = np.minimum(self.proj_W - 1, j_img_mapping)
        j_img_mapping = np.maximum(0, j_img_mapping).astype(np.int32)  # in [0,W-1]

        lidx = (i_img_mapping * self.proj_W) + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self):

        return self.proj_H, self.proj_W


class LaserProjector(AbstractProjector):
    def __init__(self, height=64, width=1024):
        self.proj_H = height
        self.proj_W = width
        super(AbstractProjector, self).__init__()

    def project_point(self, points):
        layers = points[:, -1]
        # x coordinates
        x = points[:, 0]
        # y coordinates
        y = points[:, 1]

        # we need to invert orientation of y coordinates
        yaw = np.arctan2(-y, x)
        i_img_mapping = layers.astype(np.int)
        j_img_mapping = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        j_img_mapping *= self.proj_W  # in [0.0, W]

        j_img_mapping = np.floor(j_img_mapping)
        j_img_mapping = np.minimum(self.proj_W - 1, j_img_mapping)
        j_img_mapping = np.maximum(0, j_img_mapping).astype(np.int32)  # in [0,W-1]

        lidx = (i_img_mapping * self.proj_W) + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self):
        return self.proj_H, self.proj_W

class Projection:
    def __init__(self,
                 proj_type,
                 height=64,
                 width=1024,
                 fov_up=85.0,
                 fov_down=115.0,
                 res=-1):
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

        height: int
            Used only for Spherical projections. This value represents the height of the spherical image

        width: int
            Used only for Spherical projection. This value represents the width of the spherical image

        fov_up: float
            Used only for Spherical projection. This value represents the upper value of the vertical field of view

        fov_down: float
            Used only for Spherical projection. This value represents the bottom value of the vertical field of view.

        res: float
            Used to chose resolution on BEV images.

        """
        self.proj_type = proj_type
        self.img_H = height
        self.img_W = width
        if self.proj_type == 'front':
            self.projector = self._init_spherical_projector(height=height, width=width,
                                                            fov_up=fov_up, fov_down=fov_down)

        elif self.proj_type == 'bev':
            self.res = res
            self.projector = self._init_linear_projector(height=height, width=width, res=self.res)

        elif self.proj_type == 'laser':
            self.projector = self._init_laser_projector(height=height, width=width)

        else:
            raise ValueError("Projection Type can be only 'front' or 'bev'")

    @staticmethod
    def _init_spherical_projector(height, width, fov_up, fov_down):
        """
        Function that initialize a spherical projector

        Parameters
        ----------
        height: int
            height of the image
        width: int
            width of the image

        Returns
        -------
        SphericalProjector: BaseProjector (object)
            Spherical projector
        """
        return SphericalProjector(height=height, width=width, fov_up=fov_up, fov_down=fov_down)

    @staticmethod
    def _init_linear_projector(height, width, res):
        """
        Function that initialize a linear projector

        Parameters
        ----------
        height: int
            height of projection image

        width: int
            width of projection image

        res: float
            resolution for

        Returns
        -------
        LinearProjection: BaseProjector (object)
            Linear projector
        """
        return BEVProjector(height=height, width=width, res=res)

    @staticmethod
    def _init_laser_projector(height, width):
        """
        Function that initialize a spherical projector by layer

        Parameters
        ----------
        height:
        width:


        Returns
        LaserProjector: BaseProjector (object)
            Spherical projector by layer
        """

        return LaserProjector(height=height, width=width)

    def project_points_values(self, points, values, aggregate_func='max', layers=None):
        """
        Function that project an array of values to an image

        Parameters
        ----------
        points: ndarray
            Array containing the point cloud

        values: ndarray
            Array containing the values to project

        aggregate_func: optional {'max', 'min', 'mean'}
            Function to use to aggregate the information in case of collision, i.e. when two or more points
            are projected to the same pixel.
            'max': take the maximum value among all the values projected to the same pixel
            'min': take the minimum value among all the values projected to the same pixel
            'mean': take the mean value among all the values projected to the same pixel


        Returns
        -------
        proj_img: ndarray
            Image containing projected values
        """

        nr, nc = self.projector.get_image_size()
        if len(values.shape) < 2:
            channel_shape = 1
            values = np.atleast_2d(values).T
        else:
            _, channel_shape = values.shape[:2]

        if channel_shape > 1:
            if type(aggregate_func) is str:
                aggregators = [aggregate_func] * channel_shape
            else:
                assert len(aggregate_func) == channel_shape
                aggregators = aggregate_func
        else:
            aggregators = [aggregate_func]
        # we verify that the length of the two arrays is the same
        # that is for each point we have a corresponding value to project
        assert len(points) == len(values)
        # project points to image
        if self.proj_type == 'laser':
            lidx, i_img_mapping, j_img_mapping = self.projector.project_point(np.c_[points, layers])
        else:
            lidx, i_img_mapping, j_img_mapping = self.projector.project_point(points)

        # initialize binned_feature map
        binned_values = np.zeros((nr * nc, values.shape[1]))

        if 'max' in aggregators or 'min' in aggregators:
            # auxiliary variables to compute minimum and maximum
            sidx = lidx.argsort()
            idx = lidx[sidx]
            # we select the indices of the first time a unique value in lidx appears
            # np.r_[True, idx[:1] != idx[1:]] is true if an element in idx is different than its successive
            # flat non zeros returns indices of values that are non zeros in the array
            m_idx = np.flatnonzero(np.r_[True, idx[:-1] != idx[1:]])

            unq_ids = idx[m_idx]

        if 'mean' in aggregators:
            # auxiliary vector to compute binned count
            count_input = np.ones_like(values[:, 0])
            binned_count = np.bincount(lidx, count_input, minlength=nr * nc)

        for i, func in zip(range(values.shape[1]), aggregators):

            if func == 'max':
                """
                Examples
                --------
                To take the running sum of four successive values:

                >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
                array([ 6, 10, 14, 18])

                """
                binned_values[unq_ids, i] = np.maximum.reduceat(values[sidx, i], m_idx)

            elif func == 'min':
                binned_values[unq_ids, i] = np.minimum.reduceat(values[sidx, i], m_idx)

            elif func == 'sum':
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)

            else:  # otherwise we compute mean values
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)
                binned_values[:, i] = np.divide(binned_values[:, i], binned_count, out=np.zeros_like(binned_count),
                                                where=binned_count != 0.0)

        # reshape binned_features to feature map
        binned_values_map = binned_values.reshape((nr, nc, values.shape[1]))

        if channel_shape == 1:
            binned_values_map = binned_values_map[:, :, 0]

        return binned_values_map

    def back_project(self, points, img, res_value=1, layers=None):
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

        layers: ndarray
            layers for the projection by laser

        Returns
        -------
        values: ndarray
            Back projected values
        """

        # shape of projection image
        nr, nc, nz = np.atleast_3d(img).shape
        # retrieving pixels coordinates
        if self.proj_type == 'laser':
            lidx, i_img_mapping, j_img_mapping = self.projector.project_point(np.c_[points, layers])
        else:
            lidx, i_img_mapping, j_img_mapping = self.projector.project_point(points)

        img_f = img.reshape(nr*nc, nz)

        values = img_f[lidx, :] * res_value

        if nz == 1:
            return values.flatten()

        return values


def back_proj_bev_pred(cloud, pred, proj):
    z = cloud.xyz[:, 2]

    z_min_img = proj.project_points_values(cloud.xyz, z, aggregate_func='min')

    lidx, i_mapping, j_mapping = proj.projector.project_point(cloud.xyz)

    labels = np.zeros_like(z)
    gt_flat = pred.flatten()
    z_img_fl = z_min_img.flatten()

    neighs = cloud.get_neighbors(r=0.5)
    for n in range(len(cloud.xyz)):
        if np.abs(z_img_fl[lidx[n]] - z[n]) < 0.15:
            local_neighs = neighs[n]
            if len(local_neighs) < 5:
                continue

            est_z = z[local_neighs].mean()
            if np.abs(est_z - z[n]) < 0.10:
                labels[n] = gt_flat[lidx[n]]

    return labels


def back_proj_front_pred(cloud, pred, proj, inference='regression'):
    """
    Parameters
    ----------
    cloud: Pyntcloud

    pred: ndarray

    proj: Projection

    inference: optional {'regression', 'classification'}
    """
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    layers = cloud.points['layers'].astype(int)

    # project gt
    acc_img = proj.project_points_values(cloud.xyz, np.ones_like(layers), aggregate_func='sum', layers=layers)

    # back project gt
    back_proj_acc_mask = proj.back_project(cloud.xyz, acc_img == 1, res_value=1, layers=layers)
    back_proj_pred = proj.back_project(cloud.xyz, pred, res_value=1, layers=layers)

    num_neighbors = 3
    if inference == 'regression':
        knn = KNeighborsRegressor(n_neighbors=num_neighbors, weights='uniform')
    elif inference == 'classification':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    else:
        raise ValueError('Inference value can only be "regression" or "classification". Passed value: {}'.format(inference))
    X_train, y_train = cloud.xyz[back_proj_acc_mask == 1, :], back_proj_pred[back_proj_acc_mask == 1]
    X_test = cloud.xyz[back_proj_acc_mask != 1, :]

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    labels = np.zeros_like(back_proj_pred)
    labels[back_proj_acc_mask == 1] = back_proj_pred[back_proj_acc_mask == 1]
    labels[back_proj_acc_mask != 1] = y_pred

    return labels

def naive_back_proj_front_pred(cloud, pred, proj):
    """
    Parameters
    ----------
    cloud: Pyntcloud

    pred: ndarray

    proj: Projection
    """
    layers = cloud.points['layers'].astype(int)

    return proj.back_project(cloud.xyz, pred, res_value=1, layers=layers)