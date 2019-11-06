import cv2
import keras
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from helpers.data_loaders import process_list, get_dataset, get_semantic_kitti_dataset, load_pc, load_bin_file
from helpers.data_loaders import load_label_file, load_semantic_kitti_config
from helpers.pointcloud import filter_points
from helpers.projection import Projection
from helpers.normals import estimate_normals_from_spherical_img
from skimage.morphology import closing, square, opening, rectangle

DATASET_LOADERS = {
    'kitti': get_dataset,
    'semantickitti': get_semantic_kitti_dataset
}


class KITTIPointCloud:
    """
    Classes that load the point clouds an generate features maps
    """
    def __init__(self, feature_parameters, path='', is_training=True, sequences=None, view='bev', dataset='kitti'):
        if len(path) == 0:
            self.dataset_path = '../'
        else:
            self.dataset_path = path
            
        self.is_training = is_training

        self.dataset = dataset
        data_loader_args = dict(path=self.dataset_path, is_training=is_training)

        if self.dataset == 'semantickitti':
            data_loader_args['sequences'] = sequences

        if self.is_training:
            self.train_set, self.valid_set = DATASET_LOADERS[dataset](**data_loader_args)
        else:
            self.test_set = DATASET_LOADERS[dataset](**data_loader_args)

        self.view = view

        self.compute_classic = feature_parameters.get('compute_classic', True)
        self.add_geometrical_features = feature_parameters.get('add_geometrical_features', False)
        self.subsample_ratio = feature_parameters.get('subsample_ratio', 1)

        self.kneighbors = feature_parameters.get('compute_eigen', 0)
        self.compute_eigen = self.kneighbors > 0
        self.model_name = feature_parameters.get('model_name', 'lodnn')

        self.n_channels = 6 if self.view == 'bev' else 3
        # reset n_channels to 0 if we do not compute classical features
        self.n_channels = self.n_channels if self.compute_classic else 0
        self.n_channels += 3 if self.add_geometrical_features else 0
        self.n_channels += 6 if self.compute_eigen else 0

        self.fov_up = (85.0 * np.pi) / 180.0
        self.fov_down = (115.0 * np.pi) / 180.0

        if self.dataset == 'kitti':
            # hard coded parameters for kitti dataset
            self.side_range = (-10, 10)
            self.fwd_range = (6, 46)
        else:
            self.config = load_semantic_kitti_config()
            self.num_classes = len(self.config['labels'].keys())
            self.max_label = max(self.config['labels'].keys()) + 1
            label_map = np.zeros(self.max_label)
            for n, k in enumerate(self.config['labels'].keys()):
                label_map[k] = n

            self.label_map = label_map

        if self.view == 'bev':
            # TODO: complete this part
            self.proj_H = 400
            self.proj_W = 200
            self.proj = Projection(proj_type='bev', height=self.proj_H, width=self.proj_W, res=0.1)
            if self.add_geometrical_features:
                aux_height = 64 // self.subsample_ratio
                self.aux_proj = Projection(proj_type='front', height=aux_height, width=1024)

        elif self.view == 'front':
            self.proj_W = 1024 if self.dataset == 'kitti' else 2048
            self.proj_H = 64 // self.subsample_ratio

            self.proj = Projection(proj_type='front', height=self.proj_H, width=self.proj_W)
            self.aux_proj = Projection(proj_type='front', height=64, width=self.proj_W)

        self.number_of_grids = self.proj_H * self.proj_W

        if self.is_training:
            self.z_min = np.inf
            self.z_max = -np.inf
            if self.view == 'bev':
                self.COUNT_MAX = -1

        else:
            self.z_max = feature_parameters.get('z_max')
            self.z_min = feature_parameters.get('z_min')
            self.COUNT_MAX = feature_parameters.get('COUNT_MAX')

            if self.z_max is None:
                self.z_max = 3.32
                raise RuntimeWarning('No z_max value passed for testing. '
                                     'This could lead to prediction error because of renormalization. '
                                     'Value for z_max is set to {}'.format(self.z_max))

            if self.z_min is None:
                self.z_min = -32.32
                raise RuntimeWarning('No z_min value passed for testing. '
                                     'This could lead to prediction error because of renormalization. '
                                     'Value for z_min is set to {}'.format(self.z_min))

            if self.COUNT_MAX is None and self.view == 'bev':
                self.COUNT_MAX = 200
                raise RuntimeWarning('No COUNT_MAX value passed for testing. '
                                     'This could lead to prediction error because of renormalization. '
                                     'Value for COUNT_MAX is set to {}'.format(self.COUNT_MAX))

    def load_dataset(self, limit_index=-1, **gt_args):

        print('Loading dataset')

        gt_key = 'gt_front' if self.view == 'front' else 'gt_bev'

        if self.is_training:
            f_train = self.load_pc_and_get_features(self.train_set["pc"], limit_index)
            f_valid = self.load_pc_and_get_features(self.valid_set["pc"], limit_index)

            print("Height ranges from {} to {}".format(self.z_min, self.z_max))

            if self.view == 'bev':
                print("Count varies from 0 to {}".format(self.COUNT_MAX))

            if self.dataset == 'kitti':
                gt_train_data_list = self.train_set[gt_key]
                gt_valid_data_list = self.valid_set[gt_key]
            else:
                gt_train_data_list = [[i1, i2] for i1, i2 in zip(self.train_set['pc'], self.train_set['labels'])]
                gt_valid_data_list = [[i1, i2] for i1, i2 in zip(self.valid_set['pc'], self.valid_set['labels'])]

            gt_train = self.fetch_gt(gt_train_data_list, limit_index, **gt_args)
            gt_valid = self.fetch_gt(gt_valid_data_list, limit_index, **gt_args)

            return np.array(f_train), np.array(gt_train), np.array(f_valid), np.array(gt_valid)
        else:
            f_test = self.load_pc_and_get_features(self.test_set["pc"], limit_index)

            if self.dataset == 'kitti':
                gt_test_data_list = self.test_set[gt_key]
            else:
                gt_test_data_list = [[i1, i2] for i1, i2 in zip(self.test_set['pc'], self.test_set['labels'])]

            gt_test = self.fetch_gt(gt_test_data_list, limit_index, **gt_args)

            return np.array(f_test), np.array(gt_test)

    def load_pc_and_get_features(self, file_list, limit_index=-1):
        """
        Function that load point cloud and generate features map

        Parameters
        ----------
        file_list: list
            list of point cloud files to load

        limit_index: int
            index of max element of list to treat

        Returns
        -------
        features_map: ndarray
            array of features maps generated from input point cloud
        """
        max_id = len(file_list) if limit_index == -1 else min(limit_index, len(file_list))
        pc_list = load_pc(file_list[:max_id])

        if self.subsample_ratio > 1:
            pc_list = process_list(pc_list, subsample_pc, sub_ratio=self.subsample_ratio)

        if self.is_training:
            z_vals = process_list(pc_list, get_z)
            z_vals = np.concatenate([z_vals])
            # todo: add code to extract z_min, z_max
            print(np.min(z_vals[:, 0]), np.max(z_vals[:, 1]))

            if self.dataset == 'kitti':
                pc_list = process_list(pc_list, filter_points,
                                       **{'side_range': self.side_range, 'fwd_range': self.fwd_range})

            self.z_min = min(self.z_min, np.min(z_vals[:, 0]))
            self.z_max = max(self.z_max, np.max(z_vals[:, 1]))
            print(self.z_min, self.z_max)
            if self.view == 'bev':
                count_max = process_list(pc_list, self.get_count_features)
                count_max = np.concatenate([c.flatten() for c in count_max])
                self.COUNT_MAX = max(self.COUNT_MAX, np.max(count_max))

        else:
            if self.dataset == 'kitti':
                pc_list = process_list(pc_list, filter_points, **{'side_range': self.side_range,
                                                                  'fwd_range': self.fwd_range})

        features_map = process_list(pc_list, self.get_features)

        return features_map

    def get_features(self, point_cloud):
        """
        Write a Function that given a point cloud return a the features maps to use for segmentation

        Parameters
        ----------
        point_cloud: numpy.ndarray
            Input point cloud

        Returns
        -------
        features_map: numpy.ndarray
            Features map to use for classification
        """
        feat = None
        if self.compute_classic:
            feat = self.get_classical_features(point_cloud)

        if self.add_geometrical_features:
            if self.view == 'bev':
                # estimate normals in front view image
                normals = self.get_normals_features(point_cloud, self.aux_proj)

                # back project points from front view image to 3D
                normals_3d = self.aux_proj.back_project(point_cloud[:, :3], normals)

                # project 3d normals to BEV image
                normals = self.proj.project_points_values(point_cloud[:, :3], normals_3d, aggregate_func='mean')
            else:
                # estimage surface normals for front view images
                normals = self.get_normals_features(point_cloud, self.proj)

            if feat is not None:
                feat = np.dstack([feat, normals])

        if self.compute_eigen:
            eigen = self.get_eigen_features(point_cloud)
            eigen_map = self.proj.project_points_values(point_cloud, eigen, aggregate_func='mean')
            if feat is not None:
                feat = np.dstack([feat, eigen_map])

        return feat

    def get_count_features(self, points):
        """
        Returns features of the point cloud as stacked grayscale images.
        Shape of the output is (400x200x6).

        Parameters
        ----------
        points: ndarray
            input point cloud

        """
        return self.proj.project_points_values(points[:, :3], np.ones_like(points[:, 0]), aggregate_func='sum')

    def get_classical_features(self, points):
        """
        Method that compute classical features from point cloud

        Parameters
        ----------
        points: ndarray
            input point cloud

        """
        z = points[:, 2]
        r = points[:, 3]

        norm_z_lidar = z.copy()
        norm_z_lidar = (norm_z_lidar - self.z_min) / (self.z_max - self.z_min)

        if self.view == 'bev':
            features = np.c_[np.ones_like(norm_z_lidar),  # to compute o_count
                             r,  # to compute o_mean_reflectance
                             norm_z_lidar,  # to compute o_max_elevation
                             norm_z_lidar,  # to compute o_min_elevation
                             norm_z_lidar,  # to compute o_mean_elevation
                             norm_z_lidar**2]  # to compute o_mean_elevation

            aggregators = ['sum', 'mean', 'max', 'min', 'mean', 'mean']
            feat_maps = self.proj.project_points_values(points[:, :3], features, aggregate_func=aggregators)
            o_mean_elevation = feat_maps[:, :, -2].astype(np.float64).copy()
            o_2_mean_elev = feat_maps[:, :, -1].astype(np.float64).copy()  # squared mean elevation
            o_std_elevation = np.abs(o_2_mean_elev.astype(np.float64) - np.square(o_mean_elevation.astype(np.float64)))
            # o_std_elevation[o_std_elevation < 0] = 0
            o_std_elevation = np.sqrt(o_std_elevation)

            # replace z_sum with z_std
            feat_maps[:, :, -1] = o_std_elevation

            # normalize o_count
            feat_maps[:, :, 0] = feat_maps[:, :, 0] / self.COUNT_MAX
        else:
            rho = np.linalg.norm(points[:, :3], axis=1)
            features = np.c_[rho, r, norm_z_lidar]
            feat_maps = self.proj.project_points_values(points[:, :3], features, aggregate_func=['max', 'mean', 'min'])

            # renormalize rho values
            feat_maps[:, :, 0] = feat_maps[:, :, 0] / 85

        return feat_maps

    def get_normals_features(self, points, projector):
        """
        Parameters
        ----------
        points: ndarray
            input point cloud

        projector: Projection
            object in charge to project points to front view images

        Returns
        -------
        normals: ndarray
            estimated surface normals in front view image
        """

        height, width = projector.projector.get_image_size()

        # compute distance from scanner
        rho = np.linalg.norm(points[:, :3], axis=1)

        rho_img = projector.project_points_values(points[:, :3], rho, aggregate_func='min')

        cl_img = closing(rho_img, square(3))

        rho_img[rho_img == 0] = cl_img[rho_img == 0]

        yaw_axis = np.linspace(0, 2*np.pi, width)

        pitch_axis = np.linspace(self.fov_up, self.fov_down, height)

        normals = estimate_normals_from_spherical_img(rho_img, yaw=yaw_axis, pitch=pitch_axis, res_rho=1.0)

        return normals

    def get_eigen_features(self, points):
        """
        Using pyntcloud library calculates various eigen properties.

        Parameters
        ----------
        points: ndarray
            input point cloud

        Returns
        -------
        eigen_features: ndarray
            spectral features computed from eigenvalues
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        r = points[:, 3]

        # convert it into Pandas df
        pc = PyntCloud(pd.DataFrame({'x': x, 'y': y, 'z': z, 'r': r}))
        k_neighbors = pc.get_neighbors(k=self.kneighbors)
        eigenvalues = pc.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        pc.points[eigenvalues] = pc.points[eigenvalues].values / pc.points[eigenvalues].values.sum(axis=1)[:, None]
        # anisotropy = pc.add_scalar_field("anisotropy", ev=eigenvalues)
        pc.add_scalar_field("curvature", ev=eigenvalues)
        pc.add_scalar_field("eigenentropy", ev=eigenvalues)
        # eigensum = pc.add_scalar_field("eigen_sum", ev=eigenvalues)
        pc.add_scalar_field("linearity", ev=eigenvalues)
        pc.add_scalar_field("omnivariance", ev=eigenvalues)
        pc.add_scalar_field("planarity", ev=eigenvalues)
        pc.add_scalar_field("sphericity", ev=eigenvalues)

        eigen_features = pc.points.values[:,7:]
        # eigen_features = pc.points.as_matrix(columns=pc.points.columns[7:])

        return eigen_features

    def load_gt(self, filename, **kwargs):
        """
        Function that load the ground truth image and return the one-hot encoded ground truth image

        Parameters
        ----------
        filename: str
            Filename of the ground truth image

        **kwargs:
            any other parameter that you want to use

        """
        # read a 16-bit image with opencv

        if self.dataset == 'kitti':
            img = cv2.imread(filename, -1)
            road = img[:, :, 0] / 255  # Road is encoded as 255 in the B plane
            non_road = 1 - road  # TODO: can we do this in training time?
            return np.dstack([road, non_road])

        elif self.dataset == 'semantickitti':
            pc_filename = filename[0]
            label_filename = filename[1]
            labels = load_label_file(label_filename)
            points = load_bin_file(pc_filename)
            # classes should be a list containing classes that we want to isolate for binary segmentation
            classes = kwargs.get('classes', None)
            if classes is not None:
                # binary class classification
                out = [labels == int(c) for c in classes]
                labels = sum(out)
                img = self.aux_proj.project_points_values(points, labels)
                return np.atleast_3d(img)
            else:
                # multiclass classification
                img = self.aux_proj.project_points_values(points, labels)
                return keras.utils.to_categorical(self.label_map[img.astype(int)], num_classes=self.num_classes)

    def fetch_gt(self, file_list, limit_index, **kwargs):
        max_id = len(file_list) if limit_index == -1 else min(limit_index, len(file_list))
        return process_list(file_list[:max_id], self.load_gt, **kwargs)


class SemanticKittiConfig:
    """
    Class that load the semantic kitti config file and helps to handle class ids
    """
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str to config file
        """
        self.config_file = config_file
        self.config = load_semantic_kitti_config(config_file)
        labels, id2label = self._remap_classes()
        self.labels = labels
        self.id2label = id2label

    def _remap_classes(self):
        labels = np.array(list(self.config['labels'].keys()))

        # this vector is the inverse function of labels array, i.e. labels[id_label_map[labels]] = labels
        id_label_map = -np.ones(labels.max() + 1, dtype=np.int)

        for n, l in enumerate(labels):
            id_label_map[l] = n

        return labels, id_label_map


def get_z(points):
    return np.array([np.min(points[:, 2]), np.max(points[:, 2])])


def retrieve_layers(points):
    """
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.

    """
    x = points[:, 0]
    y = points[:, 1]

    # compute the theta angles
    thetas = np.arctan2(y, x)
    op_thetas = opening(thetas.reshape(-1, 1), rectangle(20, 1))
    thetas = op_thetas.flatten()
    idx = np.ones(len(thetas))

    idx_pos = idx.copy()
    idx_pos[thetas < 0] = 0

    # since each layer is stocked in a clockwise fashion each time we do a 2*pi angle we can change layer
    # so we identify each time we do a round
    changes = np.arange(len(thetas) - 1)[np.ediff1d(idx_pos) == 1]
    changes += 1  # we add one for indexes reason

    # Stocking intervals. Each element of intervals contains min index and max index of points in the same layer
    intervals = []
    for i in range(len(changes)):
        if i == 0:
            intervals.append([0, changes[i]])
        else:
            intervals.append([changes[i - 1], changes[i]])

    intervals.append([changes[-1], len(thetas)])

    # for each element in interval we assign a label that identifies the layer
    layers = np.zeros(len(thetas), dtype=np.uint8)

    for n, el in enumerate(intervals):
        layers[el[0]:el[1]] = n

    return layers


def subsample_pc(points, sub_ratio=2):
    """
    Return sub sampled point cloud

    Parameters
    ----------
    points: ndarray
        input point cloud

    sub_ratio: int
        ratio to use to subsample point cloud

    Returns
    -------
    points: ndarray
        subsampled pointcloud
    """
    layers = retrieve_layers(points)
    new_points = np.c_[points, layers]
    # sampling only points with even id
    return new_points[layers % sub_ratio == 0]
