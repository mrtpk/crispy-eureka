import os
import cv2
from time import time
import numpy as np
import datetime
from helpers import data_loaders as dls
from helpers import pointcloud as pc
from helpers.normals import estimate_normals_from_spherical_img
from helpers.calibration import get_lidar_in_image_fov
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from skimage.morphology import opening, rectangle
import keras
from keras.callbacks import TensorBoard, EarlyStopping

from pyntcloud import PyntCloud
import pandas as pd


# memory cache now pre-evalutes get_feature functions.
# Attention if you change the code make sure you empty the cachefolder to re-evalaute these features.
def get_unique_id():
    return datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')


def create_run_dir(path, run_id):
    path = path.replace("*run_id", str(run_id))
    model_dir = "{}/model/".format(path)
    output_dir = "{}/output/".format(path)
    log_dir = "{}/log/".format(path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return path


def get_basic_callbacks(path):
    # tensorboard
    # to visualise `tensorboard --logdir="./path_to_log_dir" --port 6006`
    log_path = "{}/log/".format(path)
    tensorboard = TensorBoard(log_dir="{}/{}".format(log_path, time()))
    # save best model
    best_model_path = "{}/model/best_model.h5".format(path) #? .hd5
    save_the_best = keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                    verbose=1, save_best_only=True)
    # save models after few epochs
    epoch_save_path = "{}/model/*.h5".format(path)
    save_after_epoch = keras.callbacks.ModelCheckpoint(filepath=epoch_save_path.replace('*', 'e{epoch:02d}-val_acc{val_acc:.2f}'),
                                                       monitor='val_acc', verbose=1, period = 1)

    stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')

    return [tensorboard, save_the_best, save_after_epoch, stopping]


def apply_argmax(res):
    return 1 - np.argmax(res, axis=2)


def get_z(points):
    # return points[:,2].reshape(-1,1)
    return np.array([np.min(points[:, 2]), np.max(points[:, 2])])


def get_first_chan(points):
    return points[:, :, 0]


def measure_perf(path, all_pred, all_gt):
    result_path = "{}/output/*".format(path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    F1, P, R, ACC = [], [], [], []
    FN, FP, TP, TN = [], [], [], []
    AP = []
    for i in range(all_pred.shape[0]):
        _f, _gt = all_pred[i], all_gt[i]
        p_road = apply_argmax(_f)       
        # get metrics
        gt_road = _gt[:, :, 0]
        fn, fp, tp, tn = get_metrics_count(pred=p_road, gt=gt_road)
        f1, recall, precision, acc = get_metrics(gt=gt_road, pred=p_road)
        ap = average_precision_score(gt_road, p_road)
        AP.append(ap)
        F1.append(f1)
        P.append(precision)
        R.append(recall)
        ACC.append(acc)
        TP.append(tp)
        FP.append(fp)
        TN.append(tn)
        FN.append(fn)
        
    eps = np.finfo(np.float32).eps        
    _acc = (sum(TP)+sum(TN))/(sum(TP)+sum(FP)+sum(TN)+sum(FN) + eps)
    _recall = sum(TP)/(sum(TP) + sum(FN)+eps)
    _precision = sum(TP)/(sum(TP) + sum(FP)+eps)
    _f1 = 2*((precision * recall)/(precision + recall))    
    return {
           "accuracy" : _acc,
           "recall" : _recall,
           "precision" : _precision,
           "F1" : _f1
           }


class KittiPointCloudClass:
    """ 
    Kitti point cloud dataset to load dataset, subsampling and feature extraction
    """
    def __init__(self, dataset_path,
                 add_geometrical_features,
                 subsample,
                 compute_HOG,
                 eigen_neighbors,
                 view,
                 subsample_ratio=1,
                 dataset='kitti',
                 sequences=None):
        # get dataset
        if dataset == 'kitti':
            self.train_set, self.valid_set, self.test_set = dls.get_dataset(dataset_path, is_training=True)
        else:
            self.train_set, self.valid_set, self.test_set = dls.get_semantic_kitti_dataset(dataset_path,
                                                                                           is_training=True,
                                                                                           sequences=sequences)

        self.add_geometrical_features = add_geometrical_features  # Flag
        self.subsample = subsample  # Flag
        self.compute_HOG = compute_HOG  # Flag
        self.compute_eigen = eigen_neighbors > 0 # Flag
        self.kneighbors = eigen_neighbors
        self.view = view
        self.side_range=(-10, 10)  # this is fixed here for KITTI
        self.fwd_range=(6, 46)  # this is fixed here for KITTI
        self.res=.1
        self.res_planar = 300
        # todo we should change type of subsample parameter from boolean to integer
        self.subsample_ratio = 1

        if self.subsample:  # in  case we subsample we fix sub_sample ratio bigger than 1
            self.subsample_ratio = 2 if subsample_ratio == 1 else subsample_ratio

        self.num_layers = 64 // self.subsample_ratio

        # calculate the image dimensions
        if self.view == 'bev':
            self.img_width = int((self.side_range[1] - self.side_range[0])/self.res)
            self.img_height = int((self.fwd_range[1] - self.fwd_range[0])/self.res)
            self.number_of_grids = self.img_height * self.img_width
        else:
            self.img_width = np.ceil(np.pi * self.res_planar).astype(int)
            self.img_height = self.num_layers
            self.number_of_grids = self.img_width * self.img_height

    def get_dataset(self, limit_index=3):
        print(self.train_set.keys())
        """ NOTE: change limit_index to -1 to train on the whole dataset """
        print('Reading cloud')
        if limit_index > 0:
            f_train = dls.load_pc(self.train_set["pc"][0:limit_index])
            f_valid = dls.load_pc(self.valid_set["pc"][0:limit_index])
            f_test = dls.load_pc(self.test_set["pc"][0:limit_index])
            if self.compute_HOG:
                print('Reading calibration files')
                cal_train = dls.process_calib(self.train_set["calib"][0:limit_index])
                cal_valid = dls.process_calib(self.valid_set["calib"][0:limit_index])
                cal_test = dls.process_calib(self.test_set["calib"][0:limit_index])

                print('Reading camera images')
                cam_img_train = dls.load_img(self.train_set["imgs"][0:limit_index])
                cam_img_valid = dls.load_img(self.valid_set["imgs"][0:limit_index])
                cam_img_test = dls.load_img(self.test_set["imgs"][0:limit_index])

        else:
            f_train = dls.load_pc(self.train_set["pc"][0:])
            f_valid = dls.load_pc(self.valid_set["pc"][0:])
            f_test = dls.load_pc(self.test_set["pc"][0:])
            if self.compute_HOG:
                print('Reading calibration files')
                cal_train = dls.process_calib(self.train_set["calib"][0:])
                cal_valid = dls.process_calib(self.valid_set["calib"][0:])
                cal_test = dls.process_calib(self.test_set["calib"][0:])

                print('Reading camera images')
                cam_img_train = dls.load_img(self.train_set["imgs"][0:])
                cam_img_valid = dls.load_img(self.valid_set["imgs"][0:])
                cam_img_test = dls.load_img(self.test_set["imgs"][0:])


        #Update min max z
        z_vals = dls.process_list(f_train + f_valid, get_z)
        z_vals = np.concatenate([z_vals])
        print(z_vals.shape)
        self.z_min, self.z_max = np.min(z_vals[:,0]), np.max(z_vals[:,1])
        print("Height ranges from {} to {}".format(self.z_min, self.z_max))
        
        if self.subsample:
            print('Read and Subsample cloud')
            t = time()
            f_train = dls.process_list(f_train, subsample_pc, sub_ratio=self.subsample_ratio)
            f_valid = dls.process_list(f_valid, subsample_pc, sub_ratio=self.subsample_ratio)
            f_test = dls.process_list(f_test, subsample_pc, sub_ratio=self.subsample_ratio)
            print('Evaluated in : '+repr(time()-t))

        # Update with maximum points found within a cell to be used for normalization later
        print('Evaluating count')
        self.COUNT_MIN = 0 
        self.COUNT_MAX = []
        for _f in f_train+f_test:
            _filtered = pc.filter_points(_f, side_range=self.side_range, 
                                         fwd_range=self.fwd_range)
            f_count = self.get_count_features(_filtered)
            self.COUNT_MAX.append(int(np.max(f_count)))
        self.COUNT_MAX = max(self.COUNT_MAX)
        print("Count varies from {} to {}".format(self.COUNT_MIN, self.COUNT_MAX))    
            
        print('Extracting features')
        t = time()
        if self.compute_HOG:
            f_cam_calib_train = [(f_t, img, calib) for f_t, img, calib in zip(f_train, cam_img_train, cal_train)]
            f_cam_calib_valid = [(f_t, img, calib) for f_t, img, calib in zip(f_valid, cam_img_valid, cal_valid)]
            f_cam_calib_test = [(f_t, img, calib) for f_t, img, calib in zip(f_test, cam_img_test, cal_test)]
        else:
            f_cam_calib_train = zip(f_train, len(f_train) * [None], len(f_train) * [None])
            f_cam_calib_valid = zip(f_valid, len(f_valid) * [None], len(f_valid) * [None])
            f_cam_calib_test = zip(f_test, len(f_test) * [None], len(f_test) * [None])

        f_train = dls.process_list(f_cam_calib_train, self.get_features)
        f_valid = dls.process_list(f_cam_calib_valid, self.get_features)
        f_test = dls.process_list(f_cam_calib_test, self.get_features)
        print('Evaluated in : '+repr(time()-t))

        gt_key = 'gt_bev' if self.view == 'bev' else 'gt_front'

        if limit_index > 0:
            gt_train = dls.process_img(self.train_set[gt_key][0:limit_index], func=lambda x: kitti_gt(x))
            gt_valid = dls.process_img(self.valid_set[gt_key][0:limit_index], func=lambda x: kitti_gt(x))
            gt_test = dls.process_img(self.test_set[gt_key][0:limit_index], func=lambda x: kitti_gt(x))
        else:
            gt_train = dls.process_img(self.train_set[gt_key][0:], func=lambda x: kitti_gt(x))
            gt_valid = dls.process_img(self.valid_set[gt_key][0:], func=lambda x: kitti_gt(x))
            gt_test = dls.process_img(self.test_set[gt_key][0:], func=lambda x: kitti_gt(x))

        return np.array(f_train), np.array(f_valid), np.array(f_test), np.array(gt_train), np.array(gt_valid), np.array(gt_test)

    def get_features(self, raw_info):
        '''
        Remove points outisde y \in [-10, 10] and x \in [6, 46]
        and
        Returns features of the point cloud as stacked grayscale images.
        Shape of the output is (400x200x6).
        '''
        points = raw_info[0]
        img = raw_info[1]
        calib = raw_info[2]
        # vector containing for each point the id of the layer that captured the point
        if self.subsample:
            layers= points[:, 4].astype(int) // self.subsample_ratio
        else:
            layers = retrieve_layers(points[:,:3])

        unique = np.unique(layers)

        # coordiantes of the points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        # radial distance from scanner
        radial = np.square( x**2 + y**2 )
        # azimuthal angles of the points
        az_angles = np.arccos(z / radial)
        # todo: check the case of a sub sampled point cloud
        az_means = np.zeros(len(unique))
        for n in unique:
            az_means[n] = az_angles[layers==n].mean()

        # adding to point cloud information about layers
        points = np.c_[points, layers]
        # keep only points in the camera FOV
        points = pc.filter_points(points, side_range=self.side_range, fwd_range=self.fwd_range)
        # layers not filtered out
        layers = points[:, -1].astype(int)

        # get all features and normalize count channel
        if self.view == 'bev':
            z = points[:, 2]
            z = (z - self.z_min) / (self.z_max - self.z_min)
            points[:, 2] = z

            out_feature_map, lidx, binned_count = self.get_classical_features(points, self.side_range, self.fwd_range,
                                                                              self.res, layers)
        else:
            out_feature_map, lidx, binned_count = self.get_classical_features(points, self.side_range, self.fwd_range,
                                                                              self.res_planar, layers)
        if self.add_geometrical_features:
            # estimate normals
            # normals = get_normals(points[:, :3], res_az=res_az, res_planar=300)
            if self.view == 'bev':
                normals = get_normals(points, self.res_planar, self.num_layers, layers, az_means, return_3d=True)
                normals_feature_map = project_features(normals, lidx, (self.img_height, self.img_width))
            else:
                normals_feature_map = get_normals(points, self.res_planar, self.num_layers, layers, az_means)

            out_feature_map = np.dstack([out_feature_map, normals_feature_map])

        if self.compute_eigen:
            # calculate eigen properties and put it in an numpy array
            eigen_features = self.get_eigen_features(points)
            # project them into specified view
            eigen_feature_map = project_features(eigen_features, lidx, (self.img_height, self.img_width))
            out_feature_map = np.dstack([out_feature_map, eigen_feature_map])

        if self.compute_HOG:
            # extract hog features from camera image
            hog_features = get_hog(points[:, :3], img, calib)
            # project hog features over the image
            hog_features_map = project_features(hog_features, lidx, (self.img_height, self.img_width))

            out_feature_map = np.dstack([out_feature_map, hog_features_map])

        # cell count normalization
        out_feature_map[:, :, 0] = out_feature_map[:, :, 0] / self.COUNT_MAX

        return out_feature_map
    
    def get_eigen_features(self, points):
        '''
        Using pyntcloud library calculates various eigen properties.
        '''
        x = points[:, 0]
        y = points[:, 0]
        z = points[:, 0]
        r = points[:, 0]

        # convert it into Pandas df 
        pc = PyntCloud(pd.DataFrame({'x': x, 'y': y, 'z': z, 'r': r}))
        k_neighbors = pc.get_neighbors(k=self.kneighbors)
        eigenvalues = pc.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        # anisotropy = pc.add_scalar_field("anisotropy", ev=eigenvalues)
        curvature = pc.add_scalar_field("curvature", ev=eigenvalues)
        eigenentropy = pc.add_scalar_field("eigenentropy", ev=eigenvalues)
        # eigensum = pc.add_scalar_field("eigen_sum", ev=eigenvalues)
        linearity = pc.add_scalar_field("linearity", ev=eigenvalues)
        omnivariance = pc.add_scalar_field("omnivariance", ev=eigenvalues)
        planarity = pc.add_scalar_field("planarity", ev=eigenvalues)
        sphericity = pc.add_scalar_field("sphericity", ev=eigenvalues)

        eigen_features = pc.points.as_matrix(columns = pc.points.columns[7:])
        return eigen_features

    def get_classical_features(self, points, side_range, fwd_range, res, layers):
        # calculate the image dimensions
        # img_width = int((side_range[1] - side_range[0]) / res)
        # img_height = int((fwd_range[1] - fwd_range[0]) / res)
        # number_of_grids = img_height * img_width

        z_lidar = points[:, 2]
        r_lidar = points[:, 3]

        norm_z_lidar = z_lidar  # assumed that the z values are normalised

        if self.view == 'bev':
            lidx, x_img_mapping, y_img_mapping = _get_lidx(points, side_range, fwd_range, res)
        else:
            lidx, x_img_mapping, y_img_mapping = _get_spherical_lidx(points, res, layers)
        # Feature extraction
        # count of points per grid
        count_input = np.ones_like(norm_z_lidar)
        binned_count = np.bincount(lidx, count_input, minlength=self.number_of_grids)
        # sum reflectance
        binned_reflectance = np.bincount(lidx, r_lidar, minlength=self.number_of_grids)
        # sum elevation
        binned_elevation = np.bincount(lidx, norm_z_lidar, minlength=self.number_of_grids)

        # Finding mean!
        binned_mean_reflectance = np.divide(binned_reflectance, binned_count, out=np.zeros_like(binned_reflectance),
                                            where=binned_count != 0.0)
        binned_mean_elevation = np.divide(binned_elevation, binned_count, out=np.zeros_like(binned_elevation),
                                          where=binned_count != 0.0)
        o_mean_elevation = binned_mean_elevation.reshape(self.img_height, self.img_width)

        # Standard deviation stuff
        if self.view == 'bev':
            binned_sum_var_elevation = np.bincount(lidx, np.square(
                norm_z_lidar - o_mean_elevation[-y_img_mapping, x_img_mapping]), minlength=self.number_of_grids)
        else:
            binned_sum_var_elevation = np.bincount(lidx, np.square(
                norm_z_lidar - o_mean_elevation[y_img_mapping, x_img_mapping]), minlength=self.number_of_grids)

        binned_divide = np.divide(binned_sum_var_elevation, binned_count, out=np.zeros_like(binned_sum_var_elevation),
                                  where=binned_count != 0.0)
        binned_std_elevation = np.sqrt(binned_divide)

        # minimum and maximum
        sidx = lidx.argsort()
        idx = lidx[sidx]
        val = norm_z_lidar[sidx]

        m_idx = np.flatnonzero(np.r_[True, idx[:-1] != idx[1:]])
        unq_ids = idx[m_idx]

        o_max_elevation = np.zeros([self.img_height, self.img_width], dtype=np.float64)
        o_min_elevation = np.zeros([self.img_height, self.img_width], dtype=np.float64)

        o_max_elevation.flat[unq_ids] = np.maximum.reduceat(val, m_idx)
        o_min_elevation.flat[unq_ids] = np.minimum.reduceat(val, m_idx)

        norm_binned_count = binned_count  # normalise the output
        # reshape all other things
        o_count = norm_binned_count.reshape(self.img_height, self.img_width)
        o_mean_reflectance = binned_mean_reflectance.reshape(self.img_height, self.img_width)
        o_std_elevation = binned_std_elevation.reshape(self.img_height, self.img_width)

        out_feature_map = np.dstack([o_count,
                                     o_mean_reflectance,
                                     o_max_elevation,
                                     o_min_elevation,
                                     o_mean_elevation,
                                     o_std_elevation])
        return out_feature_map, lidx, binned_count

    def get_count_features(self, points):
        '''
        Returns features of the point cloud as stacked grayscale images.
        Shape of the output is (400x200x6).

        Parameters
        ----------
        points: ndarray
            input point cloud
        '''
        z_lidar = points[:, 2]
        norm_z_lidar = z_lidar  # assumed that the z values are normalised
        if self.view == 'bev':
            lidx, _, _ = _get_lidx(points, self.side_range, self.fwd_range, self.res)
        else:
            if self.subsample:
                l = points[:,4].astype(int) // self.subsample_ratio
                lidx, _, _ = _get_spherical_lidx(points, self.res_planar, layers=l)
            else:
                lidx, _, _ = _get_spherical_lidx(points, self.res_planar)
        # Feature extraction
        # count of points per grid
        count_input = np.ones_like(norm_z_lidar)
        binned_count = np.bincount(lidx, count_input, minlength=self.number_of_grids)
        norm_binned_count = binned_count  # normalise the output
        # reshape all other things
        o_count = norm_binned_count.reshape(self.img_height, self.img_width)
        return o_count


def get_metrics_count(pred, gt):
    '''
    Get true positive, true negative, false positive, false negative counts
    '''
    diff = pred - gt
    # get tp, tn, fp, fn
    fn = diff[diff == -1].size
    fp = diff[diff == 1].size
    tp = diff[gt == 1].size - fn
    tn = diff[diff == 0].size - tp
    return fn, fp, tp, tn


def get_metrics(pred, gt):
    '''
    Get true positive, true negative, false positive, false negative counts
    
    recall = true positives/(true positives + false negatives)
    recall = road pixels correctly classified/(road pixels correctly classified + road pixels incorrectly classified as non-road)
    
    precision = true positives/(true positives + fasle positives)
    precision = road pixels correctly classified/(road pixels correctly classified + non road incorrectly classified as road)
    
    F1/2 = (precision * recall)/(precision + recall)
    '''
    acc = accuracy_score(gt, pred)
    recall = recall_score(gt, pred, average='micro')
    precision = precision_score(gt, pred, average='micro')
    f1 = f1_score(gt, pred, average='micro')

    return f1, recall, precision, acc


#Dataset utils
def kitti_gt(img):
    road = img[:, :, 0] / 255  # Road is encoded as 255 in the B plane
    non_road = 1 - road  # TODO: can we do this in training time?
    return np.dstack([road, non_road])


# to add, get ground truth from lodnn dataset
def get_normals(points, res_planar, num_layers, layers=None, az=None, return_3d=False):
    '''
    New version of the function that computes the normals

    Parameters
    ----------
    points: ndarray
        input point cloud

    res_planar: float
        resolution along planar angle

    num_layers: int
        number of layers used

    layers: ndarray
        array containing for each point the id of the layer that did the acquisition

    az: ndarray
        array containing for each layer the mean value for the azimuthal angles

    return_3d: bool
        if True project back normals to 3D points

    Returns
    -------
    normals: ndarray
        array containing estimated normals
    '''
    from skimage.morphology import dilation, square

    img_width = np.ceil(np.pi * res_planar).astype(int)
    img_height = num_layers

    if layers is None:
        layers = retrieve_layers(points)

    # project point cloud to front view image
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    planar_angles = np.arctan2(-y, x) + np.pi / 2
    radial_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if az is None:
        azimuthal_angles = np.arccos(z / np.sqrt(x**2 + y ** 2))
        az = np.zeros(num_layers)
        for n in range(num_layers):
            az[n] = azimuthal_angles[layers == n].mean()

    I = layers
    J = np.floor(planar_angles * res_planar).astype(int)
    # initialize image
    rho_img = np.zeros((img_height, img_width))
    # assign values to rho_img
    for n in range(len(points)):
        rho_img[I[n], J[n]] = max(rho_img[I[n],J[n]],radial_dist[n])

    # dilate rho_image to fill empty cells
    dil_rho = dilation(rho_img, square(3))

    rho_img[rho_img == 0] = dil_rho[rho_img == 0]

    normals = estimate_normals_from_spherical_img(rho_img, az=az, res_rho=1, res_planar=res_planar)
    # estimate normals

    if return_3d:
        # if return 3d we back project normals to 3d point cloud
        normals_3d = np.zeros_like(points[:, :3])
        for n in range(len(normals_3d)):
            normals_3d[n] = normals[I[n], J[n]]

        return normals_3d

    # else we return the front view image
    return np.abs(normals)


def _get_lidx(points, side_range, fwd_range, res):
    # calculate the image dimensions
    img_width = int((side_range[1] - side_range[0])/res)
    img_height = int((fwd_range[1] - fwd_range[0])/res)
    
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    
    # MAPPING
    # Mappings from one point to grid 
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
    x_img_mapping = (-y_lidar/res).astype(np.int32)  # x axis is -y in LIDAR
    y_img_mapping = (x_lidar/res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img_mapping -= int(np.floor(side_range[0]/res))
    y_img_mapping -= int(np.floor(fwd_range[0]/res))

    # Linerize the mappings to 1D
    lidx = ((-y_img_mapping) % img_height) * img_width + x_img_mapping
    
    return lidx, x_img_mapping, y_img_mapping


def _get_spherical_lidx(points, res_planar, layers=None):

    if layers is None:
        # if we do not have information on layers we retrieve it
        layers = retrieve_layers(points[:,:3])

    # number of columns in the image
    img_width = np.ceil(np.pi* res_planar).astype(int)

    # x coordinates
    x = points[:, 0]
    # y coordinates
    y = points[:, 1]
    # we need to invert orientation of y coordinates
    planar = np.arctan2(-y, x) + np.pi/2
    # x_img_mapping <--> J
    x_img_mapping = np.floor(planar * res_planar).astype(int)
    # y_img_mapping <--> I
    y_img_mapping = layers

    lidx = (y_img_mapping * img_width) + x_img_mapping

    return lidx, x_img_mapping, y_img_mapping


def project_features(features, lidx, img_shape, aggregate_func='mean'):
    '''
    General function to compute features images
    :param features:
    :param lidx:
    :param img_shape:
    :param aggregate_func:
    :return:
    '''
    nr, nc = img_shape[:2]
    # if features is a 1 dimensional vector we augment its dimension
    if len(features.shape) < 2:
        features = np.atleast_2d(features).T

    # auxiliary vector to compute binned_count
    count_input = np.ones_like(features[:, 0])

    binned_count = np.bincount(lidx, count_input, minlength=nr*nc)

    # initialize binned_feature map
    binned_features = np.zeros((nr*nc, features.shape[1]))

    # summing features
    for i in range(features.shape[1]):
        binned_features[:, i] = np.bincount(lidx, features[:, i], minlength=nr*nc)
        # TODO: add other aggregation functions
        if aggregate_func == 'mean':
            # for each cell we compute mean value of the features in cell
            binned_features[:, i] = np.divide(binned_features[:, i], binned_count, out=np.zeros_like(binned_count),
                                             where=binned_count!=0.0)

    # reshape binned_features to feature map
    binned_feature_map = binned_features.reshape((nr, nc, features.shape[1]))

    # return binned_feature_map
    return binned_feature_map


def project_points_on_img(points, img, calib):
    '''
    Function that  project point cloud on fron view image and retrieve RGB information
    '''
    height, width = img.shape[:2]
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points, calib, 0, 0, width, height, return_more=True)
    # the following array contains for each point in the point cloud the corrisponding pixel of the gt_img
    velo_to_pxls = np.floor(pts_2d[fov_inds, :]).astype(int)
    points_to_pxls = -1 * np.ones((len(points), 2), dtype=int)
    points_to_pxls[fov_inds] = velo_to_pxls[:, [1, 0]]

    return points_to_pxls


def get_hog(points, img, calib):
    '''
    A basic function that compute HOG features on the front image
    '''
    points_to_pxls = project_points_on_img(points, img, calib)

    # hardcoded parameters for HOGDescriptor
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = 4.
    histogram_norm_type = 0
    l2_hys_threshold = 2.0000000000000001e-01
    gamma_correction = 0
    nlevels = 64

    num_features = nbins * ((win_size[0] // cell_size[0] - 1) * (win_size[1] // cell_size[1] - 1) * 4)

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture, win_sigma,
                            histogram_norm_type, l2_hys_threshold, gamma_correction, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    win_stride = (8, 8)
    padding = (8, 8)

    # padding img to compute HOG features
    pad_img = np.pad(img, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), 'constant')

    # compute hog features
    hog_features = np.zeros((len(points), num_features))

    idx = points_to_pxls[:, 0] >= 0  # getting idx of points projected on img
    # retrieve location for each point
    locations = tuple([tuple(x) for x in (points_to_pxls[idx] + np.array(padding)).tolist()])
    # compute histogram
    hist = hog.compute(pad_img, win_stride, padding, locations)

    # reshape hist vector to associate at each point its HOG vector
    hist = hist.reshape((-1, num_features), order='F')

    # points that do not fall in the FOV of image will have zero values
    hog_features[idx] = hist

    # reduce features using pca
    # first of all we eliminate the null features
    nn_idx = np.abs(hog_features).sum(axis=0) != 0
    nn_feats = hog_features[:, nn_idx]

    # this choice of component is heuristic
    pca = PCA(n_components=6)

    hog_features = pca.fit_transform(nn_feats)

    return hog_features


def retrieve_layers(points):
    '''
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.
    '''
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
    '''
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
    '''
    layers = retrieve_layers(points)
    new_points = np.c_[points, layers]
    # sampling only points with even id
    return new_points[layers % sub_ratio == 0]

def get_rgb(points, img, calib):
    '''
    Function that  project point cloud on fron view image and retrieve RGB information

    Parameters
    ----------
    points: ndarray
        input point cloud
    img: ndarray
        camera image
    calib: Calibration
        calibration object used to associate points to pixels in img
    '''

    # TODO : Is this function unused?
    height, width = img.shape[:2]

    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points, calib, 0, 0, width, height, return_more=True)

    # the following array contains for each point in the point cloud the corrisponding pixel of the gt_img
    velo_to_pxls = np.floor(pts_2d[fov_inds, :]).astype(int)

    # for each point in FOV of the image we retrieve the value of the pixel corresponding to the point
    back_rgb = img[velo_to_pxls[:, 1], velo_to_pxls[:, 0]]

    # initialize RGB feature vector
    rgb = np.zeros((len(points), 3))

    # for points in the FOV we assign RGB values
    rgb[fov_inds] = back_rgb

    return rgb
