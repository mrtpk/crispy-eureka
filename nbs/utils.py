import os
import cv2
from time import time
import numpy as np
#import copy
#import pathlib
import datetime
from helpers import data_loaders as dls
from helpers import pointcloud as pc
from helpers.projection import Projection
from helpers.normals import estimate_normals_from_spherical_img
from helpers.calibration import get_lidar_in_image_fov
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import keras
from keras.callbacks import TensorBoard
from tqdm import tqdm
import multiprocessing as mp

#from keras.callbacks import ModelCheckpoint, LearningRateScheduler

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
    return [tensorboard, save_the_best, save_after_epoch]


def apply_argmax(res):
    return 1 - np.argmax(res, axis=2)

def get_z(points):
#    return points[:,2].reshape(-1,1)
    return np.array([np.min(points[:,2]), np.max(points[:,2])])

def get_first_chan(points):
    return points[:,:,0]

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
    def __init__(self, dataset_path, add_geometrical_features, subsample, compute_HOG):
        # get dataset
        self.train_set, self.valid_set, self.test_set = dls.get_dataset(dataset_path, is_training=True)
        self.add_geometrical_features=add_geometrical_features #Flag
        self.subsample = subsample #Flag
        self.compute_HOG = compute_HOG  # Flag
        self.side_range=(-10, 10) #this is fixed here for KITTI
        self.fwd_range=(6, 46) #this is fixed here for KITTI
        self.res=.1

    def subsample_pc(self, points, sub_ratio=2):
        '''
        Return sub sampled point cloud
        '''
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        aux_sum = np.sqrt(np.square(x) + np.square(y))
        # azimuthal angles
        phis = np.arctan2(z, aux_sum)
        # max and min values of the azimuthal angles
        phi_center_max, phi_center_min = phis.max() - (0.2 / 180.0 * np.pi), phis.min() + (0.2 / 180.0 * np.pi)

        # angle of corresponding to inclination of layers
        phi_centers = np.linspace(phi_center_min, phi_center_max, 64)

        # compute first nearest neighbor for each phis
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(phi_centers.reshape(-1, 1))

        # for each azimuthal angle we compute its nearest neighbors in the phi_centers vector
        # in this way at each points we associate a layer of the velodyne
        _, ind = nbrs.kneighbors(phis.reshape(-1, 1))

        ind = ind.reshape(-1)

        # sampling only points with even id
        return points[ind % sub_ratio == 0]

    def _project_points_on_img(self, points, img, calib):

        '''
        Function that  project point cloud on fron view image and retrieve RGB information
        '''
        height, width = img.shape[:2]

        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points, calib, 0, 0, width, height,
                                                                  return_more=True)

        # the following array contains for each point in the point cloud the corrisponding pixel of the gt_img
        velo_to_pxls = np.floor(pts_2d[fov_inds, :]).astype(int)

        points_to_pxls = np.ones((len(points), 2), dtype=int)

        points_to_pxls[fov_inds] = velo_to_pxls[:, [1, 0]]

        return points_to_pxls

    def get_RGB(self, points, img, calib):
        '''
        Function that  project point cloud on fron view image and retrieve RGB information
        '''
        height, width = img.shape[:2]

        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points, calib, 0, 0, width, height,
                                                                  return_more=True)

        # the following array contains for each point in the point cloud the corrisponding pixel of the gt_img
        velo_to_pxls = np.floor(pts_2d[fov_inds, :]).astype(int)

        # for each point in FOV of the image we retrieve the value of the pixel corresponding to the point
        back_RGB = img[velo_to_pxls[:, 1], velo_to_pxls[:, 0]]

        # initialize RGB feature vector
        RGB = np.zeros((len(points), 3))

        # for points in the FOV we assign RGB values
        RGB[fov_inds] = back_RGB

        return RGB

    def get_HOG(self, points, img, calib):
        '''
        A basic function that compute HOG features on the front image
        '''
        points_to_pxls = self._project_points_on_img(points, img, calib)

        # parameters for HOGDescriptor
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64

        num_features = nbins * ( (winSize[0] // cellSize[0] - 1) * (winSize[1] // cellSize[1] - 1) * 4)

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8, 8)
        padding = (8, 8)

        # padding img to compute HOG features
        pad_img = np.pad(img, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), 'constant')

        # compute hog features
        hog_features = np.zeros((len(points), num_features))

        idx = points_to_pxls[:, 0] >= 0  # getting idx of points projected on img
        # retrieve location for each point
        locations = tuple([tuple(x) for x in (points_to_pxls[idx] +  np.array(padding)).tolist()])
        # compute histogram
        hist = hog.compute(pad_img, winStride, padding, locations)

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

    def get_dataset(self, limit_index = 3):
        print(self.train_set.keys())
        """ NOTE: change limit_index to -1 to train on the whole dataset """
        print('Reading cloud')
        f_train = dls.load_pc(self.train_set["pc"][0:limit_index])
        f_valid = dls.load_pc(self.valid_set["pc"][0:limit_index])
        f_test = dls.load_pc(self.test_set["pc"][0:limit_index])

        print('Reading calibration files')
        cal_train = dls.process_calib(self.train_set["calib"][0:limit_index])
        cal_valid = dls.process_calib(self.valid_set["calib"][0:limit_index])
        cal_test = dls.process_calib(self.test_set["calib"][0:limit_index])

        print('Reading camera images')
        cam_img_train = dls.load_img(self.train_set["imgs"][0:limit_index])
        cam_img_valid = dls.load_img(self.valid_set["imgs"][0:limit_index])
        cam_img_test = dls.load_img(self.test_set["imgs"][0:limit_index])

        #Update min max z
        z_vals = dls.process_list(f_train + f_valid, get_z)
        z_vals = np.concatenate([z_vals])
        print(z_vals.shape)
        self.z_min, self.z_max = np.min(z_vals[:,0]), np.max(z_vals[:,1])
        print("Height ranges from {} to {}".format(self.z_min, self.z_max))
        
        if self.subsample:
            print('Read and Subsample cloud')
            t = time()
            f_train = dls.process_list(f_train, self.subsample_pc)
            f_valid = dls.process_list(f_valid, self.subsample_pc)
            f_test = dls.process_list(f_test, self.subsample_pc)
            print('Evaluated in : '+repr(time()-t))
        #Update with maximum points found within a cell to be used for normalization later
        print('Evaluating count')
        self.COUNT_MIN = 0 
        self.COUNT_MAX = []
        for _f in f_train+f_test:
            _filtered = pc.filter_points(_f, side_range=self.side_range, 
                                         fwd_range=self.fwd_range)
            f_count = self._get_count_features(_filtered)
            self.COUNT_MAX.append(int(np.max(f_count)))
        self.COUNT_MAX = max(self.COUNT_MAX)
        print("Count varies from {} to {}".format(self.COUNT_MIN, self.COUNT_MAX))    
            
        print('Extracting features')
        t = time()

        f_cam_calib_train = [(f_t, img, calib) for f_t, img, calib in zip(f_train, cam_img_train, cal_train)]
        f_cam_calib_valid = [(f_t, img, calib) for f_t, img, calib in zip(f_valid, cam_img_valid, cal_valid)]
        f_cam_calib_test = [(f_t, img, calib) for f_t, img, calib in zip(f_test, cam_img_test, cal_test)]


        f_train = dls.process_list(f_cam_calib_train, self.get_features)
        f_valid = dls.process_list(f_cam_calib_valid, self.get_features)
        f_test = dls.process_list(f_cam_calib_test, self.get_features)
        print('Evaluated in : '+repr(time()-t))
        gt_train = dls.process_img(self.train_set["gt_bev"][0:limit_index], func=lambda x: kitti_gt(x))
        gt_valid = dls.process_img(self.valid_set["gt_bev"][0:limit_index], func=lambda x: kitti_gt(x))
        gt_test = dls.process_img(self.test_set["gt_bev"][0:limit_index], func=lambda x: kitti_gt(x))
        return np.array(f_train), np.array(f_valid), np.array(f_test), np.array(gt_train), np.array(gt_valid), np.array(gt_test)

    def get_features(self, raw_info):
        """
        Remove points outisde y \in [-10, 10] and x \in [6, 46]
        """
        points = raw_info[0]
        img = raw_info[1]
        calib = raw_info[2]
        points = pc.filter_points(points, side_range=self.side_range, fwd_range=self.fwd_range)
        z = points[:, 2]
        z = (z - self.z_min)/(self.z_max - self.z_min)
        points[:, 2] = z

        #get all features and normalize count channel
        f = self._get_features(points, img, calib)
        f[:, :, 0] = f[:, :, 0] / self.COUNT_MAX
        return f
    
    def _get_count_features(self, points):
        '''
        Returns features of the point cloud as stacked grayscale images.
        Shape of the output is (400x200x6).
        '''
#        side_range=(-10, 10)
#        fwd_range=(6, 46)
#        res=.1

        # calculate the image dimensions
        img_width = int((self.side_range[1] - self.side_range[0])/self.res)
        img_height = int((self.fwd_range[1] - self.fwd_range[0])/self.res)
        number_of_grids = img_height * img_width

        x_lidar = points[:, 0]
        y_lidar = points[:, 1]
        z_lidar = points[:, 2]

        norm_z_lidar = z_lidar # assumed that the z values are normalised
        
        # MAPPING
        # Mappings from one point to grid 
        # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
        x_img_mapping = (-y_lidar/self.res).astype(np.int32) # x axis is -y in LIDAR
        y_img_mapping = (x_lidar/self.res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img_mapping -= int(np.floor(self.side_range[0]/self.res))
        y_img_mapping -= int(np.floor(self.fwd_range[0]/self.res))

        # Linerize the mappings to 1D
        lidx = ((-y_img_mapping) % img_height) * img_width + x_img_mapping

        # Feature extraction
        # count of points per grid
        count_input = np.ones_like(norm_z_lidar)
        binned_count = np.bincount(lidx, count_input, minlength = number_of_grids)        
        norm_binned_count = binned_count # normalise the output
        # reshape all other things
        o_count            = norm_binned_count.reshape(img_height, img_width)
        return o_count
    
    def _get_features(self, points, img=None, calib=None):
        '''
        Returns features of the point cloud as stacked grayscale images.
        Shape of the output is (400x200x6).
        '''
#        side_range=(-10, 10)
#        fwd_range=(6, 46)
#        res=.1

        # calculate the image dimensions
        img_width = int((self.side_range[1] - self.side_range[0])/self.res)
        img_height = int((self.fwd_range[1] - self.fwd_range[0])/self.res)
        number_of_grids = img_height * img_width

        x_lidar = points[:, 0]
        y_lidar = points[:, 1]
        z_lidar = points[:, 2]
        r_lidar = points[:, 3]

        norm_z_lidar = z_lidar # assumed that the z values are normalised
        
        # MAPPING
        # Mappings from one point to grid 
        # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
        x_img_mapping = (-y_lidar/self.res).astype(np.int32) # x axis is -y in LIDAR
        y_img_mapping = (x_lidar/self.res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img_mapping -= int(np.floor(self.side_range[0]/self.res))
        y_img_mapping -= int(np.floor(self.fwd_range[0]/self.res))

        # Linerize the mappings to 1D
        lidx = ((-y_img_mapping) % img_height) * img_width + x_img_mapping

        # Feature extraction
        # count of points per grid
        count_input = np.ones_like(norm_z_lidar)
        binned_count = np.bincount(lidx, count_input, minlength = number_of_grids)
        # sum reflectance
        binned_reflectance =  np.bincount(lidx, r_lidar, minlength = number_of_grids)

        # sum elevation 
        binned_elevation = np.bincount(lidx, norm_z_lidar, minlength = number_of_grids)

        # Finding mean!
        binned_mean_reflectance = np.divide(binned_reflectance, binned_count, out=np.zeros_like(binned_reflectance), where=binned_count!=0.0)
        binned_mean_elevation = np.divide(binned_elevation, binned_count, out=np.zeros_like(binned_elevation), where=binned_count!=0.0)
        o_mean_elevation = binned_mean_elevation.reshape(img_height, img_width)

        # Standard devation stuff
        binned_sum_var_elevation = np.bincount(lidx, np.square(norm_z_lidar - o_mean_elevation[-y_img_mapping, x_img_mapping]), minlength = number_of_grids)
        binned_divide = np.divide(binned_sum_var_elevation, binned_count, out=np.zeros_like(binned_sum_var_elevation), where=binned_count!=0.0)
        binned_std_elevation = np.sqrt(binned_divide)

        # minimum and maximum
        sidx = lidx.argsort()
        idx = lidx[sidx]
        val = norm_z_lidar[sidx]

        m_idx = np.flatnonzero(np.r_[True,idx[:-1] != idx[1:]])
        unq_ids = idx[m_idx]

        o_max_elevation = np.zeros([img_height, img_width], dtype=np.float64)
        o_min_elevation = np.zeros([img_height, img_width], dtype=np.float64)

        o_max_elevation.flat[unq_ids] = np.maximum.reduceat(val, m_idx)
        o_min_elevation.flat[unq_ids] = np.minimum.reduceat(val, m_idx)

        norm_binned_count = binned_count # normalise the output
        # reshape all other things
        o_count            = norm_binned_count.reshape(img_height, img_width)
        o_mean_reflectance = binned_mean_reflectance.reshape(img_height, img_width)
        o_std_elevation    = binned_std_elevation.reshape(img_height, img_width)

        if self.add_geometrical_features:
            if self.subsample:
                res_az = 50
            else:
                res_az = 100
            # estimate normals
            normals = _get_normals(points[:, :3], res_az=res_az, res_planar=300)

            nx_lidar = normals[:, 0]
            ny_lidar = normals[:, 1]
            nz_lidar = normals[:, 2]

            # sum normals
            binned_nx = np.bincount(lidx, nx_lidar, minlength=number_of_grids)
            binned_ny = np.bincount(lidx, ny_lidar, minlength=number_of_grids)
            binned_nz = np.bincount(lidx, nz_lidar, minlength=number_of_grids)

            binned_mean_nx = np.divide(binned_nx, binned_count, out=np.zeros_like(binned_nx), where=binned_count != 0.0)
            binned_mean_ny = np.divide(binned_ny, binned_count, out=np.zeros_like(binned_ny), where=binned_count != 0.0)
            binned_mean_nz = np.divide(binned_nz, binned_count, out=np.zeros_like(binned_nz), where=binned_count != 0.0)

            o_mean_nx = binned_mean_nx.reshape(img_height, img_width)
            o_mean_ny = binned_mean_ny.reshape(img_height, img_width)
            o_mean_nz = binned_mean_nz.reshape(img_height, img_width)

            out_feature_map = np.dstack([o_count,
                                        o_mean_reflectance,
                                        o_max_elevation,
                                        o_min_elevation,
                                        o_mean_elevation,
                                        o_std_elevation,
                                        o_mean_nx,
                                        o_mean_ny,
                                        o_mean_nz])

        else:
            out_feature_map = np.dstack([o_count,
                                        o_mean_reflectance,
                                        o_max_elevation,
                                        o_min_elevation,
                                        o_mean_elevation,
                                        o_std_elevation])

        if self.compute_HOG:
            # extract hog features from camera image
            hog_features = self.get_HOG(points[:,:3], img, calib)

            hog_features_map = np.zeros((img_height, img_width, hog_features.shape[1]))

            for i in range(hog_features.shape[1]):
                binned_i = np.bincount(lidx, hog_features[:,i], minlength=number_of_grids)
                binned_mean_i = np.divide(binned_i, binned_count, out=np.zeros_like(binned_i), where=binned_count != 0.0)
                hog_features_map[:,:,i] = binned_mean_i.reshape(img_height, img_width)

            out_feature_map = np.dstack([out_feature_map, hog_features_map])

        return out_feature_map

def get_metrics_count(pred, gt):
    '''
    Get true positive, true negative, false positive, false negative counts
    '''
    diff = pred - gt
    # get tp, tn, fp, fn
    fn = diff[diff==-1].size
    fp = diff[diff==1].size
    tp = diff[gt==1].size - fn
    tn = diff[diff==0].size - tp
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
    road = img[:, :, 0] / 255 # Road is encoded as 255 in the B plane
    non_road = 1 - road # TODO: can we do this in training time?
    return np.dstack([road, non_road])

# to add, get ground truth from lodnn dataset

def _get_normals(points, res_az=100, res_planar=300):
    '''
    Estimate surface normals on spherical image

    # azimuthal resolution --> defines spherical image height
    res_az = 100

    # planar resolution --> defines spherical image width
    res_planar = 300

    '''
    from skimage.morphology import dilation, square


    # initializing projector
    proj = Projection(proj_type='spherical', res_azimuthal=res_az, res_planar=res_planar)

    # on spherical image we project points lenght
    values = np.linalg.norm(points, axis=1)

    # get spherical image
    rho_img = proj.project_points_values(points, values, res_values=1, dtype=np.float)

    # dilate spherical image to interpolate information where no points is projected
    dil_rho = dilation(rho_img, square(3))

    rho_img[rho_img==0] = dil_rho[rho_img==0]

    # image that contains normals
    normals = estimate_normals_from_spherical_img(rho_img, res_planar=res_planar, res_az=res_az, res_rho=1)

    # project back normals to point cloud
    pc_normals = proj.back_project(points, np.abs(normals), res_value=1, dtype=np.float)

    return pc_normals
