from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helpers import data_loaders as dls
from helpers import pointcloud as pc
from helpers.projection import Projection
from helpers.normals import estimate_normals_from_spherical_img

import numpy as np


class KittiPointCloudClass:
    """ 
    Quick hack, need a good pointcloud structure to perform feature extraction, truncation etc
    
    """
    def __init__(self, train_set, valid_set):
        
        self.train_set = train_set
        self.valid_set = valid_set

        z_vals = dls.process_pc(self.train_set["pc"] + self.valid_set["pc"], lambda x: x[:, 2])
        z_vals = np.concatenate(z_vals)
        self.z_min, self.z_max = np.min(z_vals), np.max(z_vals)
        print("Height ranges from {} to {}".format(self.z_min, self.z_max))
        
        self.side_range=(-10, 10) #this is fixed here for KITTI
        self.fwd_range=(6, 46) #this is fixed here for KITTI
        self.res=.1
    
        """
        Update with maximum points found within a cell to be used for normalization later
        """
        _filter = lambda x: pc.filter_points(x, side_range=self.side_range, fwd_range=self.fwd_range)
        f_count = dls.process_pc(self.train_set["pc"] + self.valid_set["pc"],
                       lambda x: _get_features(_filter(x))[:,:,0])
        f_count = np.concatenate(f_count)
        self.COUNT_MIN, self.COUNT_MAX = 0, np.max(f_count)
        print("Count varies from {} to {}".format(self.COUNT_MIN, self.COUNT_MAX))

    
    def get_features(self, points):
        """
        Remove points outisde y \in [-10, 10] and x \in [6, 46]
        """
        points = pc.filter_points(points, side_range=self.side_range, fwd_range=self.fwd_range)
        z = points[:, 2]
        z = (z - self.z_min)/(self.z_max - self.z_min)
        points[:, 2] = z
        #get all features and normalize count channel
        f = _get_features(points)
        f[:, :, 0] = f[:, :, 0] / self.COUNT_MAX
        return f

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

def _get_normals(points):
    '''
    Estimate surface normals on spherical image
    '''
    from skimage.morphology import dilation, square

    # azimuthal resolution --> defines spherical image height
    res_az = 100

    # planar resolution --> defines spherical image width
    res_planar = 300

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


def _get_features(points, add_geometrical_features=True):
    '''
    Returns features of the point cloud as stacked grayscale images.
    Shape of the output is (400x200x6).
    '''
    side_range=(-10, 10)
    fwd_range=(6, 46)
    res=.1

    # calculate the image dimensions
    img_width = int((side_range[1] - side_range[0])/res)
    img_height = int((fwd_range[1] - fwd_range[0])/res)
    number_of_grids = img_height * img_width

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]



    norm_z_lidar = z_lidar # assumed that the z values are normalised

    # MAPPING
    # Mappings from one point to grid 
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
    x_img_mapping = (-y_lidar/res).astype(np.int32) # x axis is -y in LIDAR
    y_img_mapping = (x_lidar/res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img_mapping -= int(np.floor(side_range[0]/res))
    y_img_mapping -= int(np.floor(fwd_range[0]/res))

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

    if add_geometrical_features:
        # estimate normals
        normals = _get_normals(points[:, :3])

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
    return out_feature_map
