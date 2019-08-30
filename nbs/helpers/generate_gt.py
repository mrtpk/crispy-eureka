import os
import numpy as np
from PIL import Image
from .calibration import Calibration, get_lidar_in_image_fov
from .data_loaders import load_bin_file, get_image, get_dataset
from skimage.morphology import closing, square


def retrieve_layers(points):
    '''
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.
    '''
    x = points[:, 0]
    y = points[:, 1]

    # compute the theta angles
    thetas = np.arctan2(y, x)
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

def project_points_on_camera_img(points, img, calib):
    """
    Function that  project point cloud on fron view image and retrieve RGB information

    Parameters
    ----------
    points: ndarray
        point cloud
    img: ndarray
        camera image
    calib:  Calibration
        calibration object

    Returns
    -------
    points_to_pxls: ndarray
        table that associate to each point in point cloud the corresponding pixel in camera image
    """
    height, width = img.shape[:2]

    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(points, calib, 0, 0, width, height,
                                                              return_more=True)

    # the following array contains for each point in the point cloud the corrisponding pixel of the gt_img
    velo_to_pxls = np.floor(pts_2d[fov_inds, :]).astype(int)

    points_to_pxls = -np.ones((len(points), 2), dtype=int)

    points_to_pxls[fov_inds] = velo_to_pxls[:, [1, 0]]

    return points_to_pxls


def project_on_bev_img(points):
    """
    Function that project point cloud to bev image

    Parameters
    ----------
    points: ndarray
        input point cloud

    Returns
    -------
    points_to_bev_pxls: ndarray
        table representing for each point the corresponding pixel in the bev image
    """
    side_range = (-10, 10)
    fwd_range = (6, 46)
    res = .1
    # calculate the image dimensions
    img_height = int((fwd_range[1] - fwd_range[0]) / res)
    # number_of_grids = img_height * img_width

    x_lidar = points[:, 0] - fwd_range[0]
    y_lidar = -points[:, 1] - side_range[0]

    # MAPPING
    # Mappings from one point to grid
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
    x_img_mapping = (y_lidar / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img_mapping = img_height - (x_lidar / res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

    points_to_bev_pxls = np.c_[y_img_mapping, x_img_mapping]

    return points_to_bev_pxls


def in_img_frame(pxl, shape):
    """
    Function that check if pixel is in the image frame

    Parameters
    ----------
    pxl: tuple
        pixels coordinates

    shape: tuple
        image shape

    Returns
    -------

    check: bool
        True if pxl is in image frame. False otherwise

    """
    i, j = pxl
    nr, nc = shape[:2]
    if (i >= 0) and (i < nr) and (j >= 0) and (j < nc):
        return True
    else:
        return False


def generate_point_cloud_gt(points, gt_img, calib):
    """
    Function used to project gt from camera image to point cloud

    Parameters
    ----------
    points: ndarray
        input point cloud

    gt_img: ndarray
        Ground truth camera image

    calib: Calibration
        Calbration object used to map point cloud to RGB image

    Returns
    -------
    points: ndarray
        point cloud with gt

    """
    points_to_cam_pxl = project_points_on_camera_img(points[:, :3], gt_img, calib)
    cam_labels = np.zeros(len(points), dtype=np.uint8)
    bev_labels = np.zeros(len(points), dtype=np.uint8)

    for n in range(len(points)):
        if points_to_cam_pxl[n, 0] > 0:
            cam_labels[n] = max(gt_img[points_to_cam_pxl[n, 0], points_to_cam_pxl[n, 1], 0] // 255, cam_labels[n])

    labels = np.logical_or(cam_labels, bev_labels)

    return np.c_[points, labels]


def project_gt_to_front(points, label_idx = -1):
    """
    Function that project gt to front (i.e. spherical) view image

    Parameters
    ----------
    points: ndarray
        input point cloud containing labels to project
    label_idx: int
        index of the columns relative to labels in the points table array

    Returns
    -------
    gt_img: ndarray
        ground truth spherical image
    """


    # todo: parametrize the following variables
    img_height = 64
    res_planar = 300
    img_width = np.ceil(np.pi * res_planar).astype(int)
    # initialize gt_img
    gt_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    gt_img[:,:,0] = 255
    # get points coordinates
    x = points[:, 0]
    y = points[:, 1]
    labels = points[:, label_idx]
    # compute planar angles
    planar_angles = np.arctan2(-y, x) + np.pi / 2

    # get pixel coordinates for each point in the cloud
    i = retrieve_layers(points)
    j = np.floor( planar_angles * res_planar).astype(int)

    idx = np.logical_and((planar_angles >= 0), (planar_angles < np.pi))

    for n in range(len(points)):
        if idx[n]:
            gt_img[i[n], j[n], 2] = 255 * labels[n].astype(np.uint8)

    cl_gt_img_2 = closing(gt_img[:,:,2], square(3))
    cl_gt_img = gt_img.copy()
    cl_gt_img[:,:,2] = cl_gt_img_2

    gt_img = Image.fromarray(cl_gt_img.astype(np.uint8))

    return gt_img


def generate_gt(path):
    """
    Function that loops over all the database and add gt to point clouds

    Parameters
    ----------
    path: str
        path to database root
    """
    train_set, valid_set, test_set = get_dataset(path, is_training=True)

    # joining all the lists together
    pc_fileslist = train_set['pc'] + valid_set['pc'] + test_set['pc']
    gt_fileslist = train_set['gt'] + valid_set['gt'] + test_set['gt']
    gt_bev_fileslist = train_set['gt_bev'] + valid_set['gt_bev'] + test_set['gt_bev']
    calib_fileslist = train_set['calib'] + valid_set['calib'] + test_set['calib']

    pc_file = train_set['pc'][0]
    pc_file_split = pc_file.split('/')
    gt_file = train_set['gt_bev'][0]
    gt_file_split = gt_file.split('/')
    pc_new_dir = '/'.join(pc_file_split[:-1]).replace('/velodyne', '/gt_velodyne')
    gt_front_new_dir = '/'.join(gt_file_split[:-1]).replace('bev', 'front')
    os.makedirs(pc_new_dir, exist_ok=True)
    os.makedirs(gt_front_new_dir, exist_ok=True)

    print(pc_new_dir)

    for pc_path, gt_img_path, gt_bev_path, calib_path in zip(pc_fileslist,
                                                             gt_fileslist,
                                                             gt_bev_fileslist,
                                                             calib_fileslist):
        # loading data from file
        pc = load_bin_file(pc_path)
        gt_img = get_image(gt_img_path, is_color=True, rgb=False)
        calib = Calibration(calib_path)
        # retrieving labels for point clouds
        points = generate_point_cloud_gt(pc, gt_img, calib)
        filename = pc_path.split('/')[-1]
        print(os.path.join(pc_new_dir, filename))
        points.astype(np.float32).tofile(os.path.join(pc_new_dir, filename))
        gt_front_img = project_gt_to_front(points, label_idx=-1)
        gt_front_img.save(os.path.join(gt_front_new_dir, gt_bev_path.split('/')[-1]))

if __name__ == "__main__":
    PATH = '../../'
    generate_gt(os.path.abspath(PATH))
