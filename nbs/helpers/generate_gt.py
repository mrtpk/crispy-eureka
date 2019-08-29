import os
import numpy as np
from .calibration import Calibration
from .data_loaders import load_bin_file, get_image, get_dataset
from .calibration import get_lidar_in_image_fov


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
    pc_new_dir = '/'.join(pc_file_split[:-1]).replace('/velodyne', '/gt_velodyne')
    os.makedirs(pc_new_dir, exist_ok=True)

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


if __name__ == "__main__":
    PATH = '../../'
    generate_gt(os.path.abspath(PATH))
