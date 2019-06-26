# Helper functions to load the data for experiments
# load_bin_file: loads point cloud stored in binary file
# load_np_file: read the numpy file into memory

# func to load the gt from KITTI and LODNN
# get_image : load the image specified in path

import numpy as np
import cv2
from glob import glob

def load_bin_file(bin_path, n_cols=4):
    '''
    Load a binary file and convert it into a numpy array.
    '''
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, n_cols)

def process_kitti_pc(path, func, is_training=True):
    '''
    Process point cloud from KITTI dataset using given function
    '''
    folder = 'training'
    file_no = 289
    if is_training is False:
        folder = 'testing'
        file_no = 290
    pc_path = '/dataset/KITTI/dataset/data_road_velodyne/?/velodyne/*.bin'.replace('?', folder)
    pc_paths = sorted(glob(path + pc_path))
    assert len(pc_paths) == file_no 
    result = []
    counter = 0
    for path in pc_paths:
        pc = load_bin_file((path))
        result.append(func(pc))
        counter += 1
    assert counter == file_no
    return result

def get_image(path, is_color=True, rgb=False):
    if is_color is False:
        return cv2.imread(path, 0)
    img = cv2.imread(path)
    if rgb is False:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_kitti_img(path, func, is_training=True, gt=None):
    '''
    Process image from KITTI dataset using given function
    @param gt can be [None, kitti, lodnn]
    '''
    assert gt in [None, 'kitti','lodnn']
    assert is_training in [True, False]
    img_path = '/dataset/KITTI/dataset/data_road/training/image_2/*.png'
    file_no = 289
    if is_training is False:
        img_path = '/dataset/KITTI/dataset/data_road/testing/image_2/*.png'
        file_no = 290
    
    if is_training and gt == 'kitti':
        img_path = '/dataset/KITTI/dataset/data_road/training/gt_bev/*.png'
    
    if is_training and gt == 'lodnn':
        img_path = '/dataset/LoDNN/dataset/gt/*.png'
    
    img_paths = sorted(glob(path + img_path))
    assert len(img_paths) == file_no 
    result = []
    counter = 0
    for path in img_paths:
        img = get_image(path, is_color=True, rgb=False)
        result.append(func(img))
        counter += 1
    assert counter == file_no
    return result

def normalize(a, min, max, scale_min=0, scale_max=255, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to scale_min-scale_max
        Optionally specify the data type of the output
    """
    return (scale_min + (((a - min) / float(max - min)) * scale_max)).astype(dtype)