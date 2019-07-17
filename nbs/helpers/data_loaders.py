# Helper functions to load the data for experiments
# load_bin_file: loads point cloud stored in binary file
# load_np_file: read the numpy file into memory

# func to load the gt from KITTI and LODNN
# get_image : load the image specified in path

import numpy as np
import cv2
from glob import glob
import os
import copy

def create_dir_struct(path, name):
    '''
    Creates necessary directory structure for doing the experiments in 
    specified @param: path with @param: name
    '''
    experiment_dir = "{}/nbs/output/{}".format(path, name)
    tmp_dir = "{}/tmp/".format(experiment_dir)
    log_dir = "{}/log/".format(experiment_dir)
    os.makedirs(os.path.dirname(tmp_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    run_path = "{}/*run_id/".format(experiment_dir)
    
    return experiment_dir, log_dir, tmp_dir, run_path

def remove_last_n(lst, n):
    '''
    Removes last @param: n elements from list @param: lst
    Returns modified list after deleting n elements and deleted elements
    '''
    return lst[:-n or None], lst[-n:]

def get_dataset(path, is_training=True):
    '''
    Returns training, test, and validation sets from KITTI training set
    if @param is_training is True. If false returns testing set of KITTI.

    For training, test set and validation set is created by taking 10 samples
    from each category(umm, um, uu) of KITTI dataset. Test set and validation
    set has 30 samples each and training set 229 samples. Each set has images,
    KITTI ground truths, BEV of KITTI ground truths, LoDNN ground truths,
    calibration files, and point clouds.
    '''
    if is_training is True:
        cat_um = (0, 95, 95)
        cat_umm = (95, 191, 96)
        cat_uu = (191, 289, 98)
        dataset = {'imgs':[],'calib':[],'gt':[],'gt_bev':[],'lodnn_gt':[], 'pc':[]}
        
        img_path = '/dataset/KITTI/dataset/data_road/training/image_2/*.png'
        gt_path = '/dataset/KITTI/dataset/data_road/training/gt_image_2/*.png'
        calib_path = '/dataset/KITTI/dataset/data_road/training/calib/*.txt'
        pc_path = '/dataset/KITTI/dataset/data_road_velodyne/training/velodyne/*.bin'
        bev_gt_path = '/dataset/KITTI/dataset/data_road/training/gt_bev/*.png'
        lodnn_path = '/dataset/LoDNN/dataset/gt/*_gt.png'
        
        img_paths = sorted(glob(path + img_path))      
        sample_names = [x.split('/')[-1].split('.')[0] for x in img_paths]
        um, umm, uu = sample_names[0: 95], sample_names[95:191], sample_names[191:289]
        test, valid, train = [], [], []
        for i in [test, valid]:
            um, _i = remove_last_n(um, 10)
            i.extend(_i)
            umm, _i = remove_last_n(umm, 10) 
            i.extend(_i)
            uu, _i = remove_last_n(uu, 10)
            i.extend(_i)
        train = um + umm + uu

        testset, validset, trainset = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
        
        for name, _dataset in zip(['train', 'test', 'valid'], [trainset, testset, validset]):
            sample_names = eval(name)
            for sample_name in sample_names:
                _dataset['imgs'].append(path + img_path.replace('*', sample_name))
                _dataset['calib'].append(path + calib_path.replace('*', sample_name))
                _dataset['gt'].append(path + gt_path.replace('*', sample_name.replace("_", "_road_")))
                _dataset['gt_bev'].append(path + bev_gt_path.replace('*', sample_name.replace("_", "_road_")))
                _dataset['lodnn_gt'].append(path + lodnn_path.replace('*', sample_name))
                _dataset['pc'].append(path + pc_path.replace('*', sample_name))
                
        return trainset, validset, testset
    else:
        # category
        # um - 0:96
        # umm - 96:190
        # uu - 190: 290
        img_path = '/dataset/KITTI/dataset/data_road/testing/image_2/*.png'
        calib_path = '/dataset/KITTI/dataset/data_road/testing/calib/*.txt'
        pc_path = '/dataset/KITTI/dataset/data_road_velodyne/testing/velodyne/*.bin'
        img_paths = sorted(glob(path + img_path))
        calib_paths = sorted(glob(path + calib_path))
        pc_paths = sorted(glob(path + pc_path))
        return {'imgs':img_paths,'calib':calib_paths,'gt':[],'gt_bev':[],'lodnn_gt':[], 'pc':pc_paths}

def load_bin_file(bin_path, n_cols=4):
    '''
    Load a binary file and convert it into a numpy array.
    '''
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, n_cols)

def get_image(path, is_color=True, rgb=False):
    if is_color is False:
        return cv2.imread(path, 0)
    img = cv2.imread(path)
    if rgb is False:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_pc(paths, func):
    '''
    Process point cloud from KITTI dataset using given function
    '''
    result = []
    counter = 0
    for path in paths:
        pc = load_bin_file((path))
        result.append(func(pc))
        counter += 1
    return result

def process_img(paths, func):
    '''
    Process image at paths using given function
    '''
    result = []
    counter = 0
    for path in paths:
        img = get_image(path, is_color=True, rgb=False)
        result.append(func(img))
        counter += 1
    return result

def normalize(a, min, max, scale_min=0, scale_max=255, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to scale_min-scale_max
        Optionally specify the data type of the output
    """
    return (scale_min + (((a - min) / float(max - min)) * (scale_max-scale_min))).astype(dtype)

