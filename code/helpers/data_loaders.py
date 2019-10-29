# Helper functions to load the data for experiments
# load_bin_file: loads point cloud stored in binary file
# load_np_file: read the numpy file into memory

# func to load the gt from KITTI and LODNN
# get_image : load the image specified in path

import numpy as np
import pandas as pd
import cv2
from glob import glob
import os
import copy
import yaml
from pyntcloud import PyntCloud
from tqdm import tqdm
import multiprocessing as mp
# from . import calibration # from helpers.calibration import Calibration
from joblib import Memory
cachedir = 'cachedir/'
memory = Memory(cachedir, verbose=0)

def create_dir_struct(path, name):
    '''
    Creates necessary directory structure for doing the experiments in 
    specified @param: path with @param: name
    '''
    experiment_dir = "{}/code/output/{}".format(path, name)
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

def get_semantic_kitti_dataset(path, is_training=True, sequences=None):

    # semantic Kitti basedir. TODO: decide if add this to parameters of the function

    sem_kitti_basedir = 'dataset/SemanticKITTI/dataset/sequences'
    train_sequences = ["{:02}".format(i) for i in range(11)]  ## according to semantic-kitti.yaml
    train_sequences.remove('08')
    # test_sequences = ["{:02}".format(i) for i in range(11, 22, 1)]
    test_sequences = ["08"]

    # initialize dataset
    dataset = {'imgs': [], 'calib': [], 'gt': [], 'gt_bev': [], 'gt_front': [], 'lodnn_gt': [], 'pc': [], 'labels':[]}

    imgs_dir = 'image_2'
    pc_dir = 'velodyne'
    label_dir = 'labels'
    gt_front_dir = 'gt_front'
    gt_bev_dir = 'gt_bev'

    if sequences is None:
        # if sequence is none we take all the sequences
        sequences = ["{:02}".format(i) for i in range(11)]

    if is_training is True:
        testset, validset, trainset = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)

        for s in sequences:
            imgs_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, imgs_dir) + '/*.png'))
            pc_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, pc_dir) + '/*.bin'))
            label_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, label_dir) + '/*.label'))
            gt_bev_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_bev_dir) + '/*.png'))
            gt_front_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_front_dir) + '/*.png'))
            calib_list = len(os.listdir(os.path.join(path, sem_kitti_basedir, s, imgs_dir))) * \
                         [ os.path.join(path, sem_kitti_basedir, s) + '/calib.txt' ]

            # the first 80% of the cases are putted in the train the remaining 20% are used for validation
            num_frames_in_seq = len(imgs_list)
            n_frames_in_train = np.floor(num_frames_in_seq * 0.8).astype(int)

            if s in train_sequences:
                trainset['imgs'] += imgs_list[:n_frames_in_train]
                trainset['pc'] += pc_list[:n_frames_in_train]
                trainset['labels'] += label_list[:n_frames_in_train]
                trainset['gt_bev'] +=  gt_bev_list[:n_frames_in_train]
                trainset['gt_front'] += gt_front_list[:n_frames_in_train]
                trainset['calib'] += calib_list[:n_frames_in_train]

                validset['imgs'] += imgs_list[n_frames_in_train:]
                validset['pc'] += pc_list[n_frames_in_train:]
                validset['labels'] += label_list[n_frames_in_train:]
                validset['gt_bev'] += gt_bev_list[n_frames_in_train:]
                validset['gt_front'] += gt_front_list[n_frames_in_train:]
                validset['calib'] += calib_list[n_frames_in_train:]

            # if s in test_sequences:
            #     testset['imgs'] += imgs_list[::5]
            #     testset['pc'] += pc_list[::5]
            #     testset['gt_bev'] += gt_bev_list[::5]
            #     testset['gt_front'] += gt_front_list[::5]
            #     testset['calib'] += calib_list[::5]

            #     raise ValueError("Sequence {} not in the dataset".format(s))

        return  trainset, validset
    else:
        for s in sequences:
            imgs_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, imgs_dir) + '/*.png'))
            pc_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, pc_dir) + '/*.bin'))
            label_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, label_dir) + '/*.label'))
            gt_bev_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_bev_dir) + '/*.png'))
            gt_front_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_front_dir) + '/*.png'))
            calib_list = len(os.listdir(os.path.join(path, sem_kitti_basedir, s, imgs_dir))) * \
                         [ os.path.join(path, sem_kitti_basedir, s) + '/calib.txt' ]

            dataset['imgs'] += imgs_list
            dataset['pc'] += pc_list
            dataset['labels'] += label_list
            dataset['gt_bev'] += gt_bev_list
            dataset['gt_front'] += gt_front_list
            dataset['calib'] += calib_list


        return dataset


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
    dataset = {'imgs':[],'calib':[],'gt':[],'gt_bev':[], 'gt_front':[], 'lodnn_gt':[], 'pc':[]}

    img_path = '/dataset/KITTI/dataset/data_road/training/image_2/*.png'
    gt_path = '/dataset/KITTI/dataset/data_road/training/gt_image_2/*.png'
    calib_path = '/dataset/KITTI/dataset/data_road/training/calib/*.txt'
    pc_path = '/dataset/KITTI/dataset/data_road_velodyne/training/velodyne/*.bin'
    bev_gt_path = '/dataset/KITTI/dataset/data_road/training/gt_bev/*.png'
    front_gt_path = '/dataset/KITTI/dataset/data_road/training/gt_front/*.png'
    lodnn_path = '/dataset/LoDNN/dataset/gt/*_gt.png'

    img_paths = sorted(glob(path + img_path))
    sample_names = [x.split('/')[-1].split('.')[0] for x in img_paths]
    um, umm, uu = sample_names[0: 95], sample_names[95:191], sample_names[191:289]
    #how are train, test and valid sets created
    # basically in the next lines we:
    # 1 ) take from um, umm and uu lists 10 files each (total 30) and add those files to test list
    # 2 ) repeat the same procedure to remove other last 10 files each from um, umm and uu and add them to valid list
    test, valid, train = [], [], []
    for i in [test, valid]:
        um, _i = remove_last_n(um, 10)
        i.extend(_i)
        umm, _i = remove_last_n(umm, 10)
        i.extend(_i)
        uu, _i = remove_last_n(uu, 10)
        i.extend(_i)

    train = um + umm + uu  # train list is composed with the remaining files in lists um, umm, uu

    testset, validset, trainset = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)

    for name, _dataset in zip(['train', 'test', 'valid'], [trainset, testset, validset]):
        sample_names = eval(name)
        for sample_name in sample_names:
            _dataset['imgs'].append(path + img_path.replace('*', sample_name))
            _dataset['calib'].append(path + calib_path.replace('*', sample_name))
            _dataset['gt'].append(path + gt_path.replace('*', sample_name.replace("_", "_road_")))
            _dataset['gt_bev'].append(path + bev_gt_path.replace('*', sample_name.replace("_", "_road_")))
            _dataset['gt_front'].append(path + front_gt_path.replace('*', sample_name.replace("_", "_road_")))
            _dataset['lodnn_gt'].append(path + lodnn_path.replace('*', sample_name))
            _dataset['pc'].append(path + pc_path.replace('*', sample_name))

    if is_training:
        return trainset, validset
    else:
        return testset

def load_label_file(bin_path):
    labels = np.fromfile(bin_path, dtype=np.uint32).reshape(-1)
    labels = labels & 0xFFFF
    return labels

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

def process_list(pc_list, func, **kwargs):
    '''
    Process point cloud from KITTI dataset using given function
    '''
    nCPU = mp.cpu_count()
    print('nCPUs = ' + repr(nCPU))
    pool = mp.Pool(processes=nCPU)
    from functools import partial
    if len(kwargs):
        func = partial(func, **kwargs)
    result = pool.map(func, [pc for pc in pc_list]) 
    pool.close()
    return result

def load_pc(paths):
    '''
    Process point cloud from KITTI dataset using given function
    '''
    result = []
    for path in tqdm(paths):
        pc = load_bin_file((path))
        result.append(pc)
    return result

def load_img(paths):
    '''
    Process image at paths using given function
    '''
    result = []
    for path in paths:
        img = get_image(path, is_color=True, rgb=False)
        result.append(img)
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

def process_calib(paths):
    '''
    Initialize calibrations at paths
    '''
    result = []

    for path in paths:
        calib = calibration.Calibration(path)
        result.append(calib)

    return result

def normalize(a, min, max, scale_min=0, scale_max=255, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to scale_min-scale_max
        Optionally specify the data type of the output
    """
    return (scale_min + (((a - min) / float(max - min)) * (scale_max-scale_min))).astype(dtype)

def load_pyntcloud(filename):
    points = load_bin_file(filename)
    cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y','z', 'i']))
    return cloud

def load_semantic_kitt_config(filename=""):

    if len(filename) == 0:
        local_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(local_path, '..', 'semantic-kitti-api','config','semantic-kitti.yaml')

    with open(filename) as f:
        config = yaml.safe_load(f)

    return config