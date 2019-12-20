# Helper functions to load the data for experiments
# load_bin_file: loads point cloud stored in binary file
# load_np_file: read the numpy file into memory

# func to load the gt from KITTI and LODNN
# get_image : load the image specified in path

import random
import h5py
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
from .pointcloud import filter_points
# from . import calibration # from helpers.calibration import Calibration
# from joblib import Memory
# cachedir = 'cachedir/'
# memory = Memory(cachedir, verbose=0)


def create_dir_struct(path, name):
    """
    Creates necessary directory structure for doing the experiments in 
    specified @param: path with @param: name
    """
    experiment_dir = "{}/code/output/{}".format(path, name)
    tmp_dir = "{}/tmp/".format(experiment_dir)
    log_dir = "{}/log/".format(experiment_dir)
    os.makedirs(os.path.dirname(tmp_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    run_path = "{}/*run_id/".format(experiment_dir)
    
    return experiment_dir, log_dir, tmp_dir, run_path


def remove_last_n(lst, n):
    """
    Removes last @param: n elements from list @param: lst
    Returns modified list after deleting n elements and deleted elements
    """
    return lst[:-n or None], lst[-n:]


def fetch_file_list(path, extension):
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, "*." + extension)))
    else:
        return []

def shuffle_dict(dictionary, seed=1):
    """
    Function that take a dictionary and return a shuffled version

    Parameters
    ----------
    dictionary: dict
        input dictionary to shuffle

    seed: int
        seed value

    Returns
    -------
    shuffled_dict: dict
    """
    shuffled_dict = {}
    for k in dictionary:
        random.seed(seed)
        shuffled_dict[k] = random.sample(dictionary[k], len(dictionary[k]))

    return shuffled_dict

def get_semantic_kitti_dataset(path, is_training=True, sequences=None, shuffle=False):

    # semantic Kitti basedir. TODO: decide if add this to parameters of the function

    # sem_kitti_basedir = 'dataset/SemanticKITTI/dataset/sequences'
    sem_kitti_basedir = ''
    train_sequences = ["{:02}".format(i) for i in range(11)]  # according to semantic-kitti.yaml
    train_sequences.remove('08')
    # test_sequences = ["{:02}".format(i) for i in range(11, 22, 1)]
    test_sequences = ["08"]

    # initialize dataset
    dataset = {'imgs': [], 'calib': [], 'gt': [], 'gt_bev': [], 'gt_front': [], 'lodnn_gt': [], 'pc': [], 'labels': []}

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
            imgs_list = fetch_file_list(os.path.join(path, sem_kitti_basedir, s, imgs_dir), 'png')
            pc_list = fetch_file_list(os.path.join(path, sem_kitti_basedir, s, pc_dir), 'bin')
            label_list = fetch_file_list(os.path.join(path, sem_kitti_basedir, s, label_dir), 'label')
            gt_bev_list = fetch_file_list(os.path.join(path, sem_kitti_basedir, s, gt_bev_dir), 'png')
            gt_front_list = fetch_file_list(os.path.join(path, sem_kitti_basedir, s, gt_front_dir), 'png')
            calib_list = len(pc_list) * [os.path.join(path, sem_kitti_basedir, s) + '/calib.txt']

            # the first 80% of the cases are putted in the train the remaining 20% are used for validation
            num_frames_in_seq = len(pc_list)
            n_frames_in_train = np.floor(num_frames_in_seq * 0.8).astype(int)

            if s in train_sequences:
                trainset['imgs'] += imgs_list[:n_frames_in_train]
                trainset['pc'] += pc_list[:n_frames_in_train]
                trainset['labels'] += label_list[:n_frames_in_train]
                trainset['gt_bev'] += gt_bev_list[:n_frames_in_train]
                trainset['gt_front'] += gt_front_list[:n_frames_in_train]
                trainset['calib'] += calib_list[:n_frames_in_train]

                validset['imgs'] += imgs_list[n_frames_in_train:]
                validset['pc'] += pc_list[n_frames_in_train:]
                validset['labels'] += label_list[n_frames_in_train:]
                validset['gt_bev'] += gt_bev_list[n_frames_in_train:]
                validset['gt_front'] += gt_front_list[n_frames_in_train:]
                validset['calib'] += calib_list[n_frames_in_train:]
        if shuffle:
            trainset = shuffle_dict(trainset, seed=1)
            validset = shuffle_dict(validset, seed=2)

        return trainset, validset
    else:
        for s in sequences:
            imgs_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, imgs_dir) + '/*.png'))
            pc_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, pc_dir) + '/*.bin'))
            label_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, label_dir) + '/*.label'))
            gt_bev_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_bev_dir) + '/*.png'))
            gt_front_list = sorted(glob(os.path.join(path, sem_kitti_basedir, s, gt_front_dir) + '/*.png'))
            calib_list = len(pc_list) * [os.path.join(path, sem_kitti_basedir, s) + '/calib.txt']

            dataset['imgs'] += imgs_list
            dataset['pc'] += pc_list
            dataset['labels'] += label_list
            dataset['gt_bev'] += gt_bev_list
            dataset['gt_front'] += gt_front_list
            dataset['calib'] += calib_list

        if shuffle:
            dataset = shuffle_dict(dataset, seed=3)

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

def load_label_file(bin_path, instances=False):
    labels = np.fromfile(bin_path, dtype=np.uint32).reshape(-1)

    seg_labels = labels & 0xFFFF

    if instances:
        inst = labels >> 16
        return seg_labels, inst

    return seg_labels

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


def process_iter(iterable, func, **kwargs):
    """
    This is an auxiliary function to call func over an iterable of any objects in multiprocess.
    Basically each element of the iterable needs to contains the parameters for the function.
    All the parameters of the function that remains constant for all the elements in the iterable can passed through
    the dict kwargs and they will be fixed at the beginning of the call.
    Basically this is a generalization of the process_list function above to the case of multi parameters to treat.

    Parameters
    ----------
    iterable: iter
        iterable of parameters to treat

    func: function
        function to call

    kwargs: dict
        dictionary of fixed elements in the call

    Return
    ------
    results: list
        list containing the results of the function func to each element in iter
    """

    nCPU = mp.cpu_count()
    print('nCPUs = ' + repr(nCPU))
    pool = mp.Pool(processes=nCPU)
    from functools import partial
    if len(kwargs):
        func = partial(func, **kwargs)
    func = partial(func, **kwargs)
    result = pool.starmap(func, iterable)
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

def load_h5_file(path, variables=None):
    """
    Function that loads h5 file from disk

    Parameters
    ----------
    path: str
        path to the h5 file

    variables: slice
        slice or list of index to use to select only a given number of channels in the output array

    Returns
    -------
    data: ndarray
        loaded array

    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf["array"])

    hf.close()

    if variables is not None:
        return data[..., variables]
    else:
        return data

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

def load_pyntcloud(filename, add_label=False, instances=False):
    """
    Parameters
    ----------
    filename: str
        path to pointcloud to read

    add_label: bool
        if True it adds also label to point cloud

    instances: bool
        if True it add also instances labels to point cloud

    Returns
    -------
    cloud: PyntCloud
        output pointcloud
    """
    points = load_bin_file(filename)
    cloud = PyntCloud(pd.DataFrame(points, columns=['x', 'y','z', 'i']))
    if add_label:
        labels = load_label_file(filename.replace('velodyne', 'labels').replace('bin', 'label'), instances=instances)
        if instances:
            cloud.points['labels'] = labels[0]
            cloud.points['instances'] = labels[1]
        else:
            cloud.points['labels'] = labels

    return cloud

def load_semantic_kitti_config(filename=""):
    """
    Function that load configuration file of semantic kitti dataset
    Parameters
    ----------
    filename: str
        file to load
    Returns
    -------
    config: dict
    """
    if len(filename) == 0:
        local_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(local_path, '..', 'semantic-kitti-api','config','semantic-kitti.yaml')

    with open(filename) as f:
        config = yaml.safe_load(f)

    return config


def load_filter_cloud(path, side_range=(-10, 10), fwd_range=(6, 46), height_range=(-4, 4), add_label=True, add_layers=False):
    """

    """
    cloud = load_pyntcloud(path, add_label=add_label)
    if add_layers:
        layers = extract_layers(cloud.xyz)
        cloud.points['layers'] = layers

    filt_points = filter_points(points=cloud.points.values, side_range=side_range, fwd_range=fwd_range,
                                height_range=height_range)

    filt_cloud = PyntCloud(pd.DataFrame(filt_points, columns=list(cloud.points)))

    if 'labels' in list(filt_cloud.points):
        road_labels = filt_cloud.points['labels'] == 40
        filt_cloud.points['road'] = road_labels

    return filt_cloud


def extract_layers(points, max_layers=64):
    """
    Function that retrieve the layer for each point. We do the hypothesis that layer are stocked one after the other.
    And each layer is stocked in a clockwise (or anticlockwise) fashion.

    """
    from skimage.morphology import opening, rectangle
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

    # check if we have retrieved all the layers
    if len(intervals) < max_layers:
        el = intervals.pop(0)
        # in case not we are going to explore again the vector of thetas on the initial part
        thex = np.copy(thetas[:el[1]])
        # we compute again the diffs between consecutive angles and we mark each time we have a negative difference
        diffs = np.ediff1d(thex)
        idx = diffs < 0
        ints = np.arange(len(idx))[idx]
        # the negative differences mark the end of a layer and the beginning of another
        new_intervals = []
        max_new_ints = min(len(ints), max_layers - len(intervals))
        for i in range(max_new_ints):
            if i == 0:
                new_intervals.append([0, ints[i]])
            elif i == max_new_ints - 1:
                new_intervals.append([ints[i], el[1]])
            else:
                new_intervals.append([ints[i], ints[i + 1]])
        intervals = new_intervals + intervals

    # for each element in interval we assign a label that identifies the layer
    layers = np.zeros(len(thetas), dtype=np.uint8)

    for n, el in enumerate(intervals[::-1]):
        layers[el[0]:el[1]] = max_layers - (n + 1)

    return layers