# %matplotlib inline
import os
import gc
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import keras  # this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from keras import backend as k
from keras_custom_loss import binary_focal_loss  # weightedLoss2
# cyclic lr
import keras_contrib
from helpers.config_parser import JSONRunConfig
# import helpers.generate_gt as generate_gt
from helpers import data_loaders as dls
from helpers.viz import plot_history  # plot
from helpers.logger import Logger

import utils
from generator import generator_from_h5, DataLoaderGenerator, load_h5_file
from data import KITTIPointCloud
import dl_models

RESCALE_VALUES = {
    'z_min': np.inf,
    'z_max': -np.inf,
    'COUNT_MAX': -1
}


def save_array_on_h5_file(array, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('array', data=array)
    h5f.close()


def generate_features_map(config_run):
    """
    Function that compute features maps and ground truth of the train and validation sets

    Parameters
    ----------
    config_run: dict
        Dictionary containing all the parameters concerning the run

    Returns
    -------
    f_train: ndarray
        Features maps of the training set
    gt_train: ndarray
        Ground Truth of the training set
    f_valid: ndarray
        Features maps of the validation set
    gt_valid: ndarray
        Ground truth of the validation set
    """
    basedir = config_run['features_basedir']
    features_parameters = config_run.get('features')
    gt_args = config_run.get('gt_args', None)
    dataset = config_run.get('dataset')
    view = config_run.get('view')
    path = config_run.get('path', '')
    sequences = config_run.get('sequences')
    # initialize KITTIPointCloud class
    kpc = KITTIPointCloud(feature_parameters=features_parameters,
                          is_training=True,
                          path=path,
                          sequences=sequences,
                          view=view,
                          dataset=dataset)
    RESCALE_VALUES['z_min'] = kpc.z_min
    RESCALE_VALUES['z_max'] = kpc.z_max
    if view == 'bev':
        RESCALE_VALUES['COUNT_MAX'] = kpc.COUNT_MAX

    # compute the maps from the dataset
    split_size = 200 # todo parametrize this parameter

    n_sample_train = len(kpc.train_set['pc'])
    # before running feature generator we verify if path exists and if we need we generate it
    os.makedirs(os.path.join(basedir, 'train', 'img'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'train', 'gt'), exist_ok=True)

    for i in range(0, n_sample_train, split_size):

        max_id = min(n_sample_train, i + split_size)
        if gt_args is None:
            f_train, gt_train = kpc.load_dataset(set_type='train', min_id=i, max_id=max_id)
        else:
            f_train, gt_train = kpc.load_dataset(set_type='train', min_id=i, max_id=max_id, **gt_args)

        filename = os.path.join(basedir, 'train', 'img', '{:03}.h5'.format(i // split_size))
        save_array_on_h5_file(f_train, filename)
        filename = os.path.join(basedir, 'train', 'gt', '{:03}.h5'.format(i // split_size))
        save_array_on_h5_file(gt_train, filename)

        gc.collect()

    n_sample_valid = len(kpc.valid_set['pc'])
    os.makedirs(os.path.join(basedir, 'valid', 'img'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'valid', 'gt'), exist_ok=True)

    for i in range(0, n_sample_valid, split_size):

        max_id = min(n_sample_valid, i + split_size)

        if gt_args is None:
            f_valid, gt_valid = kpc.load_dataset(set_type='valid', min_id=i, max_id=max_id)
        else:
            f_valid, gt_valid = kpc.load_dataset(set_type='valid', min_id=i, max_id=max_id, **gt_args)
        filename = os.path.join(basedir, 'valid', 'img', '{:03}.h5'.format(i // split_size))
        save_array_on_h5_file(f_valid, filename)
        filename = os.path.join(basedir, 'valid', 'gt', '{:03}.h5'.format(i // split_size))
        save_array_on_h5_file(gt_valid, filename)

        gc.collect()

    np.savez(os.path.join(basedir, 'rescale_values.npz'), **RESCALE_VALUES)


def get_shape_and_variables(config_run):

    view = config_run.get('view')
    features = config_run.get('features', None)
    features_idx = []
    n_channels = 0
    if features['compute_classic']:
        if view == 'bev':
            n_channels += 6
            features_idx = np.concatenate((features_idx, np.arange(6))).astype(int)
        else:
            n_channels += 3
            features_idx = np.concatenate((features_idx, np.arange(3))).astype(int)

    if features['add_geometrical_features']:
        n_channels += 3
        features_idx = np.concatenate((features_idx, np.arange(3, 6))).astype(int)

    if features['compute_eigen']:
        n_channels += 6
        features_idx = np.concatenate((features_idx, np.arange(6, 12))).astype(int)

    print("Number of features: {}".format(len(features_idx)))

    return features_idx, n_channels


def initialize_model(modelname, shape, subsample_ratio=1, output_channels=1):
    """
    Function that initialize model given its model name

    Parameters
    ----------
    modelname: optional {'lodnn', 'unet', 'unet6', 'squeeze'}

    shape: tuple
        input shape
    subsample_ratio: int
        ratio to used for subsampling

    Returns
    -------
    model:
        compiled keras model

    """
    if modelname == 'lodnn':
        model = dl_models.get_lodnn_model(shape=shape)
    elif modelname == 'unet':
        model = dl_models.get_unet_model(shape=shape,
                                         subsample_ratio=subsample_ratio,
                                         output_channels=output_channels)

    elif modelname == 'unet6':
        model = dl_models.u_net6(shape=shape,
                                 filters=512,
                                 int_space=32,
                                 output_channels=2)
    elif modelname == 'squeeze':
        # TODO this is not a squeezeSeg network, it is a simple classifier (encoder part only)
        model = dl_models.SqueezeNet(2, input_shape=shape)
    else:
        raise ValueError("Acceptable values for model parameter are 'lodnn', 'unet', 'unet6'.")

    model.summary()

    return model


def run_training(json_run):
    """
    Function used to train a DL model

    Parameters
    ----------
    json_run:
    """

    config_run = json_run.get_config_run()
    basedir = config_run['features_basedir']
    training_config = config_run.get('training_config')
    features = config_run.get('features')
    output_channels = config_run.get('output_channels')
    shape = config_run.get('shape')
    subsample_ratio = features['subsample_ratio']
    test_name = json_run.get_test_name()
    print("Test Name: ", test_name)
    view = json_run.view
    dataset = json_run.dataset
    modelname = json_run.model

    print(RESCALE_VALUES)
    print("Running ", test_name)
    experiment_name = 'experiment0'  # name of experiment
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct('../', experiment_name)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')
    # paths
    run_id = modelname + '_' + utils.get_unique_id()
    path = utils.create_run_dir(_RUN_PATH, run_id)
    callbacks = utils.get_basic_callbacks(path)

    traindir = os.path.join(basedir, 'train')
    validdir = os.path.join(basedir, 'valid')
    variables, n_channels = get_shape_and_variables(config_run)
    shape = shape + (n_channels, )

    # # train_iterator = generator_from_h5(traindir, variables=variables, batch_size=training_config["batch_size"])
    train_iterator = DataLoaderGenerator(traindir, variables=variables, batch_size=training_config["batch_size"], n_samples_per_file=200)
    # valid_iterator = generator_from_h5(validdir, variables=variables, batch_size=training_config["batch_size"], jump_after=300)
    valid_iterator = DataLoaderGenerator(validdir, variables=variables, batch_size=training_config["batch_size"], n_samples_per_file=200)

    # X_valid = keras.utils.io_utils.HDF5Matrix(os.path.join(validdir, 'all_img.h5'), 'array')
    # y_valid = keras.utils.io_utils.HDF5Matrix(os.path.join(validdir, 'all_gt.h5'), 'array')
    model = initialize_model(modelname, shape, subsample_ratio=subsample_ratio, output_channels=output_channels)

    # Add more callbacks
    clr_custom = keras_contrib.callbacks.CyclicLR(base_lr=0.0001, max_lr=0.01,
                                                  mode='triangular', gamma=0.99994,
                                                  step_size=120000)

    callbacks = callbacks + [clr_custom]
    model.compile(loss=binary_focal_loss(),
                  optimizer=eval(training_config["optimizer"])(lr=training_config["learning_rate"]),
                  metrics=['accuracy'])

    m_history = model.fit_generator(train_iterator,
                                    epochs=training_config['epochs'],
                                    # steps_per_epoch=300,
                                    verbose=1,
                                    callbacks=callbacks,
                                    validation_data=valid_iterator,
                                    # validation_steps=100,
                                    use_multiprocessing=True,
                                    workers=8
                                    )

    model.save("{}/model/final_model.h5".format(path))
    # plot_history(m_history)
    #
    # png_name = '{}.png'.format(path + '/' + test_name)
    #
    # plt.savefig(png_name)
    # plt.close()

    result = {"name": experiment_name,
              "test_name": test_name,
              "run_id": run_id,
              "dataset": dataset,
              "features": features,
              "training_config": training_config,
              "z_min": str(RESCALE_VALUES['z_min']),
              "z_max": str(RESCALE_VALUES['z_max']),
              }
    if view == 'bev':
        result['COUNT_MAX'] = str(RESCALE_VALUES['COUNT_MAX'])
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(result, f)

    print("Model Trained")


def initialize_session(cuda_device='0', gpu_memory_fraction=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    print("GPU id: {}".format(cuda_device))
    # Tensorflow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    if gpu_memory_fraction < 1.0:
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    return run_opts


def train_model(config_filename, path, compute_features=False, cuda_device='0', gpu_memory_fraction=1.0):


    json_run = JSONRunConfig(config_filename)
    config_run = json_run.get_config_run()
    config_run['path'] = path
    # load the features map for the current experiment
    features_basedir = config_run.get('features_basedir')
    # checking if the features_map for this set of experiments has already been computed
    exists = os.path.isfile(os.path.join(features_basedir, 'rescale_values.npz'))

    if not exists or compute_features:  # if not we compute and store them
        generate_features_map(config_run)

    if exists:
        rescale = np.load(os.path.join(features_basedir, 'rescale_values.npz'))

        RESCALE_VALUES['z_min'] = float(rescale['z_min'])
        RESCALE_VALUES['z_max'] = float(rescale['z_max'])
        RESCALE_VALUES['COUNT_MAX'] = int(rescale['COUNT_MAX'])

    experiments = config_run.get('experiments')

    initialize_session(cuda_device, gpu_memory_fraction)

    if experiments is None:
        run_training(json_run)
    else:
        # iterate over all possible experiments and run them
        for e in experiments:
            print("Loading experiment ", e)
            config_exp = json_run.update_config_run(e)
            print(config_exp)
            print(json_run)
            # features_maps = load_features_maps_from_file(features_basedir, features, view)
            # run current experiment
            run_training(json_run)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--path', default='../', type=str, help='path to dataset')
    parser.add_argument('--config_file', default="", type=str, help='config file')
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--gpu_fraction', default=1.0, type=float, help='Fraction of GPU memory to allocate')

    args = parser.parse_args()

    config_file = args.config_file
    force_compute_features = args.features

    train_model(config_file, args.path, force_compute_features,
                cuda_device=args.cuda_device, gpu_memory_fraction=args.gpu_fraction)
