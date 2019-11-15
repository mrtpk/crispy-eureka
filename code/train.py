# %matplotlib inline
import os
import argparse
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

# import helpers.generate_gt as generate_gt
from helpers import data_loaders as dls
from helpers.viz import plot_history  # plot
from helpers.logger import Logger
import utils
from data import KITTIPointCloud
import dl_models

RESCALE_VALUES = {
    'z_min': np.inf,
    'z_max': -np.inf,
    'COUNT_MAX': -1
}


def get_features_map(config_run, force_compute):
    """
    Parameters
    ----------
    config_run: dict
    force_compute: bool

    Returns
    -------
    """
    if force_compute:
        f_train, gt_train, f_valid, gt_valid = generate_features_map(config_run)
    else:
        features_parameters = config_run['features']
        view = config_run['view']
        basedir = config_run['features_basedir']
        exists = os.path.isfile(os.path.join(basedir, 'gt_train.npz'))

        if exists:
            f_train, gt_train, f_valid, gt_valid = load_features_maps_from_file(basedir, features_parameters, view)
        else:
            f_train, gt_train, f_valid, gt_valid = generate_features_map(config_run)
            store_features_map(basedir, (f_train, gt_train, f_valid, gt_valid))

    return f_train, gt_train, f_valid, gt_valid


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

    features_parameters = config_run.get('features')
    gt_args = config_run.get('gt_args', {})
    dataset = config_run.get('dataset')
    view = config_run.get('view')
    sequences = config_run.get('sequences')
    # initializet KITTIPointCloud class
    kpc = KITTIPointCloud(feature_parameters=features_parameters,
                          is_training=True,
                          sequences=sequences,
                          view=view,
                          dataset=dataset)

    # compute the maps from the dataset
    f_train, gt_train, f_valid, gt_valid = kpc.load_dataset(limit_index=-1, **gt_args)

    RESCALE_VALUES['z_min'] = kpc.z_min
    RESCALE_VALUES['z_max'] = kpc.z_max
    if view == 'bev':
        RESCALE_VALUES['COUNT_MAX'] = kpc.COUNT_MAX

    return f_train, gt_train, f_valid, gt_valid


def store_features_map(save_path, feature_maps):
    """
    Function that generates and store training and validation set features and ground truth images.

    Parameters
    ----------
    save_path: str

    feature_maps: tuple
        f_train, gt_train, f_valid, gt_valid
    """
    # if save_path dir does not exists we create it
    os.makedirs(save_path, exist_ok=True)
    f_train, gt_train, f_valid, gt_valid = feature_maps

    np.savez(os.path.join(save_path, 'f_train.npz'), **{'data': f_train})
    np.savez(os.path.join(save_path, 'f_valid.npz'), **{'data': f_valid})
    np.savez(os.path.join(save_path, 'gt_train.npz'), **{'data': gt_train})
    np.savez(os.path.join(save_path, 'gt_valid.npz'), **{'data': gt_valid})
    np.savez(os.path.join(save_path, 'rescale_values.npz'), **RESCALE_VALUES)


def load_features_maps_from_file(pathdir, features, view):
    """
    Parameters
    ----------
    pathdir: str
        path to the directory
    features: dict
        dictionary containing all parameters concernign the features to load
    view: option {'bev', 'front'}

    Returns
    -------
    f_train: ndarray
        Features map of training set

    gt_train: ndarray
        Ground truth of the training set

    f_valid: ndarray
        Features map of validation set

    gt_valid: ndarray
        Ground truth of validation set
    """
    print("Loading data from {}".format(pathdir))

    f_train = np.load(os.path.join(pathdir, 'f_train.npz'))['data']
    gt_train = np.load(os.path.join(pathdir, 'gt_train.npz'))['data']
    f_valid = np.load(os.path.join(pathdir, 'f_valid.npz'))['data']
    gt_valid = np.load(os.path.join(pathdir, 'gt_valid.npz'))['data']

    features_idx = []

    if features['compute_classic']:
        if view == 'bev':
            features_idx = np.concatenate((features_idx, np.arange(6))).astype(int)
        else:
            features_idx = np.concatenate((features_idx, np.arange(3))).astype(int)

    if features['add_geometrical_features']:
        features_idx = np.concatenate((features_idx, np.arange(3, 6))).astype(int)

    if features['compute_eigen']:
        features_idx = np.concatenate((features_idx, np.arange(6, 12))).astype(int)

    print("Number of features: {}".format(len(features_idx)))

    f_train = f_train[..., features_idx]
    f_valid = f_valid[..., features_idx]

    rescale = np.load(os.path.join(pathdir, 'rescale_values.npz'))

    RESCALE_VALUES['z_min'] = rescale['z_min']
    RESCALE_VALUES['z_max'] = rescale['z_max']
    RESCALE_VALUES['COUNT_MAX'] = rescale['COUNT_MAX']

    return f_train, gt_train, f_valid, gt_valid


def parse_config_file(config_filename) -> dict:
    """
    helper function that parse the config file and return the paramters for the training
    Parameters
    ----------
    config_filename: str
        config filename

    Returns
    -------
    model: str
        name of the model to use for training

    dataset: option {'kitti', 'semantickitti'}
        name of dataset to use for training

    view: option {'bev', 'front'}

    Returns
    -------
    config_run: dict

    """
    with open(config_filename) as f:
        json_config = json.load(f)

    model = json_config.get('model', 'lodnn')
    dataset = json_config.get('dataset', 'kitti')
    view = json_config.get('view', 'bev')
    parameters = json_config.get('parameters', None)
    sequences = json_config.get('sequences', None)
    gt_args = json_config.get('gt_args', None)
    store_features_basedir = json_config.get('store_features_basedir', None)
    training_config = json_config.get('training_config')
    experiments = json_config.get('experiments', None)
    if sequences is not None:
        sequences = ['{:02}'.format(i) for i in sequences]

    features = dict(
        compute_classic=str2bool(parameters.get('compute_classic', 0)),
        add_geometrical_features=str2bool(parameters.get('add_geometrical_features', 0)),
        subsample_ratio=parameters.get('subsample_ratio', 1),
        compute_eigen=parameters.get('compute_eigen', 0)
    )

    if store_features_basedir is not None:
        layer_dir = str(64 // features['subsample_ratio'])
        features_basedir = os.path.join(store_features_basedir, dataset, view, layer_dir)
    else:
        features_basedir = ''

    config_run = dict(model=model,
                      dataset=dataset,
                      view=view,
                      sequences=sequences,
                      features_basedir=features_basedir,
                      training_config=training_config,
                      experiments=experiments,
                      features=features,
                      gt_args=gt_args)

    return config_run


def generate_feature_config(experiment_name, config_run) -> dict:
    """
    Function that generate the config dictionary to run experiment

    Parameters
    ----------
    experiment_name: str
        name of the experiment for which we want to generate the feature dict

    config_run: dict
        current config_run dictionary containing all the information about the values for the parameters

    Returns
    -------
    d: dict
        dictionary containing the configuration to run the current experiment
    """
    d = copy.deepcopy(config_run)
    base_feat = d.get('features')
    features = dict(
        compute_classic=False,
        add_geometrical_features=False,
        subsample_ratio=base_feat.get('subsample_ratio', 1),
        compute_eigen=0
    )

    if 'classic' in experiment_name.lower():
        features['compute_classic'] = True

    if 'geometry' in experiment_name.lower():
        features['add_geometrical_features'] = True

    if 'eigen' in experiment_name.lower():
        features['compute_eigen'] = base_feat.get('compute_eigen')

    d['features'] = features

    return d


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


def str2bool(v):
    """
    Function that return a boolean from a string.

    Parameters
    ----------
    v: str
        input string

    Returns
    -------
    bool
    """
    return v.lower() in ("yes", "true", "t", "1")


def get_test_name(features):
    """
    Function that return test name based on the features parameters

    Parameters
    ----------
    features: dict
        dictionary containing all the possible configuration on the test

    Return
    ------
    test_name: str
        name of the test file

    """
    test_name = 'Classical' if features['compute_classic'] else ''
    test_name += '_Geometric' if features['add_geometrical_features'] else ''
    test_name += '_Eigen' if features['compute_eigen'] else ''

    if features['subsample_ratio'] == 2:
        test_name += '_Subsampled_32'

    if features['subsample_ratio'] == 4:
        test_name += '_Subsampled_16'

    return test_name


def run_training(features_maps, config_run):
    """
    Function used to train a DL model

    Parameters
    ----------
    features_maps: tuple

    config_run: dict
    """
    f_train, gt_train, f_valid, gt_valid = features_maps
    print("Train set: features map shape --> ", f_train.shape)
    print("Train set: ground truth shape --> ", gt_train.shape)
    print("Validation Set: features map shape --> ", f_valid.shape)
    print("Validation Set: ground truth shape --> ", gt_valid.shape)
    training_config = config_run.get('training_config')
    features = config_run.get('features')
    subsample_ratio = features['subsample_ratio']
    test_name = get_test_name(features)
    view = config_run.get('view')
    dataset = config_run.get('dataset')
    modelname = config_run.get('model')

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

    # this is the augmentation configuration we will use for training
    dict_aug_args = dict(horizontal_flip=True)
    seed = 1
    train_datagen = ImageDataGenerator(**dict_aug_args)
    mask_datagen = ImageDataGenerator(**dict_aug_args)

    # Provide the same seed and keyword arguments to the fit and flow methodsf_
    train_generator = train_datagen.flow(f_train, batch_size=training_config['batch_size'], shuffle=True, seed=seed)
    mask_generator = mask_datagen.flow(gt_train, batch_size=training_config['batch_size'], shuffle=True, seed=seed)

    train_iterator = zip(train_generator, mask_generator)

    # Validation
    seed = 2
    valid_datagen = ImageDataGenerator(**dict_aug_args)
    valid_mask_datagen = ImageDataGenerator(**dict_aug_args)
    valid_generator = valid_datagen.flow(f_valid, batch_size=1, shuffle=True, seed=seed)
    valid_mask_generator = valid_mask_datagen.flow(gt_valid, batch_size=1, shuffle=True, seed=seed)

    valid_iterator = zip(valid_generator, valid_mask_generator)

    model = initialize_model(modelname, f_train.shape[1:], subsample_ratio=subsample_ratio, output_channels=gt_train.shape[-1])

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
                                    steps_per_epoch=int(f_train.shape[0] // training_config["batch_size"]),
                                    verbose=1,
                                    callbacks=callbacks,
                                    validation_data=valid_iterator,
                                    validation_steps=f_valid.shape[0])

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


def train_model(config_filename, compute_features=False):

    config_run = parse_config_file(config_filename)
    # load the features map for the current experiment
    features_basedir = config_run.get('features_basedir')
    # checking if the features_map for this set of experiments has already been computed
    exists = os.path.isfile(os.path.join(features_basedir, 'gt_train.npz'))

    if not exists:  # if not we compute and store them
        features_maps = get_features_map(config_run, compute_features)
    else:
        features_maps = None

    experiments = config_run.get('experiments')

    if experiments is None:
        if features_maps is None:
            features_maps = get_features_map(config_run, compute_features)

        run_training(features_maps, config_run)
    else:
        # iterate over all possible experiments and run them
        for e in experiments:
            print("Loading experiment ", e)
            config_exp = generate_feature_config(e, config_run)
            features = config_exp.get('features')
            view = config_exp.get('view')
            print(config_exp)
            features_maps = load_features_maps_from_file(features_basedir, features, view)
            # run current experiment
            run_training(features_maps, config_exp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--config_file', default="", type=str, help='config file')
    parser.add_argument('--features', action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print(args.cuda_device)
    # Tensorflow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.85

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    config_file = args.config_file
    force_compute_features = args.features

    train_model(config_file, force_compute_features)
