# %matplotlib inline
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
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

def parse_config_file(config_filename):
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

    """
    with open(config_filename) as f:
        config = json.load(f)

    model = config.get('model', 'lodnn')
    dataset = config.get('dataset', 'kitti')
    view = config.get('view', 'bev')
    parameters = config.get('parameters', None)
    sequences = config.get('sequences', None)
    gt_args = config.get('gt_args', None)

    if sequences is not None:
        sequences = ['{:02}'.format(i) for i in sequences]

    feature_parameters = dict(
        compute_classic=str2bool(parameters.get('compute_classic', 0)),
        add_geometrical_features=str2bool(parameters.get('add_geometrical_features', 0)),
        subsample_ratio=parameters.get('subsample_ratio', 1),
        compute_eigen=parameters.get('compute_eigen', 0)
    )

    return model, dataset, view, sequences, feature_parameters, gt_args

def initialize_model(modelname, shape, subsample_ratio=1):
    if modelname == 'lodnn':
        model = dl_models.get_lodnn_model(shape=shape)
    elif modelname == 'unet':
        model = dl_models.get_unet_model(shape=shape,
                                         subsample_ratio=subsample_ratio)

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
    return v.lower() in ("yes", "true", "t", "1")

def get_test_name(features_parameters):
    """
    Function that return test name based on the features parameters

    Parameters
    ----------
    features_parameters: dict
        dictionary containing all the possible configuration on the test

    Return
    ------
    test_name: str
        name of the test file

    """
    test_name = 'Classical_' if features_parameters['compute_classic'] else ''
    test_name += 'Geometric_' if features_parameters['add_geometrical_features'] else ''
    test_name += 'Eigen_' if features_parameters['compute_eigen'] else ''

    if features_parameters['subsample_ratio'] == 2:
        test_name += 'Subsampled_32'

    if features_parameters['subsample_ratio'] == 4:
        test_name += 'Subsample_16'

    return test_name


def train_model(modelname, feature_parameters, gt_args, view, dataset, sequences=None):
    # todo: write a function that
    kpc = KITTIPointCloud(feature_parameters=feature_parameters,
                          is_training=True,
                          sequences=sequences,
                          view=view,
                          dataset=dataset)
    f_train, gt_train, f_valid, gt_valid = kpc.load_dataset(limit_index=-1, **gt_args)

    NAME = 'experiment0'  # name of experiment
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct(kpc.dataset_path, NAME)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')
    # paths
    run_id = modelname + '_' + utils.get_unique_id()
    path = utils.create_run_dir(_RUN_PATH, run_id)
    callbacks = utils.get_basic_callbacks(path)

    # All training params to be added here
    training_config = {
        "loss_function": "binary_crossentropy",
        "learning_rate": 1e-4,
        "batch_size": 3,
        "epochs": 100,
        "optimizer": "keras.optimizers.Nadam"  # "keras.optimizers.Nadam"
    }

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

    model = initialize_model(modelname, f_train.shape[1:], subsample_ratio=feature_parameters.get('subsample_ratio', 1))

    # Add more callbacks
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    # callbacks = callbacks + [early_stopping, reduce_lr]

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
    plot_history(m_history)

    test_name = get_test_name(feature_parameters)

    png_name = '{}.png'.format(path + '/' + test_name)

    plt.savefig(png_name)
    plt.close()

    result = {"name": NAME,
              "test_name": test_name,
              "run_id": run_id,
              "dataset": "KITTI",
              "training_config": training_config,
              "z_min": str(kpc.z_min),
              "z_max": str(kpc.z_max),
              }
    if view == 'bev':
        result['COUNT_MAX'] = str(kpc.COUNT_MAX)
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--config_file', default="", type=str, help='config file')
    parser.add_argument('--store', action='store_true')

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

    model, dataset, view, sequences, feature_parameters, gt_args = parse_config_file(config_file)
    train_model(model, feature_parameters, gt_args, view, dataset, sequences)
