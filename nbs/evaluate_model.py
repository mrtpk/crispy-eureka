
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
# from helpers import data_loaders as dls
# from helpers.viz import plot, plot_history
# from helpers.logger import Logger
import utils
import dl_models
import keras # this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as k
from keras.optimizers import *

def load_dataset(add_geometrical_features=True,
                  subsample_flag=True,
                  compute_HOG=False):
    PATH = '../'  # path of the repo.
    _NAME = 'experiment0'  # name of experiment

    # create dataclass
    KPC = utils.KittiPointCloudClass(dataset_path=PATH,
                                     add_geometrical_features=add_geometrical_features,
                                     subsample=subsample_flag,
                                     compute_HOG=compute_HOG)
    # number of channels in the images
    n_channels = 6
    if add_geometrical_features:
        n_channels += 3
    if compute_HOG:
        n_channels += 6

    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset(limit_index = -1)

    return f_test, gt_test, n_channels




def get_info_from_test_name(filename):
    '''
    Functions that reads the details.json file and return the number of features used during training, and if
    the point cloud has been sampled
    '''
    with open(filename) as f:
        d = json.load(f)

    test_name = d['test_name'].lower()

    geometric = False
    if 'geometric' in test_name:
        geometric = True

    hog = False
    if 'hog' in test_name:
        hog = True

    sampled = False
    if 'sampled' in test_name:
        sampled = True
    print(d)

    return geometric, hog, sampled, d['training_config']

def initialize_model(model_name, weights, training_config, shape):
    # initialize model
    if model_name == 'lodnn':
        model = dl_models.get_lodnn_model(shape=shape)
    elif model_name == 'unet':
        model = dl_models.get_unet_model(input_size=shape)
    else:
        raise ValueError("model_name nor lodnn neither unet")

    model.summary()

    model.load_weights(weights)

    # retrieve optimizer
    optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])

    # compile model
    model.compile(loss=training_config["loss_function"],
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def evaluate_model(model_name, weights):
    # getting dir
    weights_dir = os.path.abspath(os.path.dirname(weights))

    # getting parend directory
    weights_par_dir = os.path.abspath(os.path.join(weights_dir, os.pardir))

    # retrieve basic info on trained model
    geometric, hog, sampled, training_config = get_info_from_test_name(os.path.join(weights_par_dir, 'details.json'))
    print(training_config)

    f_test, gt_test, n_channels = load_dataset(add_geometrical_features=geometric,
                                   subsample_flag=sampled,
                                   compute_HOG=hog)

    # Test
    test_datagen = ImageDataGenerator() #horizontal_flip=True #add this ?
    test_iterator = test_datagen.flow(f_test, gt_test,
                    batch_size=1, shuffle=True)

    model = initialize_model(model_name, weights, training_config, (400, 200, n_channels))

    all_pred = model.predict_generator(test_iterator,
                                       steps=f_test.shape[0])
    all_pred = np.array(all_pred)

    for n in range(len(all_pred)):
        f, ax = plt.subplots(1,4)
        ax[0].imshow(gt_test[n, :,:, 0])
        ax[0].set_title('gt_{}'.format(n))
        ax[1].imshow(all_pred[n, :, :, 0])
        ax[1].set_title('pred_0_{}'.format(n))
        ax[2].imshow(all_pred[n, :, :, 1])
        ax[2].set_title('pred_1_{}'.format(n))
        ax[3].imshow(1 - np.argmax(all_pred[n, :, :, :], axis=2))
        ax[3].set_title('pred_argmax_{}'.format(n))
        plt.show()

    return all_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--model_weights', type=str, help='path to model weights')
    parser.add_argument('--model', default='lodnn', type=str, help='architecture to use for evaluation')
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Tensorflow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.85

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    evaluate_model(args.model, args.model_weights)
