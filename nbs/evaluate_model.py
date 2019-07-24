import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import utils
import dl_models
from helpers.timer import timer
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

    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset(limit_index = -1)

    return f_test, gt_test


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
    elif model_name == 'unet6':
        model = dl_models.u_net6(shape=shape,
                                 filters=512,
                                 int_space=32,
                                 output_channels=2)
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

@timer
def predict_sample(model, sample):
    prediction = model.predict(sample)
    return prediction[0, :, :, :]

def predict_test_set(model, f_test):
    all_pred = []

    for f in f_test:
        prediction = predict_sample(model, np.expand_dims(f, axis=0))
        all_pred.append(prediction)

    all_pred = np.array(all_pred)

    return all_pred

def compute_scores(pred, gt):

    all_f1, all_recall, all_prec, all_acc = [], [], [], []

    for n in range(len(pred)):

        f1, recall, precision, acc = utils.get_metrics(gt=gt[n, :, :, 0], pred=utils.apply_argmax(pred[n, :, :, :]))
        print("Scores for sample {}: F1 -> {}, Recall -> {}, Precision -> {}, Accuracy -> {}".format(n, f1, recall, precision, acc))
        all_f1.append(f1)
        all_recall.append(recall)
        all_prec.append(precision)
        all_acc.append(precision)

    print('------------------------------------------------')

    scores = {'f1': all_f1,
              'recall': all_recall,
              'precision': all_prec,
              'acc': all_acc}

    return scores

def evaluate_model(model_name, weights):
    # getting dir
    weights_dir = os.path.abspath(os.path.dirname(weights))

    # getting parend directory
    weights_par_dir = os.path.abspath(os.path.join(weights_dir, os.pardir))

    # retrieve basic info on trained model
    geometric, hog, sampled, training_config = get_info_from_test_name(os.path.join(weights_par_dir, 'details.json'))
    print(training_config)

    f_test, gt_test = load_dataset(add_geometrical_features=geometric,
                                   subsample_flag=sampled,
                                   compute_HOG=hog)


    # image shape
    n_row, n_col, n_channels = f_test.shape[1:]

    if 'unet' in model_name:
        multiple_of = 16 if model_name =='unet' else 32
        nrpad =  multiple_of - n_row % multiple_of if n_row % multiple_of != 0 else 0
        ncpad =  multiple_of - n_col % multiple_of if n_col % multiple_of != 0 else 0

        f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')

    # initializing model
    model = initialize_model(model_name, weights, training_config, (n_row, n_col, n_channels))

    pred = predict_test_set(model, f_test)

    if 'unet' in model_name:
        scores = compute_scores(pred[:, :n_row, :n_col, :], gt_test)
    else:
        scores = compute_scores(pred, gt_test)

    for n in range(len(pred)):
        f, ax = plt.subplots(1, 4)
        ax[0].imshow(gt_test[n, :, :, 0])
        ax[0].set_title('gt_{}'.format(n))
        ax[1].imshow(pred[n, :, :, 0])
        ax[1].set_title('pred_0_{}'.format(n))
        ax[2].imshow(pred[n, :, :, 1])
        ax[2].set_title('pred_1_{}'.format(n))
        ax[3].imshow(1 - np.argmax(pred[n, :, :, :], axis=2))
        ax[3].set_title('pred_argmax_{}'.format(n))
        plt.show()

    return scores

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
