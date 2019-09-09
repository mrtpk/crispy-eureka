import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import utils
import dl_models
from helpers.timer import timer
from helpers.viz import plot_confusion_matrix, ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, jaccard_score, recall_score, precision_score, f1_score
import keras # this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as k
from keras.optimizers import *
from keras_custom_loss import jaccard2_loss

def remove_bg(gt_train):
    gt_train = gt_train[:,:,:,0]
    gt_train = gt_train[:,:,:,np.newaxis]
    return gt_train

def load_dataset(add_geometrical_features=True,
                 subsample_flag=True,
                 compute_HOG=False,
                 view='bev',
                 dataset='kitti',
                 subsample_ratio=1):

    PATH = '../'  # path of the repo.
    _NAME = 'experiment0'  # name of experiment
    sequences = None
    if dataset != 'kitti':
        sequences =['04', '08']
    # create dataclass
    KPC = utils.KittiPointCloudClass(dataset_path=PATH,
                                     add_geometrical_features=add_geometrical_features,
                                     subsample=subsample_flag,
                                     compute_HOG=compute_HOG,
                                     view=view,
                                     subsample_ratio=subsample_ratio,
                                     dataset=dataset,
                                     sequences=sequences)

    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset(limit_index=-1)
    gt_test = remove_bg(gt_test)
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
    sampled_ratio = 1
    if 'sampled' in test_name:
        sampled = True
        if '32' in test_name:
            sampled_ratio = 2
        if '16' in test_name:
            sampled_ratio = 4
    print(d)

    return geometric, hog, sampled, sampled_ratio, d['training_config']


def initialize_model(model_name, weights, training_config, shape, subsample_ratio=1):
    # initialize model
    if model_name == 'lodnn':
        model = dl_models.get_lodnn_model(shape=shape)
    elif model_name == 'unet':
        model = dl_models.get_unet_model(input_size=shape, subsample_ratio=subsample_ratio)
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
    model.compile(loss=jaccard2_loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


@timer
def predict_sample(model, sample):
    prediction = model.predict(sample)
    return prediction[0, :, :, :]


def predict_test_set(model, f_test):
    all_pred = []
    all_times =[]
    for f in f_test:
        prediction, time = predict_sample(model, np.expand_dims(f, axis=0))
        all_pred.append(prediction)
        all_times.append(time)

    all_pred = np.array(all_pred)
    all_times = np.array(all_times)
    print("Mean prediction time: {}. Std prediction time: {}".format(all_times.mean(), all_times.std()))

    return all_pred, all_times


def compute_scores(pred, gt, threshold=0.5):

    gt_all = gt[:,:,:,0].flatten()
    # argmax_pred = 1 - np.argmax(pred, axis=3)
    # argmax_pred = argmax_pred.flatten()
    argmax_pred = (pred > threshold).flatten()
    # f1, recall, precision, acc, jaccard = utils.get_metrics(gt=gt_all, pred=argmax_pred)
    acc = accuracy_score(gt_all, argmax_pred)
    recall = recall_score(gt_all, argmax_pred)
    precision = precision_score(gt_all, argmax_pred)
    f1 = f1_score(gt_all, argmax_pred)
    jaccard = jaccard_score(gt_all, argmax_pred)

    print("Scores: "
          "F1 -> {},"
          " Recall -> {},"
          " Precision -> {},"
          " Accuracy -> {},"
          " Jaccard -> {}".format(f1, recall, precision, acc, jaccard))


    print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores


def evaluate_model(model_name, weights, view, plot_result_flag, dataset):
    # getting dir
    weights_dir = os.path.abspath(os.path.dirname(weights))

    # getting parend directory
    weights_par_dir = os.path.abspath(os.path.join(weights_dir, os.pardir))

    # retrieve basic info on trained model
    geometric, hog, sampled, subsample_ratio, training_config = get_info_from_test_name(os.path.join(weights_par_dir, 'details.json'))
    print(training_config)

    f_test, gt_test = load_dataset(add_geometrical_features=geometric,
                                   subsample_flag=sampled,
                                   compute_HOG=hog,
                                   view=view,
                                   subsample_ratio=subsample_ratio,
                                   dataset=dataset)

    print("Test set shape {}".format(f_test.shape))
    print("GT set shape {}".format(gt_test.shape))

    # image shape
    n_row, n_col, n_channels = f_test.shape[1:]

    if 'unet' in model_name:
        multiple_of = 16 if model_name =='unet' else 32
        nrpad = multiple_of - (n_row % multiple_of) if (n_row % multiple_of) != 0 else 0
        ncpad = multiple_of - (n_col % multiple_of) if (n_col % multiple_of) != 0 else 0

        f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')

    # initializing model
    model = initialize_model(model_name, weights, training_config, f_test.shape[1:], subsample_ratio=subsample_ratio)

    pred, times = predict_test_set(model, f_test)

    pred = pred[:, :(subsample_ratio*n_row), :n_col, :]  # setting same dimension as before

    print(pred.shape, gt_test.shape)
    if 'unet' in model_name:
        scores = compute_scores(pred[:, :(subsample_ratio*n_row), :n_col, :], gt_test)
    else:
        scores = compute_scores(pred, gt_test)

    if plot_result_flag:
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
    print(gt_test.shape)
    output = np.zeros_like(gt_test[:, :, :, 0])
    print(pred.shape)
    for i in range(pred.shape[0]):
        output[i] = 1 - np.argmax(pred[i], axis=2)

    cm = confusion_matrix(gt_test[:, :, :, 0].flatten(), output.flatten())
    plot_confusion_matrix(cm, ['not_road', 'road'], normalize=True)
    conf_mat = ConfusionMatrix(number_of_labels=2)
    conf_mat.confusion_matrix = cm
    print("Overall Accuracy: {}".format(conf_mat.get_overall_accuracy()))
    print("IoU: {}".format(conf_mat.get_average_intersection_union()))

    return f_test, gt_test, model, scores, pred, times, conf_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--model_weights', type=str, help='path to model weights')
    parser.add_argument('--model', default='lodnn', type=str, help='architecture to use for evaluation')
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--plot_result_flag', default=False, help='flag to show images of road segmentation')
    parser.add_argument('--view', type=str, default='bev', help='projection to use')
    parser.add_argument('--dataset', default='kitti', type=str, help='Dataset to use for training')

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

    evaluate_model(args.model, args.model_weights, args.view, args.plot_result_flag, args.dataset)
